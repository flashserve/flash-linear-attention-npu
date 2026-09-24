from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_chunk_kda_bwd import OP_NAME, PROFILES, validate_profiles

torch.set_num_threads(max(1, int(os.environ.get("KDA_BWD_ATK_CPU_THREADS", "1"))))
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
OUTPUT_NAMES = ("dq", "dk", "dv", "db", "dg", "dh0", "dA", "dbias")


def sequences(spec):
    if spec["mode"] == "dense":
        return [(b, 0, spec["T"], 0) for b in range(spec["B"])]
    result, start, chunk = [], 0, 0
    for length in spec["seqlens"]:
        if length:
            result.append((0, start, start + length, chunk))
        start += length
        chunk += (length + 63) // 64
    return result


def _scores(q, k, cumulative):
    """Natural-log gates; mask before exponentiation to avoid upper overflow."""
    n = q.shape[0]
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    diff = cumulative[:, None, :] - cumulative[None, :, :]
    decay = torch.exp(diff.masked_fill(~causal[..., None], 0)) * causal[..., None]
    return (q[:, None, :] * k[None, :, :] * decay).sum(-1), \
        (k[:, None, :] * k[None, :, :] * decay).sum(-1), decay


def build_inputs(spec, *, exact_caches=False):
    """Internal CPU layout is always B,H,T,D; h is B,Nc,H,K,V."""
    gen = torch.Generator(device="cpu").manual_seed(int(spec["seed"]))
    shape = (spec["B"], spec["H"], spec["T"], 128)

    def randn(dims, std, mean=0., dtype=torch.bfloat16):
        return (torch.randn(dims, generator=gen) * std + mean).to(dtype)

    x = {name: randn(shape, spec["qk_std"]) for name in ("q", "k")}
    for name in ("q", "k"):
        rstd = torch.rsqrt(x[name].float().square().sum(-1) + 1e-6)
        x[name + "_rstd"] = rstd if spec["with_l2"] else None
        if spec["with_l2"]:
            x[name] = (x[name].float() * rstd[..., None]).to(torch.bfloat16)
    x["v"] = randn(shape, spec["v_std"])
    x["d_o"] = randn(shape, spec["do_std"])
    lo, hi = spec["beta_range"]
    x["beta"] = (torch.rand(shape[:-1], generator=gen) * (hi-lo) + lo).to(DTYPES[spec["beta_dtype"]])
    x["raw_g"] = randn(shape, spec["raw_g_std"], spec["raw_g_mean"], DTYPES[spec["raw_g_dtype"]])
    x["A_log"] = randn((spec["H"],), 0.2, -0.5, DTYPES[spec["a_log_dtype"]])
    x["dt_bias"] = randn((spec["H"], 128), 0.15, dtype=torch.float32) if spec["with_bias"] else None
    cache_dtype = torch.float64 if exact_caches else torch.bfloat16
    calc = torch.float64 if exact_caches else torch.float32
    nc = sum((n + 63) // 64 for n in spec["seqlens"]) if spec["mode"] == "varlen" else (spec["T"]+63)//64
    for name in ("w", "qg", "kg", "v_new"):
        x[name] = torch.zeros(shape, dtype=cache_dtype)
    x["gk"] = torch.zeros(shape, dtype=calc)
    for name in ("Aqk", "Akk"):
        x[name] = torch.zeros((*shape[:-1], 64), dtype=cache_dtype)
    x["h"] = torch.zeros((spec["B"], nc, spec["H"], 128, 128), dtype=cache_dtype)
    for b, start, end, chunk0 in sequences(spec):
        for head in range(spec["H"]):
            state = torch.zeros((128, 128), dtype=calc)
            for chunk, a in enumerate(range(start, end, 64), chunk0):
                z = min(a+64, end)
                ix = (b, head, slice(a, z))
                q, k, v, beta, raw = (x[name][ix].to(calc) for name in ("q", "k", "v", "beta", "raw_g"))
                bias = 0 if x["dt_bias"] is None else x["dt_bias"][head].to(calc)
                gate = spec["lower_bound"] * torch.sigmoid(x["A_log"][head].to(calc).exp() * (raw + bias))
                g = gate.cumsum(0)
                qk, kk, _ = _scores(q, k, g)
                eye = torch.eye(z-a, dtype=calc)
                inv = torch.linalg.solve_triangular(eye + (beta[:, None] * kk).tril(-1), eye, upper=False)
                # Saved BF16 caches model a valid low-precision forward, including
                # the BF16 inverse consumed by recompute mode.
                inv = inv.to(cache_dtype).to(calc)
                w = (inv @ (beta[:, None] * k * g.exp())).to(cache_dtype).to(calc)
                u = (inv @ (beta[:, None] * v)).to(cache_dtype).to(calc)
                qg = (q * g.exp()).to(cache_dtype).to(calc)
                kg = (k * (g[-1] - g).exp()).to(cache_dtype).to(calc)
                x["h"][b, chunk, head] = state.to(cache_dtype)
                vn = (u - w @ state.to(cache_dtype).to(calc)).to(cache_dtype).to(calc)
                x["Aqk"][ix][..., :z-a] = (qk * spec["scale"]).to(cache_dtype)
                x["Akk"][ix][..., :z-a] = inv
                x["gk"][ix] = g / math.log(2)
                for name, value in (("w", w), ("qg", qg), ("kg", kg), ("v_new", vn)):
                    x[name][ix] = value
                state = g[-1].exp()[:, None] * state + kg.T @ vn
    return x


def cpu_reference(x, spec, high_precision=False):
    """Analytic chunk adjoints, including WY inverse, scan and safe gate."""
    if not spec["disable_recompute"]:
        x = recompute_cpu_caches(x, spec, high_precision)
    calc = torch.float64 if high_precision else torch.float32
    out = {name: torch.zeros_like(x[src], dtype=calc) for name, src in
           (("dq", "q"), ("dk", "k"), ("dv", "v"), ("db", "beta"), ("dg", "raw_g"), ("dA", "A_log"))}
    out["dh0"] = None
    out["dbias"] = None if x["dt_bias"] is None else torch.zeros_like(x["dt_bias"], dtype=calc)

    def stage(value):
        return value if high_precision else value.to(torch.bfloat16).float()

    for b, start, end, chunk0 in sequences(spec):
        for head in range(spec["H"]):
            dh = torch.zeros((128, 128), dtype=calc)
            spans = list(enumerate(range(start, end, 64), chunk0))
            for chunk, a in reversed(spans):
                z = min(a+64, end)
                ix = (b, head, slice(a, z))
                q, k, v, beta, raw, do, vn, w, qg, kg = (
                    x[name][ix].to(calc) for name in
                    ("q", "k", "v", "beta", "raw_g", "d_o", "v_new", "w", "qg", "kg"))
                g = x["gk"][ix].to(calc) * math.log(2)
                h = x["h"][b, chunk, head].to(calc)
                aqk = x["Aqk"][ix][..., :z-a].to(calc)
                inv = x["Akk"][ix][..., :z-a].to(calc)
                _, kk, decay = _scores(q, k, g)
                # Prepare + reverse state scan. dh is the gradient of the state
                # AFTER this chunk; each sequence starts its reverse scan at 0.
                daqk = (do @ vn.T).tril() * spec["scale"]
                dv0 = stage(aqk.T @ do)
                ds = stage(dv0 + kg @ dh)
                dh_prev = stage(g[-1].exp()[:, None] * dh + spec["scale"] * qg.T @ do - w.T @ ds)
                # WY representation: u=A(beta*v), w=A(beta*k*exp(g)).
                dw = -(ds @ h.T)
                du_src = inv.T @ ds
                dw_src = inv.T @ dw
                da = ds @ (beta[:, None] * v).T + dw @ (beta[:, None] * k * g.exp()).T
                dl = (-(inv.T @ da @ inv.T)).tril(-1)
                dv = beta[:, None] * du_src
                db = (du_src * v + dw_src * k * g.exp()).sum(-1) + (dl * kk).sum(-1)
                dk = beta[:, None] * dw_src * g.exp()
                dg = dk * k
                dq = spec["scale"] * (do @ h.T) * g.exp()
                dg += dq * q
                dk_state = (vn @ dh.T) * (g[-1] - g).exp()
                dk += dk_state
                dg -= dk_state * k
                dg[-1] += (dk_state * k).sum(0) + (dh * h).sum(-1) * g[-1].exp()
                # Intra-chunk causal QK and strict-lower KK score derivatives.
                pair_qk = daqk[..., None] * decay
                dq += (pair_qk * k[None, :, :]).sum(1)
                dk += (pair_qk * q[:, None, :]).sum(0)
                gate_qk = pair_qk * q[:, None, :] * k[None, :, :]
                pair_kk = (dl * beta[:, None])[..., None] * decay
                dk += (pair_kk * k[None, :, :]).sum(1) + (pair_kk * k[:, None, :]).sum(0)
                gate_kk = pair_kk * k[:, None, :] * k[None, :, :]
                dg += (gate_qk + gate_kk).sum(1) - (gate_qk + gate_kk).sum(0)
                # gk uses log2 storage; dg above is d/d natural-log cumsum.
                gate_grad = dg.flip(0).cumsum(0).flip(0)
                bias = 0 if x["dt_bias"] is None else x["dt_bias"][head].to(calc)
                raw_bias = raw + bias
                eig = x["A_log"][head].to(calc).exp()
                sig = torch.sigmoid(eig * raw_bias)
                dr = gate_grad * spec["lower_bound"] * eig * sig * (1-sig)
                out["dA"][head] += (dr * raw_bias).sum()
                if out["dbias"] is not None:
                    out["dbias"][head] += dr.sum(0)
                if spec["with_l2"]:
                    dq = (dq - q * (dq*q).sum(-1, keepdim=True)) * x["q_rstd"][ix].to(calc)[..., None]
                    dk = (dk - k * (dk*k).sum(-1, keepdim=True)) * x["k_rstd"][ix].to(calc)[..., None]
                for name, value in (("dq", dq), ("dk", dk), ("dv", dv), ("db", db), ("dg", dr)):
                    out[name][ix] = value
                dh = dh_prev
    if not high_precision:
        for name in ("dq", "dk", "dv"):
            out[name] = out[name].to(torch.bfloat16)
        out["db"] = out["db"].to(x["beta"].dtype)
    if spec["mode"] == "varlen":
        for name in ("dq", "dk", "dv", "db", "dg"):
            out[name] = out[name].squeeze(0)
    return tuple(out[name] for name in OUTPUT_NAMES)


def recompute_cpu_caches(inputs, spec, high_precision):
    """Regenerate internal caches; only Aqk/Akk are public forward caches here.

    The low-precision beta*k*exp(g) and beta*v matmul operands are BF16,
    matching the recompute reference in test_npu_chunk_kda_bwd_recompute.py.
    This quantization is absent from the FP64 golden. Never use DUT outputs.
    """
    x = dict(inputs)
    calc = torch.float64 if high_precision else torch.float32
    cache = torch.float64 if high_precision else torch.bfloat16
    for name in ("w", "qg", "kg", "v_new", "h"):
        x[name] = torch.empty_like(inputs[name], dtype=cache)
    x["gk"] = torch.empty_like(inputs["gk"], dtype=calc)

    def quant(value):
        return value.to(cache).to(calc)

    for b, start, end, chunk0 in sequences(spec):
        for head in range(spec["H"]):
            state = torch.zeros(128, 128, dtype=calc)
            for chunk, a in enumerate(range(start, end, 64), chunk0):
                z = min(a+64, end)
                ix = (b, head, slice(a, z))
                q, k, v, beta, raw = (x[n][ix].to(calc) for n in ("q", "k", "v", "beta", "raw_g"))
                bias = 0 if x["dt_bias"] is None else x["dt_bias"][head].to(calc)
                g = (spec["lower_bound"] * torch.sigmoid(x["A_log"][head].to(calc).exp() * (raw+bias))).cumsum(0)
                inv = x["Akk"][ix][..., :z-a].to(calc)
                w = quant(inv @ quant(beta[:, None] * k * g.exp()))
                u = quant(inv @ quant(beta[:, None] * v))
                qg = quant(q * g.exp())
                kg = quant(k * (g[-1]-g).exp())
                x["h"][b, chunk, head] = state.to(cache)
                vn = quant(u - w @ quant(state))
                for name, value in (("w", w), ("qg", qg), ("kg", kg), ("v_new", vn)):
                    x[name][ix] = value
                x["gk"][ix] = g / math.log(2)
                state = g[-1].exp()[:, None] * state + kg.T @ vn
    return x


def _finite_outputs(outputs):
    if len(outputs) != 8 or outputs[5] is not None:
        raise RuntimeError("V2 output contract drift (expected 8 slots and dh0=None)")
    visible = []
    for name, value in zip(OUTPUT_NAMES, outputs):
        if value is None:
            if name not in {"dh0", "dbias"}:
                raise RuntimeError(f"missing output {name}")
            continue
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"{OP_NAME}: non-finite {name}")
        visible.append(value)
    return tuple(visible)


def npu_inputs(x, spec, device):
    packed = spec["mode"] == "varlen"
    result = {}
    for name, value in x.items():
        if not spec["disable_recompute"] and name in {"gk", "w", "qg", "kg", "v_new", "h"}:
            result[name] = None
        elif value is None:
            result[name] = None
        else:
            if packed and name not in {"A_log", "dt_bias"}:
                value = value.squeeze(0)
            result[name] = value.contiguous().to(device)
    cu = indices = None
    if packed:
        cu, indices = [0], []
        for seq, length in enumerate(spec["seqlens"]):
            cu.append(cu[-1] + length)
            for chunk in range((length+63)//64):
                indices.extend((seq, chunk))
        if spec["metadata"] == "auto":
            indices = None
    result.update(scale=spec["scale"], chunk_size=64, safe_gate=True,
                  use_gate_in_kernel=True, lower_bound=spec["lower_bound"],
                  disable_recompute=spec["disable_recompute"], use_exp2=True,
                  state_v_first=False, initial_state=None, dht=None,
                  cu_seqlens=cu, chunk_indices=indices, implementation="optimized")
    return result


def self_test():
    """Independent token-autograd oracle, including a 64/65 chunk boundary."""
    validate_profiles()
    for length in (1, 7, 65):
        spec = dict(PROFILES[0], B=1, H=1, T=length, mode="dense", with_l2=False,
                    with_bias=True, scale=0.125, lower_bound=-0.5)
        x = build_inputs(spec, exact_caches=True)
        names = ("q", "k", "v", "beta", "raw_g", "A_log", "dt_bias")
        leaves = {n: x[n].double().detach().requires_grad_() for n in names}
        state = torch.zeros(128, 128, dtype=torch.float64)
        loss = torch.zeros((), dtype=torch.float64)
        for t in range(length):
            q, k, v, beta, raw = (leaves[n][0, 0, t] for n in names[:5])
            gate = spec["lower_bound"] * torch.sigmoid(leaves["A_log"][0].exp() * (raw + leaves["dt_bias"][0]))
            decayed = gate.exp()[:, None] * state
            delta = beta * (v - k @ decayed)
            state = decayed + k[:, None] * delta[None, :]
            loss = loss + ((q @ state) * spec["scale"] * x["d_o"][0, 0, t].double()).sum()
        expected = torch.autograd.grad(loss, tuple(leaves.values()))
        # Short lengths exercise the CPU algebra only, not the restricted NPU
        # recompute interface. Production profiles remain aligned and H%8=0.
        for saved in (True, False):
            actual = cpu_reference(x, dict(spec, disable_recompute=saved), True)
            for name, a, e in zip(("dq", "dk", "dv", "db", "dg", "dA", "dbias"),
                                  (actual[0], actual[1], actual[2], actual[3], actual[4], actual[6], actual[7]), expected):
                torch.testing.assert_close(a, e, rtol=1e-8, atol=1e-10, msg=f"T={length} saved={saved} {name}")
    # Repack two independent dense sequences, including empty packed entries.
    # This checks state reset, chunk-major h, head/sequence offsets, L2 and
    # parameter-gradient reductions without assuming the same memory layout.
    dense = dict(PROFILES[0], B=2, H=2, T=65, mode="dense", with_l2=True,
                 with_bias=True, scale=0.125, lower_bound=-0.5)
    xd = build_inputs(dense)
    packed = dict(dense, B=1, T=130, mode="varlen", seqlens=[0, 65, 0, 65, 0])
    xp = {}
    for name, value in xd.items():
        if value is None or name in {"A_log", "dt_bias"}:
            xp[name] = value
        elif name == "h":
            xp[name] = value.flatten(0, 1).unsqueeze(0)
        else:
            xp[name] = torch.cat(tuple(value.unbind(0)), dim=1).unsqueeze(0)
    for high in (False, True):
        od, op = cpu_reference(xd, dense, high), cpu_reference(xp, packed, high)
        for i, (d, p) in enumerate(zip(od, op)):
            if d is None:
                assert p is None
                continue
            expected = torch.cat(tuple(d.unbind(0)), dim=1) if i < 5 else d
            torch.testing.assert_close(p, expected, rtol=0, atol=0)
    # Exercise all four tiling classes, dtypes, optional outputs and both modes.
    for index in (0, 16, 20, 36, 40, 49):
        spec = PROFILES[index]
        x = build_inputs(spec)
        for high in (False, True):
            visible = _finite_outputs(cpu_reference(x, spec, high))
            assert len(visible) == (7 if spec["with_bias"] else 6)
            if high:
                assert all(t.dtype == torch.float64 for t in visible)
    # Absent caches are not inputs to the recompute API. Poisoning their
    # placeholders must not affect either CPU role (regression for ATK R1).
    spec = next(s for s in PROFILES if not s["disable_recompute"])
    x = build_inputs(spec)
    poisoned = dict(x)
    for name in ("gk", "w", "qg", "kg", "v_new", "h"):
        poisoned[name] = torch.full_like(x[name], float("nan"))
    for high in (False, True):
        clean = _finite_outputs(cpu_reference(x, spec, high))
        dirty = _finite_outputs(cpu_reference(poisoned, spec, high))
        for a, b in zip(clean, dirty):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    print("PASS: token-autograd FP64 adjoints (T=1,7,65), dense/packed equivalence, CPU dual paths and profile contracts")


try:
    from atk.tasks.api_execute import register
    from atk.tasks.api_execute.base_api import BaseApi
except ModuleNotFoundError as exc:
    # Permit standalone CPU checks when only the local atk/ directory exists.
    if exc.name not in {"atk", "atk.tasks"}:
        raise
else:
    @register("executor_chunk_kda_bwd")
    class FunctionApi(BaseApi):
        def __init__(self, task_result):
            super().__init__(task_result)
            self.high_precision = self.device == "cpu"
            self.prepared_id = None
            self.prepared = None

        def init_by_input_data(self, input_data):
            case_id = int(input_data.kwargs["case_id"])
            if not 0 <= case_id < len(PROFILES):
                raise ValueError(f"invalid case_id {case_id}")
            spec = PROFILES[case_id]
            self.prepared = None  # Release previous case before allocating next.
            self.prepared_id = None
            cpu = build_inputs(spec)
            if self.device == "npu":
                device = input_data.kwargs["low_precision_marker"].device
                if device.type != "npu":
                    raise RuntimeError("NPU marker is not on NPU")
                self.prepared = npu_inputs(cpu, spec, device)
            elif self.device == "cpu":
                self.prepared = cpu
            else:
                raise RuntimeError("Use the NPU DUT and CPU benchmark backends")
            self.prepared_id = case_id

        def __call__(self, input_data, with_output=False):
            case_id = int(input_data.kwargs["case_id"])
            if self.prepared_id != case_id:
                self.init_by_input_data(input_data)
            if self.device == "npu":
                from fla_npu.ops.ascendc import npu_chunk_kda_bwd
                outputs = npu_chunk_kda_bwd(**self.prepared)
            else:
                outputs = cpu_reference(self.prepared, PROFILES[case_id], self.high_precision)
                # Match the repository's single-golden ATK convention:
                # evaluate in FP64, transport golden tensors in FP32.
                outputs = tuple(value.float() if value is not None else None for value in outputs)
            return _finite_outputs(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()
