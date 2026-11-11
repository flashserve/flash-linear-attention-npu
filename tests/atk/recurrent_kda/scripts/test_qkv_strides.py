"""A5 QKV stride regression: public fallback and direct tiling boundaries.

Requires a qkv_stride_tiling executable built against the current CANN headers
and the current operator's kernel directory. Uses the existing CPU reference,
input ranges and tolerances, plus bitwise comparison with a dense direct call.
Unsupported varlen batch strides must be rejected before a direct launch;
the public Python entry must still accept them through L2 materialization.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import torch
import torch_npu

os.environ.setdefault("FLA_NPU_STABLE_ABI", "stable")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executor_recurrent_kda_longseq import make_inputs, specifications
from executor_recurrent_kda import recurrent_kda_reference
from test_l2norm_epsilon import compare
from fla_npu.ops.ascendc import recurrent_kda


class Kernels:
    def __init__(self, directory):
        self.rt = ct.CDLL("libascendcl.so")
        for name, args, result in [
            ("aclrtCreateBinary", [ct.c_void_p, ct.c_size_t], ct.c_void_p),
            ("aclrtBinaryLoad", [ct.c_void_p, ct.POINTER(ct.c_void_p)], ct.c_int),
            ("aclrtBinaryGetFunction", [ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_void_p)], ct.c_int),
            ("aclrtLaunchKernelWithHostArgs", [ct.c_void_p, ct.c_uint32, ct.c_void_p,
                 ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t], ct.c_int),
            ("aclrtBinaryUnLoad", [ct.c_void_p], ct.c_int),
            ("aclrtDestroyBinary", [ct.c_void_p], ct.c_int),
        ]:
            function = getattr(self.rt, name)
            function.argtypes, function.restype = args, result
        self.metadata = [json.loads(p.read_text()) | {"path": p}
                         for p in Path(directory).glob("*.json")]
        self.cache = {}

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"ACL runtime returned {status}")

    def function(self, state_bf16, mode):
        key = state_bf16, mode
        if key not in self.cache:
            wanted = {"query": "bfloat16", "key": "bfloat16", "value": "bfloat16",
                      "gate": "float32", "beta": "float32",
                      "initial_state": "bfloat16" if state_bf16 else "float32",
                      "cu_seqlens": "int32", "ssm_state_indices": "int32",
                      "num_accepted_tokens": "int32"}
            matches = [m for m in self.metadata if all(
                next((i["dtype"] for i in m.get("supportInfo", {}).get("inputs", [])
                      if i["name"] == name), None) == dtype for name, dtype in wanted.items())
                and all(i["dtype"] == wanted["initial_state"]
                        for i in m["supportInfo"]["outputs"] if i["name"] != "attn_out")]
            if len(matches) != 1:
                raise RuntimeError(f"expected one dtype binary, found {len(matches)}")
            meta = matches[0]
            blob = ct.create_string_buffer(meta["path"].with_suffix(meta["binFileSuffix"]).read_bytes())
            binary = self.rt.aclrtCreateBinary(blob, len(blob) - 1)
            if not binary:
                raise RuntimeError("aclrtCreateBinary failed")
            handle, function = ct.c_void_p(), ct.c_void_p()
            self.check(self.rt.aclrtBinaryLoad(binary, ct.byref(handle)))
            name = f'{meta["kernelName"]}_{mode}'.encode()
            self.check(self.rt.aclrtBinaryGetFunction(handle, name, ct.byref(function)))
            self.cache[key] = blob, binary, handle, function
        return self.cache[key][3]

    def close(self):
        for _, binary, handle, _ in self.cache.values():
            self.check(self.rt.aclrtBinaryUnLoad(handle))
            self.check(self.rt.aclrtDestroyBinary(binary))

    def run(self, x, tiling, blocks, mode):
        output = torch.empty_like(x["v"], memory_format=torch.contiguous_format)
        final = torch.empty_like(x["initial_state"])
        workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="npu")
        tiling_tensor = torch.tensor(list(tiling), dtype=torch.uint8, device="npu")
        values = [x.get(name) for name in ["q", "k", "v", "g", "beta", "initial_state",
                  "cu_seqlens", "ssm_state_indices", "A_log", "dt_bias", "num_accepted_tokens"]]
        values += [output, x["initial_state"], final, workspace, tiling_tensor]
        arguments = (ct.c_void_p * len(values))(*[v.data_ptr() if v is not None else 0 for v in values])
        function = self.function(x["initial_state"].dtype == torch.bfloat16, mode)
        signal.alarm(60)
        try:
            self.check(self.rt.aclrtLaunchKernelWithHostArgs(
                function, blocks, torch.npu.current_stream().npu_stream, None,
                arguments, ct.sizeof(arguments), None, 0))
            torch.npu.synchronize()
        finally:
            signal.alarm(0)
        return output, final


def view_like(tensor, order):
    b, t, h, f = tensor.shape
    if order == "dense":
        return tensor.contiguous()
    strides = (t * h * f + 128, h * f, f, 1) if order == "gap" else (t * h * f + 128, f, t * f, 1)
    # Nonzero storage offset and nonzero guards expose reads from the wrong row.
    count = 11 + sum((dim - 1) * stride for dim, stride in zip(tensor.shape, strides)) + 1
    storage = torch.full((count,), 3.0, dtype=tensor.dtype, device=tensor.device)
    view = storage.as_strided(tensor.shape, strides, storage_offset=11)
    view.copy_(tensor)
    return view


def describe(x, kw, fused):
    return dict(q_shape=list(x["q"].shape), v_shape=list(x["v"].shape),
                state_shape=list(x["initial_state"].shape),
                **{name + "_stride": list(x[name].stride()) for name in ["q", "k", "v"]},
                sequences=x["initial_state"].shape[0], has_cu=x["cu_seqlens"] is not None,
                state_bf16=x["initial_state"].dtype == torch.bfloat16,
                v_first=kw["state_v_first"], fused=fused)


def packed_qkv_views(x, conv_decode=False):
    names = ["q", "k", "v"]
    widths = [x[name].shape[-2] * x[name].shape[-1] for name in names]
    packed = torch.cat([x[name].flatten(-2) for name in names], dim=-1)
    result, start = {}, 0
    for name, width in zip(names, widths):
        view = packed[..., start:start + width].view(x[name].shape)
        if conv_decode:
            # [B, C, 1] convolution output transposed to [B, 1, C].
            view = view.as_strided(view.shape, (view.stride(0), 1, view.stride(2), 1))
        result[name] = view
        start += width
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiling-probe", type=Path, required=True)
    parser.add_argument("--kernel-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    kernels, results = Kernels(args.kernel_dir), []

    def tile(spec, tag):
        path = args.output / f"{tag}.json"
        path.write_text(json.dumps(spec))
        binary = path.with_suffix(".bin")
        result = json.loads(subprocess.check_output([str(args.tiling_probe), str(path), str(binary)]))
        return result, binary.read_bytes() if result["success"] else None

    try:
        for number, (vf, bf16, fused, vdim, scenario) in enumerate(itertools.product(
                [False, True], [False, True], [False, True], [128, 256], range(7))):
            spec = dict(specifications()[0], B=3, T=5, H=2, HV=4, V=vdim,
                        layout="BSND", state_v_first=vf, state_dtype="bf16" if bf16 else "fp32",
                        raw_gate=fused, norm=fused, beta_sigmoid=fused, negative=False, safe_gate=False,
                        gate_dtype="fp32", beta_dtype="fp32", metadata_dtype="int32",
                        cu=scenario != 3, output=True, inplace=False)
            if scenario == 4:
                spec.update(B=7, T=1)
            elif scenario == 6:
                spec.update(T=1)
            cpu, kw, _ = make_inputs(spec, torch.device("cpu"))
            if scenario in [0, 1, 2, 5]:
                cpu["cu_seqlens"] = torch.tensor([0, 1, 13, 15], dtype=torch.int32)
            elif scenario == 4:
                cpu["cu_seqlens"] = torch.tensor([0, 5, 5, 7], dtype=torch.int32)
                cpu["initial_state"] = cpu["initial_state"][:3].contiguous()
            x = {name: value.to("npu") if isinstance(value, torch.Tensor) else value for name, value in cpu.items()}
            before = x["initial_state"].clone()
            orders = [("gap", "gap", "gap"), ("head", "head", "head"),
                      ("dense", "head", "gap"), ("head", "gap", "head"),
                      ("head", "gap", "head"), ("dense", "dense", "dense"),
                      ("dense", "dense", "dense")][scenario]
            views = packed_qkv_views(x, conv_decode=scenario == 6) if scenario in [5, 6] else {
                name: view_like(x[name], order) for name, order in zip(["q", "k", "v"], orders)}
            strided = dict(x, **views)
            mode = (1 if vf else 2) + (2 if fused else 0)
            print(f"CASE {number}: mode={mode} BF16={bf16} V={vdim} scenario={scenario}", flush=True)
            dense_info, dense_tiling = tile(describe(x, kw, fused), f"{number}-dense")
            info, tiling = tile(describe(strided, kw, fused), f"{number}-view")
            assert dense_info["success"]
            direct = scenario in [3, 5]
            assert info["success"] == direct, "unexpected direct tiling acceptance"
            if direct:
                assert info["batch_strides"] == ([0, 0, 0] if spec["cu"] else
                    [strided[name].stride(0) for name in ["q", "k", "v"]])
            expected = kernels.run(x, dense_tiling, dense_info["blocks"], mode)
            outputs = [recurrent_kda(**x, **kw), recurrent_kda(**strided, **kw)]
            if direct:
                outputs.append(kernels.run(strided, tiling, info["blocks"], mode))
            assert torch.equal(before, x["initial_state"]), "non-inplace call modified its input"
            # The empty request's independent output slot has unspecified contents.
            slots = [0, 2] if scenario == 4 else list(range(3))
            reference = recurrent_kda_reference(**cpu, **kw)
            for actual in outputs:
                for a, e in zip((actual[0], actual[1][slots]), (expected[0], expected[1][slots])):
                    assert torch.equal(a, e), "public/direct strided and dense calls differ"
                compare((actual[0].cpu(), actual[1][slots].cpu()), (reference[0], reference[1][slots]))
            results.append(dict(case=number, mode=mode, bf16=bf16, V=vdim, scenario=scenario,
                                direct=direct, passed=True))

        for name in ["q", "k", "v"]:
            only_one_gap = dict(x, **{name: view_like(x[name], "gap")})
            assert not tile(describe(only_one_gap, kw, fused), f"only-{name}-batch-gap")[0]["success"]

        negative = describe(strided, kw, fused)
        negative.update(q_shape=[2, 2, 2, 128], v_shape=[2, 2, 4, 128],
                        state_shape=[2, 4, 128, 128], sequences=2, has_cu=False,
                        q_stride=[384, 128, 256, 1], k_stride=[512, 256, 128, 1],
                        v_stride=[1024, 512, 128, 1])
        assert not tile(negative, "overlapping-batches")[0]["success"]
        negative["q_stride"] = [512, 128, 256, 1]
        negative["is_a5"] = False
        assert not tile(negative, "a2-head-major")[0]["success"]
    finally:
        kernels.close()
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
    print(f"PASS {len(results)} public stride cases, "
          f"{sum(r['direct'] for r in results)} direct views, "
          f"{sum(not r['direct'] for r in results)} direct tiling rejections, "
          "and 5 negative tiling cases", flush=True)


if __name__ == "__main__":
    main()
