"""Public finalize ACLNN validation; frozen FP32/FP64 equations and seeds."""
import os
import json
import ctypes as c
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from _kda_profiles import case_profiles
from gen_chunk_kda_bwd_finalize import PROFILES, ALL_PROFILES

"""Independent full Finalize equations, with no device/capture dependencies."""
import math
import torch

NAMES = ('dq', 'dk', 'dv', 'db', 'dg', 'da_log', 'ddt_bias')


def build_inputs(spec, seed):
    gen = torch.Generator().manual_seed(seed)
    b, h, t = (spec[n] for n in ('B', 'H', 'T'))
    variable = spec['lengths'] is not None
    shape = (h, t, 128) if variable else (b, h, t, 128)

    def rand(dims, scale=.125, bf=False):
        x = torch.randn(dims, generator=gen) * scale
        return x.bfloat16() if bf else x

    data = {n: rand(shape, bf=True) for n in ('q', 'k', 'v', 'v_new', 'dv_scan')}
    for name in ('q', 'k'):
        if spec['norm']:
            x = data[name].float()
            rstd = (x.square().sum(-1) + 1e-6).rsqrt()
            data[name] = (x * rstd[..., None]).bfloat16()
            data[name + '_rstd'] = rstd
    raw = rand(shape, 1.)
    if spec['gate_mode'] == 'saturated':
        raw[..., ::4] = 30.
        raw[..., 1::4] = -30.
    data.update(raw_g=raw, a_log=rand((h,), .25), dt_bias=rand((h, 128), .125),
                dq_raw=rand(shape), dAqk=rand((*shape[:-1], 64)))
    logits = rand(shape[:-1], 2.)
    logits.reshape(-1)[:min(5, logits.numel())] = torch.tensor([-8., 0., 8., -16., 16.])[:min(5, logits.numel())]
    data['beta'] = (logits.sigmoid() * (2 if spec['beta_mode'] == 2 else 1)).to(
        torch.bfloat16 if spec.get('beta_dtype') == 'bf16' else torch.float32)
    cu, chunks = [0], []
    lengths = spec['lengths'] if variable else [t] * b
    for seq, length in enumerate(lengths):
        begin = cu[-1] if variable else 0
        for local, start in enumerate(range(begin, begin + length, 64)):
            chunks.append((seq, local, start, min(64, begin + length - start)))
        cu.append(cu[-1] + length)
    state_shape = (len(chunks), h, 128, 128) if variable else (b, (t+63)//64, h, 128, 128)
    data['h'], data['dh'] = rand(state_shape, bf=True), rand(state_shape, bf=True)
    data['akk'] = torch.zeros((*shape[:-1], 64), dtype=torch.bfloat16)
    data['gk'] = torch.empty(shape)
    for seq, _, first, rows in chunks:
        region = (slice(None), slice(first, first+rows)) if variable else (seq, slice(None), slice(first, first+rows))
        z = (raw[region] + data['dt_bias'][:, None, :]) * data['a_log'].exp()[:, None, None]
        data['gk'][region] = (spec['lower_bound'] * z.sigmoid()).cumsum(-2) / math.log(2)
        # A is unit lower triangular; inactive columns stay zero.
        a = torch.tril(rand((h, rows, rows), .025), -1) + torch.eye(rows)
        data['akk'][region][..., :rows] = a.bfloat16()
    return data, cu if variable else [], chunks


def round_bfloat16(value):
    """Round FP64 once, including subnormals; torch's cast can pass through FP32."""
    _, exponent = torch.frexp(value)
    step = (exponent - 8).clamp(min=-133)
    rounded = torch.ldexp(torch.ldexp(value, -step).round(), step)
    return rounded.bfloat16()


def reference(data, spec, chunks, high_precision=True):
    """FP64 mathematical golden / FP32 benchmark with documented BF16 handoffs.

    The exponential difference is factored per CPU output tile. FP64 can
    handle an entire chunk; FP32 uses eight rows to keep positive exponents
    bounded. There is no BF16 residual split or device intermediate reuse.
    """
    dtype = torch.float64 if high_precision else torch.float32
    variable = spec['lengths'] is not None
    heads = spec['H']
    shape = data['q'].shape
    # Round the FP64 golden directly to the interface dtype. An intervening
    # FP32 store can turn a near-midpoint value into a BF16 tie.
    out = {n: torch.empty(shape, dtype=torch.bfloat16) for n in ('dq', 'dk', 'dv')}
    out['dg'] = torch.empty(shape, dtype=data['raw_g'].dtype)
    out['db'] = torch.empty(shape[:-1], dtype=data['beta'].dtype)
    da = torch.zeros(heads, dtype=dtype)
    bias = torch.zeros(heads, 128, dtype=dtype)

    def handoff(x):
        return x if high_precision else x.bfloat16().float()

    for ci, (seq, local, first, rows) in enumerate(chunks):
        for head in range(heads):
            region = (head, slice(first, first+rows)) if variable else (seq, head, slice(first, first+rows))
            q, k, v, vn, ds, g, beta, aq, qr = [data[n][region].to(dtype) for n in
                ('q', 'k', 'v', 'v_new', 'dv_scan', 'gk', 'beta', 'dAqk', 'dq_raw')]
            a = data['akk'][region][:, :rows].to(dtype)
            hh = (data['h'][ci, head] if variable else data['h'][seq, local, head]).to(dtype)
            dh = (data['dh'][ci, head] if variable else data['dh'][seq, local, head]).to(dtype)
            e = torch.exp2(g)
            dw = handoff(ds @ hh.T)
            ke = handoff(k * e)
            dvb = a.T @ ds
            dkg = a.T @ dw
            zb = handoff(torch.tril((ds @ v.T - dw @ ke.T) * beta[None, :], -1))
            ak = -(a.T @ handoff(zb @ a.T)).tril(-1)
            state = (vn @ dh.T) * torch.exp2(g[-1] - g)
            dq = spec['scale'] * qr * e
            dk = state - beta[:, None] * e * dkg
            db = (v * dvb).sum(-1) - (k * e * dkg).sum(-1)
            dg = q * dq - k * state - beta[:, None] * k * e * dkg
            dg[-1] += (k * state).sum(0) + torch.exp2(g[-1]) * (hh * dh).sum(-1)
            aq_lower = aq[:, :rows].tril(-1)
            local_q, left, right = (torch.empty_like(k) for _ in range(3))
            width = rows if high_precision else 8
            for start in range(0, rows, width):
                end = min(start + width, rows)
                center = (g[start] + g[end - 1]) * .5
                factor = torch.exp2(g[start:end] - center)
                neg = k[:end] * torch.exp2(center - g[:end])
                pos = torch.exp2(g[start:] - center)
                local_q[start:end] = (aq_lower[start:end, :end] @ neg) * factor
                left[start:end] = (ak[start:end, :end] @ neg) * factor
                right[start:end] = (aq_lower[start:, start:end].T @ (q[start:] * pos) +
                                    ak[start:, start:end].T @ (beta[start:, None] * k[start:] * pos)) / factor
            diag = aq[:, :rows].diag()[:, None]
            dq = dq + local_q + diag * k
            dk = dk + beta[:, None] * left + right + diag * q
            db = db + (left * k).sum(-1)
            dg = dg + q * local_q + k * (beta[:, None] * left - right)
            scan = dg.flip(0).cumsum(0).flip(0)
            x = data['raw_g'][region].to(dtype) + data['dt_bias'][head].to(dtype)
            ea = data['a_log'][head].to(dtype).exp()
            z = (-(x * ea).abs()).exp()
            dg = scan * (spec['lower_bound'] * ea * z / (1 + z).square())
            if spec['norm']:
                dq = (dq - q * (dq * q).sum(-1, keepdim=True)) * data['q_rstd'][region].to(dtype)[:, None]
                dk = (dk - k * (dk * k).sum(-1, keepdim=True)) * data['k_rstd'][region].to(dtype)[:, None]
            if spec['beta_mode']:
                db *= beta * (1 - beta / (2 if spec['beta_mode'] == 2 else 1))
            for n, value in zip(('dq', 'dk', 'dv', 'db', 'dg'), (dq, dk, beta[:, None]*dvb, db, dg)):
                out[n][region] = (round_bfloat16(value) if high_precision and out[n].dtype == torch.bfloat16
                                   else value.to(out[n].dtype))
            da[head] += (dg * x).sum()
            bias[head] += dg.sum(0)
    for n in ('dq', 'dk', 'dv', 'dg'):
        if spec['output_layout'] in ('BSND', 'TND'):
            out[n] = out[n].transpose(-3, -2).contiguous()
    if spec['output_layout'] in ('BSND', 'TND'):
        out['db'] = out['db'].transpose(-2, -1).contiguous()
    out['da_log'], out['ddt_bias'] = da.float(), bias.float()
    return interface_outputs(tuple(out[n] for n in NAMES), spec)


torch.set_num_threads(1)


def interface_outputs(outputs, spec):
    source_rank = 3 if spec['lengths'] is not None else 4
    if source_rank == spec['rank']:
        return outputs
    return tuple((x.unsqueeze(0) if spec['rank'] == 4 else x.squeeze(0)) if i < 5 else x
                 for i, x in enumerate(outputs))


def run_npu(data, spec, cu, chunks, device):
    """Call the public two-stage ACLNN with the exact typed C signature."""
    import torch_npu
    from fla_npu.ops.ascendc import _runtime
    variable = spec['lengths'] is not None
    dev = {}
    for name, value in data.items():
        if name not in ('a_log', 'dt_bias'):
            if variable and spec['rank'] == 4:
                value = value.unsqueeze(0)
            elif not variable and spec['rank'] == 3:
                value = value.squeeze(0)
        dev[name] = value.contiguous().to(device)
    token_major = spec['output_layout'] in ('BSND', 'TND')
    qshape = dev['q'].transpose(-3, -2).shape if token_major else dev['q'].shape
    outputs = tuple([torch.empty(qshape, dtype=torch.bfloat16, device=device) for _ in range(3)] + [
        torch.empty(qshape[:-1], dtype=dev['beta'].dtype, device=device),
        torch.empty(qshape, dtype=dev['raw_g'].dtype, device=device),
        torch.empty_like(dev['a_log']), torch.empty_like(dev['dt_bias'])])
    signature = ([c.c_void_p] * 19 + [c.c_double, c.c_double, c.c_int64] + [c.c_bool] * 6 +
                 [c.c_char_p] + [c.c_void_p] * 7 + [c.POINTER(c.c_uint64), c.POINTER(c.c_void_p)])

    def build_args(ctx):
        def tensor(value, name):
            if value is None:
                return None
            return ctx.tensor(value, name, acl_format_override=2, storage_shape_override=tuple(value.shape))
        names = ('q', 'k', 'v', 'gk', 'raw_g', 'beta', 'a_log', 'dt_bias', 'akk', 'v_new',
                 'h', 'dh', 'dv_scan', 'dAqk', 'dq_raw', 'q_rstd', 'k_rstd')
        args = [tensor(dev.get(n), n) for n in names]
        indices = [n for x in chunks for n in x[:2]] if cu and spec['metadata'] != 'auto' else None
        args += [ctx.int_array(cu if cu else None), ctx.int_array(indices)]
        args += [spec['scale'], spec['lower_bound'], 64, True, True, True, False,
                 spec['beta_mode'] != 0, spec['beta_mode'] == 2, spec['output_layout'].encode()]
        args += [tensor(t, n) for n, t in zip(NAMES, outputs)]
        return args

    _runtime.call_aclnn('aclnnChunkKdaBwdFinalize', build_args, outputs, get_workspace_argtypes=signature)
    torch.npu.synchronize()
    return tuple(t.cpu() for t in outputs)


try:
    from atk.tasks.api_execute import register
    from atk.tasks.api_execute.base_api import BaseApi
    from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
except ModuleNotFoundError as exc:
    if exc.name not in ('atk', 'atk.tasks'):
        raise
else:
    @register('executor_chunk_kda_bwd_finalize')
    class FunctionApi(BaseApi):
        def __init__(self, task_result, aclnn_manager=None):
            super().__init__(task_result)

        def __call__(self, input_data, with_output=False):
            torch.set_num_threads(1)
            profiles = case_profiles(self.task_result, Path(__file__).with_name("chunk_kda_bwd_finalize_10000.json"), ALL_PROFILES)
            spec = profiles[int(input_data.kwargs["case_id"])]
            seed = spec['seed'] + 10000 * int(os.getenv('KDA_SEED_ROUND', '0'))
            data, cu, chunks = build_inputs(spec, seed)
            high = bool(self.task_result.is_benchmark_task)
            print(f"FINALIZE_ROLE case={spec['case_id']} seed={seed} backend={self.device} "
                  f"golden={high} expected_key={spec['expected_tiling_key']} "
                  f"beta={spec['beta_dtype']} layout={spec['output_layout']}", flush=True)
            if self.device in ('npu', 'pyaclnn'):
                output = run_npu(data, spec, cu, chunks, input_data.kwargs['marker'].device)
            else:
                output = reference(data, spec, chunks, high)
            for name, tensor in zip(NAMES, output):
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"case {spec['case_id']}: nonfinite {name}")
            return output


if 'AclnnBaseApi' in globals():
    @register('aclnn_chunk_kda_bwd_finalize')
    class FinalizeAclnnApi(AclnnBaseApi):
        def init_by_input_data(self, input_data):
            import torch_npu
            profiles = case_profiles(self.task_result, Path(__file__).with_name("chunk_kda_bwd_finalize_10000.json"), ALL_PROFILES)
            spec = profiles[int(input_data.kwargs["case_id"])]
            seed = spec['seed'] + 10000 * int(os.getenv('KDA_SEED_ROUND', '0'))
            data, cu, chunks = build_inputs(spec, seed)
            device = f"npu:{int(os.environ['KDA_TEST_DEVICE'])}"
            torch.npu.set_device(device)
            warm = torch.ones((16, 16), dtype=torch.bfloat16, device=device)
            warm = warm @ warm
            torch.npu.synchronize()
            torch.set_num_threads(1)
            self.dev = {}
            variable = spec['lengths'] is not None
            for name, value in data.items():
                if name not in ('a_log', 'dt_bias'):
                    if variable and spec['rank'] == 4:
                        value = value.unsqueeze(0)
                    elif not variable and spec['rank'] == 3:
                        value = value.squeeze(0)
                self.dev[name] = value.contiguous().to(device)
            dev = self.dev
            qshape = dev['q'].transpose(-3, -2).shape if spec['output_layout'] in ('BSND', 'TND') else dev['q'].shape
            self.outputs = tuple([torch.empty(qshape, dtype=torch.bfloat16, device=device) for _ in range(3)] + [
                torch.empty(qshape[:-1], dtype=dev['beta'].dtype, device=device),
                torch.empty(qshape, dtype=dev['raw_g'].dtype, device=device),
                torch.empty_like(dev['a_log']), torch.empty_like(dev['dt_bias'])])
            names = ('q', 'k', 'v', 'gk', 'raw_g', 'beta', 'a_log', 'dt_bias', 'akk', 'v_new',
                     'h', 'dh', 'dv_scan', 'dAqk', 'dq_raw', 'q_rstd', 'k_rstd')
            from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor, AclIntArray
            args = [self.torch_tensor_to_acl(dev[n]) if n in dev else c.POINTER(AclTensor)() for n in names]
            indices = [n for x in chunks for n in x[:2]] if cu and spec['metadata'] != 'auto' else None
            if cu:
                from fla_npu.ops.ascendc import _runtime
                # ATK releases descriptors after the call; retain their backing
                # storage here and do not also call ctx.destroy().
                self.ctx = _runtime._CallContext(_runtime.runtime(), torch.device(device))
                args += [c.cast(self.ctx.int_array(cu), c.POINTER(AclIntArray)),
                         c.cast(self.ctx.int_array(indices), c.POINTER(AclIntArray))]
            else:
                # Dense inputs have no sequence metadata. Typed null pointers
                # do not require the private runtime removed from newer wheels.
                args += [c.POINTER(AclIntArray)(), c.POINTER(AclIntArray)()]
            args += [c.c_double(spec['scale']), c.c_double(spec['lower_bound']), c.c_int64(64),
                     c.c_bool(True), c.c_bool(True), c.c_bool(True), c.c_bool(False),
                     c.c_bool(spec['beta_mode'] != 0), c.c_bool(spec['beta_mode'] == 2),
                     c.c_char_p(spec['output_layout'].encode())]
            packages = [self.torch_tensor_to_acl(t) for t in self.outputs]
            self.input_args = args + packages
            print(f"FINALIZE_ROLE case={spec['case_id']} seed={seed} backend=pyaclnn golden=False "
                  f"expected_key={spec['expected_tiling_key']} beta={spec['beta_dtype']} "
                  f"layout={spec['output_layout']}", flush=True)
            return self.input_args, packages

        def after_call(self, output_packages):
            torch.npu.synchronize()
            outputs = tuple(t.cpu() for t in self.outputs)
            for name, value in zip(NAMES, outputs):
                if not torch.isfinite(value).all():
                    raise RuntimeError(f'nonfinite {name}')
            return outputs
