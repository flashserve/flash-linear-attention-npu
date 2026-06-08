# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
import torch

try:
    import torch.distributed.tensor as _torch_dtensor
    if not hasattr(_torch_dtensor, "DTensor"):
        class DTensor:  # Compatibility for ATK post-process under torch 2.1.
            pass

        _torch_dtensor.DTensor = DTensor
except Exception:
    pass

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import AccuracyConfig, TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.base_compare import BaseAccuracyCompare


DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


@ACCURACY_REGISTRY.register("causal_conv1d_bwd_allclose")
class CausalConv1dBwdAccuracyCompare(BaseAccuracyCompare):
    def compute_accuracy_result(self, local_output, remote_output, data_file):
        local = local_output.detach().cpu()
        remote = remote_output.detach().cpu()
        filename = str(data_file)

        if local.shape != remote.shape:
            return AccuracyConfig(
                filename=data_file,
                result=False,
                error_info=f"shape mismatch: local={tuple(local.shape)} remote={tuple(remote.shape)}",
            )

        if torch.isnan(local).all() and torch.isnan(remote).all():
            return AccuracyConfig(filename=data_file, result=True, error_info="all NaN pass")

        if "output_1" in filename or "output_2" in filename:
            rtol, atol = 1e-3, 1e-2
        elif local.dtype == torch.bfloat16 or remote.dtype == torch.bfloat16:
            rtol, atol = 2e-2, 2e-2
        elif local.dtype == torch.float16 or remote.dtype == torch.float16:
            rtol, atol = 2e-3, 2e-3
        else:
            rtol, atol = 1e-4, 1e-4

        local_f = local.to(torch.float32)
        remote_f = remote.to(torch.float32)
        passed = torch.allclose(local_f, remote_f, rtol=rtol, atol=atol, equal_nan=True)
        diff = (local_f - remote_f).abs()
        max_abs = diff.max().item() if diff.numel() > 0 else 0.0
        mean_abs = diff.mean().item() if diff.numel() > 0 else 0.0
        rel = diff / (remote_f.abs() + 1e-12)
        max_rel = rel.max().item() if rel.numel() > 0 else 0.0
        return AccuracyConfig(
            filename=data_file,
            result=passed,
            error_info=(
                f"allclose={passed}, rtol={rtol}, atol={atol}, "
                f"max_abs={max_abs}, mean_abs={mean_abs}, max_rel={max_rel}"
            ),
        )


def generate_tensor(shape, dtype, data_max=2.0):
    tensor = torch.rand(shape, dtype=torch.float32) * (data_max * 2.0) - data_max
    return tensor.to(dtype)


def _activation_name(activation):
    activation = int(activation)
    if activation == 0:
        return "none"
    if activation == 1:
        return "silu"
    if activation == 2:
        return "swish"
    raise ValueError(f"unsupported activation: {activation}")


def _to_compute_tensor(tensor, dtype):
    # The device op receives fp16/bf16 storage tensors but computes in fp32.
    return tensor.to(dtype).to(torch.float32).contiguous()


def _cast_output_to_input_dtype(tensor, dtype):
    return tensor.to(dtype).contiguous()


def _pattern_tensor(shape, dtype, stride, mod, scale):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    base = torch.arange(numel, dtype=torch.float32).reshape(tuple(shape))
    values = ((base * stride) % mod - (mod // 2)) * scale
    return values.to(dtype).contiguous()


def _normalize_layout(input_layout):
    layout = str(input_layout or "BSND").upper()
    if layout in ("BSH", "BSND"):
        return "BSND"
    if layout in ("TND", "BNSD", "NTD"):
        return layout
    raise ValueError(f"unsupported inputLayout: {input_layout}")


def _input_to_logical(tensor, input_layout):
    layout = _normalize_layout(input_layout)
    if layout == "BNSD":
        b, n, t, h = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(b, t, n * h).contiguous()
    if layout == "NTD":
        n, t, h = tensor.shape
        return tensor.permute(1, 0, 2).reshape(t, n * h).contiguous()
    return tensor.contiguous()


def _logical_to_input(tensor, input_layout, n_heads=None, head_dim=None):
    layout = _normalize_layout(input_layout)
    if layout == "BNSD":
        b, t, d = tensor.shape
        n = int(n_heads)
        h = int(head_dim)
        return tensor.reshape(b, t, n, h).permute(0, 2, 1, 3).contiguous()
    if layout == "NTD":
        t, d = tensor.shape
        n = int(n_heads)
        h = int(head_dim)
        return tensor.reshape(t, n, h).permute(1, 0, 2).contiguous()
    return tensor.contiguous()


def _layout_meta(tensor, input_layout):
    layout = _normalize_layout(input_layout)
    if layout == "BNSD":
        b, n, t, h = tensor.shape
        return b, t, n * h, n, h
    if layout == "NTD":
        n, t, h = tensor.shape
        return 1, t, n * h, n, h
    if tensor.dim() == 2:
        t, d = tensor.shape
        return 1, t, d, 1, d
    b, t, d = tensor.shape
    return b, t, d, 1, d


def _query_start_loc_for_layout(input_data, input_layout):
    x = input_data.kwargs["x"]
    t = x.shape[0] if x.dim() == 2 else x.shape[1]
    # The aclnn signature expects aclIntArray even when fixed-layout kernels
    # ignore queryStartLoc. Passing a list keeps ATK's pyaclnn converter on the
    # aclCreateIntArray path instead of converting the YAML int range to c_long.
    return [0, int(t)]


def causal_conv1d_preactivation(x, weight, initial_state=None):
    b, t, d = x.shape
    w = weight.shape[0]
    y = torch.zeros((b, t, d), dtype=torch.float32, device=x.device)

    x_f = x.to(torch.float32)
    weight_f = weight.to(torch.float32)
    state_f = initial_state.to(torch.float32) if initial_state is not None else None

    for k in range(w):
        shift = k - (w - 1)
        dst_start = -shift
        if dst_start < t:
            y[:, dst_start:, :] += x_f[:, : t - dst_start, :] * weight_f[k].view(1, 1, d)

        if state_f is not None:
            max_head = min(t, w - 1 - k)
            if max_head > 0:
                slot = torch.arange(max_head, device=x.device, dtype=torch.long) + k + 1
                state = state_f[:, slot, :]
                y[:, :max_head, :] += state * weight_f[k].view(1, 1, d)

    return y


def activation_backward(dy, y, activation):
    act = _activation_name(activation)
    dy_f = dy.to(torch.float32)
    if act == "none":
        return dy_f.contiguous()

    y_f = y.to(torch.float32)
    sig = torch.sigmoid(y_f)
    return (dy_f * sig * (1.0 + y_f * (1.0 - sig))).contiguous()


def _sum_batch_time(tensor, max_work_elems=4 * 1024 * 1024):
    bsz, t, d = tensor.shape
    step = max(1, max_work_elems // max(1, bsz * d))
    out = torch.zeros((d,), dtype=torch.float32, device=tensor.device)
    for start in range(0, t, step):
        out += tensor[:, start : start + step, :].sum(dim=(0, 1))
    return out


def _sum_batch_time_product(lhs, rhs, max_work_elems=4 * 1024 * 1024):
    bsz, t, d = lhs.shape
    step = max(1, max_work_elems // max(1, bsz * d))
    out = torch.zeros((d,), dtype=torch.float32, device=lhs.device)
    for start in range(0, t, step):
        end = min(t, start + step)
        out += (lhs[:, start:end, :] * rhs[:, start:end, :]).sum(dim=(0, 1))
    return out


def compute_dw_db_kernel_order(x, dy_eff, initial_state, w):
    bsz, t, d = x.shape
    x_f = x.to(torch.float32)
    state_f = initial_state.to(torch.float32) if initial_state is not None else None
    dw = torch.zeros((w, d), dtype=torch.float32, device=x.device)
    db = _sum_batch_time(dy_eff)

    for i_w in range(w):
        w_idx = w - 1 - i_w
        valid_t = t - i_w
        if valid_t > 0:
            dw[w_idx, :] += _sum_batch_time_product(
                dy_eff[:, i_w : i_w + valid_t, :], x_f[:, :valid_t, :]
            )

    # The ATK cases initialize initial_state to zero.  Keep the mathematically
    # complete term here so non-zero state cases remain valid if enabled later.
    if state_f is not None:
        head = min(t, w - 1)
        for i_w in range(1, w):
            rows = min(head, i_w)
            if rows == 0:
                continue
            w_idx = w - 1 - i_w
            state_rows = state_f[:, w - i_w : w - i_w + rows, :]
            dw[w_idx, :] += _sum_batch_time_product(dy_eff[:, :rows, :], state_rows)

    return dw.contiguous(), db.contiguous()


def _causal_conv1d_bwd_cpu_fixed(
    x,
    y,
    weight,
    dy,
    initial_state,
    dht,
    activation,
    input_dtype=None,
    cast_reduction_outputs=True,
):
    b, t, d = x.shape
    w = weight.shape[0]
    input_dtype = input_dtype or x.dtype

    x_f = _to_compute_tensor(x, input_dtype)
    y_f = _to_compute_tensor(y, input_dtype)
    weight_f = _to_compute_tensor(weight, input_dtype)
    dy_f = _to_compute_tensor(dy, input_dtype)
    state_f = _to_compute_tensor(initial_state, input_dtype) if initial_state is not None else None
    dht_f = _to_compute_tensor(dht, input_dtype) if dht is not None else None
    dy_eff = activation_backward(dy_f, y_f, activation)

    dx = torch.zeros((b, t, d), dtype=torch.float32, device=x.device)
    for i_w in range(w):
        n = t - i_w
        if n > 0:
            dx[:, :n, :] += dy_eff[:, i_w : i_w + n, :] * weight_f[w - 1 - i_w].view(1, 1, d)

    if dht_f is not None and w > 1:
        start = max(0, t - (w - 1))
        count = t - start
        if count > 0:
            dx[:, start:t, :] += dht_f[:, 1 : 1 + count, :].to(torch.float32)

    dw, db = compute_dw_db_kernel_order(x_f, dy_eff, state_f, w)

    dh0 = torch.zeros((b, w, d), dtype=torch.float32, device=x.device)
    if state_f is not None:
        head = min(t, w - 1)
        for slot in range(1, w):
            for row in range(min(head, slot)):
                k = slot - 1 - row
                dh0[:, slot, :] += dy_eff[:, row, :] * weight_f[k].view(1, d)

    if cast_reduction_outputs:
        dw = _cast_output_to_input_dtype(dw, input_dtype)
        db = _cast_output_to_input_dtype(db, input_dtype)
    return (
        _cast_output_to_input_dtype(dx, input_dtype),
        dw,
        db,
        _cast_output_to_input_dtype(dh0, input_dtype),
    )


def causal_conv1d_bwd_cpu(x, y, weight, dy, initial_state, dht, query_start_loc, activation,
                          input_layout="BSND", input_dtype=None):
    layout = _normalize_layout(input_layout)
    input_dtype = input_dtype or x.dtype
    x_logic = x.contiguous()
    y_logic = _input_to_logical(y, layout)
    dy_logic = _input_to_logical(dy, layout)

    if x_logic.dim() == 3:
        return _causal_conv1d_bwd_cpu_fixed(
            x_logic, y_logic, weight, dy_logic, initial_state, dht, activation, input_dtype
        )

    total_tokens, d = x_logic.shape
    if query_start_loc is None or isinstance(query_start_loc, int):
        qsl = [0, total_tokens]
    elif torch.is_tensor(query_start_loc):
        qsl = [int(v) for v in query_start_loc.detach().cpu().flatten().tolist()]
    else:
        qsl = [int(v) for v in query_start_loc]

    bsz = len(qsl) - 1
    w = weight.shape[0]
    dx = torch.zeros((total_tokens, d), dtype=input_dtype, device=x.device)
    dw = torch.zeros((w, d), dtype=torch.float32, device=x.device)
    db = torch.zeros((d,), dtype=torch.float32, device=x.device)
    dh0 = torch.zeros((bsz, w, d), dtype=input_dtype, device=x.device)
    for b in range(bsz):
        start, end = qsl[b], qsl[b + 1]
        if end <= start:
            continue
        init_b = initial_state[b : b + 1] if initial_state is not None else None
        dht_b = dht[b : b + 1] if dht is not None else None
        dx_b, dw_b, db_b, dh0_b = _causal_conv1d_bwd_cpu_fixed(
            x_logic[start:end].unsqueeze(0),
            y_logic[start:end].unsqueeze(0),
            weight,
            dy_logic[start:end].unsqueeze(0),
            init_b,
            dht_b,
            activation,
            input_dtype,
            cast_reduction_outputs=False,
        )
        dx[start:end] = dx_b.squeeze(0)
        dw += dw_b.to(torch.float32)
        db += db_b.to(torch.float32)
        dh0[b : b + 1] = dh0_b

    return (
        dx.contiguous(),
        dw.to(input_dtype).contiguous(),
        db.to(input_dtype).contiguous(),
        dh0.contiguous(),
    )


@register("executor_causal_conv1d_bwd")
class CausalConv1dBwdApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(CausalConv1dBwdApi, self).__init__(task_result)

    def _target_input_dtype(self):
        case_config = getattr(self.task_result, "case_config", None)
        inputs = getattr(case_config, "inputs", None) or []
        for input_config in inputs:
            if getattr(input_config, "name", None) == "x":
                dtype = DTYPE_MAP.get(str(getattr(input_config, "dtype", "")).lower())
                if dtype is not None:
                    return dtype
        return torch.float32

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        return causal_conv1d_bwd_cpu(
            input_data.kwargs["x"],
            input_data.kwargs["y"],
            input_data.kwargs["weight"],
            input_data.kwargs["dy"],
            input_data.kwargs["initial_state"],
            input_data.kwargs["dht"],
            input_data.kwargs["queryStartLoc"],
            input_data.kwargs["activation"],
            input_data.kwargs.get("inputLayout", "BSND"),
            self._target_input_dtype(),
        )

    def init_by_input_data(self, input_data: InputDataset):
        dtype = self._target_input_dtype()
        input_layout = _normalize_layout(input_data.kwargs.get("inputLayout", "BSND"))
        x = input_data.kwargs["x"].to(dtype).contiguous()
        weight = input_data.kwargs["weight"].to(dtype).contiguous()
        dy = input_data.kwargs["dy"].to(dtype).contiguous()
        b, t, d, n_heads, head_dim = _layout_meta(dy, input_layout)
        w = weight.shape[0]
        activation = int(input_data.kwargs["activation"])

        initial_state = input_data.kwargs["initial_state"].to(dtype).contiguous()
        dht = input_data.kwargs["dht"].to(dtype).contiguous()

        x_logic = x
        if input_layout in ("TND", "NTD"):
            query_start_loc = [0, t]
        else:
            query_start_loc = [0, t]

        if activation == 0:
            y = torch.zeros_like(dy)
        else:
            if x_logic.dim() == 2:
                y_logic = causal_conv1d_preactivation(
                    x_logic.detach().cpu().unsqueeze(0),
                    weight.detach().cpu(),
                    initial_state.detach().cpu(),
                ).squeeze(0).to(dtype)
            else:
                y_logic = causal_conv1d_preactivation(
                    x_logic.detach().cpu(), weight.detach().cpu(), initial_state.detach().cpu()
                ).to(dtype)
            y = _logical_to_input(y_logic, input_layout, n_heads, head_dim).to(dtype)

        if self.device == "pyaclnn":
            x = x.npu()
            y = y.npu()
            weight = weight.npu()
            dy = dy.npu()
            initial_state = initial_state.npu()
            dht = dht.npu()

        input_data.kwargs["x"] = x
        input_data.kwargs["y"] = y
        input_data.kwargs["weight"] = weight
        input_data.kwargs["dy"] = dy
        input_data.kwargs["initial_state"] = initial_state
        input_data.kwargs["dht"] = dht
        input_data.kwargs["queryStartLoc"] = query_start_loc
        input_data.kwargs["activation"] = activation
        input_data.kwargs["inputLayout"] = input_layout

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND


@register("aclnn_causal_conv1d_bwd")
class CausalConv1dBwdAclnnApi(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_layout = _normalize_layout(input_data.kwargs.get("inputLayout", "BSND"))
        input_data.kwargs["inputLayout"] = input_layout
        input_data.kwargs["queryStartLoc"] = _query_start_loc_for_layout(input_data, input_layout)
        return super().init_by_input_data(input_data)

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND

    def get_cpp_func_signature_type(self):
        return """aclnnStatus aclnnCausalConv1dBwdGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *yOptional,
    const aclTensor *weight,
    const aclTensor *dy,
    const aclTensor *initialStateOptional,
    const aclTensor *dhtOptional,
    const aclIntArray *queryStartLocOptional,
    int64_t activation,
    char *inputLayoutOptional,
    const aclTensor *dxOut,
    const aclTensor *dwOutOptional,
    const aclTensor *dbOutOptional,
    const aclTensor *dh0OutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);"""
