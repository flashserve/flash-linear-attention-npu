#!/usr/bin/env python3
"""chunk_gated_delta_rule_fwd 的 ATK 原生双标杆执行器。"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

from atk_role_contract_new_path import role_for_atk_task
from gdn_reference import (
    GdnCase,
    canonical_chunk_indices,
    deterministic_initial_state,
    run_golden_reference,
)


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _uses_l2norm() -> bool:
    value = os.environ.get("GDN_ATK_USE_QK_L2NORM", "false").strip().lower()
    if value not in {"true", "false"}:
        raise RuntimeError("GDN_ATK_USE_QK_L2NORM 必须为 true 或 false")
    return value == "true"


def _normalized_qk(inputs, *, same_precision: bool):
    q, k, v, g, beta = inputs
    if same_precision:
        from fla.modules.l2norm import l2norm_fwd

        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
    else:
        q64 = q.double()
        k64 = k.double()
        q = q64 * torch.rsqrt((q64 * q64).sum(dim=-1, keepdim=True) + 1.0e-6)
        k = k64 * torch.rsqrt((k64 * k64).sum(dim=-1, keepdim=True) + 1.0e-6)
    return q, k, v, g, beta


def _scalar(value: Any):
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    return value


def _bool(value: Any) -> bool:
    value = _scalar(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int(value: Any) -> int:
    return int(_scalar(value))


def _float(value: Any) -> float:
    return float(_scalar(value))


def _text(value: Any) -> str:
    return str(_scalar(value)).strip()


def _cu_seqlens(specification: str, tokens: int, enabled: bool):
    if not enabled:
        return None
    values = tuple(int(value) for value in specification.split(",") if value.strip())
    if len(values) < 2 or values[0] != 0 or values[-1] != tokens:
        raise ValueError(
            f"cu_seqlens 必须从 0 开始并以 T={tokens} 结束：{values}"
        )
    return values


def build_inputs(values: dict[str, Any]):
    """把 ATK 原始输入冻结为三路共用的公开输入和执行合同。"""

    dtype_name = _text(values["qkv_dtype"]).lower()
    try:
        public_dtype = DTYPES[dtype_name]
    except KeyError as exc:
        raise ValueError(f"qkv_dtype 仅支持 {sorted(DTYPES)}，实际为 {dtype_name}") from exc

    q = values["q"].detach().cpu().to(public_dtype).contiguous()
    k = values["k"].detach().cpu().to(public_dtype).contiguous()
    v = values["v"].detach().cpu().to(public_dtype).contiguous()
    g = values["g"].detach().cpu().float().contiguous()
    beta = values["beta"].detach().cpu().to(public_dtype).contiguous()
    is_varlen = _bool(values["is_varlen"])
    scenario = _text(values["scenario"]).lower()
    case = GdnCase(
        batch=q.shape[0],
        k_heads=q.shape[1],
        v_heads=v.shape[1],
        tokens=q.shape[2],
        key_dim=q.shape[3],
        value_dim=v.shape[3],
        chunk_size=_int(values["chunk_size"]),
        scale=_float(values["scale"]),
        scenario=scenario,
        cu_seqlens=_cu_seqlens(
            _text(values["cu_seqlens_spec"]), q.shape[2], is_varlen
        ),
    )
    case.validate()
    if scenario == "dense" and is_varlen:
        raise ValueError("dense 场景不能设置 is_varlen=true")
    if scenario == "varlen" and not is_varlen:
        raise ValueError("varlen 场景必须设置 is_varlen=true")
    return case, public_dtype, (q, k, v, g, beta)


def _npu_device(device_id: int):
    import torch_npu  # noqa: F401

    # ATK 的 NPU worker 已绑定 node 中的设备；这里只构造显式 device，
    # 不修改进程级 current device。
    return torch.device(f"npu:{device_id}")


def _gpu_device(device_id: int):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU Triton golden 节点未检测到 CUDA 设备")
    return torch.device(f"cuda:{device_id}")


def _public_outputs(outputs, case: GdnCase, target: str):
    o, final_state, _, _ = outputs
    if case.output_final_state:
        if final_state is None:
            raise RuntimeError(f"请求 final_state，但 {target} 返回 None")
        return o, final_state
    return (o,)


def _normalize_o_for_comparison(output, case: GdnCase, role: str):
    """将三路结果的 o 统一为公开 BSND 布局。"""

    if role == "dut":
        expected_shape = (case.batch, case.tokens, case.v_heads, case.value_dim)
    elif role in {"benchmark", "golden"}:
        expected_shape = (case.batch, case.v_heads, case.tokens, case.value_dim)
    else:
        raise RuntimeError(f"无法识别输出角色：{role!r}")
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"{role} 的 o shape 应为 {expected_shape}，实际为 {tuple(output.shape)}"
        )
    if role in {"benchmark", "golden"}:
        return output.transpose(1, 2).contiguous()
    return output


def run_cpu(inputs, case: GdnCase, public_dtype):
    """运行 CPU FP64 benchmark。"""

    if _uses_l2norm():
        inputs = _normalized_qk(inputs, same_precision=False)
    outputs = run_golden_reference(*inputs, case, public_dtype)
    if not case.output_final_state:
        o, g_cumsum, a = outputs
        outputs = (o, None, g_cumsum, a)
    return _public_outputs(outputs, case, "CPU FP64 benchmark")


def run_triton(inputs, case: GdnCase):
    """运行 FLA Triton forward，作为 GPU golden。"""

    if case.chunk_size != 64:
        raise RuntimeError("GPU Triton 双标杆当前仅覆盖 chunk_size=64")
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_fwd_o
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2

    q, k, v, g, beta = inputs
    if _uses_l2norm():
        q, k, v, g, beta = _normalized_qk(inputs, same_precision=True)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    cu_seqlens = None
    chunk_indices = None
    if case.cu_seqlens is not None:
        cu_seqlens = torch.tensor(case.cu_seqlens, dtype=torch.int64, device=q.device)
        chunk_indices = torch.tensor(
            canonical_chunk_indices(case.cu_seqlens, case.chunk_size),
            dtype=torch.int64, device=q.device).reshape(-1, 2)
    g_cumsum = chunk_local_cumsum(g, chunk_size=case.chunk_size,
                                  cu_seqlens=cu_seqlens,
                                  chunk_indices=chunk_indices)
    g_exp2 = g_cumsum * RCP_LN2
    w, u, a = chunk_gated_delta_rule_fwd_intra(
        k, v, g_exp2, beta, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, chunk_size=case.chunk_size)
    initial_state = deterministic_initial_state(case)
    if initial_state is not None:
        initial_state = initial_state.to(q.device)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k, w, u, g=g_exp2, initial_state=initial_state,
        output_final_state=case.output_final_state, cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices, chunk_size=case.chunk_size)
    o = chunk_fwd_o(q, k, v_new, h, g=g_exp2, scale=case.scale,
                    cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                    chunk_size=case.chunk_size)
    outputs = (o.transpose(1, 2).contiguous(), final_state, g_cumsum, a)
    return _public_outputs(outputs, case, "GPU Triton golden")


def run_npu(role: str, inputs, case: GdnCase):
    """运行融合 DUT。"""

    from fla_npu.ops import ascendc

    q, k, v, g, beta = inputs
    initial_state = deterministic_initial_state(case)
    if initial_state is not None:
        initial_state = initial_state.to(q.device).contiguous()

    if role != "dut":
        raise RuntimeError(f"run_npu 只支持 DUT 角色，实际为：{role}")
    if not hasattr(ascendc, "chunk_gated_delta_rule_fwd"):
        raise RuntimeError("当前 fla_npu 包未提供 chunk_gated_delta_rule_fwd")
    cu_values = None if case.cu_seqlens is None else list(case.cu_seqlens)
    chunk_indices = canonical_chunk_indices(case.cu_seqlens, case.chunk_size)
    if q.dtype != torch.bfloat16 or case.value_dim != 128 or case.chunk_size != 64:
        raise RuntimeError("A5 新路径要求 BF16、K=V=128、chunk_size=64")
    if case.v_heads // case.k_heads > 4:
        raise RuntimeError("A5 新路径要求 HV/HK <= 4")
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    return _public_outputs(
        ascendc.chunk_gated_delta_rule_fwd(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=case.output_final_state,
            chunk_size=case.chunk_size,
            cu_seqlens=cu_values,
            chunk_indices=chunk_indices,
            scale=case.scale,
            use_exp2=False,
            use_qk_l2norm_in_kernel=_uses_l2norm(),
            disable_recompute=True,
            return_intermediate_states=False,
            state_v_first=False,
            layout="BSND",
        ),
        case,
        "融合算子",
    )


@register("executor_chunk_gated_delta_rule_fwd")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self._task_name = str(task_result.name or "")
        self._is_benchmark_task = bool(task_result.is_benchmark_task)
        self._case_id = int(task_result.case_config.id)
        self._case = None
        self._public_dtype = None
        self._inputs = None
        self._role = "uninitialized"
        self._npu_inputs = None
        self._gpu_inputs = None
        self._output_names = ()
        self._execution_device_id = None

    def init_by_input_data(self, input_data: InputDataset):
        values = input_data.kwargs
        case, public_dtype, inputs = build_inputs(values)
        q, k, v, g, beta = inputs

        self._case = case
        self._public_dtype = public_dtype
        self._inputs = inputs
        self._output_names = ("o", "final_state") if case.output_final_state else ("o",)

        # 保存并复用三路完全一致的有效输入，而不是生成器的 raw g/beta。
        values["q"] = q
        values["k"] = k
        values["v"] = v
        values["g"] = g
        values["beta"] = beta

        self._role = role_for_atk_task(
            self.device,
            self._task_name,
            self._is_benchmark_task,
        )
        if self._role == "benchmark":
            print(
                f"[gdn-double-atk] case={self._case_id} "
                "role=benchmark target=cpu_fp64_recurrence",
                flush=True,
            )
            return

        if self._role == "golden":
            device_id = int(self.device_id)
            self._execution_device_id = device_id
            device = _gpu_device(device_id)
            self._gpu_inputs = tuple(tensor.to(device).contiguous() for tensor in inputs)
            print(
                f"[gdn-double-atk] case={self._case_id} "
                "role=golden target=triton_gpu "
                f"node={self._task_name} device={device_id}",
                flush=True,
            )
            return

        device_id = int(self.device_id)
        self._execution_device_id = device_id
        device = _npu_device(device_id)
        self._npu_inputs = tuple(tensor.to(device).contiguous() for tensor in inputs)

        print(
            f"[gdn-double-atk] case={self._case_id} "
            "role=dut target=chunk_gated_delta_rule_fwd "
            f"node={self._task_name} device={device_id}",
            flush=True,
        )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self._case is None or self._inputs is None or self._public_dtype is None:
            raise RuntimeError("GDN ATK 执行器尚未初始化")
        with torch.no_grad():
            if self._role == "dut":
                if self._npu_inputs is None:
                    raise RuntimeError("dut NPU 输入未初始化")
                outputs = run_npu(self._role, self._npu_inputs, self._case)
            elif self._role == "benchmark":
                outputs = run_cpu(self._inputs, self._case, self._public_dtype)
            elif self._role == "golden":
                if self._gpu_inputs is None:
                    raise RuntimeError("golden GPU 输入未初始化")
                outputs = run_triton(self._gpu_inputs, self._case)
            else:
                raise RuntimeError(f"未知执行角色：{self._role}")

        if not with_output:
            return None
        normalized = []
        for index, output in enumerate(outputs):
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(f"output[{index}] 不是 Tensor：{type(output)!r}")
            output = output.detach().cpu().contiguous()
            if self._output_names[index] == "o":
                output = _normalize_o_for_comparison(output, self._case, self._role)
            normalized.append(output.contiguous())
        return normalized[0] if len(normalized) == 1 else tuple(normalized)

    def export_custom_data(self, *_args, **_kwargs):
        return {
            "output_names": list(self._output_names),
            "role": self._role,
            "target": {
                "dut": "chunk_gated_delta_rule_fwd",
                "benchmark": "cpu_fp64_recurrence",
                "golden": "triton_gpu",
            }.get(self._role, "uninitialized"),
            "benchmark_ops": [],
            "varlen_cumsum_transport": "",
            "benchmark_role_transport": "atk_cpu_benchmark_gpu_golden",
            "execution_device_id": self._execution_device_id,
            "a_padding_policy": "zero_non_contract_tail",
        }
