# Copyright (c) Tianjin University, Ltd. 2026. All rights reserved.
# chunk_bwd_dqkwg 算子执行适配 executor
# 职责:
#   1. 按 ATK plugin/executor 合同构造输入 tensor
#   2. 根据 case 中的 route 调用对应 DUT (ascendc / aclnn)
#   3. 提供双标杆所需的 golden (cpu) 和 benchmark (cpu_benchmark)
#   4. 处理输出命名、有效区和 ATK 需要的结构转换
# 注意: direct_launch 路径暂不实现, 待 ATK 社区提供通用 backend 后再补充
import os
import sys
import math
from typing import List, Optional, Tuple

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

# 将算子实现目录下的 CPU 参考实现加入 import 路径
_OP_TEST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "fla", "ops", "ascendc", "gdn", "chunk_gdn_bwd", "chunk_bwd_dqkwg", "tests",
)
if _OP_TEST_DIR not in sys.path:
    sys.path.insert(0, _OP_TEST_DIR)
from chunk_bwd_dqkwg_cpu import chunk_bwd_dqkwg_cpu  # noqa: E402


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def create_gate_g(B: int, H: int, T: int, gtype) -> torch.Tensor:
    """生成满足约束的 g: 负数且沿 T 单调递减.

    复刻原代码逻辑: 使用 torch.linspace 生成从接近 0 到较负的递减序列,
    添加微小 margin 避免 linspace 端点精度问题.
    """
    lo, hi = -5e-2, -5e-5
    span = hi - lo
    margin = max(span * 1e-7, 1e-12)
    g_t = torch.linspace(
        float(hi) - margin,
        float(lo) + margin,
        T,
        dtype=torch.float64,
    )
    return g_t.unsqueeze(0).unsqueeze(0).expand(B, H, T).contiguous().to(gtype)


def generate_tensor(shape, data_type, data_max) -> torch.Tensor:
    """生成均匀分布的随机 tensor, 范围 [-data_max, data_max]."""
    tensor = torch.rand(shape) * (data_max * 2) - data_max
    return tensor.to(data_type)


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def cdiv(a, b: int):
    return (a + b - 1) // b


def prepare_chunk_indices(cu_seqlens: List[int], chunk_size: int) -> List[int]:
    """基于 cu_seqlens 生成扁平化的 chunk 索引.

    逻辑复刻原代码:
    1. 计算每个序列的长度: lens[i] = cu_seqlens[i+1] - cu_seqlens[i]
    2. 计算每个序列需要的 chunk 数: ceil(lens[i] / chunk_size)
    3. 生成对应的 (sequence_id, chunk_id) 对并扁平化为 [s0, c0, s1, c1, ...]
    """
    indices = []
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        length = end - start
        if length <= 0:
            continue
        num_chunks = (length + chunk_size - 1) // chunk_size
        for chunk_id in range(num_chunks):
            indices.append(i)
            indices.append(chunk_id)
    return indices


def cumsum_cu_seqlens(cu_seqlens: torch.LongTensor) -> List[int]:
    """对 cu_seqlens 做 cumsum 并左 pad 一个 0."""
    return torch.nn.functional.pad(
        torch.cumsum(cu_seqlens, dim=0),
        (1, 0),
        value=0,
    ).tolist()


def _as_int_list(val) -> Optional[List[int]]:
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return [int(x) for x in val.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in val]


# ---------------------------------------------------------------------------
# CPU 参考实现 (golden / benchmark)
# ---------------------------------------------------------------------------

def chunk_bwd_dqkwg_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    w: Optional[torch.Tensor],
    g: Optional[torch.Tensor],
    dv: torch.Tensor,
    scale: Optional[float],
    cu_seqlens: Optional[List[int]],
    chunk_size: int = 64,
    benchmark: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """CPU 参考实现, 同时用于 golden (fp32) 和 benchmark (fp64)."""
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    do_t = do.transpose(1, 2).contiguous()
    dv_t = dv.transpose(1, 2).contiguous()
    g_t = g.transpose(1, 2).contiguous() if g is not None else None
    w_t = w.transpose(1, 2).contiguous() if w is not None else None
    h_t = h.permute(0, 2, 1, 3, 4).contiguous()
    dh_t = dh.permute(0, 2, 1, 3, 4).contiguous()

    cu_seqlens_tensor = (
        torch.tensor(cu_seqlens, dtype=torch.int64) if cu_seqlens is not None else None
    )

    dq, dk, dw, dg = chunk_bwd_dqkwg_cpu(
        q_t, k_t, v_t, do_t, h_t, dh_t, w_t, g_t, dv_t,
        scale, cu_seqlens_tensor, chunk_size,
        benchmark=benchmark,
    )

    dq = dq.transpose(1, 2).contiguous()
    dk = dk.transpose(1, 2).contiguous()
    dw = dw.transpose(1, 2).contiguous()
    if dg is not None:
        dg = dg.transpose(1, 2).contiguous()

    return dq, dk, dw, dg


# ---------------------------------------------------------------------------
# ATK Executor
# ---------------------------------------------------------------------------

@register("executor_chunk_bwd_dqkwg")
class FunctionApi(BaseApi):
    """chunk_bwd_dqkwg 算子 ATK 执行适配.

    支持的 route:
      - ascendc: 调用 fla_npu.ops.ascendc.npu_chunk_bwd_dqkwg
      - aclnn:   调用 aclnnChunkBwdDqkwg 两段式接口
      - direct_launch: 暂不支持, 待 ATK 社区提供通用 backend

    golden (cpu) 使用 fp32 参考实现, benchmark (cpu_benchmark) 使用 fp64 参考实现.
    """

    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.qkv_type = None

    # ------------------------------------------------------------------
    # golden: fp32 CPU 参考实现
    # ------------------------------------------------------------------
    def cpu(self, input_data: InputDataset, with_output: bool = False):
        q = input_data.kwargs["q"]
        k = input_data.kwargs["k"]
        v = input_data.kwargs["v"]
        do = input_data.kwargs["do"]
        h = input_data.kwargs["h"]
        dh = input_data.kwargs["dh"]
        w = input_data.kwargs.get("w", None)
        g = input_data.kwargs["g"]
        dv = input_data.kwargs["dv"]
        cu_seqlens = input_data.kwargs.get("cu_seqlens", None)
        chunk_size = input_data.kwargs["chunk_size"]
        scale = input_data.kwargs["scale"]

        dq, dk, dw_out, dg = chunk_bwd_dqkwg_torch(
            q, k, v, do, h, dh, w, g, dv, scale, cu_seqlens, chunk_size
        )

        # 将输出 dtype 对齐到输入 dtype
        if self.qkv_type == "bf16":
            dq = dq.to(torch.bfloat16)
            dk = dk.to(torch.bfloat16)
            dw_out = dw_out.to(torch.bfloat16) if dw_out is not None else None
        elif self.qkv_type == "fp16":
            dq = dq.to(torch.float16)
            dk = dk.to(torch.float16)
            dw_out = dw_out.to(torch.float16) if dw_out is not None else None

        # is_mix=False 时 dg 也要对齐到 qkv dtype
        is_mix = input_data.kwargs.get("is_mix", True)
        if not is_mix:
            if self.qkv_type == "bf16":
                dg = dg.to(torch.bfloat16)
            elif self.qkv_type == "fp16":
                dg = dg.to(torch.float16)

        return dq, dk, dw_out, dg

    # ------------------------------------------------------------------
    # benchmark: fp64 CPU 参考实现
    # ------------------------------------------------------------------
    def cpu_benchmark(self, input_data: InputDataset, with_output: bool = False):
        q = input_data.kwargs["q"].to(torch.float64)
        k = input_data.kwargs["k"].to(torch.float64)
        v = input_data.kwargs["v"].to(torch.float64)
        do = input_data.kwargs["do"].to(torch.float64)
        h = input_data.kwargs["h"].to(torch.float64)
        dh = input_data.kwargs["dh"].to(torch.float64)
        w = input_data.kwargs.get("w", None)
        if w is not None:
            w = w.to(torch.float64)
        g = input_data.kwargs["g"].to(torch.float64)
        dv = input_data.kwargs["dv"].to(torch.float64)
        cu_seqlens = input_data.kwargs.get("cu_seqlens", None)
        chunk_size = input_data.kwargs["chunk_size"]
        scale = input_data.kwargs["scale"]

        dq, dk, dw_out, dg = chunk_bwd_dqkwg_torch(
            q, k, v, do, h, dh, w, g, dv, scale, cu_seqlens, chunk_size,
            benchmark=True
        )

        return dq, dk, dw_out, dg

    # ------------------------------------------------------------------
    # DUT 调用: 根据 route 分发到 ascendc 或 aclnn
    # ------------------------------------------------------------------
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        q = input_data.kwargs["q"]
        # float64 输入走 benchmark 路径
        if q.dtype == torch.float64:
            return self.cpu_benchmark(input_data, with_output)

        route = input_data.kwargs.get("route", "ascendc")
        if route == "ascendc":
            return self._call_ascendc(input_data, with_output)
        elif route == "aclnn":
            return self._call_aclnn(input_data, with_output)
        else:
            raise NotImplementedError(
                f"route '{route}' is not supported, "
                "currently only 'ascendc' and 'aclnn' are implemented; "
                "'direct_launch' is pending ATK community backend"
            )

    # ------------------------------------------------------------------
    # ascendc route: 调用 fla_npu.ops.ascendc.npu_chunk_bwd_dqkwg
    # ------------------------------------------------------------------
    def _call_ascendc(self, input_data: InputDataset, with_output: bool = False):
        from fla_npu.ops import ascendc as ascendc_ops

        q = input_data.kwargs["q"].npu()
        k = input_data.kwargs["k"].npu()
        v = input_data.kwargs["v"].npu()
        g = input_data.kwargs["g"].npu()
        h = input_data.kwargs["h"].npu()
        do = input_data.kwargs["do"].npu()
        dh = input_data.kwargs["dh"].npu()
        dv = input_data.kwargs["dv"].npu()
        chunk_size = input_data.kwargs["chunk_size"]
        scale = input_data.kwargs.get("scale", None)
        cu_seqlens = input_data.kwargs.get("cu_seqlens", None)
        chunk_indices = input_data.kwargs.get("chunk_indices", None)

        # cu_seqlens / chunk_indices 转 NPU tensor
        if cu_seqlens is not None:
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int64, device=q.device)
        if chunk_indices is not None:
            chunk_indices = torch.tensor(chunk_indices, dtype=torch.int64, device=q.device)

        dq, dk, dw, dg = ascendc_ops.npu_chunk_bwd_dqkwg(
            q, k, v, g, h, do, dh, dv, chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            w=None,
            g_gamma=None,
            scale=scale,
            use_exp2=False,
            transpose_state_layout=False,
        )
        return dq, dk, dw, dg

    # ------------------------------------------------------------------
    # aclnn route: 调用 aclnnChunkBwdDqkwg 两段式接口
    # ------------------------------------------------------------------
    def _call_aclnn(self, input_data: InputDataset, with_output: bool = False):
        from fla_npu.ops.ascendc._aclnn_ctypes import npu_chunk_bwd_dqkwg as _aclnn_call

        q = input_data.kwargs["q"].npu()
        k = input_data.kwargs["k"].npu()
        v = input_data.kwargs["v"].npu()
        g = input_data.kwargs["g"].npu()
        h = input_data.kwargs["h"].npu()
        do = input_data.kwargs["do"].npu()
        dh = input_data.kwargs["dh"].npu()
        dv = input_data.kwargs["dv"].npu()
        chunk_size = input_data.kwargs["chunk_size"]
        scale = input_data.kwargs.get("scale", None)
        cu_seqlens = input_data.kwargs.get("cu_seqlens", None)
        chunk_indices = input_data.kwargs.get("chunk_indices", None)

        # cu_seqlens / chunk_indices 转 NPU tensor
        if cu_seqlens is not None:
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int64, device=q.device)
        if chunk_indices is not None:
            chunk_indices = torch.tensor(chunk_indices, dtype=torch.int64, device=q.device)

        dq, dk, dw, dg = _aclnn_call(
            q, k, v, g, h, do, dh, dv, chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            w=None,
            g_gamma=None,
            scale=scale,
            use_exp2=False,
            transpose_state_layout=False,
        )
        return dq, dk, dw, dg

    # ------------------------------------------------------------------
    # 输入预处理: 构造满足算子约束的 tensor
    # ------------------------------------------------------------------
    def init_by_input_data(self, input_data: InputDataset):
        B, HK, T_json, K = input_data.kwargs["q"].shape
        HV = input_data.kwargs["v"].shape[1]
        V = input_data.kwargs["v"].shape[3]
        n_ratio = HV // HK

        q = input_data.kwargs["q"]
        k = input_data.kwargs["k"]
        v = input_data.kwargs["v"]
        do = input_data.kwargs["do"]
        h = input_data.kwargs["h"]
        dh = input_data.kwargs["dh"]
        w = input_data.kwargs.get("w", None)
        g = input_data.kwargs.get("g", None)
        dv = input_data.kwargs["dv"]
        cu_seqlens = input_data.kwargs.get("cu_seqlens", None)
        chunk_indices = input_data.kwargs.get("chunk_indices", None)
        chunk_size = input_data.kwargs["chunk_size"]
        scale = input_data.kwargs.get("scale", None)

        is_fix = input_data.kwargs.get("is_fix", True)
        self.qkv_type = input_data.kwargs.get("qkv_type", None)

        # 确定 dtype
        if self.qkv_type is None:
            qkv_type = q.dtype
        elif self.qkv_type == "bf16":
            qkv_type = torch.bfloat16
        elif self.qkv_type == "fp16":
            qkv_type = torch.float16
        else:
            qkv_type = q.dtype

        # 确定 g dtype
        is_mix = input_data.kwargs.get("is_mix", True)
        if is_mix:
            g_type = torch.float32
        else:
            g_type = qkv_type

        # 根据定长/变长模式构造输入
        if not is_fix:
            # 变长模式
            if cu_seqlens is not None and not isinstance(cu_seqlens, list):
                cu_seqlens = _as_int_list(cu_seqlens)
            if cu_seqlens is None:
                # 默认变长配置
                cu_seqlens = [0, T_json // 2, T_json]
            cu_seqlens = cumsum_cu_seqlens(torch.tensor(cu_seqlens, dtype=torch.int64))
            T = cu_seqlens[-1]
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
            num_chunks = len(chunk_indices) // 2

            # 变长模式 B 必须为 1
            B = 1
            q = torch.rand((B, HK, T, K), dtype=qkv_type) * 5e-7
            k = torch.rand((B, HK, T, K), dtype=qkv_type) * 5e-2
            v = torch.rand((B, HV, T, V), dtype=qkv_type) * 5e-2
            do = torch.rand((B, HV, T, V), dtype=qkv_type) * 5e-7
            dv = torch.rand((B, HV, T, V), dtype=qkv_type) * 5e-1
            w = None
            g = create_gate_g(B, HV, T, g_type)
            h = torch.rand((B, HV, num_chunks, K, V), dtype=qkv_type) * 5e-2
            dh = torch.rand((B, HV, num_chunks, K, V), dtype=qkv_type) * 5e-2
        else:
            # 定长模式
            cu_seqlens = None
            chunk_indices = None
            num_chunks = (T_json + chunk_size - 1) // chunk_size
            T = T_json
            dtype = qkv_type
            Gtype = g_type

            q = torch.randn(B, HK, T, K, dtype=dtype, requires_grad=True)
            k = torch.randn(B, HK, T, K, dtype=dtype, requires_grad=True)
            v = torch.randn(B, HV, T, V, dtype=dtype, requires_grad=True)

            # g 必须递减且为负数
            g = -torch.sort(torch.rand(B * T * HV) * 10, descending=False)[0]
            g = g.reshape((B, HV, T)).to(Gtype)

            do = torch.randn(B, HV, T, V, dtype=dtype, requires_grad=True)
            dv = torch.randn(B, HV, T, V, dtype=dtype, requires_grad=True)
            w = None

            h = torch.randn(B, HV, num_chunks, K, V, dtype=dtype, requires_grad=True)
            dh = torch.randn(B, HV, num_chunks, K, V, dtype=dtype, requires_grad=True)

        # 统一 dtype 对齐
        q = q.to(qkv_type)
        k = k.to(qkv_type)
        v = v.to(qkv_type)
        do = do.to(qkv_type)
        dv = dv.to(qkv_type)
        w = w.to(qkv_type) if w is not None else None
        h = h.to(qkv_type)
        dh = dh.to(qkv_type)
        g = g.to(g_type)

        # 回写处理后的输入
        input_data.kwargs["q"] = q
        input_data.kwargs["k"] = k
        input_data.kwargs["v"] = v
        input_data.kwargs["do"] = do
        input_data.kwargs["h"] = h
        input_data.kwargs["dh"] = dh
        input_data.kwargs["w"] = None
        input_data.kwargs["g"] = g
        input_data.kwargs["dv"] = dv
        input_data.kwargs["cu_seqlens"] = cu_seqlens
        input_data.kwargs["chunk_indices"] = chunk_indices
        input_data.kwargs["scale"] = scale
        input_data.kwargs["chunk_size"] = chunk_size
        input_data.kwargs["g_gamma"] = None
        input_data.kwargs["use_exp2"] = False
        input_data.kwargs["transpose_state_layout"] = False
        # 清理仅用于生成的中间参数
        input_data.kwargs.pop("is_mix", None)
        input_data.kwargs.pop("is_fix", None)
        input_data.kwargs.pop("qkv_type", None)
