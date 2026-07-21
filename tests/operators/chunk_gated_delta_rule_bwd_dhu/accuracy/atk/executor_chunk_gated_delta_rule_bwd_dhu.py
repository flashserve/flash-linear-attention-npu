# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
import torch
from typing import List, Optional, Tuple

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


def create_gate_g(B: int, H: int, T: int, gtype):
    """与 test_chunk_gated_delta_rule_bwd_dhu.create_gate_g 一致，避免 g∈[-5,5] 时 exp 爆炸。"""
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


def prepare_chunk_indices(cu_seqlens, chunk_size=64):
    """chunk indices 生成（chunk_idx 从 0 开始），与 test_chunk_gated_delta_rule_bwd_dhu 一致。"""
    chunk_indices = []
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        chunk_num = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(chunk_num):
            chunk_indices.append(seq_idx)
            chunk_indices.append(chunk_idx)
    return chunk_indices


def cumsum_cu_seqlens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.nn.functional.pad(
        torch.cumsum(cu_seqlens, dim=0),
        (1, 0),
        value=0,
    )


def _as_int_list_cu_seqlens(cu_seqlens) -> Optional[List[int]]:
    if cu_seqlens is None:
        return None
    if isinstance(cu_seqlens, torch.Tensor):
        return [int(x) for x in cu_seqlens.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in cu_seqlens]


def _as_int_list_chunk_indices(chunk_indices) -> Optional[List[int]]:
    if chunk_indices is None:
        return None
    if isinstance(chunk_indices, torch.Tensor):
        return [int(x) for x in chunk_indices.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in chunk_indices]


def chunk_gated_delta_rule_bwd_dhu_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    cu_seqlens: Optional[List[int]] = None,
    chunk_indices: Optional[List[int]] = None,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    PyTorch 标杆：与 flash-linear-attention-npu 中 test_chunk_gated_delta_rule_bwd_dhu.py 保持一致。
    输入形状: [B, H, T, D]；算法见 docs/design.md。
    """
    device = q.device
    dtype = q.dtype

    B, H, T, K = q.shape
    V = do.shape[-1]
    BT = chunk_size

    cu_seqlens = _as_int_list_cu_seqlens(cu_seqlens)
    chunk_indices = _as_int_list_chunk_indices(chunk_indices)

    if cu_seqlens is not None:
        seq_total = cu_seqlens[-1]
        if seq_total > T:
            raise ValueError(
                f"cu_seqlens 末元 {seq_total} 超过 q 的时间维 T={T}，无法对齐 [B,H,T,V] 的 dv2。"
            )
        NT = len(chunk_indices) // 2
    else:
        seq_total = T
        NT = (T + BT - 1) // BT

    if scale is None:
        scale = 1.0

    chunk_info = []
    for i_t in range(NT):
        if cu_seqlens is not None:
            i_n = chunk_indices[i_t * 2]
            block_idx_in_token = chunk_indices[i_t * 2 + 1]
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            token_length = eos - bos
        else:
            i_n = 0
            block_idx_in_token = i_t
            bos = 0
            token_length = T

        start_t = block_idx_in_token * BT
        end_t = min((block_idx_in_token + 1) * BT, token_length)
        block_size_t = end_t - start_t
        global_start_t = bos + start_t
        global_end_t = bos + end_t

        chunk_info.append(
            {
                "i_t": i_t,
                "i_n": i_n,
                "block_idx_in_token": block_idx_in_token,
                "bos": bos,
                "token_length": token_length,
                "start_t": start_t,
                "end_t": end_t,
                "block_size_t": block_size_t,
                "global_start_t": global_start_t,
                "global_end_t": global_end_t,
            }
        )

    dh = torch.zeros(B, H, NT, K, V, device=device, dtype=torch.float32)
    if cu_seqlens is not None:
        dv2 = dv.clone()
    else:
        dv2 = torch.zeros(B, H, T, V, device=device, dtype=dv.dtype)

    for b in range(B):
        for i_h in range(H):
            if cu_seqlens is not None:
                num_tokens = len(cu_seqlens) - 1
            else:
                num_tokens = 1
            b_dh_buffers = {}
            for i_n in range(num_tokens):
                b_dh_buffers[i_n] = torch.zeros(K, V, device=device, dtype=torch.float32)

            for i_t in range(NT - 1, -1, -1):
                info = chunk_info[i_t]
                i_n = info["i_n"]
                b_dh = b_dh_buffers[i_n]

                dh[b, i_h, i_t] = b_dh

                global_start_t = info["global_start_t"]
                global_end_t = info["global_end_t"]
                block_size_t = info["block_size_t"]

                last_idx = min((info["block_idx_in_token"] + 1) * BT, info["token_length"]) - 1
                global_last_idx = info["bos"] + last_idx

                bg_last = bg_last_exp = b_g = b_g_exp = None
                if g is not None:
                    bg_last = g[b, i_h, global_last_idx]
                    b_g = g[b, i_h, global_start_t:global_end_t]
                    bg_last_exp = torch.exp(bg_last)
                    b_g_exp = torch.exp(b_g)

                b_do = do[b, i_h, global_start_t:global_end_t, :]
                b_dv_existing = dv[b, i_h, global_start_t:global_end_t, :]

                b_k = k[b, i_h, global_start_t:global_end_t, :]
                b_dv = torch.matmul(b_k.to(torch.float), b_dh.to(torch.float))

                if g is not None:
                    m_t = torch.arange(block_size_t, device=device) < block_size_t
                    gate_factor = torch.exp(bg_last - b_g).unsqueeze(-1)
                    mask_expanded = m_t.unsqueeze(-1).float()
                    b_dv *= gate_factor * mask_expanded

                b_dv += b_dv_existing.to(torch.float32)
                dv2[b, i_h, global_start_t:global_end_t, :] = b_dv.to(dv2.dtype)

                b_q = q[b, i_h, global_start_t:global_end_t, :]
                b_w = w[b, i_h, global_start_t:global_end_t, :]

                b_q_t = b_q.transpose(0, 1)
                b_w_t = b_w.transpose(0, 1)
                b_dh_for_update = b_dh.clone()
                if g is not None:
                    b_dh_for_update = b_dh_for_update * bg_last_exp

                b_q_gated = b_q_t
                if g is not None:
                    b_q_gated = b_q_t * b_g_exp.unsqueeze(0)

                term1 = torch.matmul(b_q_gated.to(torch.float), b_do.to(torch.float)) * scale
                term2 = torch.matmul(b_w_t.to(torch.float), b_dv.to(torch.float))

                new_b_dh = b_dh_for_update + term1 - term2
                b_dh_buffers[i_n] = new_b_dh

    return dh, None, dv2


@register("executor_chunk_gated_delta_rule_bwd_dhu")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.qkv_type = None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        q = input_data.kwargs["q"]
        k = input_data.kwargs["k"]
        w = input_data.kwargs["w"]
        d_o = input_data.kwargs["d_o"]
        dv = input_data.kwargs["dv"]
        g = input_data.kwargs["g"]
        cu_seqlens = input_data.kwargs["cu_seqlens"]
        chunk_indices = input_data.kwargs["chunk_indices"]
        chunk_size = input_data.kwargs["chunk_size"]
        scale = input_data.kwargs["scale"]

        dh, _, dv2 = chunk_gated_delta_rule_bwd_dhu_torch(
            q,
            k,
            w,
            d_o,
            dv,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            g=g,
            scale=scale,
            chunk_size=chunk_size,
        )
        if self.qkv_type == "bf16":
            dh = dh.to(torch.bfloat16)
            dv2 = dv2.to(torch.bfloat16)
        if self.qkv_type == "fp16":
            dh = dh.to(torch.float16)
            dv2 = dv2.to(torch.float16)

        return dh, dv2

    def init_by_input_data(self, input_data: InputDataset):
        B, H, T_json, K = input_data.kwargs["q"].shape
        V = input_data.kwargs["d_o"].shape[3]
        q = input_data.kwargs["q"]
        k = input_data.kwargs["k"]
        w = input_data.kwargs["w"]
        d_o = input_data.kwargs["d_o"]
        dv = input_data.kwargs["dv"]
        g = input_data.kwargs["g"]
        cu_seqlens = input_data.kwargs["cu_seqlens"]
        chunk_indices = input_data.kwargs["chunk_indices"]
        chunk_size = input_data.kwargs["chunk_size"]

        self.qkv_type = input_data.kwargs["qkv_type"]

        qkv_type = input_data.kwargs["q"].dtype
        g_type = input_data.kwargs["g"].dtype
        is_mix = input_data.kwargs["is_mix"]
        if not is_mix:
            g_type = qkv_type

        is_fix = input_data.kwargs["is_fix"]
        is_fix = False
        if not is_fix:
            cu_seqlens = cumsum_cu_seqlens(cu_seqlens)
            T = cu_seqlens[-1].item()
            q = torch.rand((B, H, T, K), dtype=qkv_type) * 5e-7
            k = torch.rand((B, H, T, K), dtype=qkv_type) * 5e-2
            w = torch.rand((B, H, T, K), dtype=qkv_type) * 5e-2
            d_o = torch.rand((B, H, T, V), dtype=qkv_type) * 5e-7
            dv = torch.rand((B, H, T, V), dtype=qkv_type) * 5e-1
            g = create_gate_g(B, H, T, g_type)
            cu_seqlens_list = cu_seqlens.tolist()
            chunk_indices = prepare_chunk_indices(cu_seqlens_list, chunk_size)
        else:
            cu_seqlens = None
            chunk_indices = None
            g = create_gate_g(B, H, T_json, g_type)

        q = q.to(qkv_type)
        k = k.to(qkv_type)
        w = w.to(qkv_type)
        d_o = d_o.to(qkv_type)
        dv = dv.to(qkv_type)
        g = g.to(g_type)

        if self.device == "pyaclnn":
            q = q.npu()
            k = k.npu()
            w = w.npu()
            d_o = d_o.npu()
            dv = dv.npu()
            g = g.npu()
            if cu_seqlens is not None:
                cu_seqlens = cu_seqlens.npu()
            if chunk_indices is not None:
                chunk_indices = torch.tensor(chunk_indices, dtype=torch.int64, device=q.device)

        input_data.kwargs["q"] = q
        input_data.kwargs["k"] = k
        input_data.kwargs["w"] = w
        input_data.kwargs["d_o"] = d_o
        input_data.kwargs["dv"] = dv
        input_data.kwargs["g"] = g
        input_data.kwargs["cu_seqlens"] = cu_seqlens
        input_data.kwargs["chunk_indices"] = chunk_indices
        input_data.kwargs.pop("is_mix")
        input_data.kwargs.pop("is_fix")
        input_data.kwargs.pop("qkv_type")
