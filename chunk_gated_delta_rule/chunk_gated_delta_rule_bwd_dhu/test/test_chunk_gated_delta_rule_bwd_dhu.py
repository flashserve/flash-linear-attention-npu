import torch
import torch_npu
from typing import Optional, Tuple, List
import random
import ct
import aclnn_extension

def prepare_chunk_indices(cu_seqlens, chunk_size=64):
    """chunk indices生成函数(chunk_idx从0开始)"""
    chunk_indices = []
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        chunk_num = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(chunk_num):
            chunk_indices.append(seq_idx)
            chunk_indices.append(chunk_idx)
    return chunk_indices


def generate_cu_seqlens(
    cu_seqlens_len: int,
    total_length: int,
) -> List[int]:
    batchsize = cu_seqlens_len - 1
    remaining = total_length
    seq_lengths = []
    for i in range(batchsize - 1):
        min_len = 1
        max_len = remaining - (batchsize - 1 - i)
        if max_len < min_len:
            max_len = min_len
        seq_len = random.randint(min_len, max_len)
        seq_lengths.append(seq_len)
        remaining -= seq_len
    seq_lengths.append(remaining)

    cu_seqlens = [0]
    for seq_len in seq_lengths:
        cu_seqlens.append(cu_seqlens[-1] + seq_len)
    return cu_seqlens


def create_tensor(shape, dtype=torch.float16):
    return torch.rand(shape, dtype=dtype)


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
    PyTorch版本的chunk_gated_delta_rule_bwd_dhu函数
    输入形状: [B, H, T, D]
    支持定长序列(cu_seqlens=None)和变长序列(cu_seqlens!=None)
    """
    device = q.device
    dtype = q.dtype

    B, H, T, K = q.shape
    V = do.shape[-1]
    BT = chunk_size

    if cu_seqlens is not None:
        total_length = cu_seqlens[-1]
        NT = len(chunk_indices) // 2
    else:
        total_length = T
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

        chunk_info.append({
            'i_t': i_t,
            'i_n': i_n,
            'block_idx_in_token': block_idx_in_token,
            'bos': bos,
            'token_length': token_length,
            'start_t': start_t,
            'end_t': end_t,
            'block_size_t': block_size_t,
            'global_start_t': global_start_t,
            'global_end_t': global_end_t,
        })

    dh = torch.zeros(B, H, NT, K, V, device=device, dtype=torch.float32)
    dv2 = torch.zeros(B, H, total_length, V, device=device, dtype=dv.dtype)

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
                i_n = info['i_n']
                b_dh = b_dh_buffers[i_n]

                dh[b, i_h, i_t] = b_dh

                global_start_t = info['global_start_t']
                global_end_t = info['global_end_t']
                block_size_t = info['block_size_t']

                last_idx = min((info['block_idx_in_token'] + 1) * BT, info['token_length']) - 1
                global_last_idx = info['bos'] + last_idx

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

    return dh.to(dtype), None, dv2.to(dtype)


def test_fix(
    B: int,
    H: int,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    scale: float,
    ktype,
    gtype,
    seed: int = 0,
):
    torch.manual_seed(seed)
    if not hasattr(test_fix, "call_count"):
        test_fix.call_count = 1
    else:
        test_fix.call_count += 1

    q = create_tensor((B, H, T, K), dtype=ktype) * 5e-7
    k = create_tensor((B, H, T, K), dtype=ktype) * 5e-2
    w = create_tensor((B, H, T, K), dtype=ktype) * 5e-2
    d_o = create_tensor((B, H, T, V), dtype=ktype) * 5e-7
    dv = create_tensor((B, H, T, V), dtype=ktype) * 5e-1
    g = torch.arange(B * H * T, 0, -1).reshape((B, H, T)).to(gtype) * 5e-2

    dh_golden, _, dv2_golden = chunk_gated_delta_rule_bwd_dhu_torch(
        q, k, w, d_o, dv,
        cu_seqlens=None, chunk_indices=None,
        g=g, scale=scale, chunk_size=chunk_size
    )
    print("dh_golden shape is", dh_golden.shape)
    print("dv2_golden shape is", dv2_golden.shape)

    q_npu = q.npu()
    k_npu = k.npu()
    w_npu = w.npu()
    d_o_npu = d_o.npu()
    dv_npu = dv.npu()
    g_npu = g.npu()

    dh_npu, dh0_npu, dv2_npu = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q_npu, k_npu, w_npu, d_o_npu, dv_npu, 
        scale=scale, 
        chunk_size=chunk_size,
        g=g_npu,
        gK=None, 
        h0=None, 
        dht=None,
        cu_seqlens=None, 
        chunk_indices=None,
        use_exp2=False, 
        transpose_state_layout=False
    )

    print("dh_npu shape is", dh_npu.shape)
    print("dv2_npu shape is", dv2_npu.shape)

    print("dh comparison:")
    ct.single(dh_npu.cpu(), dh_golden)
    print("dv2 comparison:")
    ct.single(dv2_npu.cpu(), dv2_golden)
    print(f"test_fix 被调用了第 {test_fix.call_count} 次")


def test_variable(
    B: int,
    H: int,
    T: int,
    K: int,
    V: int,
    chunk_size: int,
    scale: float,
    cu_seqlens_len: int,
    ktype,
    gtype,
    seed: int = 0,
):
    torch.manual_seed(seed)
    if not hasattr(test_variable, "call_count"):
        test_variable.call_count = 1
    else:
        test_variable.call_count += 1

    q = create_tensor((B, H, T, K), dtype=ktype) * 5e-7
    k = create_tensor((B, H, T, K), dtype=ktype) * 5e-2
    w = create_tensor((B, H, T, K), dtype=ktype) * 5e-2
    d_o = create_tensor((B, H, T, V), dtype=ktype) * 5e-7
    dv = create_tensor((B, H, T, V), dtype=ktype) * 5e-1
    g = torch.arange(B * H * T, 0, -1).reshape((B, H, T)).to(gtype) * 5e-2

    cu_seqlens = generate_cu_seqlens(cu_seqlens_len, T)
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    dh_golden, _, dv2_golden = chunk_gated_delta_rule_bwd_dhu_torch(
        q, k, w, d_o, dv,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        g=g, scale=scale, chunk_size=chunk_size
    )


    q_npu = q.npu()
    k_npu = k.npu()
    w_npu = w.npu()
    d_o_npu = d_o.npu()
    dv_npu = dv.npu()
    g_npu = g.npu()

    dh_npu, dh0_npu, dv2_npu = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q_npu, k_npu, w_npu, d_o_npu, dv_npu, 
        scale=scale, 
        chunk_size=chunk_size,
        g=g_npu,
        gK=None, 
        h0=None, 
        dht=None,
        cu_seqlens=cu_seqlens, 
        chunk_indices=chunk_indices,
        use_exp2=False, 
        transpose_state_layout=False
    )


    print("dh comparison:")
    ct.single(dh_npu.cpu(), dh_golden)
    print("dv2 comparison:")
    ct.single(dv2_npu.cpu(), dv2_golden)
    print(f"test_variable 被调用了第 {test_variable.call_count} 次")


if __name__ == "__main__":
    # Fix length tests
    test_fix(B=2, H=2, T=65, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.float16, gtype=torch.float16)
    test_fix(B=4, H=4, T=128, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.float16, gtype=torch.float32)
    test_fix(B=8, H=8, T=256, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.float16, gtype=torch.float16)
    test_fix(B=16, H=16, T=512, K=128, V=128, chunk_size=64, scale=0.0625, ktype=torch.bfloat16, gtype=torch.bfloat16)
    test_fix(B=32, H=16, T=1024, K=128, V=128, chunk_size=64, scale=0.088, ktype=torch.bfloat16, gtype=torch.bfloat16)

    # # # Variable length tests
    test_variable(B=1, H=32, T=128, K=128, V=128, chunk_size=64, scale=0.011, cu_seqlens_len=2, ktype=torch.float16, gtype=torch.float16)
    test_variable(B=1, H=16, T=256, K=128, V=128, chunk_size=64, scale=0.011, cu_seqlens_len=3, ktype=torch.float16, gtype=torch.float32)
    test_variable(B=1, H=8, T=512, K=128, V=128, chunk_size=64, scale=0.011, cu_seqlens_len=4, ktype=torch.bfloat16, gtype=torch.bfloat16)
    test_variable(B=1, H=4, T=1024, K=128, V=128, chunk_size=64, scale=0.011, cu_seqlens_len=5, ktype=torch.float16, gtype=torch.float16)
    test_variable(B=1, H=32, T=2048, K=128, V=128, chunk_size=64, scale=0.011, cu_seqlens_len=8, ktype=torch.bfloat16, gtype=torch.bfloat16)
    test_variable(B=1, H=32, T=65536, K=128, V=128, chunk_size=64, scale=0.011, cu_seqlens_len=739, ktype=torch.bfloat16, gtype=torch.bfloat16)
    test_variable(B=1, H=32, T=65536, K=128, V=128, chunk_size=128, scale=0.011, cu_seqlens_len=739, ktype=torch.bfloat16, gtype=torch.bfloat16)
