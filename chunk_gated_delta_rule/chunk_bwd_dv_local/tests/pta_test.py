import torch
import torch_npu


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def cdiv(a: torch.LongTensor
    , b : int):
    torch.empty
    return (a + b - 1) // b

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    print("cu_seqlens is ", cu_seqlens)
    print("indices is ", indices)

    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

if __name__ == "__main__":
    torch.manual_seed(0)
    
    B, T, H, K, V = 1, 64, 1, 128, 128
    chunk_size=64
    scale = 1.0

    q = torch.randn(B, H, T, K, dtype=torch.float16, requires_grad=True).npu()
    k = torch.randn(B, H, T, K, dtype=torch.float16, requires_grad=True).npu()
    w = torch.randn(B, H, T, K, dtype=torch.float16, requires_grad=True).npu()
    do = torch.randn(B, H, T, V, dtype=torch.float16, requires_grad=True).npu()
    upper_tri_matrix = (torch.randn(chunk_size, chunk_size) > 0).to(torch.bool).npu()
    # dv = torch.randn(B, H, T, V, dtype=torch.float16, requires_grad=True).npu()
    g = torch.randn(B, H, T, dtype=torch.float16, requires_grad=True).npu()
    cu_seqlens = q.new_tensor([0, 64], dtype=torch.long).npu()

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunk_indices = chunk_indices.view(-1).npu()

    dv = torch_npu.npu_chunk_bwd_dv_local(q, k, do, g,upper_tri_matrix=None, g_gamma=None, A=None,cu_seqlens=cu_seqlens, chunk_indices = chunk_indices, scale=scale, chunk_size =chunk_size)
    # dv = torch_npu.npu_chunk_bwd_dv_local(q, k, do, g,upper_tri_matrix=upper_tri_matrix, g_gamma=g, A=q,cu_seqlens=cu_seqlens, chunk_indices = chunk_indices, scale=scale, chunk_size =chunk_size)
    print(f"==== dv.shape = {dv.shape} ",dv)
