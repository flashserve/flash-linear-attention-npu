import torch
import torch_npu
import os
os.environ['TBE_PARALLEL_COMPILE_ENABLE'] = '0'
os.environ['PARALLEL_COMPILE'] = '0'

def prepare_chunk_indices(cu_seqlens, chunk_size=64):
    """简化的chunk indices生成函数"""
    chunk_indices = []
    
    # 遍历每个序列
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        chunk_num = (seq_len + chunk_size - 1) // chunk_size  # 向上取整
        
        # 添加 [seq_idx, chunk_idx] 对
        for chunk_idx in range(chunk_num):
            chunk_indices.append(seq_idx)
            chunk_indices.append(chunk_idx + 1)  # chunk_idx从1开始
    
    return chunk_indices

B, T, H, K, V = 1, 32768, 32, 128, 128
isVarLen = True
if B > 1:
    isVarLen = False
chunk_size=64
torch.manual_seed(0)
torch_npu.npu.set_device(4)
q = torch.randn(B, H, T, K, dtype=torch.float16)
k = torch.randn(B, H, T, K, dtype=torch.float16)
w = torch.randn(B, H, T, K, dtype=torch.float16)
do = torch.randn(B, H, T, V, dtype=torch.float16)
dv = torch.randn(B, H, T, V, dtype=torch.float16)
g = torch.randn(B, H, T, dtype=torch.float16)
scale = k.shape[-1] ** -0.5

cu_seqlens = None
chunk_indices = None
if isVarLen:
    cu_seqlens = [0, 1024, 2048, 4096, 8192, 10240, 20480, 32768]
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    print("chunk_indices shape is ", len(chunk_indices))

dh, dh0, dv2 = torch_npu.npu_chunk_gated_delta_rule_bwd_dhu(q.npu(),k.npu(),w.npu(),do.npu(),dv.npu(),g.npu(),gK=None,h0=None,dht=None,cu_seqlens=cu_seqlens,chunk_indices=chunk_indices,scale=scale,chunk_size=chunk_size)
print("dh shape ",dh.shape)
print("dv2 shape ",dv2.shape)
