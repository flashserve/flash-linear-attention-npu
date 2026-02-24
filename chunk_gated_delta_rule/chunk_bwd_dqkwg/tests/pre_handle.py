## 1. 生成chunk_indices
## 2. 转置左右T，H
## 3. 从pt生成bin文件
## 4. 写配置文件 B=？T=？H=？K=？V=？chunk_size=？num_chunks=？再从c++读取

import numpy as np
import torch
import sys
def compute_chunk_mapping(Lens, chunk_size=10):
    # Step 1: Compute how many chunks each sequence needs

    num_chunks = torch.zeros(Lens.shape)
    for i in range(Lens.shape[0]):
        if i == 0:
            num_chunks[0] = Lens[0]
        else:
            num_chunks[i] = Lens[i]- Lens[i-1]
    print(num_chunks)
    
    # Step 2: Generate indices (which sequence each chunk belongs to)
    indices = []
    for seq_idx, num in enumerate(num_chunks):
        indices.extend([seq_idx] * num)
    print(indices)
    
    # Step 3: Compute sequence IDs and chunk indices
    sequence_ids = []
    chunk_indices = []
    current_counts = {}
    
    for idx in indices:
        if idx not in current_counts:
            current_counts[idx] = 0
        else:
            current_counts[idx] += 1
        
        sequence_ids.append(idx)
        chunk_indices.append(current_counts[idx])
    
    # Combine into final result
    result = np.array(list(zip(sequence_ids, chunk_indices)))
    
    return result


from typing import Optional
import pickle
import math

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]

def cdiv(a: torch.LongTensor
    , b : int):
    torch.empty
    return (a + b - 1) // b

def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunkSize: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv(prepare_lens(cu_seqlens), chunkSize).tolist()])
    # print("cu_seqlens is ", cu_seqlens)
    # print("indices is ", indices)

    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

# # Example usage
# Lens = torch.tensor([   0, 1023, 1025, 1536, 2048])
# chunk_size = 16
# result = compute_chunk_mapping(Lens, chunk_size)
# print(result)

def get_inputs(pkl_path, transpose = True, dtype=torch.float16, gdtype=torch.float32):
    import pickle
    import json
    with open(f"{pkl_path}/input.pkl", 'rb') as f:
        data = pickle.load(f)
    q = data['q'].cpu().to(dtype)
    k = data['k'].cpu().to(dtype)
    v = data['v'].cpu().to(dtype)
    h = data['h'].cpu().to(dtype)
    g = data['g'].cpu().to(gdtype)
    do = data['do'].cpu().to(dtype)
    dh = data['dh'].cpu().to(dtype)
    dv = data['dv'].cpu().to(dtype)
    w = data['w'].cpu().to(dtype)
    if transpose == True:       ## 交换H，T
        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)
        v = torch.transpose(v, 1, 2)
        w = torch.transpose(w, 1, 2)
        g = torch.transpose(g, 1, 2)
        h = torch.transpose(h, 1, 2)
        dv = torch.transpose(dv, 1, 2)
        do = torch.transpose(do, 1, 2)
        dh = torch.transpose(dh, 1, 2)

    cu_seqlens = data['cu_seqlens'].cpu().to(torch.int64) if data['cu_seqlens'] is not None else None
    scale = data['scale']
    chunk_size = data['chunk_size']
    # chunk_indices = compute_chunk_mapping(cu_seqlens, chunk_size).astype(np.int64)
    # chunk_indices = torch.load(f"{pkl_path}/chunk_indices.pt").numpy()
    chunk_indices = data['chunk_indices'].numpy() if data['chunk_indices'] is not None else None
    num_chunks = chunk_indices.shape[0] if chunk_indices is not None else q.shape[2] // chunk_size
    isVarLen = 0 if cu_seqlens is None else 1
    if True:
        print("q", q.dtype, q.shape)
        print("k", k.dtype, k.shape)
        print("v", v.dtype, v.shape)
        print("w", w.dtype, w.shape)
        print("g", g.dtype, g.shape)
        print("h", h.dtype, h.shape)
        print("dv", dv.dtype, dv.shape)
        print("do", do.dtype, do.shape)
        print("dh", dh.dtype, dh.shape)
        print(f"scale = {scale}")
        print(f"chunk_size = {chunk_size}")
        print(f"num_chunks = {num_chunks}")
        print(f"seqlen_nums = {cu_seqlens.shape[0]}")
        if isVarLen == 1:
            print("cu_seqlens", cu_seqlens.dtype, cu_seqlens.shape, cu_seqlens)
            print("chunk_indices", chunk_indices.dtype, chunk_indices.shape)
        else:
            print("isVarLen is False!")
        

    q.numpy().tofile(f"{pkl_path}/gen/q.bin")
    k.numpy().tofile(f"{pkl_path}/gen/k.bin")
    v.numpy().tofile(f"{pkl_path}/gen/v.bin")
    h.numpy().tofile(f"{pkl_path}/gen/h.bin")
    g.numpy().tofile(f"{pkl_path}/gen/g.bin")
    do.numpy().tofile(f"{pkl_path}/gen/do.bin")
    dh.numpy().tofile(f"{pkl_path}/gen/dh.bin")
    dv.numpy().tofile(f"{pkl_path}/gen/dv.bin")
    w.numpy().tofile(f"{pkl_path}/gen/w.bin")
    if cu_seqlens != None:
        cu_seqlens.numpy().tofile(f"{pkl_path}/gen/cu_seqlens.bin")
    chunk_indices.tofile(f"{pkl_path}/gen/chunk_indices.bin")
    # 写入配置文件
    with open(f"{pkl_path}/gen/config.cfg", 'w') as f:
        f.write(f"scale = {scale}\n")
        f.write(f"chunk_size = {chunk_size}\n")
        f.write(f"num_chunks = {num_chunks}\n")
        # f.write(f"num_chunks = {q.shape[2] // chunk_size}\n")
        f.write(f"seqlen_nums = {cu_seqlens.shape[0]}\n")
        f.write(f"B = {q.shape[0]}\n")
        f.write(f"H = {q.shape[1]}\n")
        f.write(f"T = {q.shape[2]}\n")
        f.write(f"K = {q.shape[3]}\n")
        f.write(f"V = {v.shape[3]}\n")
        f.write(f"isVarLen = {isVarLen}\n")
    print(f"path is {pkl_path}")

path = '/data/huangjunzhe/GDN/ops-transformer_GDN/chunk_gated_delta_rule/chunk_bwd_dqkwg/tests/result/cpu_for_test'
# 示例: python script.py arg1 arg2
if len(sys.argv) > 1:
    path = sys.argv[1]
    print("path", path)
get_inputs(path)