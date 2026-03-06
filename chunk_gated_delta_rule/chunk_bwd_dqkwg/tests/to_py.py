import numpy as np 
import torch

import sys
from ml_dtypes import bfloat16

device_list = ('cpu',)
type_map = {
    torch.bfloat16: bfloat16,
    torch.float16: np.float16,  # np.float16
    torch.float32: np.float32,  # np.float32
    torch.float64: np.float64,  # np.float64
    torch.int32: np.int32,      # np.int32
    torch.int64: np.int64       # np.int64
}

path = "/data/huangjunzhe/GDN/ops-transformer_GDN/chunk_gated_delta_rule/chunk_bwd_dqkwg/tests/result/cpu_for_test"
# 示例: python script.py arg1 arg2
if len(sys.argv) > 1:
    path = sys.argv[1]
    print("path", path)

if True:
    with open(f'{path}/gen/config.cfg', 'r') as f:
        for line in f:
            if '=' in line:
                exec(line.strip())
else:
    scale = 0.08838834764831845
    chunk_size = 64
    num_chunks = 44
    seqlen_nums = 5
    B = 1
    H = 4
    T = 2816
    K = 128
    V = 128

# /root/data_nvme0n1/huangjunzhe/GDN/target/result/gen/dg_npu.bin
# for name in ["dg_npu", "dk_npu", "dq_npu", "dw_npu"]:
for name in ["dg", "dk", "dq", "dw"]:
    if name == "dg":
        # dtype = np.float32
        dtype = type_map[gtype]
        torchtype = gtype
    else:
        # dtype = np.float16
        dtype = type_map[datatype]
        torchtype = datatype

    binx = np.fromfile(f"{path}/gen/{name}_npu.bin", dtype).astype(np.float32)
    tp = torch.from_numpy(binx).to(torchtype) #.reshape(B,H,T,K)
    if name == "dg":
        tp = tp.reshape(B,H,T)
        # tp = torch.transpose(tp, 1, 2)
    else:
        tp = tp.reshape(B,H,T,K)
        # tp = torch.transpose(tp, 1, 2)
    torch.save(tp, f"{path}/gen/{name}_npu.pt")
    print(f"saved {tp.dtype} {path}/gen/{name}_npu.pt, shape {tp.shape}.")
    for device_dtype in device_list:
        tb = torch.load(f"{path}/{name}_{device_dtype}.pt")
        tb = tb.transpose(1, 2)
        # if tb.dtype == torch.bfloat16:
        #     print("converting bf16 to fp32 for ct tool verify")
        #     tb = tb.to(torch.float32)
        torch.save(tb, f"{path}/gen/{name}_{device_dtype}_ht.pt")      ##cpu ht转置
        print(f"saved {tb.dtype} {path}/gen/{name}_{device_dtype}_ht.pt, shape {tb.shape}.")