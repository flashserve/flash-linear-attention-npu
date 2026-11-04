# Copyright (c) 2026 Tianjin University, Ltd.
"""Phase A scaffold. CPU zeros describe the ABI only, never numerical accuracy."""

from chunk_kda_bwd_intra_cpu import chunk_kda_bwd_intra_cpu
import torch


def main():
    count = 0
    for layout, variable in (("BNSD",False),("BSND",False),("BSND",True),("TND",True)):
        for t in (1,63,64,65,129):
            for k in ((128,) if variable else (64,128,256)):
                for beta_type in (torch.bfloat16,torch.float32):
                    b,h = (1 if variable else 2),3
                    shape = ((h,t,k) if layout == "BNSD" else (t,h,k))
                    if layout != "TND": shape = (b,*shape)
                    q = torch.empty(shape,dtype=torch.bfloat16)
                    g = torch.empty(shape,dtype=torch.float32)
                    scalar = torch.empty(shape[:-1],dtype=torch.float32)
                    beta = torch.empty(shape[:-1],dtype=beta_type)
                    a = torch.empty((*shape[:-1],64),dtype=torch.float32)
                    outputs = chunk_kda_bwd_intra_cpu(q,q,g,beta,a,a,g,g,scalar,g,
                                     layout=layout,cu_seqlens=[0,0,t//2,t] if variable else None)
                    assert [tuple(o.shape) for o in outputs] == [shape,shape,shape[:-1],shape]
                    assert all(o.dtype == torch.float32 for o in outputs)
                    count += 1
    print(f"intra: {count} layout/mode/K/beta/tail ABI shape checks passed (zero placeholder only)")


if __name__ == "__main__":
    main()
