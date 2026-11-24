# Copyright (c) 2026 Tianjin University, Ltd.
"""Phase A scaffold. CPU zeros describe the ABI only, never numerical accuracy."""

from chunk_kda_bwd_finalize_cpu import chunk_kda_bwd_finalize_cpu
import torch


def main():
    count = 0
    for layout in ("BNSD","BSND","NTD","TND"):
        for t in (1,63,64,65,129):
            for beta_type in (torch.bfloat16,torch.float32):
                for variable in (False,True):
                    for rstd in (False,True):
                        b,h = (1 if variable else 2),3
                        rank3 = layout in ("NTD","TND")
                        shape = (h,t,128) if rank3 else (b,h,t,128)
                        token = (t,h,128) if rank3 else (b,t,h,128)
                        q = torch.empty(shape,dtype=torch.bfloat16)
                        f = torch.empty(shape,dtype=torch.float32)
                        beta = torch.empty(shape[:-1],dtype=beta_type)
                        a = torch.empty((h,),dtype=torch.float32)
                        bias = torch.empty((h,128),dtype=torch.float32)
                        matrix = torch.empty((*shape[:-1],64),dtype=torch.bfloat16)
                        da = matrix.float()
                        n = sum((n+63)//64 for n in (0,t//2,t-t//2)) if variable else (t+63)//64
                        state = (n,h,128,128) if rank3 else (b,n,h,128,128)
                        s = torch.empty(state,dtype=torch.bfloat16)
                        outputs = chunk_kda_bwd_finalize_cpu(q,q,q,f,f,beta,a,bias,matrix,q,s,s,q,da,f,
                                         scale=0.125,output_layout=layout,
                                         q_rstd=beta.float() if rstd else None,
                                         k_rstd=beta.float() if rstd else None,
                                         cu_seqlens=[0,0,t//2,t] if variable else None)
                        qkv = token if layout in ("BSND","TND") else shape
                        assert [tuple(o.shape) for o in outputs] == [qkv,qkv,qkv,token[:-1],token,(h,),(h,128)]
                        assert [o.dtype for o in outputs] == [torch.bfloat16]*3+[beta_type]+[torch.float32]*3
                        count += 1
    print(f"finalize: {count} output-layout/beta/rstd/mode/tail ABI shape checks passed (zero placeholder only)")


if __name__ == "__main__":
    main()
