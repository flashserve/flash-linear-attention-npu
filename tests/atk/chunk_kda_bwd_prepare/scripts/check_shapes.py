# Copyright (c) 2026 Tianjin University, Ltd.
"""Phase A scaffold. CPU zeros describe the ABI only, never numerical accuracy."""

from chunk_kda_bwd_prepare_cpu import chunk_kda_bwd_prepare_cpu
import torch


def main():
    for layout in ("BNSD", "BSND", "NTD", "TND"):
        for t in (1, 63, 64, 65, 129):
            for variable in (False, True):
                b, heads = (1 if variable else 2), 3
                if layout in ("NTD", "TND"):
                    b = 1
                shapes = {"BNSD": (b, heads, t, 64), "BSND": (b, t, heads, 64),
                          "NTD": (heads, t, 64), "TND": (t, heads, 64)}
                aqk = torch.empty(shapes[layout], dtype=torch.bfloat16)
                val = torch.empty((*aqk.shape[:-1], 128), dtype=torch.bfloat16)
                expected = ((heads, t) if aqk.ndim == 3 else (b, heads, t))
                chunks = sum((n + 63)//64 for n in (0,t//2,t-t//2)) if variable else (t+63)//64
                state = (chunks, heads, 128, 128) if aqk.ndim == 3 else (b,chunks,heads,128,128)
                for transpose in (False, True):
                    outputs = chunk_kda_bwd_prepare_cpu(aqk,val,val,torch.empty(state,dtype=torch.bfloat16),
                                     scale=0.125,layout=layout,state_v_first=transpose,
                                     cu_seqlens=[0,0,t//2,t] if variable else None)
                    assert [tuple(o.shape) for o in outputs] == [(*expected,64),(*expected,128),(*expected,128)]
                    assert [o.dtype for o in outputs] == [torch.float32,torch.bfloat16,torch.float32]
    print("prepare: 80 layout/mode/tail/state ABI shape checks passed (zero placeholder only)")


if __name__ == "__main__":
    main()
