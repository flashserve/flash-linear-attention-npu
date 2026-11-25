"""Public negative cases must be rejected by ACLNN, before device execution."""
import pytest
import torch
import torch_npu
from fla_npu.ops.ascendc import npu_chunk_kda_fwd,npu_chunk_kda_bwd
@pytest.mark.parametrize("bad",["scale","chunk","beta","state","sigmoid","v_dtype"])
def test_aclnn_rejects(bad):
    device="npu";q=torch.zeros(1,1,1,64,device=device,dtype=torch.float16)
    v=torch.zeros(1,1,1,32,device=device,dtype=torch.float16)
    g=torch.zeros_like(q,dtype=torch.float32);beta=torch.ones(1,1,1,device=device,dtype=torch.float16)
    r=npu_chunk_kda_fwd(q,q,v,g,beta,.37,disable_recompute=True)
    _,_,G,aqk,akk,w,u,qg,kg,vn,h,_=r
    kwargs=dict(raw_g=g,layout="BSND",use_gate_in_kernel=False,safe_gate=False,disable_recompute=True)
    scale=.37
    if bad=="scale":scale=float("nan")
    if bad=="chunk":kwargs["chunk_size"]=32
    if bad=="beta":beta=torch.ones(1,1,2,device=device,dtype=torch.float16)
    if bad=="state":kwargs["initial_state"]=torch.zeros(1,1,64,32,device=device,dtype=torch.float32)
    if bad=="sigmoid":kwargs["allow_neg_eigval"]=True
    if bad=="v_dtype":v=v.float()
    with pytest.raises(RuntimeError,match="aclnnChunkKdaBwdGetWorkspaceSize failed: 161002"):
        npu_chunk_kda_bwd(q,q,v,beta,G,aqk,akk,w,qg,kg,vn,h,torch.zeros_like(v),scale,**kwargs)
