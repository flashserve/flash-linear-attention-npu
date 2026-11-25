"""Independent public stage interfaces and the two physical state layouts."""
import pytest
import torch
import torch.nn.functional as F
from test_public_backward_expanded import upstream
from fla_npu.ops.ascendc import (npu_chunk_kda_fwd,npu_chunk_kda_bwd_recompute,
    npu_chunk_kda_bwd_prepare,npu_chunk_gated_delta_rule_bwd_dhu,npu_chunk_kda_bwd_finalize)
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@pytest.mark.parametrize("v_first",[False,True])
def test_public_stages(dtype,v_first):
    torch.manual_seed(9);device=upstream.device
    q=F.normalize(torch.randn(1,65,2,64),dim=-1).to(device,dtype).requires_grad_()
    k=F.normalize(torch.randn_like(q.float()),dim=-1).to(dtype).requires_grad_()
    v=torch.randn(1,65,2,32,device=device,dtype=dtype,requires_grad=True)
    g=(-torch.rand_like(q.float())*.1).requires_grad_()
    beta=torch.rand(1,65,2,device=device,dtype=dtype,requires_grad=True)
    head=lambda x:x.transpose(1,2).contiguous()
    r=npu_chunk_kda_fwd(q,k,v,g,beta,.37,disable_recompute=True,use_exp2=True)
    o,ht,G,aqk,akk,w,u,qg,kg,vn,h,_=r
    rr=npu_chunk_kda_bwd_recompute(head(q.float()),head(k.float()),head(v),G,head(beta.float()),akk,64,use_gate_in_kernel=False,use_exp2=True)
    for a,b in zip(rr[1:],(w,u,qg,kg)):torch.testing.assert_close(a,b,rtol=0,atol=0)
    state=h.transpose(-1,-2).contiguous() if v_first else h
    do=torch.randn_like(v);doh=head(do)
    da,dv0,dqr=npu_chunk_kda_bwd_prepare(aqk,vn,doh,state,scale=.37,layout="BNSD",state_v_first=v_first)
    dh,dh0,dvs=npu_chunk_gated_delta_rule_bwd_dhu(qg,kg,w,doh,dv0,.37,64,gK=G,use_exp2=True,transpose_state_layout=v_first)
    assert dh.shape==state.shape and dh0 is None
    assert torch.count_nonzero(dh[:, -1])==0
    zero=torch.zeros(2,device=device);bias=torch.zeros(2,64,device=device)
    grad=npu_chunk_kda_bwd_finalize(head(q.float()),head(k.float()),head(v),G,head(g),head(beta.float()),zero,bias,akk,vn,state,dh,dvs,da,dqr,
        scale=.37,use_gate_in_kernel=False,safe_gate=False,use_exp2=True,state_v_first=v_first)
    torch.npu.synchronize()
    ref,_=upstream.naive_recurrent_kda(q.float(),k.float(),v.float(),g,beta.float(),scale=.37)
    gold=torch.autograd.grad(ref,(q,k,v,beta,g),do.float())
    for name,a,b in zip(("dq","dk","dv","db","dg"),gold,grad[:5]):
        upstream.assert_close(name,head(a).float(),b.float(),.025)
