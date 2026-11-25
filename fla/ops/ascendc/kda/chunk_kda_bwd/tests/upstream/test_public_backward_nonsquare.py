"""Supplementary K64/V32 matrix using fla-org's recurrent reference and checks."""
import pytest
import torch
import torch.nn.functional as F
from test_public_backward_expanded import upstream
from public_backward_plugin import public_chunk
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@pytest.mark.parametrize("save",[True,False])
@pytest.mark.parametrize("norm",[True,False])
@pytest.mark.parametrize("beta_mode",[0,1,2])
def test_k64_v32(dtype,save,norm,beta_mode):
    torch.manual_seed(42)
    B,T,H,K,V=1,129,2,64,32
    device=upstream.device
    q=F.normalize(torch.randn(B,T,H,K),dim=-1).to(device,dtype).requires_grad_()
    k=F.normalize(torch.randn(B,T,H,K),dim=-1).to(device,dtype).requires_grad_()
    v=(torch.randn(B,T,H,V)*.5).to(device,dtype).requires_grad_()
    g=(torch.randn(B,T,H,K)*.5).to(device,dtype).requires_grad_()
    beta=(torch.randn(B,T,H)*.5 if beta_mode else torch.zeros(B,T,H)).to(device,dtype).requires_grad_()
    a=(torch.randn(H)*.2).to(device).requires_grad_()
    bias=(torch.randn(H*K)*.3-1.5).to(device).requires_grad_()
    inputs=(q,k,v,g,beta,a,bias)
    cu=[0,0,1,66,66,129]
    out,_=public_chunk(q,k,v,g,beta,A_log=a,dt_bias=bias,scale=.37,
        cu_seqlens=torch.tensor(cu),use_qk_l2norm_in_kernel=norm,use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=bool(beta_mode),allow_neg_eigval=beta_mode==2,
        safe_gate=True,lower_bound=-5.,disable_recompute=save)
    do=torch.randn_like(v)
    got=torch.autograd.grad(out,inputs,do)
    gate=upstream.naive_kda_lowerbound_gate(g.float(),a,bias,lower_bound=-5.)
    br=beta.float() if not beta_mode else beta.float().sigmoid()*(2 if beta_mode==2 else 1)
    qr=F.normalize(q.float(),dim=-1) if norm else q.float()
    kr=F.normalize(k.float(),dim=-1) if norm else k.float()
    outputs=[]
    for s,e in zip(cu,cu[1:]):
        if s==e:continue
        o,_=upstream.naive_recurrent_kda(qr[:,s:e],kr[:,s:e],v[:,s:e].float(),gate[:,s:e],br[:,s:e],scale=.37)
        outputs.append(o)
    ref=torch.cat(outputs,dim=1)
    gold=torch.autograd.grad(ref,inputs,do.float())
    upstream.assert_close("o",ref.to(dtype),out,.008)
    for name,r,x in zip(("dq","dk","dv","dg","db","dA","dbias"),gold,got):
        assert x.dtype==r.dtype and torch.isfinite(x).all()
        upstream.assert_close(name,r,x,.025)
