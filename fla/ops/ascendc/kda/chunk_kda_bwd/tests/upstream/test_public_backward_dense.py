"""Dense multi-batch/preactivated gate coverage with upstream recurrent reference."""
import pytest
import torch
import torch.nn.functional as F
from test_public_backward_expanded import upstream
from public_backward_plugin import public_chunk
@pytest.mark.parametrize("dtype",[torch.float16,torch.bfloat16])
@pytest.mark.parametrize("save",[True,False])
@pytest.mark.parametrize("norm",[True,False])
def test_dense_preactivated(dtype,save,norm):
    torch.manual_seed(63)
    B,T,H,K,V=2,65,2,64,32;device=upstream.device
    q=F.normalize(torch.randn(B,T,H,K),dim=-1).to(device,dtype).requires_grad_()
    k=F.normalize(torch.randn(B,T,H,K),dim=-1).to(device,dtype).requires_grad_()
    v=torch.randn(B,T,H,V).to(device,dtype).requires_grad_()
    g=(-torch.rand(B,T,H,K)*.1).to(device,torch.float32).requires_grad_()
    beta=torch.rand(B,T,H).to(device,dtype).requires_grad_()
    inputs=(q,k,v,g,beta)
    out,_=public_chunk(q,k,v,g,beta,scale=.37,use_qk_l2norm_in_kernel=norm,
        use_gate_in_kernel=False,use_beta_sigmoid_in_kernel=False,safe_gate=False,disable_recompute=save)
    do=torch.randn_like(v);got=torch.autograd.grad(out,inputs,do)
    ref,_=upstream.naive_recurrent_kda(F.normalize(q.float(),dim=-1) if norm else q.float(),
        F.normalize(k.float(),dim=-1) if norm else k.float(),v.float(),g,beta.float(),scale=.37)
    gold=torch.autograd.grad(ref,inputs,do.float())
    upstream.assert_close("o",ref.to(dtype),out,.008)
    for name,r,x in zip(("dq","dk","dv","dg","db"),gold,got):
        assert x.dtype==r.dtype and torch.isfinite(x).all()
        upstream.assert_close(name,r,x,.025)
