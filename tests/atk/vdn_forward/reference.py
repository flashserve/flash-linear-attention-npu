"""Independent CPU oracle for the VDN alpha bridge, in value-by-key layout."""
import torch
import torch.nn.functional as F


def features(x, silu, norm):
    if silu:
        x = F.silu(x.float()).to(x.dtype)
    if norm:
        length = x.float().norm(dim=-1, keepdim=True).to(x.dtype).clamp_min(1e-6)
        x = (x.float() / length.float()).to(x.dtype)
    return x


@torch.no_grad()
def reference(q, k, v, beta, alpha, bounds, text_state=None, *, silu=False, qk_l2norm=False):
    """Solve the linear systems directly; use products across inclusive windows."""
    if any(x.device.type != "cpu" for x in (q,k,v,beta,alpha,bounds)):
        raise ValueError("reference expects CPU tensors")
    q = features(q, silu, qk_l2norm)
    k = features(k, silu, qk_l2norm)
    v = features(v, silu, False)
    frames, heads, _, dim = q.shape
    seed = torch.zeros(heads, dim, dim) if text_state is None else text_state.float()
    systems, injections = [], []
    for t in range(frames):
        key = k[t].float()
        weight = beta[t].unsqueeze(-1)
        gram = key.mT @ (weight.float() * key)
        systems.append(torch.eye(dim) + (gram + gram.mT) * 0.5)
        weighted_v = (v[t] * weight.to(v.dtype)).float()
        injections.append((weighted_v.mT @ key).to(v.dtype).float())

    def step(state, t):
        rhs = state * alpha[t].unsqueeze(-2) + injections[t]
        return torch.linalg.solve(systems[t].mT, rhs.mT).mT

    left = [seed]
    for t in range(frames):
        left.append(step(left[-1], t))
    right = [None] * frames + [seed]
    for t in reversed(range(frames)):
        right[t] = step(right[t+1], t)
    decay = alpha.clamp_min(1e-12)
    out = []
    for t, (lo, hi) in enumerate(bounds.tolist()):
        lhs = left[lo] * decay[lo:t+1].prod(0).unsqueeze(-2)
        rhs = right[hi+1] * decay[t:hi+1].prod(0).unsqueeze(-2)
        state = (lhs + rhs).to(q.dtype).float()
        out.append((q[t].float() @ state.mT).to(q.dtype))
    return torch.stack(out)

