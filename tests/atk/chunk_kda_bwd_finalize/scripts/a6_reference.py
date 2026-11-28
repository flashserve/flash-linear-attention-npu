# Copyright (c) 2026 Tianjin University, Ltd.
"""Development FP64 equations for one Finalize chunk/head (no kernel imports).

This is separate from the ATK ABI placeholder. Tests validate the equations
against differentiation of a scalar WY/state/intra surrogate. It has not been
calibrated against an A5/A6 device and is not a formal ATK acceptance golden.
"""
import math
import torch


def finalize_chunk(q, k, v, g, raw, beta, a_log, bias, akk, v_new, h, dh,
                   dv_scan, d_aqk, dq_raw, scale, lower_bound,
                   q_rstd=None, k_rstd=None):
    q, k, v, g, raw, beta, a_log, bias, akk, v_new, h, dh, p, d, dq_raw = [
        x.to(torch.float64) for x in
        (q, k, v, g, raw, beta, a_log, bias, akk, v_new, h, dh, dv_scan, d_aqk, dq_raw)]
    rows = q.shape[0]
    e = torch.exp(g * math.log(2))
    dk_state = (v_new @ dh.T) * torch.exp((g[-1] - g) * math.log(2))
    dv_b = akk.T @ p
    dw = p @ h.T
    d_kgb = akk.T @ dw
    zb = torch.tril((p @ v.T - dw @ (k * e).T) * beta[None, :], diagonal=-1)
    c = -torch.tril(akk.T @ (zb @ akk.T), diagonal=-1)
    x = e * d_kgb
    dq = scale * dq_raw * e
    dk = dk_state - beta[:, None] * x
    db = (v * dv_b).sum(-1) - (k * x).sum(-1)
    dg = q * dq - k * dk_state - beta[:, None] * k * x
    dg[-1] += (k * dk_state).sum(0) + e[-1] * (h * dh).sum(-1)
    dq_local = torch.zeros_like(q)
    left, right = torch.zeros_like(k), torch.zeros_like(k)
    # Evaluate only causal exp(G_i-G_j), avoiding noncausal positive exponents.
    # This directly expresses the mathematical result, independently of the
    # device's 32-row centering, FP32 NZ operands, and power-of-two scaling.
    for i in range(rows):
        for j in range(i):
            decay = torch.exp((g[i] - g[j]) * math.log(2))
            dq_local[i] += d[i, j] * k[j] * decay
            left[i] += c[i, j] * k[j] * decay
            right[j] += (d[i, j] * q[i] + c[i, j] * beta[i] * k[i]) * decay
    dq += dq_local + d.diagonal()[:, None] * k
    dk += beta[:, None] * left + right + d.diagonal()[:, None] * q
    db += (left * k).sum(-1)
    dg += q * dq_local + k * (beta[:, None] * left - right)
    suffix = dg.flip(0).cumsum(0).flip(0)
    gate_x = raw + bias
    t = torch.exp(-(a_log.exp() * gate_x).abs())
    d_raw = suffix * lower_bound * a_log.exp() * t / (1 + t).square()
    if q_rstd is not None:
        dq = q_rstd.double()[:, None] * (dq - q * (dq * q).sum(-1, keepdim=True))
        dk = k_rstd.double()[:, None] * (dk - k * (dk * k).sum(-1, keepdim=True))
    return dq, dk, beta[:, None] * dv_b, db, d_raw, (d_raw * gate_x).sum(), d_raw.sum(0)


def check_reference():
    """Check signs, triangular direction, scale, gate chain and reductions."""
    torch.manual_seed(42)
    torch.set_num_threads(1)
    for rows in (1, 3, 17, 33):
        dim = 8  # Independent equations support any D; device D remains 128.
        q, k, v = [torch.randn(rows, dim, dtype=torch.float64).mul(.1).requires_grad_() for _ in range(3)]
        raw = torch.randn(rows, dim, dtype=torch.float64).mul(.1).requires_grad_()
        beta = torch.rand(rows, dtype=torch.float64).requires_grad_()
        a_log = torch.tensor(-1., dtype=torch.float64, requires_grad=True)
        bias = torch.randn(dim, dtype=torch.float64).mul(.1).requires_grad_()
        h, dh = [torch.randn(dim, dim, dtype=torch.float64).mul(.1) for _ in range(2)]
        vn, p, qr = [torch.randn(rows, dim, dtype=torch.float64).mul(.1) for _ in range(3)]
        d = torch.randn(rows, rows, dtype=torch.float64).mul(.1)
        lower, scale = -.1, .37
        natural = (lower * torch.sigmoid(a_log.exp() * (raw + bias))).cumsum(0)
        # Compute a true WY inverse in the surrogate, rather than reproducing
        # its backward formula or the kernel's partial-derivative stages.
        aq = torch.zeros(rows, rows, dtype=torch.float64)
        kk = torch.zeros_like(aq)
        end_state = natural[-1].exp()[:, None] * h
        for i in range(rows):
            end_state = end_state + (k[i] * (natural[-1]-natural[i]).exp())[:, None] * vn[i][None, :]
            for j in range(i+1):
                decay = (natural[i]-natural[j]).exp()
                aq[i,j] = (q[i] * k[j] * decay).sum()
                if j < i:
                    kk[i,j] = beta[i] * (k[i] * k[j] * decay).sum()
        akk = torch.linalg.inv(torch.eye(rows, dtype=torch.float64) + kk)
        u = akk @ (beta[:, None] * v)
        w = akk @ (beta[:, None] * k * natural.exp())
        loss = ((u - w @ h) * p).sum() + (q * natural.exp() * qr * scale).sum()
        loss = loss + (aq * d).sum() + (end_state * dh).sum()
        gradients = torch.autograd.grad(loss, (q,k,v,beta,raw,a_log,bias))
        got = finalize_chunk(q,k,v,natural/math.log(2),raw,beta,a_log,bias,akk,vn,h,dh,p,d,qr,scale,lower)
        for name, actual, expected in zip(('dq','dk','dv','db','dg','da','dbias'), got, gradients):
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12, msg=lambda m: f'{name} M={rows}: {m}')
    print('FP64 Finalize equations match independent autograd surrogate: 4 shapes, 7 gradients each')


def check_saturated_gate():
    """A one-row state-only loss isolates the saturated gate chain rule."""
    torch.set_num_threads(1)
    dim, lower = 8, -.1
    z = torch.zeros(1, dim, dtype=torch.float64)
    state = torch.full((dim, dim), .125, dtype=torch.float64)
    beta = torch.ones(1, dtype=torch.float64)
    a_log = torch.tensor(0., dtype=torch.float64)
    bias = torch.zeros(dim, dtype=torch.float64)
    matrix = torch.ones(1, 1, dtype=torch.float64)
    for x in (-80., -40., 0., 40., 80.):
        raw = torch.full_like(z, x)
        g = lower * torch.sigmoid(raw) / math.log(2)
        got = finalize_chunk(z, z, z, g, raw, beta, a_log, bias,
                             matrix, z, state, state, z, matrix * 0, z, 1., lower)
        # log-sigmoid uses a separate stable formulation of the derivative.
        log_slope = (torch.nn.functional.logsigmoid(raw) +
                     torch.nn.functional.logsigmoid(-raw))
        expected = lower * torch.exp(g * math.log(2)) * (state * state).sum(-1) * log_slope.exp()
        for actual, want in zip(got[4:], (expected, (expected * raw).sum(), expected.sum(0))):
            torch.testing.assert_close(actual, want, rtol=1e-12, atol=0.)
        assert torch.count_nonzero(got[4]) == dim
    print('FP64 saturated gate chain matches log-sigmoid derivative: x=-80/-40/0/40/80')


if __name__ == '__main__':
    check_reference()
    check_saturated_gate()
