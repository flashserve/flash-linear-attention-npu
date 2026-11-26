import torch


def bridge(op, m, h, s, inp, cpu, high, obj):
    if op == "causal_conv1d_bwd":
        if not cpu:
            return h.ConvBackward.__call__(obj, inp, True)
        dtype = getattr(
            torch,
            {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[s["dtype"]],
        )

        def make(shape, off):
            return h.gold.make_tensor(shape, dtype=dtype, offset=s["seed"] % 37 + off)

        b, t, d, w = (s[k] for k in ("B", "T", "D", "W"))
        x = make((b, t, d), 9)
        wt = make((w, d), 11)
        dy = make(x.shape, 17)
        dht = make((b, w, d), 7) if s["with_dht"] else None
        ys = (
            h.gold.causal_conv1d_preactivation_ref(x, wt, None).to(dtype)
            if s["with_y"] or s["activation"]
            else None
        )

        def cast(v):
            return None if v is None else v.double() if high else v

        return m._causal_conv1d_bwd_ref(
            dict(x=cast(x), weight=cast(wt), dy=cast(dy)),
            y_optional=cast(ys),
            dht=cast(dht),
            activation=s["activation"],
            input_layout=s["input_layout"],
        )
    if not cpu:
        return h.run_npu(s, inp)
    if op == "pre_process_bwd_kernel_merged":
        tensors = h.build_inputs(s, torch.device("cpu"))
        return m._call_reference(
            tensors,
            s.get("cu_seqlens"),
            s.get("scale", 1),
            s.get("chunk_size", 64),
            high,
        )
    q, k, v, vn, do, g, beta, br, du, ht, dh, a, cu, ix, _, _ = h._cpu_prepared(inp, s)

    def rstd(x):
        return (
            torch.rsqrt(x.float().square().sum(-1) + 1.0e-6)
            if s["use_qk_l2norm"]
            else None
        )

    return m.chunk_gated_delta_rule_bwd_finalize_golden(
        q,
        k,
        v,
        vn,
        do,
        du,
        g,
        beta,
        ht,
        dh,
        a,
        q_rstd=rstd(q),
        k_rstd=rstd(k),
        beta_raw=br if s["use_beta_sigmoid"] else None,
        cu_seqlens=cu,
        chunk_indices=ix,
        scale=float(s.get("scale", h.SCALE)),
        chunk_size=s["chunk_size"],
        use_qk_l2_norm_in_kernel=s["use_qk_l2norm"],
        use_beta_sigmoid_in_kernel=s["use_beta_sigmoid"],
        use_gate_in_kernel=False,
        state_v_first=s["state_v_first"],
        use_exp2=True,
        nt_first=True,
        high_precision=high,
    )
