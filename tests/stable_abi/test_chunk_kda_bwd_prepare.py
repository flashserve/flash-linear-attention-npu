#!/usr/bin/env python3
"""Verify the public Stable-ABI entry against an exact, independent CPU reference.

Run after installing the current wheel:
    python tests/stable_abi/test_chunk_kda_bwd_prepare.py --device 0

Integer BF16 inputs and a dyadic scale make all FP32 intermediates exact. This
checks indexing, tails and state orientation; ATK remains the accuracy gate for
the original random input ranges and native accuracy criteria.
"""
import argparse
import itertools

import torch


def make_case(layout, tokens, state_v_first=False, *, batch=1, heads=3, variable=None, random=False, seed=0, high_precision=False,
              compute_expected=True):
    packed = layout in {"NTD", "TND"}
    variable = packed if variable is None else variable
    batch = 1 if packed or variable else batch
    generator = torch.Generator().manual_seed(1000 + tokens + batch + heads + seed)

    def values(shape):
        if random:
            return (torch.randn(shape, generator=generator) * 0.05).bfloat16()
        return torch.randint(-2, 3, shape, generator=generator).to(torch.bfloat16)

    prefix = (heads, tokens) if packed else (batch, heads, tokens)
    aqk, v_new, d_o = (values(prefix + (width,)) for width in (64, 128, 128))
    if variable:
        # An empty middle sequence, a one-token sequence and a possible tail.
        cu = [0, 1, 1, tokens] if tokens > 1 else [0, 0, 1]
        chunks = [(seq, chunk) for seq in range(len(cu) - 1)
                  for chunk in range((cu[seq + 1] - cu[seq] + 63) // 64)]
        h = values((len(chunks), heads, 128, 128))
        if not packed:
            h = h.unsqueeze(0)
    else:
        cu = None
        chunks = [(b, chunk) for b in range(batch)
                  for chunk in range((tokens + 63) // 64)]
        h = values((batch, (tokens + 63) // 64, heads, 128, 128))
        if packed:
            h = h.squeeze(0)

    expected = [torch.zeros(prefix + (width,), dtype=torch.float64 if high_precision else dtype)
                for width, dtype in ((64, torch.float32), (128, torch.bfloat16),
                                     (128, torch.float32))]
    for index, (seq, chunk) in enumerate(chunks if compute_expected else []):
        begin = (cu[seq] if variable else 0) + chunk * 64
        end = min(begin + 64, cu[seq + 1] if variable else tokens)
        for head in range(heads):
            pos = (head, slice(begin, end)) if packed else (0 if variable else seq, head, slice(begin, end))
            do = d_o[pos].double()
            vn = v_new[pos].double()
            a = aqk[pos][..., :end - begin].double()
            state_view = h.unsqueeze(0) if packed else h
            state = state_view[0 if variable else seq, index if variable else chunk, head].double()
            expected[0][pos][..., :end - begin] = (do @ vn.T).tril() * 0.125
            expected[1][pos] = (a.T @ do) if high_precision else (a.T @ do).bfloat16()
            expected[2][pos] = do @ (state if state_v_first else state.T)

    if layout == "BSND":
        aqk, v_new, d_o = (x.transpose(1, 2).contiguous() for x in (aqk, v_new, d_o))
    elif layout == "TND":
        aqk, v_new, d_o = (x.transpose(0, 1).contiguous() for x in (aqk, v_new, d_o))
    kwargs = dict(layout=layout, scale=0.125, chunk_size=64,
                  state_v_first=state_v_first, cu_seqlens=cu,
                  chunk_indices=[item for pair in chunks for item in pair] if variable else None)
    return (aqk, v_new, d_o, h), kwargs, expected


def main():
    import torch_npu  # noqa: F401
    from fla_npu.ops import ascendc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--extended", action="store_true",
                        help="include fractal tails, DMA stride boundaries and slot reuse")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    count = 0
    matrix = list(itertools.product(("BNSD", "BSND", "NTD", "TND"),
                                    (1, 64, 65), (False, True)))
    matrix += [(layout, 129, flag) for layout in ("BNSD", "BSND", "NTD", "TND")
               for flag in (False, True)]
    cases = [(layout, tokens, flag, 2 if tokens == 129 else 1, 5 if tokens == 129 else 3)
             for layout, tokens, flag in matrix]
    if args.extended:
        cases += [(layout, tokens, flag, 1, heads)
                  for layout in ("BNSD", "BSND", "NTD", "TND")
                  for tokens, heads in ((15, 1), (16, 2), (17, 5), (63, 3),
                                        (513, 3), (17, 513), (8193, 1))
                  for flag in (False, True)]
    for layout, tokens, flag, batch, heads in cases:
        inputs, kwargs, expected = make_case(layout, tokens, flag, batch=batch, heads=heads)
        print(f"RUN {layout} B={batch} H={heads} T={tokens} state={flag}", flush=True)
        actual = ascendc.chunk_kda_bwd_prepare(*(x.npu() for x in inputs), **kwargs)
        for got, want in zip(actual, expected):
            assert got.is_contiguous()
            torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
        assert ascendc.BACKENDS.get("npu_chunk_kda_bwd_prepare") == "stable", ascendc.BACKENDS
        assert not ascendc.FALLBACKS.get("npu_chunk_kda_bwd_prepare", 0), ascendc.FALLBACKS
        count += 1
        print(f"PASS {layout} B={batch} H={heads} T={tokens} state_v_first={flag}", flush=True)
    for layout, variable, flag, omit_indices, tokens in itertools.product(
            ("BNSD", "BSND", "NTD", "TND"), (False, True), (False, True), (False, True), (64, 129)):
        inputs, kwargs, expected = make_case(layout, tokens, flag, heads=5, variable=variable)
        if omit_indices:
            kwargs["chunk_indices"] = None
        print(f"RUN {layout} T={tokens} variable={variable} state={flag} auto_indices={omit_indices}",
              flush=True)
        actual = ascendc.chunk_kda_bwd_prepare(*(x.npu() for x in inputs), **kwargs)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got.cpu(), want, rtol=0, atol=0)
        count += 1
        print(f"PASS {layout} T={tokens} variable={variable} state={flag} auto_indices={omit_indices}", flush=True)
    print(f"PASS {count} cases, {3 * count} exact output comparisons; backend=stable, fallback=0")
    if args.extended:
        for layout in ("BNSD", "BSND", "NTD", "TND"):
            # One dense chunk even for packed layouts: the invalid upper
            # triangle contains 0 * Inf, which must be selected out, not
            # multiplied by a zero mask. H=1 makes both input orders equal.
            inputs, kwargs, _ = make_case(layout, 2, heads=1)
            if layout in {"NTD", "TND"}:
                kwargs.update(cu_seqlens=[0, 2], chunk_indices=[0, 0])
                inputs = (*inputs[:3], inputs[3][:1].contiguous())
            inputs[1].reshape(2, 128)[1].fill_(float("inf"))
            inputs[2].reshape(2, 128)[0].zero_()
            d_aqk, _, _ = ascendc.chunk_kda_bwd_prepare(*(x.npu() for x in inputs), **kwargs)
            rows = d_aqk.cpu().reshape(2, 64)
            assert torch.count_nonzero(rows[0, 1:]).item() == 0
            assert torch.count_nonzero(rows[1, 2:]).item() == 0
        print("PASS 4 nonfinite upper-triangle/tail masking cases", flush=True)


if __name__ == "__main__":
    main()
