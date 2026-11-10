def npu_fused_recurrent_rwkv8(q, w, k, v, z, b, *, scale=1.0,
                              initial_state=None, reverse=False,
                              output_chunk_state=False, output_sa=False,
                              chunk_len=16):
    """RWKV-v8 (WKV7) fused recurrent forward, token by token.

    Layout is BHTC: q/w/k/z/b are (B,H,T,K), v is (B,H,T,V) (K and V are
    independent; K==V is the special case).  `initial_state` is (B,H,K,V)
    fp32 (the kernel's ledger orientation, same as the `s` snapshots);
    None means a zero initial state.

    Returns (o, s, sa): o is (B,H,T,V) with q's dtype; s is the chunk
    snapshot (B,H,T//chunk_len,K,V) fp32 when `output_chunk_state` is set,
    else None; sa is the per-token state@z (B,H,T,V) fp32 when `output_sa`
    is set, else None.
    """

    return _op("npu_fused_recurrent_rwkv8")(
        q, w, k, v, z, b, initial_state, float(scale), bool(reverse),
        bool(output_chunk_state), bool(output_sa), int(chunk_len),
        _current_stream_ptr())
