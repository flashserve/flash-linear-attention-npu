from __future__ import annotations
from typing import Any
import torch

def _user_cpu_golden(inputs: dict[str, Any], spec: dict[str, Any], high_precision: bool) -> tuple[torch.Tensor, ...]:
    """Independent CPU formulation of dAqk, dv and dq_raw.

    For each valid in-chunk row range ``r``:
      dAqk = tril(dO @ v_new.T) * scale
      dv   = aqk[:, :r].T @ dO
      dq   = dO @ h          (or dO @ h.T for state_v_first=False)
    Matrix products and scale application use FP64 for Golden (the automatic
    ATK is_benchmark_task) and FP32 for Bench (the explicit ordinary CPU node).
    Both receive identical BF16 inputs, then convert outputs to the public
    FP32/BF16/FP32 dtypes required for the native ATK comparison.
    """
    aqk, v_new, d_o, h = (inputs[key].cpu() for key in ("aqk", "v_new", "d_o", "h"))
    if spec["layout"] in ("BSND", "TND"):
        axes = (0, 2, 1, 3) if spec["mode"] == "dense" else (1, 0, 2)
        aqk, v_new, d_o = (tensor.permute(axes) for tensor in (aqk, v_new, d_o))
    calc_dtype = torch.float64 if high_precision else torch.float32
    mode = str(spec["mode"])
    chunks = int(spec["chunk_size"])
    scale = float(spec["scale"])
    state_v_first = bool(spec["state_v_first"])
    d_aqk = torch.zeros_like(aqk, dtype=torch.float32, memory_format=torch.contiguous_format)
    dv = torch.empty_like(v_new, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
    dq_raw = torch.empty_like(d_o, dtype=torch.float32, memory_format=torch.contiguous_format)

    if mode == "dense":
        batch, heads, tokens, _ = aqk.shape
        sequence_ranges = [(batch_id, 0, tokens, 0) for batch_id in range(batch)]
        state_base = 0
        for batch_id, begin_seq, end_seq, _ in sequence_ranges:
            for head in range(heads):
                for local_chunk, begin in enumerate(range(begin_seq, end_seq, chunks)):
                    end = min(begin + chunks, end_seq)
                    rows = end - begin
                    _write_golden_tile(
                        d_aqk, dv, dq_raw, aqk, v_new, d_o, h,
                        batch_id, head, begin, end, rows, local_chunk,
                        scale, state_v_first, calc_dtype,
                        dense=True,
                    )
        return d_aqk, dv, dq_raw

    heads, tokens, _ = aqk.shape
    del tokens
    offset = 0
    state_index = 0
    for sequence_length in (int(value) for value in spec["seqlens"]):
        sequence_end = offset + sequence_length
        for begin in range(offset, sequence_end, chunks):
            end = min(begin + chunks, sequence_end)
            rows = end - begin
            for head in range(heads):
                _write_golden_tile(
                    d_aqk, dv, dq_raw, aqk, v_new, d_o, h,
                    0, head, begin, end, rows, state_index,
                    scale, state_v_first, calc_dtype,
                    dense=False,
                )
            state_index += 1
        offset = sequence_end
    return d_aqk, dv, dq_raw


def _write_golden_tile(
    d_aqk: torch.Tensor, dv: torch.Tensor, dq_raw: torch.Tensor,
    aqk: torch.Tensor, v_new: torch.Tensor, d_o: torch.Tensor, h: torch.Tensor,
    batch: int, head: int, begin: int, end: int, rows: int, state_index: int,
    scale: float, state_v_first: bool, calc_dtype: torch.dtype, *,
    dense: bool,
) -> None:
    if dense:
        aqk_tile = aqk[batch, head, begin:end]
        v_tile = v_new[batch, head, begin:end]
        do_tile = d_o[batch, head, begin:end]
        h_tile = h[batch, state_index, head]
    else:
        aqk_tile = aqk[head, begin:end]
        v_tile = v_new[head, begin:end]
        do_tile = d_o[head, begin:end]
        h_tile = h[state_index, head]
    aqk_calc, v_calc, do_calc, h_calc = (tile.to(calc_dtype) for tile in (aqk_tile, v_tile, do_tile, h_tile))
    tri = torch.tril(torch.ones((rows, rows), dtype=calc_dtype))
    a_raw = do_calc @ v_calc.transpose(0, 1)
    a_tile = torch.where(tri.bool(), a_raw * scale, 0.0)
    h_right = h_calc if state_v_first else h_calc.transpose(0, 1)
    q_tile = do_calc @ h_right
    d_tile = aqk_calc[:, :rows].transpose(0, 1) @ do_calc
    if dense:
        d_aqk[batch, head, begin:end, :rows] = a_tile.float()
        dv[batch, head, begin:end] = d_tile.to(torch.bfloat16)
        dq_raw[batch, head, begin:end] = q_tile.float()
    else:
        d_aqk[head, begin:end, :rows] = a_tile.float()
        dv[head, begin:end] = d_tile.to(torch.bfloat16)
        dq_raw[head, begin:end] = q_tile.float()


def _cpu_golden(inputs, spec, high_precision):
    # Normalize only CPU rank representation for the supplied user's formula.
    variable = spec["mode"] != "dense"
    rank3 = spec["layout"] in ("NTD", "TND")
    token_major = spec["layout"] in ("BSND", "TND")
    adapted = dict(inputs)
    if variable and not rank3:
        for name in ("aqk", "v_new", "d_o", "h"):
            adapted[name] = adapted[name].squeeze(0)
    elif not variable and rank3:
        for name in ("aqk", "v_new", "d_o", "h"):
            adapted[name] = adapted[name].unsqueeze(0)
    contract = dict(spec)
    contract["layout"] = ("TND" if token_major else "NTD") if variable else ("BSND" if token_major else "BNSD")
    result = _user_cpu_golden(adapted, contract, high_precision)
    if variable and not rank3:
        return tuple(x.unsqueeze(0) for x in result)
    if not variable and rank3:
        return tuple(x.squeeze(0) for x in result)
    return result

def chunk_kda_bwd_prepare_cpu(aqk, v_new, d_o, h, *, scale, layout,
                            chunk_size=64, state_v_first=False,
                            cu_seqlens=None, chunk_indices=None, gva_ratio=1):
    """Public CPU adapter; all mathematics lives in _user_cpu_golden above."""
    spec = {"layout": layout, "mode": "varlen" if cu_seqlens is not None else "dense",
            "scale": scale, "chunk_size": chunk_size, "state_v_first": state_v_first}
    if cu_seqlens is not None:
        cu = [int(value) for value in cu_seqlens]
        spec["seqlens"] = [end-begin for begin,end in zip(cu,cu[1:])]
    return _cpu_golden(dict(aqk=aqk, v_new=v_new, d_o=d_o, h=h), spec, False)
