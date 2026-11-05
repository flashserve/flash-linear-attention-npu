"""Build the chunk32 ATK matrices.

Accuracy covers the combinations chunk32 accepts: BNSD/BSND/NTD/TND, MHA/GVA,
aligned and tail lengths, packed sequences, ``allow_neg_eigval``,
``state_v_first``, initial state none/bf16/fp32/zero, and the optional
exports (L2 hats and rstd, beta_eff, g_cumsum and A, h). ``use_exp2`` stays
false and the in-kernel L2 and beta sigmoid stay on. T is at most 8192.
``g_mode`` is ``normal`` (``randn * 0.02``), ``strong`` (every token -4),
``spike`` (prefix -100 then -99) or ``mix`` (chunk 0 normal, chunk 1 strong).
"""

from __future__ import annotations

import json
from pathlib import Path


OP_NAME = "chunk_gated_delta_rule_fwd_chunk32"
SEED_BASE = 20261028
STANDARD = {
    "acc": {
        "mixed_tolerance_bm": {
            "output_dtype_overrides": {"0": "bf16", "1": "bf16"}
        }
    },
    "perf": "not_key",
    "mem": 1.1,
}
HERE = Path(__file__).resolve().parent
SCALE = 128 ** -0.5


def _spec(
    *,
    case_key: str,
    layout: str,
    batch: int,
    k_heads: int,
    v_heads: int,
    tokens: int,
    state: str = "none",
    seqlens: str = "",
    allow_neg: bool = False,
    state_v_first: bool = False,
    out_l2: bool = False,
    disable_recompute: bool = False,
    out_h: bool = False,
    g_mode: str = "normal",
    tags: str = "accuracy",
    seed: int,
    max_tokens: int = 8192,
) -> dict:
    if tokens > max_tokens:
        raise ValueError(f"T={tokens} exceeds {max_tokens}")
    return {
        "case_key": case_key,
        "tags": tags,
        "route": "ascendc",
        "soc": "ascend950",
        "dtype": "bf16",
        "layout": layout,
        "B": batch,
        "HK": k_heads,
        "HV": v_heads,
        "T": tokens,
        "K": 128,
        "V": 128,
        "state_dtype": state,
        "seqlens": seqlens,
        "allow_neg_eigval": allow_neg,
        "state_v_first": state_v_first,
        "out_l2": out_l2,
        "disable_recompute": disable_recompute,
        "out_h": out_h,
        "g_mode": g_mode,
        "scale": SCALE,
        "seed": seed,
    }


def _key(layout: str, k_heads: int, v_heads: int, tokens: int, state: str,
         seqlens: str, allow_neg: bool, state_v_first: bool, batch: int,
         out_l2: bool = False, disable_recompute: bool = False, out_h: bool = False,
         g_mode: str = "normal") -> str:
    head = "gva" if v_heads > k_heads else "mha"
    packed = "var" if seqlens else "fix"
    seq = "" if not seqlens else "_seq" + seqlens.replace(",", "-")
    exports = f"_l2{int(out_l2)}_ga{int(disable_recompute)}_h{int(out_h)}"
    gate = "" if g_mode == "normal" else f"_{g_mode}"
    return (
        f"{layout.lower()}_{head}_{packed}_B{batch}_H{k_heads}x{v_heads}"
        f"_T{tokens}_{state}{seq}_neg{int(allow_neg)}_sv{int(state_v_first)}{exports}{gate}"
    )


def accuracy_specs() -> list[dict]:
    """Covering set. BNSD/NTD are fully crossed; BSND/TND and the exports are sampled.

    Optional exports are ``out_l2`` (q_hat/k_hat/rstd and beta_eff),
    ``disable_recompute`` (g_cumsum and A) and ``out_h``. The inference launch
    leaves all three off. T stays at or under 8192.
    """

    negs = (False, True)
    svs = (False, True)
    states = ("none", "bf16", "fp32", "zero")
    heads = ((4, 4), (2, 8))  # MHA, GVA
    specs: list[dict] = []

    def add(layout, hk, hv, batch, tokens, state, seqlens, allow_neg, state_v_first,
            out_l2=False, disable_recompute=False, out_h=False, g_mode="normal") -> None:
        specs.append(
            _spec(
                case_key=_key(
                    layout, hk, hv, tokens, state, seqlens, allow_neg, state_v_first, batch,
                    out_l2, disable_recompute, out_h, g_mode,
                ),
                layout=layout, batch=batch, k_heads=hk, v_heads=hv, tokens=tokens,
                state=state, seqlens=seqlens, allow_neg=allow_neg, state_v_first=state_v_first,
                out_l2=out_l2, disable_recompute=disable_recompute, out_h=out_h,
                g_mode=g_mode, seed=SEED_BASE + len(specs),
            )
        )

    def cross(layout, profiles) -> None:
        for allow_neg in negs:
            for state_v_first in svs:
                for state in states:
                    for hk, hv in heads:
                        for profile in profiles:
                            add(
                                layout, hk, hv, profile["batch"], profile["tokens"],
                                state, profile["seqlens"], allow_neg, state_v_first,
                            )

    # Head-major fixed: aligned, tail, and a second batch.
    cross("BNSD", (
        {"batch": 1, "tokens": 128, "seqlens": ""},
        {"batch": 1, "tokens": 100, "seqlens": ""},
        {"batch": 2, "tokens": 64, "seqlens": ""},
    ))
    # Head-major packed: aligned pair, unaligned pair, and one fixed tail.
    cross("NTD", (
        {"batch": 1, "tokens": 128, "seqlens": "64,64"},
        {"batch": 1, "tokens": 100, "seqlens": "40,60"},
        {"batch": 1, "tokens": 100, "seqlens": ""},
    ))
    # Sequence-major spot check, both layouts, aligned and tail.
    for layout, profiles in (
        ("BSND", (
            {"batch": 1, "tokens": 128, "seqlens": ""},
            {"batch": 1, "tokens": 100, "seqlens": ""},
        )),
        ("TND", (
            {"batch": 1, "tokens": 128, "seqlens": "64,64"},
            {"batch": 1, "tokens": 96, "seqlens": "48,48"},
        )),
    ):
        for allow_neg in negs:
            for state_v_first in svs:
                for state in ("none", "bf16"):
                    for hk, hv in heads:
                        for profile in profiles:
                            add(
                                layout, hk, hv, profile["batch"], profile["tokens"],
                                state, profile["seqlens"], allow_neg, state_v_first,
                            )
    # Cap. Still at or under 8K, one head-major fixed and one packed tail.
    add("BNSD", 4, 4, 1, 8192, "none", "", False, False)
    add("NTD", 2, 8, 1, 8000, "fp32", "3000,5000", True, True)

    # Each export switch, and the combinations, on the layouts those switches touch.
    export_modes = (
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    )
    export_rows = (
        # layout, hk, hv, batch, tokens, state, seqlens, neg, sv
        ("BNSD", 4, 4, 1, 64, "none", "", False, False),
        ("BNSD", 2, 8, 1, 40, "bf16", "", True, True),
        ("NTD", 2, 8, 1, 64, "fp32", "24,40", False, True),
        ("NTD", 4, 4, 1, 96, "zero", "64,32", True, False),
        ("BSND", 2, 8, 1, 96, "none", "", True, False),
        ("TND", 4, 4, 1, 80, "bf16", "48,32", False, True),
    )
    for out_l2, disable_recompute, out_h in export_modes:
        for row in export_rows:
            layout, hk, hv, batch, tokens, state, seqlens, allow_neg, state_v_first = row
            add(
                layout, hk, hv, batch, tokens, state, seqlens, allow_neg, state_v_first,
                out_l2, disable_recompute, out_h,
            )
    # The block above is the original 300. Everything after it fills the holes
    # and keeps those seeds stable.
    def cross_rows(layout, profiles, state_names) -> None:
        for allow_neg in negs:
            for state_v_first in svs:
                for state in state_names:
                    for hk, hv in heads:
                        for profile in profiles:
                            add(
                                layout, hk, hv, profile["batch"], profile["tokens"],
                                state, profile["seqlens"], allow_neg, state_v_first,
                            )

    # BSND/TND had only none/bf16. FP32 and explicit zero match the BNSD cross.
    cross_rows("BSND", (
        {"batch": 1, "tokens": 128, "seqlens": ""},
        {"batch": 1, "tokens": 100, "seqlens": ""},
    ), ("fp32", "zero"))
    cross_rows("TND", (
        {"batch": 1, "tokens": 128, "seqlens": "64,64"},
        {"batch": 1, "tokens": 96, "seqlens": "48,48"},
    ), ("fp32", "zero"))
    # Fixed TND, plus an uneven packed pair. NTD already has 40+60.
    cross_rows("TND", (
        {"batch": 1, "tokens": 128, "seqlens": ""},
        {"batch": 1, "tokens": 100, "seqlens": ""},
        {"batch": 1, "tokens": 100, "seqlens": "40,60"},
    ), states)
    # B=2 on the sequence-major layout. Varlen stays at physical B=1.
    cross_rows("BSND", (
        {"batch": 2, "tokens": 64, "seqlens": ""},
    ), states)
    # One chunk: a full block of 32 and a partial block. Packed sequences are
    # each at most 32, so every sequence takes the single-chunk prologue.
    cross_rows("BNSD", (
        {"batch": 1, "tokens": 32, "seqlens": ""},
        {"batch": 1, "tokens": 20, "seqlens": ""},
    ), states)
    cross_rows("NTD", (
        {"batch": 1, "tokens": 32, "seqlens": ""},
        {"batch": 1, "tokens": 20, "seqlens": ""},
        {"batch": 1, "tokens": 32, "seqlens": "16,16"},
        {"batch": 1, "tokens": 20, "seqlens": "12,8"},
    ), states)
    cross_rows("BSND", (
        {"batch": 1, "tokens": 32, "seqlens": ""},
        {"batch": 1, "tokens": 20, "seqlens": ""},
    ), states)
    cross_rows("TND", (
        {"batch": 1, "tokens": 32, "seqlens": ""},
        {"batch": 1, "tokens": 20, "seqlens": ""},
        {"batch": 1, "tokens": 32, "seqlens": "16,16"},
        {"batch": 1, "tokens": 20, "seqlens": "12,8"},
    ), states)
    # Exports on the long cap and on B=2. The seven switches already exist on
    # the small shapes.
    export_shapes = (
        ("BNSD", 4, 4, 1, 8192, "none", "", False, False),
        ("BNSD", 4, 4, 2, 64, "none", "", False, False),
        ("BSND", 4, 4, 2, 64, "none", "", False, False),
    )
    for out_l2, disable_recompute, out_h in export_modes:
        for row in export_shapes:
            layout, hk, hv, batch, tokens, state, seqlens, allow_neg, state_v_first = row
            add(
                layout, hk, hv, batch, tokens, state, seqlens, allow_neg, state_v_first,
                out_l2, disable_recompute, out_h,
            )

    # g=-4 pushes every prefix past -80, so gate and v_new take exp(diff).
    # Spike is the 0*Inf pair inside one chunk. Mix keeps chunk 0 on the
    # multiply and chunk 1 on exp(diff).
    def cross_gate(layout, profiles, g_mode, state_names=("none", "bf16")) -> None:
        for allow_neg in negs:
            for state_v_first in svs:
                for state in state_names:
                    for hk, hv in heads:
                        for profile in profiles:
                            add(
                                layout, hk, hv, profile["batch"], profile["tokens"],
                                state, profile["seqlens"], allow_neg, state_v_first,
                                g_mode=g_mode,
                            )

    single_fixed = ({"batch": 1, "tokens": 32, "seqlens": ""},)
    multi_fixed = ({"batch": 1, "tokens": 128, "seqlens": ""},)
    single_packed = ({"batch": 1, "tokens": 32, "seqlens": "16,16"},)
    multi_packed = ({"batch": 1, "tokens": 128, "seqlens": "64,64"},)
    for layout in ("BNSD", "BSND"):
        cross_gate(layout, single_fixed + multi_fixed, "strong")
    for layout in ("NTD", "TND"):
        cross_gate(layout, single_fixed + single_packed + multi_packed, "strong")
    for layout in ("BNSD", "BSND"):
        cross_gate(layout, multi_fixed, "spike", ("none",))
    for layout in ("NTD", "TND"):
        cross_gate(layout, multi_packed, "spike", ("none",))
    mix_fixed = ({"batch": 1, "tokens": 64, "seqlens": ""},)
    mix_packed = ({"batch": 1, "tokens": 64, "seqlens": "32,32"},)
    for layout in ("BNSD", "BSND"):
        cross_gate(layout, mix_fixed, "mix", ("none",))
    for layout in ("NTD", "TND"):
        cross_gate(layout, mix_packed, "mix", ("none",))

    keys = [row["case_key"] for row in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate accuracy case_key")
    if any(row["T"] > 8192 for row in specs):
        raise RuntimeError("accuracy T exceeds 8192")
    return specs


def slim_specs() -> list[dict]:
    """One case per scenario, for a fast mixed-tolerance pass.

    Each row is a single combination. The full matrix stays in accuracy_specs.
    """

    # layout, B, HK, HV, T, state, seqlens, neg, sv, l2, ga, h, g_mode, key
    rows = [
        ("BNSD", 1, 4, 4, 128, "none", "", False, False, False, False, False, "normal", "bnsd_mha_aligned"),
        ("BNSD", 1, 2, 8, 128, "none", "", False, False, False, False, False, "normal", "bnsd_gva"),
        ("BNSD", 1, 4, 4, 100, "none", "", False, False, False, False, False, "normal", "bnsd_tail"),
        ("BNSD", 2, 4, 4, 64, "none", "", False, False, False, False, False, "normal", "bnsd_batch2"),
        ("BSND", 1, 4, 4, 128, "none", "", False, False, False, False, False, "normal", "bsnd_aligned"),
        ("BSND", 1, 4, 4, 100, "none", "", False, False, False, False, False, "normal", "bsnd_tail"),
        ("BSND", 2, 4, 4, 64, "none", "", False, False, False, False, False, "normal", "bsnd_batch2"),
        ("NTD", 1, 4, 4, 128, "none", "", False, False, False, False, False, "normal", "ntd_fixed"),
        ("NTD", 1, 4, 4, 128, "none", "64,64", False, False, False, False, False, "normal", "ntd_var_aligned"),
        ("NTD", 1, 4, 4, 100, "none", "40,60", False, False, False, False, False, "normal", "ntd_var_uneven"),
        ("TND", 1, 4, 4, 128, "none", "", False, False, False, False, False, "normal", "tnd_fixed"),
        ("TND", 1, 4, 4, 128, "none", "64,64", False, False, False, False, False, "normal", "tnd_var_aligned"),
        ("TND", 1, 4, 4, 100, "none", "40,60", False, False, False, False, False, "normal", "tnd_var_uneven"),
        ("BNSD", 1, 4, 4, 32, "none", "", False, False, False, False, False, "normal", "single_chunk"),
        ("BNSD", 1, 4, 4, 20, "none", "", False, False, False, False, False, "normal", "single_chunk_tail"),
        ("NTD", 1, 4, 4, 32, "none", "16,16", False, False, False, False, False, "normal", "varlen_single"),
        ("NTD", 1, 4, 4, 20, "none", "12,8", False, False, False, False, False, "normal", "varlen_single_partial"),
        ("TND", 1, 4, 4, 32, "none", "16,16", False, False, False, False, False, "normal", "tnd_varlen_single"),
        ("BNSD", 1, 4, 4, 64, "bf16", "", False, False, False, False, False, "normal", "state_bf16"),
        ("BNSD", 1, 4, 4, 64, "fp32", "", False, False, False, False, False, "normal", "state_fp32"),
        ("BNSD", 1, 4, 4, 64, "zero", "", False, False, False, False, False, "normal", "state_zero"),
        ("BNSD", 1, 4, 4, 64, "none", "", True, False, False, False, False, "normal", "allow_neg"),
        ("BNSD", 1, 4, 4, 64, "none", "", False, True, False, False, False, "normal", "state_v_first"),
        ("BNSD", 1, 4, 4, 64, "none", "", False, False, True, False, False, "normal", "export_l2"),
        ("BNSD", 1, 4, 4, 64, "none", "", False, False, False, True, False, "normal", "export_ga"),
        ("BNSD", 1, 4, 4, 64, "none", "", False, False, False, False, True, "normal", "export_h"),
        ("BNSD", 1, 4, 4, 64, "none", "", False, False, True, True, False, "normal", "export_l2_ga"),
        ("BNSD", 1, 4, 4, 64, "none", "", False, False, True, False, True, "normal", "export_l2_h"),
        ("BNSD", 1, 4, 4, 64, "none", "", False, False, False, True, True, "normal", "export_ga_h"),
        ("BNSD", 1, 4, 4, 64, "none", "", False, False, True, True, True, "normal", "export_all"),
        ("BNSD", 1, 4, 4, 128, "none", "", False, False, False, False, False, "strong", "g_strong"),
        ("BNSD", 1, 4, 4, 128, "none", "", False, False, False, False, False, "spike", "g_spike"),
        ("BNSD", 1, 4, 4, 64, "none", "", False, False, False, False, False, "mix", "g_mix"),
        ("BNSD", 1, 4, 4, 8192, "none", "", False, False, False, False, False, "normal", "cap_t8192"),
        ("NTD", 1, 2, 8, 8000, "fp32", "3000,5000", True, True, False, False, False, "normal", "cap_varlen"),
    ]
    specs = []
    for index, row in enumerate(rows):
        layout, batch, hk, hv, tokens, state, seqlens, neg, sv, l2, ga, out_h, g_mode, key = row
        specs.append(
            _spec(
                case_key=f"slim_{key}", tags="slim", layout=layout, batch=batch,
                k_heads=hk, v_heads=hv, tokens=tokens, state=state, seqlens=seqlens,
                allow_neg=neg, state_v_first=sv, out_l2=l2, disable_recompute=ga,
                out_h=out_h, g_mode=g_mode, seed=SEED_BASE + 5000 + index,
            )
        )
    keys = [row["case_key"] for row in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate slim case_key")
    return specs


def perf_specs() -> list[dict]:
    rows = [
        ("BNSD", 1, 8, 8, 8192, "none", "", False, False, "perf_bnsd_h8_t8192"),
        ("BSND", 1, 4, 8, 4096, "none", "", True, True, "perf_bsnd_gva_t4096_neg_sv"),
    ]
    specs = []
    for index, (layout, batch, hk, hv, tokens, state, seqlens, allow_neg, state_v_first, key) in enumerate(rows):
        specs.append(
            _spec(
                case_key=key, tags="performance", layout=layout, batch=batch,
                k_heads=hk, v_heads=hv, tokens=tokens, state=state, seqlens=seqlens,
                allow_neg=allow_neg, state_v_first=state_v_first, seed=SEED_BASE + 1000 + index,
            )
        )
    return specs


def mss_specs() -> list[dict]:
    rows = [
        ("BNSD", 1, 4, 4, 64, "none", "", False, False),
        ("BSND", 1, 2, 4, 96, "bf16", "", True, True),
        ("NTD", 1, 2, 4, 96, "fp32", "64,32", False, False),
        ("TND", 1, 2, 4, 80, "zero", "48,32", True, True),
    ]
    specs = []
    for index, (layout, batch, hk, hv, tokens, state, seqlens, allow_neg, state_v_first) in enumerate(rows):
        specs.append(
            _spec(
                case_key=_key(layout, hk, hv, tokens, state, seqlens, allow_neg, state_v_first, batch),
                tags="mss", layout=layout, batch=batch, k_heads=hk, v_heads=hv, tokens=tokens,
                state=state, seqlens=seqlens, allow_neg=allow_neg, state_v_first=state_v_first,
                seed=SEED_BASE + 2000 + index,
            )
        )
    return specs


def _input(name: str, dtype: str, value, *, input_type: str = "attr", shape=None) -> dict:
    return {
        "name": name,
        "type": input_type,
        "required": True,
        "dtype": dtype,
        "shape": shape,
        "range_values": value,
        "backward": False,
    }


def _case_payload(case_id: int, spec: dict, standard: dict | None = None) -> dict:
    metadata = dict(spec)
    metadata["case_id"] = case_id
    inputs = [
        _input("low_precision_marker", "bf16", [0, 0], input_type="tensor", shape=[1]),
        _input("fp32_marker", "fp32", [0, 0], input_type="tensor", shape=[1]),
        _input(
            "case_spec",
            "non_param",
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ),
        _input("layout", "string", metadata["layout"]),
        _input("dtype", "string", metadata["dtype"]),
        _input("B", "int", metadata["B"]),
        _input("HK", "int", metadata["HK"]),
        _input("HV", "int", metadata["HV"]),
        _input("T", "int", metadata["T"]),
        _input("K", "int", metadata["K"]),
        _input("V", "int", metadata["V"]),
        _input("state_dtype", "string", metadata["state_dtype"]),
        _input("seqlens", "string", metadata["seqlens"]),
        _input("soc", "string", metadata["soc"]),
        _input("seed", "int", metadata["seed"]),
    ]
    return {
        "id": case_id,
        "default_seed": metadata["seed"],
        "name": f"{OP_NAME}_{case_id:04d}_{metadata['case_key']}",
        "aclnn_name": None,
        "version": "v2.1",
        "api": "pytorch",
        "api_type": "executor_chunk_gated_delta_rule_fwd_chunk32",
        "expected_error_msg": None,
        "backward": False,
        "standard": STANDARD if standard is None else standard,
        "outputs": None,
        "inputs": inputs,
        "save_name": OP_NAME,
    }


def _write(name: str, specs: list[dict], standard: dict | None = None) -> None:
    payloads = [
        _case_payload(case_id, spec, standard) for case_id, spec in enumerate(specs)
    ]
    path = HERE / name
    path.write_text(json.dumps(payloads, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{name}: {len(payloads)} cases")


def main() -> None:
    specs = accuracy_specs()
    print(f"accuracy: {len(specs)}")
    for row in specs:
        print(
            f"  {row['layout']} B={row['B']} H={row['HK']}x{row['HV']} T={row['T']} "
            f"state={row['state_dtype']} seq={row['seqlens'] or '-'} "
            f"neg={int(row['allow_neg_eigval'])} sv={int(row['state_v_first'])}"
        )
    _write(f"atk_{OP_NAME}.json", specs)
    _write(f"atk_{OP_NAME}_perf.json", perf_specs())
    _write(f"atk_{OP_NAME}_mss.json", mss_specs())
    slim = slim_specs()
    print(f"slim: {len(slim)}")
    for row in slim:
        print(f"  {row['case_key']}")
    _write(f"atk_{OP_NAME}_slim.json", slim)


if __name__ == "__main__":
    main()
