"""Constrain chunk32 ATK cases for ``atk case``.

The YAML declares ``q/k/v/g/beta`` and one typed attribute per switch.
``after_case_config`` writes the shape and the attribute value for each row.
Accuracy covers the combinations chunk32 accepts: BNSD/BSND/NTD/TND, MHA/GVA,
aligned and tail lengths, packed sequences, ``allow_neg_eigval``,
``state_v_first``, initial state none/bf16/fp32/zero, and the optional
exports (L2 hats and rstd, beta_eff, g_cumsum and A, h). ``use_exp2`` stays
false and the in-kernel L2 and beta sigmoid stay on. T is at most 8192.
``g_mode`` is ``normal``, ``strong`` (every token -4), ``spike`` (first token
-100, second token 1) or ``mix`` (chunk 0 sampled, chunk 1 set to -4).
The executor applies those three patterns; ``normal`` keeps the sampled ``g``.
"""

from __future__ import annotations

import os

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig


OP_NAME = "chunk_gated_delta_rule_fwd_chunk32"
SEED_BASE = 20261028
SCALE = 128 ** -0.5
SUITES = ("accuracy", "slim", "perf", "mss", "dc")


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
    disable_recompute: bool = False,
    out_h: bool = False,
    g_mode: str = "normal",
    output_final_state: bool = True,
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
        "disable_recompute": disable_recompute,
        "out_h": out_h,
        "g_mode": g_mode,
        "output_final_state": output_final_state,
        "scale": SCALE,
        "seed": seed,
    }


def _key(layout: str, k_heads: int, v_heads: int, tokens: int, state: str,
         seqlens: str, allow_neg: bool, state_v_first: bool, batch: int,
         disable_recompute: bool = False, out_h: bool = False,
         g_mode: str = "normal") -> str:
    head = "gva" if v_heads > k_heads else "mha"
    packed = "var" if seqlens else "fix"
    seq = "" if not seqlens else "_seq" + seqlens.replace(",", "-")
    exports = f"_ga{int(disable_recompute)}_h{int(out_h)}"
    gate = "" if g_mode == "normal" else f"_{g_mode}"
    return (
        f"{layout.lower()}_{head}_{packed}_B{batch}_H{k_heads}x{v_heads}"
        f"_T{tokens}_{state}{seq}_neg{int(allow_neg)}_sv{int(state_v_first)}{exports}{gate}"
    )


def accuracy_specs() -> list[dict]:
    """Covering set. BNSD/NTD are fully crossed; BSND/TND and the exports are sampled.

    ``disable_recompute`` exports g_cumsum, A and the q/k hats together.
    ``out_h`` still exports h on its own. T stays at or under 8192.
    """

    negs = (False, True)
    svs = (False, True)
    states = ("none", "bf16", "fp32", "zero")
    heads = ((4, 4), (2, 8))  # MHA, GVA
    specs: list[dict] = []

    def add(layout, hk, hv, batch, tokens, state, seqlens, allow_neg, state_v_first,
            disable_recompute=False, out_h=False, g_mode="normal") -> None:
        specs.append(
            _spec(
                case_key=_key(
                    layout, hk, hv, tokens, state, seqlens, allow_neg, state_v_first, batch,
                    disable_recompute, out_h, g_mode,
                ),
                layout=layout, batch=batch, k_heads=hk, v_heads=hv, tokens=tokens,
                state=state, seqlens=seqlens, allow_neg=allow_neg, state_v_first=state_v_first,
                disable_recompute=disable_recompute, out_h=out_h,
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
        (True, False),
        (False, True),
        (True, True),
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
    for disable_recompute, out_h in export_modes:
        for row in export_rows:
            layout, hk, hv, batch, tokens, state, seqlens, allow_neg, state_v_first = row
            add(
                layout, hk, hv, batch, tokens, state, seqlens, allow_neg, state_v_first,
                disable_recompute, out_h,
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
    for disable_recompute, out_h in export_modes:
        for row in export_shapes:
            layout, hk, hv, batch, tokens, state, seqlens, allow_neg, state_v_first = row
            add(
                layout, hk, hv, batch, tokens, state, seqlens, allow_neg, state_v_first,
                disable_recompute, out_h,
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

    # layout, B, HK, HV, T, state, seqlens, neg, sv, ga, h, g_mode, key
    rows = [
        ('BNSD', 1, 4, 4, 128, 'none', '', False, False, False, False, 'normal', 'bnsd_mha_aligned'),
        ('BNSD', 1, 2, 8, 128, 'none', '', False, False, False, False, 'normal', 'bnsd_gva'),
        ('BNSD', 1, 4, 4, 100, 'none', '', False, False, False, False, 'normal', 'bnsd_tail'),
        ('BNSD', 2, 4, 4, 64, 'none', '', False, False, False, False, 'normal', 'bnsd_batch2'),
        ('BSND', 1, 4, 4, 128, 'none', '', False, False, False, False, 'normal', 'bsnd_aligned'),
        ('BSND', 1, 4, 4, 100, 'none', '', False, False, False, False, 'normal', 'bsnd_tail'),
        ('BSND', 2, 4, 4, 64, 'none', '', False, False, False, False, 'normal', 'bsnd_batch2'),
        ('NTD', 1, 4, 4, 128, 'none', '', False, False, False, False, 'normal', 'ntd_fixed'),
        ('NTD', 1, 4, 4, 128, 'none', '64,64', False, False, False, False, 'normal', 'ntd_var_aligned'),
        ('NTD', 1, 4, 4, 100, 'none', '40,60', False, False, False, False, 'normal', 'ntd_var_uneven'),
        ('TND', 1, 4, 4, 128, 'none', '', False, False, False, False, 'normal', 'tnd_fixed'),
        ('TND', 1, 4, 4, 128, 'none', '64,64', False, False, False, False, 'normal', 'tnd_var_aligned'),
        ('TND', 1, 4, 4, 100, 'none', '40,60', False, False, False, False, 'normal', 'tnd_var_uneven'),
        ('BNSD', 1, 4, 4, 32, 'none', '', False, False, False, False, 'normal', 'single_chunk'),
        ('BNSD', 1, 4, 4, 20, 'none', '', False, False, False, False, 'normal', 'single_chunk_tail'),
        ('NTD', 1, 4, 4, 32, 'none', '16,16', False, False, False, False, 'normal', 'varlen_single'),
        ('NTD', 1, 4, 4, 20, 'none', '12,8', False, False, False, False, 'normal', 'varlen_single_partial'),
        ('TND', 1, 4, 4, 32, 'none', '16,16', False, False, False, False, 'normal', 'tnd_varlen_single'),
        ('BNSD', 1, 4, 4, 64, 'bf16', '', False, False, False, False, 'normal', 'state_bf16'),
        ('BNSD', 1, 4, 4, 64, 'fp32', '', False, False, False, False, 'normal', 'state_fp32'),
        ('BNSD', 1, 4, 4, 64, 'zero', '', False, False, False, False, 'normal', 'state_zero'),
        ('BNSD', 1, 4, 4, 64, 'none', '', True, False, False, False, 'normal', 'allow_neg'),
        ('BNSD', 1, 4, 4, 64, 'none', '', False, True, False, False, 'normal', 'state_v_first'),
        ('BNSD', 1, 4, 4, 64, 'none', '', False, False, False, False, 'normal', 'export_off'),
        ('BNSD', 1, 4, 4, 64, 'none', '', False, False, True, False, 'normal', 'export_ga'),
        ('BNSD', 1, 4, 4, 64, 'none', '', False, False, False, True, 'normal', 'export_h'),
        ('BNSD', 1, 4, 4, 64, 'none', '', False, False, True, True, 'normal', 'export_ga_h'),
        ('BNSD', 1, 4, 4, 128, 'none', '', False, False, False, False, 'strong', 'g_strong'),
        ('BNSD', 1, 4, 4, 128, 'none', '', False, False, False, False, 'spike', 'g_spike'),
        ('BNSD', 1, 4, 4, 64, 'none', '', False, False, False, False, 'mix', 'g_mix'),
        ('BNSD', 1, 4, 4, 8192, 'none', '', False, False, False, False, 'normal', 'cap_t8192'),
        ('NTD', 1, 2, 8, 8000, 'fp32', '3000,5000', True, True, False, False, 'normal', 'cap_varlen'),
    ]
    specs = []
    for index, row in enumerate(rows):
        layout, batch, hk, hv, tokens, state, seqlens, neg, sv, ga, out_h, g_mode, key = row
        specs.append(
            _spec(
                case_key=f"slim_{key}", tags="slim", layout=layout, batch=batch,
                k_heads=hk, v_heads=hv, tokens=tokens, state=state, seqlens=seqlens,
                allow_neg=neg, state_v_first=sv, disable_recompute=ga,
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
    """The 60 sanitizer rows. ``seqlens`` is empty when the length is fixed."""

    # key, layout, B, HK, HV, T, state, seqlens, neg, sv, ga, h, final
    rows = [
        ('mss_src00_bsnd_dense_T8192', 'BSND', 1, 16, 32, 8192, 'none', '', False, False, True, False, False),
        ('mss_src01_bsnd_varlen_T8088', 'BSND', 1, 32, 32, 8088, 'none', '83,219,1069,249,1126,5342', False, False, True, False, False),
        ('mss_src02_bnsd_state_zero_final_T1', 'BNSD', 1, 1, 1, 1, 'zero', '1', True, True, False, False, True),
        ('mss_src03_ntd_state_initial_final_T1', 'NTD', 1, 1, 1, 1, 'fp32', '', True, False, False, True, True),
        ('mss_src04_bsnd_varlen_T1', 'BSND', 1, 1, 2, 1, 'none', '1', True, True, False, True, False),
        ('mss_src05_ntd_state_initial_final_T1', 'NTD', 1, 1, 2, 1, 'fp32', '1', True, True, False, False, True),
        ('mss_src06_bsnd_state_zero_final_T1', 'BSND', 1, 1, 2, 1, 'zero', '1', False, True, True, True, True),
        ('mss_src07_ntd_varlen_T1', 'NTD', 1, 1, 2, 1, 'none', '1', False, False, False, True, True),
        ('mss_src08_bsnd_state_zero_final_T1', 'BSND', 1, 1, 2, 1, 'zero', '1', True, False, True, True, True),
        ('mss_src09_ntd_varlen_T1', 'NTD', 1, 1, 2, 1, 'none', '1', False, True, True, True, True),
        ('mss_src10_tnd_state_initial_final_T1', 'TND', 1, 1, 2, 1, 'fp32', '1', False, True, False, True, True),
        ('mss_src11_tnd_state_initial_only_T1', 'TND', 1, 1, 2, 1, 'fp32', '1', False, True, True, True, True),
        ('mss_src12_tnd_state_initial_final_T1', 'TND', 1, 1, 2, 1, 'fp32', '1', False, True, False, False, False),
        ('mss_src13_bnsd_state_initial_only_T1', 'BNSD', 1, 1, 3, 1, 'fp32', '1', False, False, False, False, True),
        ('mss_src14_ntd_dense_T1', 'NTD', 1, 1, 3, 1, 'none', '', True, False, True, False, True),
        ('mss_src15_tnd_state_zero_final_T1', 'TND', 1, 1, 3, 1, 'zero', '1', False, False, False, False, True),
        ('mss_src16_ntd_varlen_T1', 'NTD', 1, 2, 2, 1, 'none', '1', True, True, True, True, False),
        ('mss_src17_tnd_state_zero_final_T1', 'TND', 2, 1, 1, 1, 'zero', '', True, False, False, False, False),
        ('mss_src18_tnd_varlen_T1', 'TND', 1, 2, 2, 1, 'none', '1', False, False, False, True, True),
        ('mss_src19_bsnd_varlen_T2', 'BSND', 1, 1, 1, 2, 'none', '2', False, True, False, True, False),
        ('mss_src20_ntd_varlen_T2', 'NTD', 1, 1, 1, 2, 'none', '2', True, False, True, True, False),
        ('mss_src21_tnd_state_initial_final_T1', 'TND', 1, 2, 2, 1, 'fp32', '1', False, False, True, False, True),
        ('mss_src22_bsnd_state_zero_final_T1', 'BSND', 1, 2, 2, 1, 'zero', '1', False, True, False, True, False),
        ('mss_src23_bsnd_state_initial_only_T1', 'BSND', 2, 1, 1, 1, 'fp32', '', False, False, True, True, True),
        ('mss_src24_tnd_dense_T2', 'TND', 1, 1, 1, 2, 'none', '', True, False, True, True, False),
        ('mss_src25_ntd_varlen_T1', 'NTD', 1, 2, 2, 1, 'none', '1', False, True, True, False, False),
        ('mss_src26_tnd_state_initial_final_T1', 'TND', 1, 2, 2, 1, 'fp32', '1', True, False, True, True, True),
        ('mss_src27_bsnd_varlen_T2', 'BSND', 1, 1, 1, 2, 'none', '2', True, True, True, True, True),
        ('mss_src28_bsnd_varlen_T2', 'BSND', 1, 1, 1, 2, 'none', '2', False, True, False, True, False),
        ('mss_src29_bnsd_state_zero_final_T1', 'BNSD', 1, 2, 2, 1, 'zero', '1', False, False, True, True, True),
        ('mss_src30_ntd_varlen_T1', 'NTD', 1, 2, 2, 1, 'none', '1', False, False, False, True, False),
        ('mss_src31_bnsd_state_initial_only_T2', 'BNSD', 1, 1, 1, 2, 'fp32', '2', False, False, False, True, True),
        ('mss_src32_bnsd_state_initial_only_T2', 'BNSD', 1, 1, 1, 2, 'fp32', '1,1', False, False, True, True, False),
        ('mss_src33_bnsd_varlen_T1', 'BNSD', 1, 1, 4, 1, 'none', '1', True, True, True, False, False),
        ('mss_src34_ntd_varlen_T1', 'NTD', 1, 1, 4, 1, 'none', '1', True, False, False, True, False),
        ('mss_src35_ntd_state_initial_only_T1', 'NTD', 1, 1, 4, 1, 'fp32', '1', False, True, False, False, False),
        ('mss_src36_bsnd_varlen_T1', 'BSND', 1, 1, 4, 1, 'none', '1', False, False, False, False, True),
        ('mss_src37_bnsd_varlen_T1', 'BNSD', 1, 2, 4, 1, 'none', '1', True, False, True, False, True),
        ('mss_src38_bnsd_state_initial_final_T1', 'BNSD', 1, 2, 4, 1, 'fp32', '1', False, True, False, False, True),
        ('mss_src39_bsnd_varlen_T2', 'BSND', 1, 1, 2, 2, 'none', '2', False, True, True, False, False),
        ('mss_src40_ntd_state_initial_only_T2', 'NTD', 1, 1, 2, 2, 'fp32', '2', False, False, True, True, True),
        ('mss_src41_bnsd_state_zero_final_T2', 'BNSD', 1, 1, 2, 2, 'zero', '1,1', False, False, False, False, True),
        ('mss_src42_ntd_state_initial_only_T2', 'NTD', 1, 1, 2, 2, 'fp32', '1,1', False, False, False, True, True),
        ('mss_src43_ntd_varlen_T2', 'NTD', 1, 1, 2, 2, 'none', '2', False, True, False, True, True),
        ('mss_src44_bsnd_state_zero_final_T1', 'BSND', 1, 2, 4, 1, 'zero', '1', False, False, False, True, True),
        ('mss_src45_tnd_varlen_T1', 'TND', 1, 3, 3, 1, 'none', '1', True, False, True, True, True),
        ('mss_src46_bsnd_state_initial_final_T1', 'BSND', 1, 3, 3, 1, 'fp32', '1', True, True, False, False, False),
        ('mss_src47_bsnd_state_initial_only_T1', 'BSND', 1, 3, 3, 1, 'fp32', '1', True, True, False, False, False),
        ('mss_src48_tnd_state_initial_final_T1', 'TND', 1, 3, 3, 1, 'fp32', '1', False, False, True, False, False),
        ('mss_src49_tnd_state_initial_final_T1', 'TND', 1, 3, 3, 1, 'fp32', '1', True, True, True, True, True),
        ('mss_src50_tnd_state_zero_final_T1', 'TND', 1, 3, 3, 1, 'zero', '1', False, True, True, True, True),
        ('mss_src51_tnd_state_initial_only_T1', 'TND', 1, 3, 3, 1, 'fp32', '1', False, False, True, True, True),
        ('mss_src52_bsnd_varlen_T1', 'BSND', 1, 3, 3, 1, 'none', '1', False, False, True, False, False),
        ('mss_src53_tnd_state_zero_final_T1', 'TND', 1, 3, 3, 1, 'zero', '1', True, False, True, False, False),
        ('mss_src54_tnd_state_initial_final_T1', 'TND', 1, 2, 6, 1, 'fp32', '1', False, False, True, True, False),
        ('mss_src55_tnd_varlen_T1', 'TND', 1, 2, 6, 1, 'none', '1', False, True, False, False, True),
        ('mss_src56_bsnd_varlen_T2', 'BSND', 1, 1, 3, 2, 'none', '1,1', True, True, False, True, True),
        ('mss_src57_bsnd_state_initial_final_T2', 'BSND', 1, 1, 3, 2, 'fp32', '1,1', False, True, False, True, False),
        ('mss_src58_tnd_varlen_T2', 'TND', 1, 1, 3, 2, 'none', '1,1', False, True, True, False, False),
        ('mss_src59_ntd_varlen_T1', 'NTD', 1, 2, 6, 1, 'none', '1', False, True, False, False, True),
    ]
    specs = []
    for index, row in enumerate(rows):
        key, layout, batch, hk, hv, tokens, state, seqlens, neg, sv, ga, out_h, final = row
        specs.append(
            _spec(
                case_key=key, tags="mss", layout=layout, batch=batch, k_heads=hk, v_heads=hv,
                tokens=tokens, state=state, seqlens=seqlens, allow_neg=neg, state_v_first=sv,
                disable_recompute=ga, out_h=out_h, output_final_state=final,
                seed=SEED_BASE + 2000 + index,
            )
        )
    return specs


def dc_specs() -> list[dict]:
    return [
        _spec(
            case_key="dc_bnsd_mha_fix_B1_H28x28_T8192_none_neg1_sv1_ga1_h1",
            tags="determinism", layout="BNSD", batch=1, k_heads=28, v_heads=28, tokens=8192,
            state="none", allow_neg=True, state_v_first=True, disable_recompute=True,
            out_h=True, output_final_state=True, seed=20261105,
        )
    ]


def suite_specs(name: str) -> list[dict]:
    builders = {
        "accuracy": accuracy_specs,
        "slim": slim_specs,
        "perf": perf_specs,
        "mss": mss_specs,
        "dc": dc_specs,
    }
    try:
        specs = builders[name]()
    except KeyError as exc:
        raise ValueError(f"CHUNK32_ATK_SUITE must be one of {SUITES}, got {name!r}") from exc
    keys = [row["case_key"] for row in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate case_key in {name}")
    return specs


Q, K, V, G, BETA = 0, 1, 2, 3, 4
SCALE_I, CHUNK, LAYOUT, STATE, SEQLENS = 5, 6, 7, 8, 9
NEG, SV, GA, OUT_H, G_MODE, FINAL = 10, 11, 12, 13, 14, 15


def _tensor_shapes(spec: dict) -> dict[str, list[int]]:
    batch, k_heads, v_heads = int(spec["B"]), int(spec["HK"]), int(spec["HV"])
    tokens, key, value = int(spec["T"]), int(spec["K"]), int(spec["V"])
    if str(spec["layout"]).upper() in {"BSND", "TND"}:
        return {
            "q": [batch, tokens, k_heads, key],
            "k": [batch, tokens, k_heads, key],
            "v": [batch, tokens, v_heads, value],
            "g": [batch, tokens, v_heads],
            "beta": [batch, tokens, v_heads],
        }
    return {
        "q": [batch, k_heads, tokens, key],
        "k": [batch, k_heads, tokens, key],
        "v": [batch, v_heads, tokens, value],
        "g": [batch, v_heads, tokens],
        "beta": [batch, v_heads, tokens],
    }


@GENERATOR_REGISTRY.register("generator_chunk_gated_delta_rule_fwd_chunk32")
class ChunkGatedDeltaRuleFwdChunk32Generator(CaseGenerator):
    def __init__(self, config) -> None:
        super().__init__(config)
        suite = os.environ.get("CHUNK32_ATK_SUITE", "accuracy")
        self._specs = suite_specs(suite)
        self._case_index = 0

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        if self._case_index >= len(self._specs):
            raise RuntimeError(
                f"CHUNK32_ATK_SUITE has {len(self._specs)} rows, "
                f"but ATK asked for case {self._case_index}"
            )
        spec = self._specs[self._case_index]
        shapes = _tensor_shapes(spec)
        case_config.inputs[Q].shape = shapes["q"]
        case_config.inputs[K].shape = shapes["k"]
        case_config.inputs[K].dtype = case_config.inputs[Q].dtype
        case_config.inputs[V].shape = shapes["v"]
        case_config.inputs[V].dtype = case_config.inputs[Q].dtype
        case_config.inputs[G].shape = shapes["g"]
        case_config.inputs[G].dtype = "fp32"
        case_config.inputs[BETA].shape = shapes["beta"]
        case_config.inputs[BETA].dtype = case_config.inputs[Q].dtype
        case_config.inputs[SCALE_I].range_values = float(spec["scale"])
        case_config.inputs[CHUNK].range_values = 32
        case_config.inputs[LAYOUT].range_values = str(spec["layout"]).upper()
        case_config.inputs[STATE].range_values = str(spec["state_dtype"])
        case_config.inputs[SEQLENS].range_values = str(spec["seqlens"])
        case_config.inputs[NEG].range_values = bool(spec["allow_neg_eigval"])
        case_config.inputs[SV].range_values = bool(spec["state_v_first"])
        case_config.inputs[GA].range_values = bool(spec["disable_recompute"])
        case_config.inputs[OUT_H].range_values = bool(spec["out_h"])
        case_config.inputs[G_MODE].range_values = str(spec["g_mode"])
        case_config.inputs[FINAL].range_values = bool(spec["output_final_state"])
        case_config.name = f"{OP_NAME}_{self._case_index:04d}_{spec['case_key']}"
        if spec["tags"] == "determinism":
            case_config.standard.acc = None
        self._case_index += 1
        return case_config


def main() -> None:
    for name in SUITES:
        print(f"{name}: {len(suite_specs(name))}")


if __name__ == "__main__":
    main()
