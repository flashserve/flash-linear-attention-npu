"""Build the chunk32 ATK matrices.

Accuracy covers the combinations chunk32 actually accepts: four layouts,
MHA/GVA, fixed and packed sequences, aligned and tail lengths, both
``allow_neg_eigval`` and ``state_v_first`` values, and initial state
none/bf16/fp32/zero. ``use_exp2``, L2 and beta sigmoid stay on the required
values. Shapes are medium; the longest sequences are about 24K tokens.
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
    tags: str = "accuracy",
    seed: int,
) -> dict:
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
        "scale": SCALE,
        "seed": seed,
    }


def _key(layout: str, k_heads: int, v_heads: int, tokens: int, state: str,
         seqlens: str, allow_neg: bool, state_v_first: bool, batch: int) -> str:
    head = "gva" if v_heads > k_heads else "mha"
    packed = "var" if seqlens else "fix"
    seq = "" if not seqlens else "_seq" + seqlens.replace(",", "-")
    return (
        f"{layout.lower()}_{head}_{packed}_B{batch}_H{k_heads}x{v_heads}"
        f"_T{tokens}_{state}{seq}_neg{int(allow_neg)}_sv{int(state_v_first)}"
    )


def accuracy_specs() -> list[dict]:
    """600 cells. Free axes are fully crossed at medium T; 16K is only the cap."""

    layouts = ("BNSD", "BSND", "NTD", "TND")
    negs = (False, True)
    svs = (False, True)
    states = ("none", "bf16", "fp32", "zero")
    heads = ((4, 4), (2, 8))  # MHA, GVA
    # Fixed aligned, fixed tail, multi-batch, packed. Varlen stays at B=1.
    profiles = (
        {"batch": 1, "tokens": 1024, "seqlens": ""},
        {"batch": 1, "tokens": 1000, "seqlens": ""},
        {"batch": 2, "tokens": 512, "seqlens": ""},
        {"batch": 1, "tokens": 1536, "seqlens": "512,1024"},
    )
    specs: list[dict] = []

    def add(layout, hk, hv, batch, tokens, state, seqlens, allow_neg, state_v_first) -> None:
        specs.append(
            _spec(
                case_key=_key(layout, hk, hv, tokens, state, seqlens, allow_neg, state_v_first, batch),
                layout=layout, batch=batch, k_heads=hk, v_heads=hv, tokens=tokens,
                state=state, seqlens=seqlens, allow_neg=allow_neg, state_v_first=state_v_first,
                seed=SEED_BASE + len(specs),
            )
        )

    for layout in layouts:
        for allow_neg in negs:
            for state_v_first in svs:
                for state in states:
                    for hk, hv in heads:
                        for profile in profiles:
                            add(
                                layout, hk, hv, profile["batch"], profile["tokens"],
                                state, profile["seqlens"], allow_neg, state_v_first,
                            )
    # Longer medium sequences, still at or under 16K. MHA only at 2048,
    # GVA on BNSD/BSND at 4096, then one 16K case per layout.
    for layout in layouts:
        for allow_neg in negs:
            for state_v_first in svs:
                for state in states:
                    add(layout, 4, 4, 1, 2048, state, "", allow_neg, state_v_first)
    for layout in ("BNSD", "BSND"):
        for allow_neg in negs:
            for state_v_first in svs:
                for state in ("none", "bf16", "fp32"):
                    add(layout, 2, 8, 1, 4096, state, "", allow_neg, state_v_first)
    long_cases = (
        ("BNSD", 4, 4, 16384, "none", "", False, False),
        ("BSND", 4, 8, 16384, "bf16", "", True, True),
        ("NTD", 2, 8, 16384, "fp32", "4096,12288", False, True),
        ("TND", 4, 4, 16000, "zero", "6000,10000", True, False),
    )
    for layout, hk, hv, tokens, state, seqlens, allow_neg, state_v_first in long_cases:
        add(layout, hk, hv, 1, tokens, state, seqlens, allow_neg, state_v_first)
    if len(specs) != 604:
        raise RuntimeError(f"expected 604 accuracy cases, got {len(specs)}")
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


def _case_payload(case_id: int, spec: dict) -> dict:
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
        "standard": STANDARD,
        "outputs": None,
        "inputs": inputs,
        "save_name": OP_NAME,
    }


def _write(name: str, specs: list[dict]) -> None:
    payloads = [_case_payload(case_id, spec) for case_id, spec in enumerate(specs)]
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


if __name__ == "__main__":
    main()
