"""Emit double-benchmark or memcheck cases for the public layout template branches."""

import argparse
from copy import deepcopy
from itertools import product
import json
from pathlib import Path


def key_profiles():
    # G=1 selects the arch22 owner schedule; G=3 selects the general schedule.
    for scalar, norm, sigmoid, exp2, owner, sequence_major in product(
        ("bf16", "fp32"), (False, True), (False, True), (False, True),
        (False, True), (False, True)
    ):
        yield dict(B=2, HK=2, HV=2 if owner else 6, T=65,
                   scalar_dtype=scalar, use_qk_l2norm=norm,
                   use_beta_sigmoid=sigmoid, use_exp2=exp2,
                   with_state=True, with_dht=True,
                   layout="BSND" if sequence_major else "BNSD")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--include-layout-regression", action="store_true")
    args = parser.parse_args()
    op_dir = Path(__file__).resolve().parents[1]
    cases = json.loads((op_dir / "atk_chunk_gated_delta_rule_bwd.json").read_text(encoding="utf-8"))
    profiles = list(key_profiles())
    if args.include_layout_regression:
        from check_layout_equivalence import supplemental_specs
        original = [json.loads(next(i["range_values"] for i in case["inputs"]
                                    if i["name"] == "case_spec")) for case in cases]
        for profile in original + supplemental_specs():
            for seq in (False, True):
                spec = dict(profile)
                packed = "seqlens" in spec
                spec["layout"] = ("TND" if seq else "NTD") if packed else (
                    "BSND" if seq else "BNSD")
                profiles.append(spec)
    result = []
    for index, profile in enumerate(profiles):
        # case_id=50 (FP32 scalar, norm, G=1 BNSD) needs accuracy_lt recheck
        # when its dg relative-error ratio fails in an individual random run.
        case = deepcopy(cases[0])
        seed = int(profile.get("seed", 20261107 + index)) + args.seed_offset
        profile = dict(profile, seed=seed, case_id=index,
                       name=f"layout_key_{index}_{profile['layout'].lower()}")
        case.update(id=index, default_seed=seed,
                    name=f"chunk_gated_delta_rule_bwd_{index:04d}_{profile['name']}")
        # Storage views and trace inputs are checked by check_layout_equivalence;
        # double benchmark cases retain the standard generated value ranges.
        profile.pop("noncontiguous", None)
        profile.pop("trace_values", None)
        next(i for i in case["inputs"] if i["name"] == "case_spec")["range_values"] = json.dumps(profile)
        result.append(case)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(result)} cases; seed offset={args.seed_offset}")


if __name__ == "__main__":
    main()
