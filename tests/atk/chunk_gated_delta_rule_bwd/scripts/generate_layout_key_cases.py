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


def remaining_public_intra_profiles():
    # The composite API promotes g/beta to a common dtype. Complete the BF16
    # fixed/packed G branches not covered by the 64 Finalize combinations.
    for packed, group in ((False, 2), (False, 4),
                          (True, 1), (True, 2), (True, 3), (True, 4)):
        for sequence_major in (False, True):
            spec = dict(B=1 if packed else 2, HK=2, HV=2 * group, T=65,
                        scalar_dtype="bf16", use_exp2=True,
                        with_state=True, with_dht=True)
            if packed:
                spec.update(seqlens=[0, 1, 64, 0],
                            layout="TND" if sequence_major else "NTD")
            else:
                spec["layout"] = "BSND" if sequence_major else "BNSD"
            yield spec


def remaining_public_fwd_h_profiles():
    # Absent initial_state resolves FwdH's state dtype to FP32. Exercise it
    # independently of gate dtype, exponent mode and dense/packed selection.
    missing = (("bf16", False, False, False), ("bf16", False, False, True),
               ("bf16", False, True, True), ("bf16", True, False, True),
               ("fp32", False, False, False), ("fp32", False, False, True))
    for scalar, exp2, state, packed in missing:
        for sequence_major in (False, True):
            spec = dict(B=1 if packed else 2, HK=2, HV=6, T=65,
                        scalar_dtype=scalar, use_exp2=exp2,
                        with_state=state, with_dht=True)
            if packed:
                spec.update(seqlens=[0, 1, 64, 0],
                            layout="TND" if sequence_major else "NTD")
            else:
                spec["layout"] = "BSND" if sequence_major else "BNSD"
            yield spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--include-layout-regression", action="store_true")
    parser.add_argument("--complete-public-keys", action="store_true",
                        help="Append the 24 remaining public Intra/FwdH key cases")
    args = parser.parse_args()
    if args.complete_public_keys and not args.include_layout_regression:
        parser.error("--complete-public-keys requires --include-layout-regression")
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
    if args.complete_public_keys:
        profiles.extend(remaining_public_intra_profiles())
        profiles.extend(remaining_public_fwd_h_profiles())
    result = []
    for index, profile in enumerate(profiles):
        # 单轮双标杆失败时需要复检（固定结构，accuracy_lt 50 轮）：
        # arch22 case_id=1,6,20,23,29,33,35,47,50,51,61,112,113。
        # arch35 case_id=10,11,46,47,50,51,55,58,59。
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
