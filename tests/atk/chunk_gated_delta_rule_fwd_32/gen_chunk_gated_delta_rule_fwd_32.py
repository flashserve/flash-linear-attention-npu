"""Build the frozen ATK matrices for chunk_gated_delta_rule_fwd_32.

Shapes come from ``chunk_gated_delta_rule_fwd/case_matrix_contract.py``.
V=256, FP16 and token-major TND are dropped: this op only accepts BF16,
K=V=128, and head-major BNSD / packed NTD. T and ``B*Hv*T`` stay in the
medium band of that contract so the CPU golden does not run out of memory.
Each kept shape is repeated on three seeds.
"""

from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path


OP_NAME = "chunk_gated_delta_rule_fwd_32"
SEED_BASE = 20261027
STANDARD = {"acc": "mixed_tolerance_bm", "perf": "not_key", "mem": 1.1}
SEEDS_PER_SHAPE = 3
HERE = Path(__file__).resolve().parent
CONTRACT = HERE.parent / "chunk_gated_delta_rule_fwd" / "case_matrix_contract.py"


def _load_contract():
    spec = importlib.util.spec_from_file_location("gdn_case_matrix_contract", CONTRACT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_dtype(scenario: str, local_index: int) -> str:
    if scenario in {"dense", "varlen"}:
        return "none"
    if scenario == "state_zero_final":
        return "zero"
    # initial_final and initial_only both become an initial tensor. The 32 op
    # always writes final_state, so the sibling's "initial only" still compares it.
    if local_index % 2 == 0:
        return "bf16" if scenario == "state_initial_final" else "fp32"
    return "fp32" if scenario == "state_initial_final" else "bf16"


def _bucket(scenario: str, batch: int, k_heads: int, v_heads: int, tokens: int, varlen: bool) -> str:
    gva = v_heads > k_heads
    aligned = tokens % 32 == 0
    packed = "ntd" if varlen else "bnsd"
    if tokens <= 33:
        return f"tiny_{packed}"
    if scenario in {"dense", "varlen"}:
        if not varlen and batch > 1:
            return "bnsd_batch"
        if gva and aligned:
            return f"{packed}_gva_align"
        if gva:
            return f"{packed}_gva_tail"
        if aligned:
            return f"{packed}_mha_align"
        return f"{packed}_mha_tail"
    if scenario == "state_zero_final":
        return f"zero_{packed}"
    return f"init_pending_{packed}"


def _lengths(cu_spec: str) -> str:
    if cu_spec in {"", "none"}:
        return ""
    offsets = [int(item) for item in cu_spec.split(",") if item.strip()]
    return ",".join(str(right - left) for left, right in zip(offsets, offsets[1:]))


def iter_filtered():
    contract = _load_contract()
    for scenario in contract.SCENARIOS:
        for local_index in range(contract.CASES_PER_SCENARIO):
            batch, k_heads, v_heads, tokens, value, _chunk, varlen, cu_spec = contract.case_contract(
                scenario, local_index
            )
            if value != 128 or tokens > 384 or batch * v_heads * tokens > 2048:
                continue
            state = _state_dtype(scenario, local_index)
            bucket = _bucket(scenario, batch, k_heads, v_heads, tokens, varlen)
            # Recompute the init bucket from the actual state, not from tokens.
            if scenario.startswith("state_") and scenario != "state_zero_final":
                bucket = f"init_{state}_{'ntd' if varlen else 'bnsd'}"
            yield {
                "scenario": scenario,
                "local_index": local_index,
                "bucket": bucket,
                "B": batch,
                "HK": k_heads,
                "HV": v_heads,
                "T": tokens,
                "state_dtype": state,
                "seqlens": _lengths(cu_spec) if varlen else "",
                "layout": "NTD" if varlen else "BNSD",
                "aligned": tokens % 32 == 0,
                "gva": v_heads > k_heads,
            }


def _row(
    scenario: str,
    local_index: int,
    bucket: str,
    batch: int,
    k_heads: int,
    v_heads: int,
    tokens: int,
    state: str,
    seqlens: str,
    layout: str,
) -> dict:
    return {
        "scenario": scenario,
        "local_index": local_index,
        "bucket": bucket,
        "B": batch,
        "HK": k_heads,
        "HV": v_heads,
        "T": tokens,
        "state_dtype": state,
        "seqlens": seqlens,
        "layout": layout,
        "aligned": tokens % 32 == 0,
        "gva": v_heads > k_heads,
    }


def select_bases() -> list[dict]:
    """Up to two sibling shapes per coverage bucket, then the aligned holes.

    The sibling contract's B=1 aligned lengths are real ``TOKEN_EDGES`` values
    (64 and 128). Under the V=128 filter they are not paired with B=1, so those
    two edges are added explicitly. Heads stay inside ``HEAD_PAIRS``.
    """

    grouped: dict[str, list[dict]] = defaultdict(list)
    filtered = list(iter_filtered())
    for row in filtered:
        grouped[row["bucket"]].append(row)
    chosen = []
    seen = set()

    def add(row: dict) -> None:
        key = (
            row["B"],
            row["HK"],
            row["HV"],
            row["T"],
            row["seqlens"],
            row["state_dtype"],
        )
        if key in seen:
            return
        seen.add(key)
        chosen.append(row)

    for bucket in sorted(grouped):
        for row in grouped[bucket][:2]:
            add(row)
    for row in filtered:
        if row["bucket"] == "bnsd_batch" and row["HK"] == row["HV"] and row["aligned"]:
            add(row)
            break
    # TOKEN_EDGES {64, 128} and HEAD_PAIRS (1,1), (2,4). Not emitted for V=128, B=1.
    for tokens, k_heads, v_heads, state, bucket in (
        (64, 1, 1, "none", "bnsd_mha_align"),
        (128, 1, 1, "none", "bnsd_mha_align"),
        (64, 2, 4, "none", "bnsd_gva_align"),
        (128, 1, 2, "fp32", "init_fp32_bnsd_align"),
        (64, 1, 1, "bf16", "init_bf16_bnsd_align"),
    ):
        add(
            _row(
                "edge",
                tokens,
                bucket,
                1,
                k_heads,
                v_heads,
                tokens,
                state,
                "",
                "BNSD",
            )
        )
    add(_row("edge", 128, "ntd_mha_align", 1, 2, 2, 128, "none", "64,64", "NTD"))
    return chosen


def _spec(base: dict, seed: int, replica: int) -> dict:
    tag = "gva" if base["gva"] else "mha"
    align = "align" if base["aligned"] else "tail"
    seq = "" if not base["seqlens"] else "_seq" + base["seqlens"].replace(",", "-")
    key = (
        f"{base['layout'].lower()}_{tag}_{align}_B{base['B']}_H{base['HK']}x{base['HV']}"
        f"_T{base['T']}_{base['state_dtype']}{seq}_s{replica}"
    )
    return {
        "case_key": key,
        "tags": "accuracy",
        "route": "ascendc",
        "soc": "ascend950",
        "dtype": "bf16",
        "layout": base["layout"],
        "B": base["B"],
        "HK": base["HK"],
        "HV": base["HV"],
        "T": base["T"],
        "K": 128,
        "V": 128,
        "state_dtype": base["state_dtype"],
        "seqlens": base["seqlens"],
        "sibling_scenario": base["scenario"],
        "sibling_index": base["local_index"],
        "scale": 0.08838834764831845,
        "seed": seed,
    }


def accuracy_specs() -> list[dict]:
    specs = []
    for index, base in enumerate(select_bases()):
        for replica in range(SEEDS_PER_SHAPE):
            specs.append(_spec(base, SEED_BASE + index * SEEDS_PER_SHAPE + replica, replica))
    return specs


def perf_specs() -> list[dict]:
    """NPU-only medium shapes. T=16K stays on the existing profiler, not in this JSON."""

    rows = [
        {
            "case_key": "perf_bnsd_h4_t512",
            "layout": "BNSD",
            "B": 1,
            "HK": 4,
            "HV": 4,
            "T": 512,
            "state_dtype": "none",
            "seqlens": "",
        },
        {
            "case_key": "perf_ntd_gva_t256",
            "layout": "NTD",
            "B": 1,
            "HK": 2,
            "HV": 4,
            "T": 256,
            "state_dtype": "fp32",
            "seqlens": "128,128",
        },
    ]
    specs = []
    for index, row in enumerate(rows):
        specs.append(
            {
                **row,
                "tags": "performance",
                "route": "ascendc",
                "soc": "ascend950",
                "dtype": "bf16",
                "K": 128,
                "V": 128,
                "sibling_scenario": "perf",
                "sibling_index": index,
                "scale": 0.08838834764831845,
                "seed": SEED_BASE + 1000 + index,
            }
        )
    return specs


def mss_specs() -> list[dict]:
    """Tiling key is 0 for every supported input. These four cover the memory paths."""

    rows = [
        ("mss_bnsd_none", "BNSD", 1, 1, 1, 64, "none", ""),
        ("mss_bnsd_bf16", "BNSD", 1, 1, 2, 40, "bf16", ""),
        ("mss_bnsd_fp32", "BNSD", 2, 2, 2, 32, "fp32", ""),
        ("mss_ntd_gva", "NTD", 1, 1, 2, 96, "none", "64,32"),
    ]
    specs = []
    for index, (key, layout, batch, hk, hv, tokens, state, seqlens) in enumerate(rows):
        specs.append(
            {
                "case_key": key,
                "tags": "mss",
                "route": "ascendc",
                "soc": "ascend950",
                "dtype": "bf16",
                "layout": layout,
                "B": batch,
                "HK": hk,
                "HV": hv,
                "T": tokens,
                "K": 128,
                "V": 128,
                "state_dtype": state,
                "seqlens": seqlens,
                "sibling_scenario": "mss",
                "sibling_index": index,
                "scale": 0.08838834764831845,
                "seed": SEED_BASE + 2000 + index,
            }
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
        "api_type": "executor_chunk_gated_delta_rule_fwd_32",
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
    bases = select_bases()
    print(f"base shapes: {len(bases)}")
    for row in bases:
        print(
            f"  {row['bucket']:22} {row['layout']} B={row['B']} H={row['HK']}x{row['HV']} "
            f"T={row['T']} state={row['state_dtype']} seqlens={row['seqlens'] or '-'}"
        )
    _write(f"atk_{OP_NAME}.json", accuracy_specs())
    _write(f"atk_{OP_NAME}_perf.json", perf_specs())
    _write(f"atk_{OP_NAME}_mss.json", mss_specs())


if __name__ == "__main__":
    main()
