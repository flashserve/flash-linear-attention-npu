#!/usr/bin/env python3
"""Repeat legacy ATK cases on NPU using only committed cache inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from persistent_reference_cache import PinnedCatalog, default_catalog_reference


HERE = Path(__file__).resolve().parent
DEFAULT_CASE_JSON = HERE / "atk_chunk_kda_fwd_pr297_48.json"


def _load_case_specs(path: Path, case_ids: list[int]) -> list[tuple[int, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("legacy ATK case JSON must contain a list")
    by_id = {int(case["id"]): case for case in payload}
    if len(by_id) != len(payload):
        raise ValueError("legacy ATK case JSON contains duplicate ids")

    selected = []
    for case_id in case_ids:
        case = by_id.get(case_id)
        if case is None:
            raise ValueError(f"legacy ATK case id {case_id} is missing")
        case_spec_inputs = [
            item for item in case.get("inputs", []) if item.get("name") == "case_spec"
        ]
        if len(case_spec_inputs) != 1:
            raise ValueError(f"case {case_id} does not contain exactly one case_spec")
        raw_spec = case_spec_inputs[0].get("range_values")
        spec = json.loads(raw_spec) if isinstance(raw_spec, str) else raw_spec
        if not isinstance(spec, dict) or int(spec.get("seed", -1)) < 0:
            raise ValueError(f"case {case_id} has an invalid case_spec")
        selected.append((case_id, spec))
    return selected


def _run_case(case_id: int, spec: dict, catalog: PinnedCatalog, device, repeats: int) -> dict:
    import torch

    from canonical_execution_runner import compare_outputs_bitwise
    from executor_chunk_kda_fwd import (
        _OUTPUT_NAMES,
        _EXECUTOR_PATH,
        _apply_input_storage,
        _prepared_inputs_from_cpu,
        _run_positive_npu,
        _select_cached_input_payload,
    )
    reader = catalog.reader_for(
        spec,
        int(spec["seed"]),
        _EXECUTOR_PATH,
        include_references=True,
    )
    for shard_name in reader.required_shards:
        reader.validate_shard_file(shard_name)
    cached_inputs = _select_cached_input_payload(reader.load_shard("inputs"), spec)
    inputs = _prepared_inputs_from_cpu(cached_inputs, device, high_precision=False)
    inputs = _apply_input_storage(inputs, spec)
    if inputs.seed != int(spec["seed"]):
        raise RuntimeError(
            f"case {case_id} cached input seed {inputs.seed} does not match {spec['seed']}"
        )

    baseline = None
    with torch.no_grad():
        for repeat in range(repeats):
            outputs = tuple(_run_positive_npu(inputs, spec))
            torch.npu.synchronize()
            if len(outputs) != len(_OUTPUT_NAMES):
                raise RuntimeError(
                    f"case {case_id} repeat {repeat} returned {len(outputs)} outputs"
                )
            for name, output in zip(_OUTPUT_NAMES, outputs):
                if (
                    output is not None
                    and output.is_floating_point()
                    and not torch.isfinite(output).all().item()
                ):
                    raise RuntimeError(
                        f"case {case_id} repeat {repeat} output {name} contains NaN or Inf"
                    )
            if baseline is None:
                baseline = tuple(
                    None if output is None else output.detach().clone()
                    for output in outputs
                )
                continue
            mismatches = compare_outputs_bitwise(torch, baseline, outputs)
            if mismatches:
                raise RuntimeError(
                    f"case {case_id} repeat {repeat} is not bitwise deterministic: "
                    f"{mismatches}"
                )
    return {
        "case_id": case_id,
        "repeats": repeats,
        "all_outputs_finite": True,
        "bitwise_equal_to_run_0": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-json", type=Path, default=DEFAULT_CASE_JSON)
    parser.add_argument("--case-id", type=int, action="append", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        default=default_catalog_reference(),
        help="externally pinned catalog SHA256, filename, or in-cache path",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if len(args.case_id) != len(set(args.case_id)):
        parser.error("--case-id values must be unique")

    import torch
    import torch_npu  # noqa: F401

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    catalog = PinnedCatalog(args.cache_dir, args.catalog)
    catalog.validate_source(
        args.case_json.resolve(),
        adapter="atk-json:v1",
    )
    results = [
        _run_case(case_id, spec, catalog, device, args.repeats)
        for case_id, spec in _load_case_specs(args.case_json.resolve(), args.case_id)
    ]
    print(
        json.dumps(
            {
                "status": "passed",
                "cache_receipt": catalog.validation_receipt,
                "cases": results,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
