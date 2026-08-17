#!/usr/bin/env python3
"""Build or validate persistent CPU references for chunk_kda_fwd ATK cases."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from persistent_reference_cache import (
    CATALOG_FORMAT_VERSION,
    CacheReader,
    INPUT_SHARD_NAMES,
    PinnedCatalog,
    REFERENCE_SHARD_NAMES,
    ReferenceCacheError,
    VARIANT_MATERIALIZER_SCHEMA,
    build_catalog,
    build_chunk_kda_metadata,
    default_catalog_reference,
    default_cache_dir,
    file_sha256,
    write_catalog,
)


HERE = Path(__file__).resolve().parent
EXECUTOR_PATH = HERE / "executor_chunk_kda_fwd.py"
DEFAULT_CASE_JSON = HERE / "atk_chunk_kda_fwd.json"


def _case_spec(case: dict) -> dict:
    for item in case.get("inputs", []):
        if item.get("name") == "case_spec":
            value = item.get("range_values")
            return json.loads(value) if isinstance(value, str) else dict(value)
    raise ValueError(f"case id={case.get('id')} has no case_spec input")


def _parse_case_ids(values: list[str]) -> set[int]:
    selected: set[int] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(f"invalid descending case range {token!r}")
                selected.update(range(start, end + 1))
            else:
                selected.add(int(token))
    return selected


def _atk_json_records(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("ATK case JSON must contain a list")
    return [{"id": int(case["id"]), "spec": _case_spec(case)} for case in raw]


def _load_adapter(target: str):
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--case-adapter must use '<python_module>:<callable>' syntax")
    callable_obj = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callable_obj):
        raise ValueError(f"case adapter is not callable: {target}")
    return callable_obj


def _adapter_digest(target: str | None) -> str | None:
    if target is None:
        return None
    module_name = target.partition(":")[0]
    module = importlib.import_module(module_name)
    source = getattr(module, "__file__", None)
    if not source:
        raise ValueError(f"case adapter has no source file: {module_name}")
    return file_sha256(Path(source))


def _load_cases(
    path: Path,
    case_ids: set[int],
    adapter: str | None,
) -> list[tuple[int, dict]]:
    records = _atk_json_records(path) if adapter is None else list(_load_adapter(adapter)(path))
    cases = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"id", "spec"}:
            raise ValueError("case adapters must return {'id': int, 'spec': dict} records")
        case_id, spec = int(record["id"]), dict(record["spec"])
        tags = {tag.strip() for tag in str(spec.get("tags", "")).split(",") if tag.strip()}
        execution_tags = tags.intersection(
            {"accuracy", "run", "msopprof", "stress", "sanitizer"}
        )
        if len(execution_tags) != 1:
            raise ValueError(
                f"case id={case_id} must declare exactly one supported execution kind"
            )
        cases.append((case_id, spec))
    available = {case_id for case_id, _ in cases}
    missing = case_ids.difference(available)
    if missing:
        raise ValueError(f"unknown case ids: {sorted(missing)}")
    return [item for item in cases if not case_ids or item[0] in case_ids]


def _include_references(spec: dict) -> bool:
    tags = {tag.strip() for tag in str(spec.get("tags", "")).split(",")}
    return "accuracy" in tags and "negative" not in tags


def _cache_specs(spec: dict) -> list[dict]:
    tags = {tag.strip() for tag in str(spec.get("tags", "")).split(",")}
    if _include_references(spec) and "canonical_300" in tags:
        try:
            from canonical_case_adapter import materialize_cache_variants
        except ModuleNotFoundError:
            from test.chunk_kda_fwd.canonical_case_adapter import (
                materialize_cache_variants,
            )
        return materialize_cache_variants(spec)
    return [
        {
            "variant": str(spec.get("materialized_variant", "default")),
            "spec": spec,
        }
    ]


def _producer_metadata(
    spec: dict,
    *,
    producer_identity: dict | None = None,
) -> dict:
    seed = int(spec["seed"])
    return build_chunk_kda_metadata(
        spec,
        seed,
        EXECUTOR_PATH,
        **(producer_identity or {}),
        include_references=_include_references(spec),
    )


def _producer_identity_from_metadata(metadata: dict) -> dict:
    return {
        "producer_torch_version": metadata["producer_torch_version"],
        "producer_executor_sha256": metadata["executor_sha256"],
        "producer_golden_executor_sha256": metadata[
            "golden_executor_sha256"
        ],
        "producer_benchmark_executor_sha256": metadata[
            "benchmark_executor_sha256"
        ],
    }


def _producer_reader(
    spec: dict,
    cache_dir: Path,
    *,
    producer_identity: dict | None = None,
) -> CacheReader:
    return CacheReader(
        cache_dir,
        _producer_metadata(
            spec,
            producer_identity=producer_identity,
        ),
    )


def _catalog_entries(
    cache_plans: list[tuple[int, dict, list[dict]]],
    cache_dir: Path,
    *,
    producer_identity: dict | None,
    catalog_format_version: int,
) -> list[dict]:
    entries = []
    for case_id, logical_spec, variants in cache_plans:
        include_references = _include_references(logical_spec)
        cache_entries = []
        for item in variants:
            metadata = _producer_metadata(
                item["spec"],
                producer_identity=producer_identity,
            )
            cache_entry = {
                "variant": item["variant"],
                "cache_key": metadata["cache_key"],
                "required_shards": list(
                    REFERENCE_SHARD_NAMES
                    if include_references
                    else INPUT_SHARD_NAMES
                ),
            }
            if catalog_format_version == CATALOG_FORMAT_VERSION:
                reader = CacheReader(cache_dir, metadata)
                cache_entry.update(reader.catalog_content_pin)
            cache_entries.append(cache_entry)
        entries.append({"case_id": case_id, "cache_entries": cache_entries})
    return entries


def _validate(
    case_id: int,
    variant: str,
    spec: dict,
    catalog: PinnedCatalog,
) -> None:
    reader = catalog.reader_for(
        spec,
        int(spec["seed"]),
        EXECUTOR_PATH,
        include_references=_include_references(spec),
    )
    reader.validate_all()
    receipt = reader.validation_receipt
    print(
        f"VALID case_id={case_id} variant={variant} "
        f"key={reader.expected_metadata['cache_key']} "
        f"producer_torch={receipt['producer_torch_version']} "
        f"consumer_torch={receipt['consumer_torch_version']}"
    )


def _build(
    case_id: int,
    variant: str,
    spec: dict,
    cache_dir: Path,
    force: bool,
    *,
    producer_identity: dict | None = None,
    catalog_anchored: bool = False,
) -> None:
    if not force:
        try:
            reader = _producer_reader(
                spec,
                cache_dir,
                producer_identity=producer_identity,
            )
            reader.validate_all()
            print(
                f"VALID case_id={case_id} variant={variant} "
                f"key={reader.expected_metadata['cache_key']}"
            )
            return
        except ReferenceCacheError as exc:
            if "is missing" not in str(exc):
                raise
            if catalog_anchored:
                raise ReferenceCacheError(
                    "catalog-anchored build refuses to create a missing producer entry"
                ) from exc
    if catalog_anchored:
        raise ReferenceCacheError(
            "catalog-anchored build may validate existing entries only"
        )
    try:
        executor = importlib.import_module("executor_chunk_kda_fwd")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("atk"):
            raise RuntimeError(
                "cache build requires the ATK Python environment because it imports the executor"
            ) from exc
        raise
    print(
        f"BUILD case_id={case_id} variant={variant} seed={spec['seed']}",
        flush=True,
    )
    manifest = executor.build_persistent_reference_cache(
        spec,
        cache_dir,
        overwrite=force,
        include_references=_include_references(spec),
    )
    print(f"BUILT case_id={case_id} variant={variant} manifest={manifest.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--case-json", type=Path, default=DEFAULT_CASE_JSON)
    parser.add_argument(
        "--case-adapter",
        metavar="MODULE:CALLABLE",
        help=(
            "adapter for a canonical materializer; it must return explicit "
            "{'id': int, 'spec': dict} records"
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument(
        "--catalog",
        help=(
            "externally pinned catalog SHA256, catalog filename, or in-cache path; "
            "required by validate (or set KDA_ATK_PERSISTENT_CACHE_CATALOG); for "
            "build it explicitly anchors a validate-only v1/v2 catalog upgrade"
        ),
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        metavar="ID[,ID|START-END]",
        help="select case ids; repeat the option or omit it to select all cases",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an already committed entry (build only)",
    )
    args = parser.parse_args()
    if args.command != "build" and args.force:
        parser.error("--force is valid only with the build command")
    if args.command == "build" and args.force and args.catalog:
        parser.error("--force cannot be combined with catalog-anchored build")

    selected_ids = _parse_case_ids(args.case_id)
    source_path = args.case_json.resolve()
    cases = _load_cases(source_path, selected_ids, args.case_adapter)
    adapter_name = args.case_adapter or "atk-json:v1"
    pinned_catalog = None
    catalog_reference = (
        args.catalog or default_catalog_reference()
        if args.command == "validate"
        else args.catalog
    )
    if args.command == "validate" or catalog_reference is not None:
        pinned_catalog = PinnedCatalog(
            args.cache_dir,
            catalog_reference,
        )
    cache_plans = [
        (case_id, spec, _cache_specs(spec)) for case_id, spec in cases
    ]
    if not cache_plans or not cache_plans[0][2]:
        raise ValueError("selected case set has no cache entries")
    producer_identity = (
        pinned_catalog.producer_identity
        if pinned_catalog is not None
        else _producer_identity_from_metadata(
            _producer_metadata(cache_plans[0][2][0]["spec"])
        )
    )
    output_catalog_format = (
        pinned_catalog.catalog_format_version
        if args.command == "validate"
        else CATALOG_FORMAT_VERSION
    )
    cache_entry_count = sum(len(variants) for _, _, variants in cache_plans)
    case_ids = [case_id for case_id, _, _ in cache_plans]
    print(
        f"CASE_SET logical_count={len(cache_plans)} "
        f"cache_entry_count={cache_entry_count} ids={case_ids}"
    )

    def current_catalog(catalog_format_version: int) -> dict:
        return build_catalog(
            source_path,
            adapter_name,
            _catalog_entries(
                cache_plans,
                args.cache_dir,
                producer_identity=producer_identity,
                catalog_format_version=catalog_format_version,
            ),
            adapter_sha256=_adapter_digest(args.case_adapter),
            variant_materializer_schema=(
                VARIANT_MATERIALIZER_SCHEMA if args.case_adapter else None
            ),
            **producer_identity,
            catalog_format_version=catalog_format_version,
        )

    catalog = None
    failures = []
    passed_entries = 0
    if pinned_catalog is not None:
        try:
            catalog = current_catalog(pinned_catalog.catalog_format_version)
            pinned_catalog.validate_expected(catalog)
            print(
                f"CATALOG_PIN_VALID key={pinned_catalog.catalog_key} "
                f"producer_torch={pinned_catalog.producer_torch_version} "
                f"producer_executor={pinned_catalog.producer_executor_sha256} "
                f"consumer_torch={pinned_catalog.consumer_torch_version}"
            )
        except Exception as exc:
            print(f"ERROR catalog: {exc}")
            print(
                f"SUMMARY command={args.command} logical_selected={len(cases)} "
                f"cache_entries={cache_entry_count} "
                "passed_entries=0 failed_entries=1"
            )
            return 1
    for case_id, _logical_spec, variants in cache_plans:
        for item in variants:
            variant, spec = item["variant"], item["spec"]
            try:
                if args.command == "build":
                    _build(
                        case_id,
                        variant,
                        spec,
                        args.cache_dir,
                        args.force,
                        producer_identity=producer_identity,
                        catalog_anchored=pinned_catalog is not None,
                    )
                else:
                    _validate(case_id, variant, spec, pinned_catalog)
                passed_entries += 1
            except Exception as exc:
                failures.append((case_id, variant, str(exc)))
                print(f"ERROR case_id={case_id} variant={variant}: {exc}")
    if not failures:
        try:
            if args.command == "build":
                catalog = current_catalog(output_catalog_format)
                path = write_catalog(args.cache_dir, catalog)
                print(f"CATALOG_WRITTEN file={path.name}")
            else:
                print(f"CATALOG_VALID file={pinned_catalog.path.name}")
        except Exception as exc:
            failures.append((-1, str(exc)))
            print(f"ERROR catalog: {exc}")
    print(
        f"SUMMARY command={args.command} logical_selected={len(cases)} "
        f"cache_entries={cache_entry_count} "
        f"passed_entries={passed_entries} failed_entries={len(failures)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
