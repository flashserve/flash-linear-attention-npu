"""Strict, atomic persistent cache for chunk_kda_fwd ATK references."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

import torch


CACHE_FORMAT_VERSION = 3
CATALOG_FORMAT_VERSION = 2
PINNED_CATALOG_ENV = "KDA_ATK_PERSISTENT_CACHE_CATALOG"
REFERENCE_SCHEMA = "chunk_kda_fwd.cpu_fp64_and_same_precision.v2"
INPUT_ONLY_SCHEMA = "chunk_kda_fwd.deterministic_inputs_only.v2"
INPUT_GENERATOR_SCHEMA = "chunk_kda_fwd.deterministic_quantized_inputs.v2"
VARIANT_MATERIALIZER_SCHEMA = "chunk_kda_fwd.canonical_variant_materializers.v2"
OUTPUT_NAMES = (
    "attn_out",
    "final_state",
    "gk",
    "Aqk",
    "Akk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "h",
    "initial_state_out",
)
OUTPUT_SCHEMA = {
    "container": "tuple",
    "names": list(OUTPUT_NAMES),
    "visibility": "exact_normalized_spec_output_policy",
}
INPUT_SHARD_NAMES = ("inputs",)
REFERENCE_SHARD_NAMES = ("inputs", "cpu_fp64", "cpu_same_precision")
SHARD_NAMES = REFERENCE_SHARD_NAMES
_NON_SEMANTIC_SPEC_KEYS = frozenset(
    {
        "case_key",
        "expected_code_name",
        "expected_message",
        "expected_return_code",
        "optional_spec",
        "profile",
        "route",
        "shape_spec",
        "soc",
        "status",
        "tags",
    }
)


class ReferenceCacheError(RuntimeError):
    """Raised when a persistent reference cache cannot be trusted."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CATALOG_NAME_RE = re.compile(r"^catalog-([0-9a-f]{64})\.json$")


def _sha256_json(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    text = str(value)
    if not _SHA256_RE.fullmatch(text):
        raise ReferenceCacheError(f"{field} must be a lowercase SHA256 digest")
    return text


def _require_torch_version(value: Any, field: str = "producer_torch_version") -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ReferenceCacheError(f"{field} must be a non-empty exact version string")
    return value


def default_catalog_reference() -> str | None:
    configured = os.environ.get(PINNED_CATALOG_ENV, "").strip()
    return configured or None


def default_cache_dir() -> Path:
    configured = os.environ.get("KDA_ATK_PERSISTENT_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "fla_npu" / "chunk_kda_fwd_atk"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cache metadata contains unsupported value {type(value).__name__}")


def normalize_spec(spec: dict) -> dict:
    """Return the semantic, JSON-canonical portion of an ATK case spec."""
    return _canonical_value(
        {key: value for key, value in spec.items() if key not in _NON_SEMANTIC_SPEC_KEYS}
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_metadata(
    spec: dict,
    seed: int,
    executor_path: Path,
    *,
    executor_digest: str | None = None,
    golden_executor_digest: str | None = None,
    benchmark_executor_digest: str | None = None,
    producer_torch_version: str | None = None,
    reference_schema: str = REFERENCE_SCHEMA,
    output_schema: dict | None = None,
    required_shards: tuple[str, ...] = REFERENCE_SHARD_NAMES,
    producer_source_digests: dict[str, str] | None = None,
    variant_materializer_schema: str | None = None,
) -> dict:
    full_executor_digest = executor_digest or file_sha256(executor_path)
    normalized_spec = normalize_spec(spec)
    producer_version = (
        str(torch.__version__)
        if producer_torch_version is None
        else _require_torch_version(producer_torch_version)
    )
    input_identity = {
        "input_generator_schema": INPUT_GENERATOR_SCHEMA,
        "normalized_spec": normalized_spec,
        "runtime_seed": int(seed),
    }
    input_digest = hashlib.sha256(
        json.dumps(input_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "reference_schema": reference_schema,
        "input_generator_schema": INPUT_GENERATOR_SCHEMA,
        "deterministic_input_sha256": input_digest,
        "output_schema": _canonical_value(output_schema or {"container": "tuple"}),
        "normalized_spec": normalized_spec,
        "runtime_seed": int(seed),
        "executor_sha256": full_executor_digest,
        "golden_executor_sha256": golden_executor_digest or full_executor_digest,
        "benchmark_executor_sha256": benchmark_executor_digest or full_executor_digest,
        "producer_torch_version": producer_version,
        "required_shards": list(required_shards),
        "producer_source_digests": _canonical_value(producer_source_digests or {}),
        "variant_materializer_schema": variant_materializer_schema,
    }
    return {**identity, "cache_key": _sha256_json(identity)}


def build_chunk_kda_metadata(
    spec: dict,
    seed: int,
    executor_path: Path,
    *,
    producer_torch_version: str | None = None,
    producer_executor_sha256: str | None = None,
    producer_golden_executor_sha256: str | None = None,
    producer_benchmark_executor_sha256: str | None = None,
    include_references: bool = True,
) -> dict:
    executor_digest = (
        file_sha256(executor_path)
        if producer_executor_sha256 is None
        else _require_sha256(
            producer_executor_sha256, "producer_executor_sha256"
        )
    )

    tags = {tag.strip() for tag in str(spec.get("tags", "")).split(",")}
    producer_source_digests = {}
    variant_materializer_schema = None
    if "canonical_300" in tags:
        adapter_name = (
            "canonical_case_adapter.py"
            if include_references
            else "canonical_execution_adapter.py"
        )
        adapter_path = Path(__file__).resolve().parent / adapter_name
        producer_source_digests[adapter_name] = file_sha256(adapter_path)
        variant_materializer_schema = VARIANT_MATERIALIZER_SCHEMA

    def domain_digest(role: str) -> str:
        return hashlib.sha256(f"{executor_digest}:{role}".encode("ascii")).hexdigest()

    golden_executor_digest = (
        domain_digest("cpu_fp64:_reference_impl")
        if producer_golden_executor_sha256 is None
        else _require_sha256(
            producer_golden_executor_sha256,
            "producer_golden_executor_sha256",
        )
    )
    benchmark_executor_digest = (
        domain_digest("cpu_same_precision:_reference_model_parallel")
        if producer_benchmark_executor_sha256 is None
        else _require_sha256(
            producer_benchmark_executor_sha256,
            "producer_benchmark_executor_sha256",
        )
    )

    return build_metadata(
        spec,
        seed,
        executor_path,
        executor_digest=executor_digest,
        golden_executor_digest=golden_executor_digest,
        benchmark_executor_digest=benchmark_executor_digest,
        producer_torch_version=producer_torch_version,
        reference_schema=REFERENCE_SCHEMA if include_references else INPUT_ONLY_SCHEMA,
        output_schema=OUTPUT_SCHEMA if include_references else {"container": "none"},
        required_shards=(
            REFERENCE_SHARD_NAMES if include_references else INPUT_SHARD_NAMES
        ),
        producer_source_digests=producer_source_digests,
        variant_materializer_schema=variant_materializer_schema,
    )


def cache_entry_dir(cache_dir: Path, metadata: dict) -> Path:
    return Path(cache_dir) / _require_sha256(metadata["cache_key"], "cache_key")


def build_catalog(
    source_path: Path,
    adapter: str,
    entries: list[dict],
    *,
    adapter_sha256: str | None = None,
    variant_materializer_schema: str | None = None,
    producer_torch_version: str | None = None,
    producer_executor_sha256: str | None = None,
    producer_golden_executor_sha256: str | None = None,
    producer_benchmark_executor_sha256: str | None = None,
    catalog_format_version: int = CATALOG_FORMAT_VERSION,
) -> dict:
    if catalog_format_version not in {1, CATALOG_FORMAT_VERSION}:
        raise ReferenceCacheError(
            f"unsupported cache catalog format version {catalog_format_version}"
        )
    producer_version = (
        str(torch.__version__)
        if producer_torch_version is None
        else _require_torch_version(producer_torch_version)
    )
    producer_executor_digests = None
    if catalog_format_version == CATALOG_FORMAT_VERSION:
        producer_executor_digests = {
            "producer_executor_sha256": _require_sha256(
                producer_executor_sha256, "producer_executor_sha256"
            ),
            "producer_golden_executor_sha256": _require_sha256(
                producer_golden_executor_sha256,
                "producer_golden_executor_sha256",
            ),
            "producer_benchmark_executor_sha256": _require_sha256(
                producer_benchmark_executor_sha256,
                "producer_benchmark_executor_sha256",
            ),
        }
    case_ids = [int(entry["case_id"]) for entry in entries]
    if len(case_ids) != len(set(case_ids)):
        raise ReferenceCacheError("cache catalog contains duplicate case ids")

    def cache_entries(entry: dict) -> list[dict]:
        variants = entry.get("cache_entries")
        if variants is None:
            variant = {
                "variant": entry.get("variant", "default"),
                "cache_key": entry["cache_key"],
                "required_shards": entry.get(
                    "required_shards", REFERENCE_SHARD_NAMES
                ),
            }
            if catalog_format_version == CATALOG_FORMAT_VERSION:
                variant.update(
                    {
                        "manifest_generation": entry.get(
                            "manifest_generation"
                        ),
                        "shard_sha256": entry.get("shard_sha256"),
                    }
                )
            variants = [variant]
        normalized = []
        for item in variants:
            required_shards = list(
                item.get("required_shards", REFERENCE_SHARD_NAMES)
            )
            if (
                not required_shards
                or any(not isinstance(name, str) for name in required_shards)
                or len(required_shards) != len(set(required_shards))
                or set(required_shards).difference(REFERENCE_SHARD_NAMES)
            ):
                raise ReferenceCacheError(
                    f"case id={entry['case_id']} has invalid required shards"
                )
            normalized_item = {
                "variant": str(item["variant"]),
                "cache_key": _require_sha256(item["cache_key"], "catalog cache_key"),
                "required_shards": required_shards,
            }
            if catalog_format_version == CATALOG_FORMAT_VERSION:
                shard_sha256 = item.get("shard_sha256")
                if not isinstance(shard_sha256, dict) or set(shard_sha256) != set(
                    required_shards
                ):
                    raise ReferenceCacheError(
                        f"case id={entry['case_id']} has invalid pinned shard digests"
                    )
                normalized_item.update(
                    {
                        "manifest_generation": _require_sha256(
                            item.get("manifest_generation"),
                            "catalog manifest_generation",
                        ),
                        "shard_sha256": {
                            name: _require_sha256(
                                shard_sha256[name], f"catalog {name} shard sha256"
                            )
                            for name in required_shards
                        },
                    }
                )
            normalized.append(normalized_item)
        names = [item["variant"] for item in normalized]
        if not normalized or len(names) != len(set(names)):
            raise ReferenceCacheError(
                f"case id={entry['case_id']} has invalid cache variants"
            )
        for item in normalized:
            required = item["required_shards"]
            if not required or len(required) != len(set(required)):
                raise ReferenceCacheError(
                    f"case id={entry['case_id']} has invalid required shards"
                )
            if set(required).difference(REFERENCE_SHARD_NAMES):
                raise ReferenceCacheError(
                    f"case id={entry['case_id']} has unknown required shards"
                )
        return normalized

    normalized_entries = [
        {
            "case_id": int(entry["case_id"]),
            "cache_entries": cache_entries(entry),
        }
        for entry in entries
    ]
    identity = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "source_sha256": file_sha256(source_path),
        "source_name": source_path.name,
        "adapter": adapter,
        "adapter_sha256": adapter_sha256,
        "variant_materializer_schema": variant_materializer_schema,
        "case_ids": case_ids,
        "case_count": len(case_ids),
        "cache_entry_count": sum(
            len(entry["cache_entries"]) for entry in normalized_entries
        ),
        "entries": normalized_entries,
    }
    if catalog_format_version == CATALOG_FORMAT_VERSION:
        identity = {
            **identity,
            "catalog_format_version": CATALOG_FORMAT_VERSION,
            "producer_torch_version": producer_version,
            **producer_executor_digests,
        }
    return {**identity, "catalog_key": _sha256_json(identity)}


def catalog_path(cache_dir: Path, catalog: dict) -> Path:
    key = _require_sha256(catalog["catalog_key"], "catalog_key")
    return Path(cache_dir) / f"catalog-{key}.json"


def write_catalog(cache_dir: Path, catalog: dict) -> Path:
    path = catalog_path(_writable_cache_root(cache_dir), catalog)
    _atomic_json_save(path, catalog)
    return path


def validate_catalog(cache_dir: Path, expected: dict) -> Path:
    path = catalog_path(cache_dir, expected)
    if not path.is_file():
        raise ReferenceCacheError(
            "cache catalog is missing for exact case ids/count: "
            f"count={expected['case_count']} ids={expected['case_ids']}"
        )
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceCacheError(f"invalid cache catalog: {exc}") from exc
    if actual != expected:
        raise ReferenceCacheError("cache catalog does not match the exact input case set")
    _validate_catalog_payload(actual, expected_pin=expected["catalog_key"])
    return path


def _cache_root(cache_dir: Path) -> Path:
    try:
        root = Path(cache_dir).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ReferenceCacheError(
            f"persistent cache directory is missing or unavailable: {exc}"
        ) from exc
    if not root.is_dir():
        raise ReferenceCacheError("persistent cache path is not a directory")
    return root


def _writable_cache_root(cache_dir: Path) -> Path:
    requested = Path(cache_dir).expanduser()
    try:
        requested.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ReferenceCacheError(f"cannot create persistent cache directory: {exc}") from exc
    return _cache_root(requested)


def _safe_entry_dir(cache_root: Path, cache_key: str) -> Path:
    key = _require_sha256(cache_key, "cache_key")
    try:
        entry_dir = (cache_root / key).resolve(strict=True)
    except OSError as exc:
        raise ReferenceCacheError(f"persistent cache entry is missing for key={key}") from exc
    if entry_dir.parent != cache_root or entry_dir.name != key or not entry_dir.is_dir():
        raise ReferenceCacheError("persistent cache entry escapes the cache directory")
    return entry_dir


def _read_json(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceCacheError(f"invalid {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReferenceCacheError(f"invalid {description}: expected a JSON object")
    return value


def _open_regular_file(directory: Path, filename: str, description: str) -> BinaryIO:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ReferenceCacheError(f"invalid {description} filename")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(directory, directory_flags)
    except OSError as exc:
        raise ReferenceCacheError(f"invalid {description} directory: {exc}") from exc
    try:
        descriptor = os.open(filename, file_flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise ReferenceCacheError(f"invalid or missing {description}: {exc}") from exc
    finally:
        os.close(directory_descriptor)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ReferenceCacheError(f"{description} is not a regular file")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def _read_json_in_dir(directory: Path, filename: str, description: str) -> dict:
    with _open_regular_file(directory, filename, description) as stream:
        before = os.fstat(stream.fileno())
        try:
            value = json.loads(stream.read().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReferenceCacheError(f"invalid {description}: {exc}") from exc
        after = os.fstat(stream.fileno())
    if not _same_file_state(before, after):
        raise ReferenceCacheError(f"{description} changed while being read")
    if not isinstance(value, dict):
        raise ReferenceCacheError(f"invalid {description}: expected a JSON object")
    return value


def _stream_sha256(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _same_file_state(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _metadata_cache_key(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        raise ReferenceCacheError("cache manifest metadata is not an object")
    actual_key = _require_sha256(metadata.get("cache_key"), "manifest cache_key")
    identity = {key: value for key, value in metadata.items() if key != "cache_key"}
    if _sha256_json(identity) != actual_key:
        raise ReferenceCacheError("cache manifest metadata key digest mismatch")
    return actual_key


def _validate_manifest_generation(manifest: dict) -> None:
    generation_identity = {
        "metadata": manifest.get("metadata"),
        "shards": manifest.get("shards"),
    }
    if manifest.get("generation") != _sha256_json(generation_identity):
        raise ReferenceCacheError("cache manifest generation digest mismatch")


def _read_catalog_manifest(
    cache_root: Path,
    cache_key: str,
    required_shards: list[str],
) -> dict:
    entry_dir = _safe_entry_dir(cache_root, cache_key)
    manifest = _read_json_in_dir(entry_dir, "manifest.json", "cache manifest")
    if _metadata_cache_key(manifest.get("metadata")) != cache_key:
        raise ReferenceCacheError("catalog cache key does not match manifest metadata")
    metadata_required = manifest["metadata"].get("required_shards")
    if metadata_required != required_shards:
        raise ReferenceCacheError("catalog required_shards do not match manifest metadata")
    _validate_manifest_generation(manifest)
    _manifest_content_pin(manifest, required_shards)
    return manifest


def _manifest_content_pin(manifest: dict, required_shards: list[str]) -> dict:
    generation = _require_sha256(
        manifest.get("generation"), "cache manifest generation"
    )
    shards = manifest.get("shards")
    if not isinstance(shards, dict) or set(shards) != set(required_shards):
        raise ReferenceCacheError("cache manifest does not declare the required shards")
    shard_sha256 = {}
    for name in required_shards:
        descriptor = shards[name]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "file",
            "sha256",
            "signature",
        }:
            raise ReferenceCacheError(f"cache shard descriptor is malformed: {name}")
        digest = _require_sha256(descriptor.get("sha256"), f"{name} shard sha256")
        if descriptor.get("file") != f"{name}-{digest}.pt":
            raise ReferenceCacheError(f"cache shard filename is invalid: {name}")
        shard_sha256[name] = digest
    return {
        "manifest_generation": generation,
        "shard_sha256": shard_sha256,
    }


def _validate_catalog_payload(catalog: dict, *, expected_pin: str) -> None:
    pin = _require_sha256(expected_pin, "pinned catalog_key")
    actual_key = _require_sha256(catalog.get("catalog_key"), "catalog_key")
    identity = {key: value for key, value in catalog.items() if key != "catalog_key"}
    if _sha256_json(identity) != actual_key or actual_key != pin:
        raise ReferenceCacheError("cache catalog key digest does not match the external pin")

    catalog_format = catalog.get("catalog_format_version", 1)
    if catalog_format not in {1, CATALOG_FORMAT_VERSION}:
        raise ReferenceCacheError(f"unsupported cache catalog format version {catalog_format}")
    if catalog.get("cache_format_version") != CACHE_FORMAT_VERSION:
        raise ReferenceCacheError("cache catalog entry format version mismatch")
    _require_sha256(catalog.get("source_sha256"), "catalog source_sha256")
    adapter_sha256 = catalog.get("adapter_sha256")
    if adapter_sha256 is not None:
        _require_sha256(adapter_sha256, "catalog adapter_sha256")
    if catalog_format == CATALOG_FORMAT_VERSION:
        _require_torch_version(catalog.get("producer_torch_version"))
        for field in (
            "producer_executor_sha256",
            "producer_golden_executor_sha256",
            "producer_benchmark_executor_sha256",
        ):
            _require_sha256(catalog.get(field), field)
    elif any(
        field in catalog
        for field in (
            "producer_torch_version",
            "producer_executor_sha256",
            "producer_golden_executor_sha256",
            "producer_benchmark_executor_sha256",
        )
    ):
        raise ReferenceCacheError("legacy cache catalog has unexpected producer metadata")

    case_ids = catalog.get("case_ids")
    entries = catalog.get("entries")
    if not isinstance(case_ids, list) or not isinstance(entries, list):
        raise ReferenceCacheError("cache catalog case ids or entries are malformed")
    if len(case_ids) != len(set(case_ids)) or any(
        not isinstance(case_id, int) for case_id in case_ids
    ):
        raise ReferenceCacheError("cache catalog contains invalid case ids")
    if catalog.get("case_count") != len(case_ids) or len(entries) != len(case_ids):
        raise ReferenceCacheError("cache catalog case count mismatch")
    if [entry.get("case_id") for entry in entries if isinstance(entry, dict)] != case_ids:
        raise ReferenceCacheError("cache catalog entry order does not match case ids")

    cache_entry_count = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"case_id", "cache_entries"}:
            raise ReferenceCacheError("cache catalog logical entry is malformed")
        variants = entry["cache_entries"]
        if not isinstance(variants, list) or not variants:
            raise ReferenceCacheError("cache catalog has an empty variant list")
        names = []
        for item in variants:
            expected_fields = {
                "variant",
                "cache_key",
                "required_shards",
            }
            if catalog_format == CATALOG_FORMAT_VERSION:
                expected_fields.update({"manifest_generation", "shard_sha256"})
            if not isinstance(item, dict) or set(item) != expected_fields:
                raise ReferenceCacheError("cache catalog variant entry is malformed")
            if not isinstance(item["variant"], str) or not item["variant"]:
                raise ReferenceCacheError("cache catalog variant name is invalid")
            names.append(item["variant"])
            _require_sha256(item["cache_key"], "catalog cache_key")
            required = item["required_shards"]
            if (
                not isinstance(required, list)
                or not required
                or any(not isinstance(name, str) for name in required)
                or len(required) != len(set(required))
                or set(required).difference(REFERENCE_SHARD_NAMES)
            ):
                raise ReferenceCacheError("cache catalog required_shards are invalid")
            if catalog_format == CATALOG_FORMAT_VERSION:
                _require_sha256(
                    item["manifest_generation"], "catalog manifest_generation"
                )
                shard_sha256 = item["shard_sha256"]
                if not isinstance(shard_sha256, dict) or set(shard_sha256) != set(
                    required
                ):
                    raise ReferenceCacheError("cache catalog shard digests are invalid")
                for name, digest in shard_sha256.items():
                    _require_sha256(digest, f"catalog {name} shard sha256")
        if len(names) != len(set(names)):
            raise ReferenceCacheError("cache catalog contains duplicate variant names")
        cache_entry_count += len(variants)
    if catalog.get("cache_entry_count") != cache_entry_count:
        raise ReferenceCacheError("cache catalog entry count mismatch")


def _resolve_pinned_catalog_path(
    cache_dir: Path,
    catalog_reference: str | Path | None,
) -> tuple[Path, Path, str]:
    if catalog_reference is None or not str(catalog_reference).strip():
        raise ReferenceCacheError(
            f"a catalog key or path must be explicitly pinned via --catalog or {PINNED_CATALOG_ENV}"
        )
    root = _cache_root(cache_dir)
    reference = str(catalog_reference).strip()
    if _SHA256_RE.fullmatch(reference):
        pin = reference
        candidate = root / f"catalog-{pin}.json"
    else:
        supplied = Path(reference).expanduser()
        candidate = supplied if supplied.is_absolute() else root / supplied
        match = _CATALOG_NAME_RE.fullmatch(candidate.name)
        if match is None:
            raise ReferenceCacheError(
                "pinned cache catalog path must use catalog-<sha256>.json"
            )
        pin = match.group(1)
    try:
        path = candidate.resolve(strict=True)
    except OSError as exc:
        raise ReferenceCacheError(f"pinned cache catalog is missing: {exc}") from exc
    if path.parent != root or not path.is_file():
        raise ReferenceCacheError("pinned cache catalog escapes the cache directory")
    if path.name != f"catalog-{pin}.json":
        raise ReferenceCacheError("pinned cache catalog filename does not match its key")
    return root, path, pin


def _signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ReferenceCacheError("persistent cache shards must contain CPU tensors only")
        return {
            "type": "tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if value is None:
        return {"type": "none"}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": {str(key): _signature(value[key]) for key in sorted(value)},
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": "tuple" if isinstance(value, tuple) else "list",
            "items": [_signature(item) for item in value],
        }
    if isinstance(value, (str, int, float, bool)):
        return {"type": type(value).__name__, "value": value}
    raise ReferenceCacheError(f"unsupported cache shard value {type(value).__name__}")


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _content_addressed_torch_save(
    directory: Path,
    shard_name: str,
    payload: dict,
) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=directory, prefix=f".{shard_name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        digest = file_sha256(temporary)
        path = directory / f"{shard_name}-{digest}.pt"
        if not path.exists() or file_sha256(path) != digest:
            os.replace(temporary, path)
            _fsync_directory(directory)
        return path, digest
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_load(source: Path | BinaryIO, display_name: str | None = None) -> dict:
    name = display_name or getattr(source, "name", "cache shard")
    name = Path(name).name if isinstance(name, (str, os.PathLike)) else "cache shard"
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise ReferenceCacheError(
            "runtime Torch cannot safely load cache shards with weights_only=True"
        ) from exc
    except Exception as exc:
        raise ReferenceCacheError(f"failed to read cache shard {name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReferenceCacheError(f"cache shard {name} does not contain an object payload")
    return payload


class CacheWriter:
    """Write shards atomically and publish the manifest only after all shards exist."""

    def __init__(self, cache_dir: Path, metadata: dict, *, overwrite: bool = False):
        self.metadata = metadata
        self.cache_key = _metadata_cache_key(metadata)
        self.cache_root = _writable_cache_root(cache_dir)
        self.entry_dir = self.cache_root / self.cache_key
        self.manifest_path = self.entry_dir / "manifest.json"
        self.lock_name = f".{self.cache_key}.lock"
        self.lock_path = self.cache_root / self.lock_name
        self.overwrite = overwrite
        self.required_shards = tuple(metadata.get("required_shards", REFERENCE_SHARD_NAMES))
        if not self.required_shards or set(self.required_shards).difference(REFERENCE_SHARD_NAMES):
            raise ReferenceCacheError("cache metadata has invalid required_shards")
        self.shards: dict[str, dict] = {}
        self._lock_descriptor: int | None = None

    def __enter__(self):
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        root_descriptor = os.open(self.cache_root, directory_flags)
        try:
            lock_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
            lock_flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                self._lock_descriptor = os.open(
                    self.lock_name,
                    lock_flags,
                    0o600,
                    dir_fd=root_descriptor,
                )
            except OSError as exc:
                raise ReferenceCacheError(
                    f"cannot safely open cache lock: {exc}"
                ) from exc
            if not stat.S_ISREG(os.fstat(self._lock_descriptor).st_mode):
                raise ReferenceCacheError("cache lock is not a regular file")
            fcntl.flock(
                self._lock_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            os.close(self._lock_descriptor)
            self._lock_descriptor = None
            os.close(root_descriptor)
            raise ReferenceCacheError(
                f"cache entry {self.metadata['cache_key']} is already being built"
            ) from exc
        except Exception:
            if self._lock_descriptor is not None:
                os.close(self._lock_descriptor)
                self._lock_descriptor = None
            os.close(root_descriptor)
            raise
        try:
            os.ftruncate(self._lock_descriptor, 0)
            os.write(
                self._lock_descriptor,
                f"pid={os.getpid()}\n".encode("ascii"),
            )
            os.fsync(self._lock_descriptor)
            try:
                os.mkdir(self.cache_key, 0o700, dir_fd=root_descriptor)
            except FileExistsError:
                pass
            try:
                entry_descriptor = os.open(
                    self.cache_key,
                    directory_flags,
                    dir_fd=root_descriptor,
                )
            except OSError as exc:
                raise ReferenceCacheError(
                    f"cache entry must be a real directory: {exc}"
                ) from exc
            try:
                if not stat.S_ISDIR(os.fstat(entry_descriptor).st_mode):
                    raise ReferenceCacheError("cache entry is not a directory")
                try:
                    manifest_stat = os.stat(
                        "manifest.json",
                        dir_fd=entry_descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    manifest_stat = None
                if manifest_stat is not None and not stat.S_ISREG(manifest_stat.st_mode):
                    raise ReferenceCacheError("cache manifest is not a regular file")
            finally:
                os.close(entry_descriptor)
            self.entry_dir = _safe_entry_dir(self.cache_root, self.cache_key)
            self.manifest_path = self.entry_dir / "manifest.json"
            if manifest_stat is not None and not self.overwrite:
                raise ReferenceCacheError(
                    f"cache entry already exists: {self.metadata['cache_key']}"
                )
        except Exception:
            self._release_lock()
            os.close(root_descriptor)
            raise
        os.close(root_descriptor)
        return self

    def write_shard(self, name: str, value: Any) -> None:
        if name not in SHARD_NAMES:
            raise ReferenceCacheError(f"unknown cache shard {name!r}")
        signature = _signature(value)
        path, digest = _content_addressed_torch_save(
            self.entry_dir,
            name,
            {
                "cache_key": self.metadata["cache_key"],
                "kind": name,
                "signature": signature,
                "value": value,
            },
        )
        self.shards[name] = {
            "file": path.name,
            "sha256": digest,
            "signature": signature,
        }

    def commit(self) -> Path:
        missing = set(self.required_shards).difference(self.shards)
        if missing:
            raise ReferenceCacheError(f"cannot commit cache; missing shards: {sorted(missing)}")
        unexpected = set(self.shards).difference(self.required_shards)
        if unexpected:
            raise ReferenceCacheError(
                f"cannot commit cache; unexpected shards: {sorted(unexpected)}"
            )
        generation_identity = {
            "metadata": self.metadata,
            "shards": self.shards,
        }
        generation = hashlib.sha256(
            json.dumps(
                generation_identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        _atomic_json_save(
            self.manifest_path,
            {
                "generation": generation,
                "metadata": self.metadata,
                "input_tensor_summary": self.shards["inputs"]["signature"],
                "reference_output_schemas": {
                    name: self.shards[name]["signature"]
                    for name in ("cpu_fp64", "cpu_same_precision")
                    if name in self.shards
                },
                "shards": self.shards,
            },
        )
        return self.manifest_path

    def _release_lock(self) -> None:
        if self._lock_descriptor is None:
            return
        descriptor = self._lock_descriptor
        self._lock_descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __exit__(self, _exc_type, _exc, _traceback):
        self._release_lock()


class CacheReader:
    """Validate cache identity before loading any PT shard."""

    def __init__(
        self,
        cache_dir: Path,
        expected_metadata: dict,
        *,
        expected_manifest_generation: str | None = None,
        expected_shard_sha256: dict[str, str] | None = None,
    ):
        self.expected_metadata = expected_metadata
        expected_key = _metadata_cache_key(expected_metadata)
        self.cache_root = _cache_root(cache_dir)
        self.entry_dir = _safe_entry_dir(self.cache_root, expected_key)
        self.manifest_path = self.entry_dir / "manifest.json"
        if not self.manifest_path.is_file():
            raise ReferenceCacheError(
                "persistent cache entry is missing for "
                f"key={expected_metadata['cache_key']} seed={expected_metadata['runtime_seed']}"
            )
        manifest = _read_json_in_dir(
            self.entry_dir, "manifest.json", "cache manifest"
        )
        if manifest.get("metadata") != expected_metadata:
            raise ReferenceCacheError(
                "cache metadata is stale or does not match the current spec, seed, schema, "
                "executor digest, and Torch version"
            )
        _validate_manifest_generation(manifest)
        self.required_shards = tuple(
            expected_metadata.get("required_shards", REFERENCE_SHARD_NAMES)
        )
        self.shards = manifest.get("shards")
        content_pin = _manifest_content_pin(manifest, list(self.required_shards))
        self.manifest_generation = content_pin["manifest_generation"]
        self.shard_sha256 = content_pin["shard_sha256"]
        if (
            expected_manifest_generation is not None
            and self.manifest_generation
            != _require_sha256(
                expected_manifest_generation, "pinned manifest_generation"
            )
        ):
            raise ReferenceCacheError(
                "cache manifest generation does not match the pinned catalog"
            )
        if expected_shard_sha256 is not None:
            if not isinstance(expected_shard_sha256, dict):
                raise ReferenceCacheError("pinned shard digests are malformed")
            pinned_shards = {
                name: _require_sha256(digest, f"pinned {name} shard sha256")
                for name, digest in expected_shard_sha256.items()
            }
            if pinned_shards != self.shard_sha256:
                raise ReferenceCacheError(
                    "cache shard digests do not match the pinned catalog"
                )
        if manifest.get("input_tensor_summary") != self.shards["inputs"].get("signature"):
            raise ReferenceCacheError("cache input tensor summary does not match its shard")
        expected_reference_schemas = {
            name: self.shards[name].get("signature")
            for name in ("cpu_fp64", "cpu_same_precision")
            if name in self.shards
        }
        if manifest.get("reference_output_schemas") != expected_reference_schemas:
            raise ReferenceCacheError("cache reference output schemas do not match their shards")
        self.validation_receipt = {
            "cache_key": expected_key,
            "manifest_generation": self.manifest_generation,
            "producer_torch_version": expected_metadata["producer_torch_version"],
            "producer_executor_sha256": expected_metadata["executor_sha256"],
            "producer_golden_executor_sha256": expected_metadata[
                "golden_executor_sha256"
            ],
            "producer_benchmark_executor_sha256": expected_metadata[
                "benchmark_executor_sha256"
            ],
            "consumer_torch_version": str(torch.__version__),
        }

    def _shard_descriptor(self, name: str) -> tuple[Path, str, str]:
        if name not in self.required_shards:
            raise ReferenceCacheError(f"unknown cache shard {name!r}")
        descriptor = self.shards[name]
        filename = descriptor["file"]
        digest = _require_sha256(descriptor.get("sha256"), f"{name} shard sha256")
        expected_filename = f"{name}-{digest}.pt"
        if not isinstance(filename, str) or filename != expected_filename:
            raise ReferenceCacheError(f"cache shard filename is invalid: {name}")
        return self.entry_dir / filename, filename, digest

    def validate_shard_file(self, name: str) -> Path:
        path, filename, digest = self._shard_descriptor(name)
        with _open_regular_file(self.entry_dir, filename, f"{name} cache shard") as stream:
            before = os.fstat(stream.fileno())
            actual_digest = _stream_sha256(stream)
            after = os.fstat(stream.fileno())
        if not _same_file_state(before, after):
            raise ReferenceCacheError(f"cache shard changed during validation: {name}")
        if actual_digest != digest:
            raise ReferenceCacheError(f"cache shard checksum mismatch: {name}")
        return path

    def load_shard(self, name: str) -> Any:
        _path, filename, digest = self._shard_descriptor(name)
        descriptor = self.shards[name]
        with _open_regular_file(self.entry_dir, filename, f"{name} cache shard") as stream:
            before = os.fstat(stream.fileno())
            actual_digest = _stream_sha256(stream)
            if actual_digest != digest:
                raise ReferenceCacheError(f"cache shard checksum mismatch: {name}")
            stream.seek(0)
            payload = _torch_load(stream, filename)
            after = os.fstat(stream.fileno())
        if not _same_file_state(before, after):
            raise ReferenceCacheError(f"cache shard changed while loading: {name}")
        if payload.get("cache_key") != self.expected_metadata["cache_key"]:
            raise ReferenceCacheError(f"cache shard belongs to another entry: {name}")
        if payload.get("kind") != name:
            raise ReferenceCacheError(f"cache shard kind mismatch: {name}")
        value = payload.get("value")
        actual_signature = _signature(value)
        if payload.get("signature") != actual_signature:
            raise ReferenceCacheError(f"cache shard embedded signature mismatch: {name}")
        if descriptor.get("signature") != actual_signature:
            raise ReferenceCacheError(f"cache manifest signature mismatch: {name}")
        return value

    def validate_all(self) -> None:
        for name in self.required_shards:
            self.load_shard(name)

    @property
    def catalog_content_pin(self) -> dict:
        return {
            "manifest_generation": self.manifest_generation,
            "shard_sha256": dict(self.shard_sha256),
        }


class PinnedCatalog:
    """Resolve cache entries only through one externally pinned catalog."""

    def __init__(self, cache_dir: Path, catalog_reference: str | Path | None):
        self.cache_root, self.path, pin = _resolve_pinned_catalog_path(
            cache_dir, catalog_reference
        )
        self.catalog = _read_json(self.path, "cache catalog")
        _validate_catalog_payload(self.catalog, expected_pin=pin)
        self.catalog_key = pin
        self.catalog_format_version = int(
            self.catalog.get("catalog_format_version", 1)
        )
        self.consumer_torch_version = str(torch.__version__)
        if self.catalog_format_version == CATALOG_FORMAT_VERSION:
            producer_identity = {
                "producer_torch_version": _require_torch_version(
                    self.catalog.get("producer_torch_version")
                ),
                "producer_executor_sha256": _require_sha256(
                    self.catalog.get("producer_executor_sha256"),
                    "producer_executor_sha256",
                ),
                "producer_golden_executor_sha256": _require_sha256(
                    self.catalog.get("producer_golden_executor_sha256"),
                    "producer_golden_executor_sha256",
                ),
                "producer_benchmark_executor_sha256": _require_sha256(
                    self.catalog.get("producer_benchmark_executor_sha256"),
                    "producer_benchmark_executor_sha256",
                ),
            }
        else:
            producer_identity = self._infer_legacy_producer_identity()
        for field, value in producer_identity.items():
            setattr(self, field, value)

    def _cache_entries(self):
        for logical_entry in self.catalog["entries"]:
            yield from logical_entry["cache_entries"]

    def _infer_legacy_producer_identity(self) -> dict:
        producer_identities = set()
        seen_keys = set()
        for entry in self._cache_entries():
            cache_key = entry["cache_key"]
            if cache_key in seen_keys:
                continue
            seen_keys.add(cache_key)
            manifest = _read_catalog_manifest(
                self.cache_root,
                cache_key,
                entry["required_shards"],
            )
            metadata = manifest["metadata"]
            producer_identities.add(
                (
                    _require_torch_version(
                        metadata.get("producer_torch_version")
                    ),
                    _require_sha256(
                        metadata.get("executor_sha256"),
                        "producer_executor_sha256",
                    ),
                    _require_sha256(
                        metadata.get("golden_executor_sha256"),
                        "producer_golden_executor_sha256",
                    ),
                    _require_sha256(
                        metadata.get("benchmark_executor_sha256"),
                        "producer_benchmark_executor_sha256",
                    ),
                )
            )
        if len(producer_identities) != 1:
            raise ReferenceCacheError(
                "legacy cache catalog must resolve to exactly one producer identity"
            )
        version, executor, golden, benchmark = next(iter(producer_identities))
        return {
            "producer_torch_version": version,
            "producer_executor_sha256": executor,
            "producer_golden_executor_sha256": golden,
            "producer_benchmark_executor_sha256": benchmark,
        }

    def validate_expected(self, expected_catalog: dict) -> None:
        if self.catalog != expected_catalog:
            raise ReferenceCacheError(
                "pinned cache catalog does not match the exact source, adapter, cases, "
                "variants, producer version, and entry keys"
            )

    def validate_source(
        self,
        source_path: Path,
        *,
        adapter: str,
        adapter_path: Path | None = None,
        variant_materializer_schema: str | None = None,
    ) -> None:
        expected_adapter_sha256 = (
            None if adapter_path is None else file_sha256(adapter_path)
        )
        expected = {
            "source_sha256": file_sha256(source_path),
            "source_name": source_path.name,
            "adapter": adapter,
            "adapter_sha256": expected_adapter_sha256,
            "variant_materializer_schema": variant_materializer_schema,
        }
        actual = {key: self.catalog.get(key) for key in expected}
        if actual != expected:
            raise ReferenceCacheError(
                "pinned cache catalog source, adapter, or materializer digest is stale"
            )

    def reader_for(
        self,
        spec: dict,
        seed: int,
        executor_path: Path,
        *,
        include_references: bool,
    ) -> CacheReader:
        metadata = build_chunk_kda_metadata(
            spec,
            seed,
            executor_path,
            producer_torch_version=self.producer_torch_version,
            producer_executor_sha256=self.producer_executor_sha256,
            producer_golden_executor_sha256=self.producer_golden_executor_sha256,
            producer_benchmark_executor_sha256=(
                self.producer_benchmark_executor_sha256
            ),
            include_references=include_references,
        )
        cache_key = metadata["cache_key"]
        matches = [
            entry for entry in self._cache_entries() if entry["cache_key"] == cache_key
        ]
        if not matches:
            raise ReferenceCacheError(
                "pinned cache catalog has no entry for the exact spec, seed, producer, and digests"
            )
        expected_shards = list(metadata["required_shards"])
        declared_shards = {tuple(entry["required_shards"]) for entry in matches}
        if declared_shards != {tuple(expected_shards)}:
            raise ReferenceCacheError(
                "pinned cache catalog has conflicting required_shards for the entry"
            )
        expected_generation = None
        expected_shard_sha256 = None
        if self.catalog_format_version == CATALOG_FORMAT_VERSION:
            content_pins = {
                (
                    entry["manifest_generation"],
                    tuple(sorted(entry["shard_sha256"].items())),
                )
                for entry in matches
            }
            if len(content_pins) != 1:
                raise ReferenceCacheError(
                    "pinned cache catalog has conflicting manifest or shard digests"
                )
            expected_generation, shard_items = next(iter(content_pins))
            expected_shard_sha256 = dict(shard_items)
        reader = CacheReader(
            self.cache_root,
            metadata,
            expected_manifest_generation=expected_generation,
            expected_shard_sha256=expected_shard_sha256,
        )
        reader.validation_receipt.update(
            {
                "catalog_key": self.catalog_key,
                "catalog_format_version": self.catalog_format_version,
                "consumer_executor_sha256": file_sha256(executor_path),
            }
        )
        return reader

    @property
    def producer_identity(self) -> dict:
        return {
            "producer_torch_version": self.producer_torch_version,
            "producer_executor_sha256": self.producer_executor_sha256,
            "producer_golden_executor_sha256": self.producer_golden_executor_sha256,
            "producer_benchmark_executor_sha256": (
                self.producer_benchmark_executor_sha256
            ),
        }

    @property
    def validation_receipt(self) -> dict:
        return {
            "catalog_key": self.catalog_key,
            "catalog_format_version": self.catalog_format_version,
            **self.producer_identity,
            "consumer_torch_version": self.consumer_torch_version,
        }
