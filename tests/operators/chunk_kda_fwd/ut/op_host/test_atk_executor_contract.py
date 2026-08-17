"""Host-only contracts for the distributed ATK chunk_kda_fwd executor."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[5]
EXECUTOR_PATH = ROOT / "test/chunk_kda_fwd/executor_chunk_kda_fwd.py"
BUILDER_PATH = ROOT / "test/chunk_kda_fwd/build_reference_cache.py"
ATK_CASE_PATH = ROOT / "test/chunk_kda_fwd/atk_chunk_kda_fwd.json"
CANONICAL_ADAPTER_PATH = ROOT / "test/chunk_kda_fwd/canonical_case_adapter.py"
CANONICAL_EXECUTION_ADAPTER_PATH = (
    ROOT / "test/chunk_kda_fwd/canonical_execution_adapter.py"
)
CANONICAL_MANIFEST_PATH = ROOT / "tests/op_cases/chunk_kda_fwd.json"


def _load_executor(monkeypatch):
    monkeypatch.syspath_prepend(str(EXECUTOR_PATH.parent))
    dataset_module = types.ModuleType("atk.configs.dataset_config")
    dataset_module.InputDataset = type("InputDataset", (), {})
    results_module = types.ModuleType("atk.configs.results_config")
    results_module.TaskResult = type("TaskResult", (), {})
    execute_module = types.ModuleType("atk.tasks.api_execute")
    execute_module.register = lambda _name: lambda cls: cls
    base_module = types.ModuleType("atk.tasks.api_execute.base_api")
    base_module.BaseApi = type("BaseApi", (), {})
    modules = {
        "atk": types.ModuleType("atk"),
        "atk.configs": types.ModuleType("atk.configs"),
        "atk.configs.dataset_config": dataset_module,
        "atk.configs.results_config": results_module,
        "atk.tasks": types.ModuleType("atk.tasks"),
        "atk.tasks.api_execute": execute_module,
        "atk.tasks.api_execute.base_api": base_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("chunk_kda_atk_executor", EXECUTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_cache_module(monkeypatch):
    monkeypatch.syspath_prepend(str(EXECUTOR_PATH.parent))
    import persistent_reference_cache

    return persistent_reference_cache


def _load_builder(monkeypatch):
    monkeypatch.syspath_prepend(str(BUILDER_PATH.parent))
    spec = importlib.util.spec_from_file_location("chunk_kda_cache_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_canonical_adapter():
    spec = importlib.util.spec_from_file_location(
        "chunk_kda_canonical_adapter_for_executor_tests", CANONICAL_ADAPTER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_spec(design_id):
    records = _load_canonical_adapter().materialize(
        ROOT / "tests/op_cases/chunk_kda_fwd.json"
    )
    return next(
        record["spec"] for record in records if record["spec"]["design_id"] == design_id
    )


def _canonical_execution_spec(design_id):
    sys.path.insert(0, str(CANONICAL_EXECUTION_ADAPTER_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "chunk_kda_canonical_execution_adapter_for_executor_tests",
            CANONICAL_EXECUTION_ADAPTER_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        records = module.materialize(CANONICAL_MANIFEST_PATH)
    finally:
        sys.path.remove(str(CANONICAL_EXECUTION_ADAPTER_PATH.parent))
    return next(
        record["spec"] for record in records if record["spec"]["design_id"] == design_id
    )


def _outputs(aqk, akk):
    values = [None] * 12
    values[3] = aqk
    values[4] = akk
    return tuple(values)


def test_reference_cache_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("KDA_ATK_REFERENCE_CACHE_ENTRIES", raising=False)
    executor = _load_executor(monkeypatch)
    assert executor._REFERENCE_CACHE_ENTRIES == 0


def test_persistent_accuracy_cache_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("KDA_ATK_PERSISTENT_CACHE_MODE", raising=False)
    executor = _load_executor(monkeypatch)
    assert executor._persistent_cache_mode() == "off"


def test_executor_readonly_cache_resolves_only_through_pinned_catalog(
    monkeypatch, tmp_path
):
    executor = _load_executor(monkeypatch)
    calls = []
    reader = object()

    class FakePinnedCatalog:
        def __init__(self, cache_dir, reference):
            calls.append(("catalog", Path(cache_dir), reference))

        def reader_for(self, spec, seed, executor_path, *, include_references):
            calls.append(
                (
                    "reader",
                    spec,
                    seed,
                    Path(executor_path),
                    include_references,
                )
            )
            return reader

    monkeypatch.setattr(executor, "PinnedCatalog", FakePinnedCatalog)
    monkeypatch.setattr(executor, "default_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        executor, "default_catalog_reference", lambda: "a" * 64
    )
    executor._PERSISTENT_CATALOGS.clear()
    spec = {"seed": 7, "tags": "accuracy"}

    assert executor._persistent_cache_reader(spec, 7) is reader
    assert executor._persistent_cache_reader(spec, 7) is reader
    assert calls[0] == ("catalog", tmp_path, "a" * 64)
    assert calls[1:] == [
        ("reader", spec, 7, executor._EXECUTOR_PATH, True),
        ("reader", spec, 7, executor._EXECUTOR_PATH, True),
    ]


def test_reference_cache_evicts_old_result_before_allocating_new_result(monkeypatch):
    executor = _load_executor(monkeypatch)
    executor._REFERENCE_CACHE_ENTRIES = 1
    executor._REFERENCE_CACHE["old"] = (torch.ones(1),)
    executor._reference_cache_key = lambda *_args: "new"
    inputs = types.SimpleNamespace(q=torch.empty(0), seed=0)
    spec = {
        "disable_recompute": True,
        "output_final_state": True,
        "return_intermediate_states": True,
        "use_gate_in_kernel": True,
    }

    def runner(_inputs, _spec):
        assert not executor._REFERENCE_CACHE
        return (torch.ones(1),) + (None,) * 11

    executor._cached_full_reference(inputs, spec, "test", runner)
    assert list(executor._REFERENCE_CACHE) == ["new"]


def test_persistent_cache_identity_binds_all_required_provenance(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("first\n", encoding="utf-8")
    spec = {"T": 64, "seed": 17, "route": "ascendc", "case_key": "case-a"}
    base = cache.build_metadata(
        spec,
        17,
        executor_path,
        producer_torch_version="2.test",
        reference_schema="schema-a",
    )

    variants = [
        ({**spec, "T": 128}, 17, "2.test", "schema-a", "first\n"),
        (spec, 18, "2.test", "schema-a", "first\n"),
        (spec, 17, "2.other", "schema-a", "first\n"),
        (spec, 17, "2.test", "schema-b", "first\n"),
        (spec, 17, "2.test", "schema-a", "second\n"),
    ]
    for variant_spec, seed, torch_version, schema, executor_text in variants:
        executor_path.write_text(executor_text, encoding="utf-8")
        actual = cache.build_metadata(
            variant_spec,
            seed,
            executor_path,
            producer_torch_version=torch_version,
            reference_schema=schema,
        )
        assert actual["cache_key"] != base["cache_key"]

    executor_path.write_text("first\n", encoding="utf-8")
    cosmetic = cache.build_metadata(
        {**spec, "route": "direct_launch", "case_key": "case-b"},
        17,
        executor_path,
        producer_torch_version="2.test",
        reference_schema="schema-a",
    )
    assert cosmetic["cache_key"] == base["cache_key"]

    chunk_metadata = cache.build_chunk_kda_metadata(
        spec,
        17,
        executor_path,
        producer_torch_version="2.test",
    )
    assert chunk_metadata["deterministic_input_sha256"]
    assert chunk_metadata["golden_executor_sha256"]
    assert chunk_metadata["benchmark_executor_sha256"]
    assert (
        chunk_metadata["golden_executor_sha256"]
        != chunk_metadata["benchmark_executor_sha256"]
    )
    assert chunk_metadata["output_schema"]["names"] == list(cache.OUTPUT_NAMES)


def test_canonical_cache_identity_binds_variant_materializer_source(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    spec = {"T": 64, "seed": 17, "tags": "accuracy,canonical_300"}
    original_sha256 = cache.file_sha256

    def source_digest(digest):
        return lambda path: (
            digest
            if Path(path).name == "canonical_case_adapter.py"
            else original_sha256(Path(path))
        )

    monkeypatch.setattr(cache, "file_sha256", source_digest("1" * 64))
    first = cache.build_chunk_kda_metadata(
        spec, 17, executor_path, producer_torch_version="2.test"
    )
    monkeypatch.setattr(cache, "file_sha256", source_digest("2" * 64))
    second = cache.build_chunk_kda_metadata(
        spec, 17, executor_path, producer_torch_version="2.test"
    )

    assert first["producer_source_digests"] == {
        "canonical_case_adapter.py": "1" * 64
    }
    assert first["variant_materializer_schema"] == cache.VARIANT_MATERIALIZER_SCHEMA
    assert first["cache_key"] != second["cache_key"]


def _write_test_cache(cache, cache_dir, metadata):
    outputs = (torch.arange(3, dtype=torch.float64),) + (None,) * 11
    with cache.CacheWriter(cache_dir, metadata) as writer:
        writer.write_shard("inputs", {"seed": 7, "q": torch.ones(2)})
        writer.write_shard("cpu_fp64", outputs)
        writer.write_shard("cpu_same_precision", outputs)
        return writer.commit()


def _write_inputs_only_cache(cache, cache_dir, metadata):
    with cache.CacheWriter(cache_dir, metadata) as writer:
        writer.write_shard("inputs", {"seed": 7, "q": torch.ones(2)})
        return writer.commit()


def _catalog_cache_entry(
    cache_key,
    required_shards=("inputs", "cpu_fp64", "cpu_same_precision"),
):
    return {
        "cache_key": cache_key,
        "required_shards": list(required_shards),
        "manifest_generation": "b" * 64,
        "shard_sha256": {name: "c" * 64 for name in required_shards},
    }


def _catalog_producer_identity(metadata=None):
    if metadata is not None:
        return {
            "producer_executor_sha256": metadata["executor_sha256"],
            "producer_golden_executor_sha256": metadata[
                "golden_executor_sha256"
            ],
            "producer_benchmark_executor_sha256": metadata[
                "benchmark_executor_sha256"
            ],
        }
    return {
        "producer_executor_sha256": "d" * 64,
        "producer_golden_executor_sha256": "e" * 64,
        "producer_benchmark_executor_sha256": "f" * 64,
    }


def _write_test_catalog(
    cache,
    cache_dir,
    source,
    metadata,
    *,
    producer_torch_version,
    catalog_format_version=2,
    case_id=1,
):
    cache_entry = {
        "cache_key": metadata["cache_key"],
        "required_shards": metadata["required_shards"],
    }
    if catalog_format_version == cache.CATALOG_FORMAT_VERSION:
        cache_entry.update(
            cache.CacheReader(cache_dir, metadata).catalog_content_pin
        )
    catalog = cache.build_catalog(
        source,
        "test:v1",
        [
            {
                "case_id": case_id,
                **cache_entry,
            }
        ],
        producer_torch_version=producer_torch_version,
        **_catalog_producer_identity(metadata),
        catalog_format_version=catalog_format_version,
    )
    return catalog, cache.write_catalog(cache_dir, catalog)


@pytest.mark.parametrize("include_references", [False, True])
def test_pinned_catalog_decouples_producer_and_consumer_torch_versions(
    monkeypatch,
    tmp_path,
    include_references,
):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    source = tmp_path / "cases.json"
    source.write_text("[]\n", encoding="utf-8")
    spec = {
        "T": 64,
        "seed": 7,
        "tags": "accuracy" if include_references else "stress",
    }
    producer_version = "2.7.1+cpu"
    metadata = cache.build_chunk_kda_metadata(
        spec,
        7,
        executor_path,
        producer_torch_version=producer_version,
        include_references=include_references,
    )
    manifest_path = (
        _write_test_cache(cache, tmp_path / "cache", metadata)
        if include_references
        else _write_inputs_only_cache(cache, tmp_path / "cache", metadata)
    )
    catalog, _ = _write_test_catalog(
        cache,
        tmp_path / "cache",
        source,
        metadata,
        producer_torch_version=producer_version,
    )
    manifest_before = manifest_path.read_bytes()

    monkeypatch.setattr(cache.torch, "__version__", "2.12.0+cpu")
    executor_path.write_text("consumer executor changed\n", encoding="utf-8")
    consumer_executor_sha256 = cache.file_sha256(executor_path)
    pinned = cache.PinnedCatalog(tmp_path / "cache", catalog["catalog_key"])
    reader = pinned.reader_for(
        spec,
        7,
        executor_path,
        include_references=include_references,
    )
    reader.validate_all()

    assert reader.expected_metadata == metadata
    assert reader.validation_receipt["producer_torch_version"] == producer_version
    assert reader.validation_receipt["consumer_torch_version"] == "2.12.0+cpu"
    assert reader.validation_receipt["producer_executor_sha256"] == metadata[
        "executor_sha256"
    ]
    assert (
        reader.validation_receipt["consumer_executor_sha256"]
        == consumer_executor_sha256
    )
    assert consumer_executor_sha256 != metadata["executor_sha256"]
    assert manifest_path.read_bytes() == manifest_before
    consumer_build = cache.build_chunk_kda_metadata(
        spec,
        7,
        executor_path,
        include_references=include_references,
    )
    assert consumer_build["cache_key"] != metadata["cache_key"]


def test_pinned_catalog_rejects_producer_conflict(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    source = tmp_path / "cases.json"
    source.write_text("[]\n", encoding="utf-8")
    spec = {"T": 64, "seed": 7, "tags": "stress"}
    metadata = cache.build_chunk_kda_metadata(
        spec,
        7,
        executor_path,
        producer_torch_version="2.8.0+cpu",
        include_references=False,
    )
    _write_inputs_only_cache(cache, tmp_path / "cache", metadata)
    catalog, _ = _write_test_catalog(
        cache,
        tmp_path / "cache",
        source,
        metadata,
        producer_torch_version="2.7.1+cpu",
    )

    pinned = cache.PinnedCatalog(tmp_path / "cache", catalog["catalog_key"])
    with pytest.raises(cache.ReferenceCacheError, match="no entry"):
        pinned.reader_for(spec, 7, executor_path, include_references=False)


def test_pinned_catalog_rejects_producer_executor_digest_conflict(
    monkeypatch,
    tmp_path,
):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    source = tmp_path / "cases.json"
    source.write_text("[]\n", encoding="utf-8")
    spec = {"T": 64, "seed": 7, "tags": "stress"}
    metadata = cache.build_chunk_kda_metadata(
        spec,
        7,
        executor_path,
        producer_torch_version="2.7.1+cpu",
        include_references=False,
    )
    _write_inputs_only_cache(cache, tmp_path / "cache", metadata)
    content_pin = cache.CacheReader(
        tmp_path / "cache", metadata
    ).catalog_content_pin
    catalog = cache.build_catalog(
        source,
        "test:v1",
        [
            {
                "case_id": 1,
                "cache_key": metadata["cache_key"],
                "required_shards": metadata["required_shards"],
                **content_pin,
            }
        ],
        producer_torch_version="2.7.1+cpu",
        producer_executor_sha256="a" * 64,
        producer_golden_executor_sha256=metadata["golden_executor_sha256"],
        producer_benchmark_executor_sha256=metadata[
            "benchmark_executor_sha256"
        ],
    )
    cache.write_catalog(tmp_path / "cache", catalog)

    pinned = cache.PinnedCatalog(tmp_path / "cache", catalog["catalog_key"])
    with pytest.raises(cache.ReferenceCacheError, match="no entry"):
        pinned.reader_for(spec, 7, executor_path, include_references=False)


def test_legacy_catalog_requires_explicit_pin_and_one_producer(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    source = tmp_path / "cases.json"
    source.write_text("[]\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    entries = []
    for case_id, producer_version in ((1, "2.7.1+cpu"), (2, "2.8.0+cpu")):
        spec = {"T": 64, "seed": case_id, "tags": "stress"}
        metadata = cache.build_chunk_kda_metadata(
            spec,
            case_id,
            executor_path,
            producer_torch_version=producer_version,
            include_references=False,
        )
        with cache.CacheWriter(cache_dir, metadata) as writer:
            writer.write_shard("inputs", {"seed": case_id})
            writer.commit()
        entries.append(
            {
                "case_id": case_id,
                "cache_key": metadata["cache_key"],
                "required_shards": metadata["required_shards"],
            }
        )
    catalog = cache.build_catalog(
        source,
        "test:v1",
        entries,
        producer_torch_version="ignored-for-v1",
        catalog_format_version=1,
    )
    cache.write_catalog(cache_dir, catalog)

    with pytest.raises(cache.ReferenceCacheError, match="explicitly pinned"):
        cache.PinnedCatalog(cache_dir, None)
    with pytest.raises(cache.ReferenceCacheError, match="exactly one producer"):
        cache.PinnedCatalog(cache_dir, catalog["catalog_key"])


def test_legacy_catalog_rejects_conflicting_producer_executor_digests(
    monkeypatch,
    tmp_path,
):
    cache = _load_cache_module(monkeypatch)
    source = tmp_path / "cases.json"
    source.write_text("[]\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    entries = []
    for case_id, content in ((1, "first executor\n"), (2, "second executor\n")):
        executor_path = tmp_path / f"executor-{case_id}.py"
        executor_path.write_text(content, encoding="utf-8")
        spec = {"T": 64, "seed": case_id, "tags": "stress"}
        metadata = cache.build_chunk_kda_metadata(
            spec,
            case_id,
            executor_path,
            producer_torch_version="2.7.1+cpu",
            include_references=False,
        )
        _write_inputs_only_cache(cache, cache_dir, metadata)
        entries.append(
            {
                "case_id": case_id,
                "cache_key": metadata["cache_key"],
                "required_shards": metadata["required_shards"],
            }
        )
    catalog = cache.build_catalog(
        source,
        "test:v1",
        entries,
        catalog_format_version=1,
    )
    cache.write_catalog(cache_dir, catalog)

    with pytest.raises(cache.ReferenceCacheError, match="exactly one producer"):
        cache.PinnedCatalog(cache_dir, catalog["catalog_key"])


def test_legacy_catalog_uses_manifest_producer_only_when_explicitly_pinned(
    monkeypatch,
    tmp_path,
):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    source = tmp_path / "cases.json"
    source.write_text("[]\n", encoding="utf-8")
    spec = {"T": 64, "seed": 7, "tags": "stress"}
    metadata = cache.build_chunk_kda_metadata(
        spec,
        7,
        executor_path,
        producer_torch_version="2.7.1+cpu",
        include_references=False,
    )
    _write_inputs_only_cache(cache, tmp_path / "cache", metadata)
    catalog, _ = _write_test_catalog(
        cache,
        tmp_path / "cache",
        source,
        metadata,
        producer_torch_version="ignored-for-v1",
        catalog_format_version=1,
    )
    monkeypatch.setattr(cache.torch, "__version__", "2.12.0+cpu")

    pinned = cache.PinnedCatalog(tmp_path / "cache", catalog["catalog_key"])
    reader = pinned.reader_for(spec, 7, executor_path, include_references=False)
    reader.validate_all()
    assert pinned.producer_torch_version == "2.7.1+cpu"
    assert pinned.consumer_torch_version == "2.12.0+cpu"

    upgraded = cache.build_catalog(
        source,
        "test:v1",
        [
            {
                "case_id": 1,
                "cache_key": metadata["cache_key"],
                "required_shards": metadata["required_shards"],
                **reader.catalog_content_pin,
            }
        ],
        **pinned.producer_identity,
    )
    assert upgraded["catalog_format_version"] == cache.CATALOG_FORMAT_VERSION
    assert upgraded["producer_torch_version"] == "2.7.1+cpu"
    assert upgraded["producer_executor_sha256"] == metadata["executor_sha256"]
    assert upgraded["catalog_key"] != catalog["catalog_key"]


def test_v2_catalog_tamper_is_rejected_by_external_pin(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    source = tmp_path / "cases.json"
    source.write_text("[]\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    catalog = cache.build_catalog(
        source,
        "test:v1",
        [{"case_id": 1, **_catalog_cache_entry("a" * 64)}],
        producer_torch_version="2.7.1+cpu",
        **_catalog_producer_identity(),
    )
    path = cache.write_catalog(cache_dir, catalog)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["producer_executor_sha256"] = "0" * 64
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(cache.ReferenceCacheError, match="external pin"):
        cache.PinnedCatalog(cache_dir, catalog["catalog_key"])


def test_builder_validate_rejects_missing_catalog_pin(monkeypatch, tmp_path):
    builder = _load_builder(monkeypatch)
    case_json = tmp_path / "cases.json"
    case_json.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "inputs": [
                        {
                            "name": "case_spec",
                            "range_values": json.dumps(
                                {"T": 64, "seed": 7, "tags": "stress"}
                            ),
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.delenv("KDA_ATK_PERSISTENT_CACHE_CATALOG", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(BUILDER_PATH),
            "validate",
            "--case-json",
            str(case_json),
            "--cache-dir",
            str(cache_dir),
        ],
    )

    with pytest.raises(builder.ReferenceCacheError, match="explicitly pinned"):
        builder.main()


def test_catalog_anchored_build_never_generates_a_missing_entry(
    monkeypatch,
    tmp_path,
):
    builder = _load_builder(monkeypatch)
    spec = {"T": 64, "seed": 7, "tags": "stress"}
    producer_identity = builder._producer_identity_from_metadata(
        builder._producer_metadata(spec)
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    imported = []

    def unexpected_import(name):
        imported.append(name)
        raise AssertionError("generator import must not be reached")

    monkeypatch.setattr(builder.importlib, "import_module", unexpected_import)
    with pytest.raises(
        builder.ReferenceCacheError,
        match="refuses to create a missing producer entry",
    ):
        builder._build(
            1,
            "default",
            spec,
            cache_dir,
            False,
            producer_identity=producer_identity,
            catalog_anchored=True,
        )
    assert imported == []
    assert list(cache_dir.iterdir()) == []


def test_safe_torch_load_never_falls_back_to_unrestricted_pickle(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    shard = tmp_path / "input.pt"
    shard.write_bytes(b"not-used")
    calls = []

    def unsupported(*args, **kwargs):
        calls.append((args, kwargs))
        raise TypeError("weights_only unsupported")

    monkeypatch.setattr(cache.torch, "load", unsupported)
    with pytest.raises(cache.ReferenceCacheError, match="weights_only=True"):
        cache._torch_load(shard)
    assert len(calls) == 1
    assert calls[0][1]["weights_only"] is True


def test_cache_reader_rejects_shard_path_traversal(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    metadata = cache.build_chunk_kda_metadata(
        {"T": 64, "seed": 7, "tags": "stress"},
        7,
        executor_path,
        producer_torch_version="2.7.1+cpu",
        include_references=False,
    )
    manifest_path = _write_inputs_only_cache(cache, tmp_path / "cache", metadata)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"]["inputs"]["file"] = "../outside.pt"
    manifest["generation"] = cache._sha256_json(
        {"metadata": manifest["metadata"], "shards": manifest["shards"]}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(cache.ReferenceCacheError, match="filename is invalid"):
        cache.CacheReader(tmp_path / "cache", metadata)


def test_pinned_catalog_path_cannot_escape_cache_root(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    source = tmp_path / "cases.json"
    source.write_text("[]\n", encoding="utf-8")
    catalog = cache.build_catalog(
        source,
        "test:v1",
        [{"case_id": 1, **_catalog_cache_entry("a" * 64)}],
        producer_torch_version="2.7.1+cpu",
        **_catalog_producer_identity(),
    )
    outside = tmp_path / "outside"
    outside_path = cache.write_catalog(outside, catalog)
    with pytest.raises(cache.ReferenceCacheError, match="escapes"):
        cache.PinnedCatalog(cache_dir, outside_path)


def test_v2_catalog_pins_manifest_generation_and_shard_content(
    monkeypatch, tmp_path
):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    source = tmp_path / "cases.json"
    source.write_text("[]\n", encoding="utf-8")
    spec = {"T": 64, "seed": 7, "tags": "stress"}
    metadata = cache.build_chunk_kda_metadata(
        spec,
        7,
        executor_path,
        producer_torch_version="2.7.1+cpu",
        include_references=False,
    )
    _write_inputs_only_cache(cache, tmp_path / "cache", metadata)
    catalog, _ = _write_test_catalog(
        cache,
        tmp_path / "cache",
        source,
        metadata,
        producer_torch_version="2.7.1+cpu",
    )
    pinned = cache.PinnedCatalog(tmp_path / "cache", catalog["catalog_key"])
    pinned.reader_for(
        spec, 7, executor_path, include_references=False
    ).validate_all()

    with cache.CacheWriter(tmp_path / "cache", metadata, overwrite=True) as writer:
        writer.write_shard("inputs", {"seed": 7, "q": torch.zeros(2)})
        writer.commit()

    with pytest.raises(cache.ReferenceCacheError, match="pinned catalog"):
        pinned.reader_for(spec, 7, executor_path, include_references=False)


def test_cache_reader_rejects_manifest_symlink(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    metadata = cache.build_chunk_kda_metadata(
        {"T": 64, "seed": 7, "tags": "stress"},
        7,
        executor_path,
        producer_torch_version="2.7.1+cpu",
        include_references=False,
    )
    manifest_path = _write_inputs_only_cache(
        cache, tmp_path / "cache", metadata
    )
    outside_manifest = tmp_path / "outside-manifest.json"
    manifest_path.replace(outside_manifest)
    manifest_path.symlink_to(outside_manifest)

    with pytest.raises(cache.ReferenceCacheError, match="cache manifest"):
        cache.CacheReader(tmp_path / "cache", metadata)


def test_cache_writer_rejects_entry_and_lock_symlinks(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    metadata = cache.build_chunk_kda_metadata(
        {"T": 64, "seed": 7, "tags": "stress"},
        7,
        executor_path,
        producer_torch_version="2.7.1+cpu",
        include_references=False,
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    outside_entry = tmp_path / "outside-entry"
    outside_entry.mkdir()
    (cache_dir / metadata["cache_key"]).symlink_to(
        outside_entry, target_is_directory=True
    )
    with pytest.raises(cache.ReferenceCacheError, match="real directory"):
        with cache.CacheWriter(cache_dir, metadata):
            pass
    assert not list(outside_entry.iterdir())

    (cache_dir / metadata["cache_key"]).unlink()
    outside_lock = tmp_path / "outside-lock"
    outside_lock.write_text("do-not-truncate", encoding="utf-8")
    (cache_dir / f".{metadata['cache_key']}.lock").unlink()
    (cache_dir / f".{metadata['cache_key']}.lock").symlink_to(outside_lock)
    with pytest.raises(cache.ReferenceCacheError, match="cache lock"):
        with cache.CacheWriter(cache_dir, metadata):
            pass
    assert outside_lock.read_text(encoding="utf-8") == "do-not-truncate"


def test_cache_reader_hashes_and_loads_the_same_open_file(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    metadata = cache.build_chunk_kda_metadata(
        {"T": 64, "seed": 7, "tags": "stress"},
        7,
        executor_path,
        producer_torch_version="2.7.1+cpu",
        include_references=False,
    )
    manifest_path = _write_inputs_only_cache(
        cache, tmp_path / "cache", metadata
    )
    reader = cache.CacheReader(tmp_path / "cache", metadata)
    shard_path = manifest_path.parent / reader.shards["inputs"]["file"]
    replacement = tmp_path / "replacement.pt"
    replacement_value = {"seed": 7, "q": torch.zeros(2)}
    torch.save(
        {
            "cache_key": metadata["cache_key"],
            "kind": "inputs",
            "signature": cache._signature(replacement_value),
            "value": replacement_value,
        },
        replacement,
    )
    original_load = cache._torch_load
    swapped = False

    def replace_after_hash(stream, display_name=None):
        nonlocal swapped
        if not swapped:
            replacement.replace(shard_path)
            swapped = True
        return original_load(stream, display_name)

    monkeypatch.setattr(cache, "_torch_load", replace_after_hash)
    with pytest.raises(cache.ReferenceCacheError, match="changed while loading"):
        reader.load_shard("inputs")
    with pytest.raises(cache.ReferenceCacheError, match="checksum"):
        reader.load_shard("inputs")


def test_persistent_cache_atomic_round_trip_and_validation(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    metadata = cache.build_metadata(
        {"T": 64, "seed": 7},
        7,
        executor_path,
        producer_torch_version="2.test",
    )
    manifest = _write_test_cache(cache, tmp_path / "cache", metadata)

    reader = cache.CacheReader(tmp_path / "cache", metadata)
    reader.validate_all()
    assert torch.equal(reader.load_shard("inputs")["q"], torch.ones(2))
    assert torch.equal(reader.load_shard("cpu_fp64")[0], torch.arange(3, dtype=torch.float64))
    assert not list(manifest.parent.glob("*.tmp"))
    lock_files = list((tmp_path / "cache").glob("*.lock"))
    assert len(lock_files) == 1
    assert lock_files[0].read_text(encoding="ascii").startswith("pid=")
    with pytest.raises(cache.ReferenceCacheError, match="already exists"):
        with cache.CacheWriter(tmp_path / "cache", metadata):
            pass


def test_force_build_publishes_manifest_last_and_keeps_old_reader_valid(
    monkeypatch, tmp_path
):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    metadata = cache.build_metadata(
        {"T": 64, "seed": 7},
        7,
        executor_path,
        producer_torch_version="2.test",
    )
    cache_dir = tmp_path / "cache"
    _write_test_cache(cache, cache_dir, metadata)
    old_reader = cache.CacheReader(cache_dir, metadata)
    old_files = {
        name: descriptor["file"] for name, descriptor in old_reader.shards.items()
    }

    replacement = (torch.full((3,), 9.0, dtype=torch.float64),) + (None,) * 11
    with cache.CacheWriter(cache_dir, metadata, overwrite=True) as writer:
        writer.write_shard("inputs", {"seed": 7, "q": torch.zeros(2)})
        writer.write_shard("cpu_fp64", replacement)
        writer.write_shard("cpu_same_precision", replacement)

        during_reader = cache.CacheReader(cache_dir, metadata)
        assert torch.equal(
            during_reader.load_shard("cpu_fp64")[0],
            torch.arange(3, dtype=torch.float64),
        )
        writer.commit()

    new_reader = cache.CacheReader(cache_dir, metadata)
    assert torch.equal(old_reader.load_shard("cpu_fp64")[0], torch.arange(3, dtype=torch.float64))
    assert torch.equal(new_reader.load_shard("cpu_fp64")[0], replacement[0])
    assert old_files["cpu_fp64"] != new_reader.shards["cpu_fp64"]["file"]
    assert all(
        descriptor["file"] == f"{name}-{descriptor['sha256']}.pt"
        for name, descriptor in new_reader.shards.items()
    )


def test_cache_lock_rejects_active_writer_and_recovers_stale_lock_file(
    monkeypatch, tmp_path
):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    metadata = cache.build_metadata(
        {"T": 64, "seed": 7},
        7,
        executor_path,
        producer_torch_version="2.test",
    )
    cache_dir = tmp_path / "cache"
    lock_path = cache_dir / f".{metadata['cache_key']}.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("pid=999999999\n", encoding="ascii")

    with cache.CacheWriter(cache_dir, metadata) as first:
        assert lock_path.read_text(encoding="ascii").startswith("pid=")
        with pytest.raises(cache.ReferenceCacheError, match="already being built"):
            with cache.CacheWriter(cache_dir, metadata):
                pass
        first.write_shard("inputs", {"seed": 7, "q": torch.ones(2)})
        first.write_shard("cpu_fp64", (torch.ones(1),) + (None,) * 11)
        first.write_shard("cpu_same_precision", (torch.ones(1),) + (None,) * 11)
        first.commit()

    cache.CacheReader(cache_dir, metadata).validate_all()


def test_persistent_cache_rejects_missing_stale_and_corrupt_entries(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    executor_path = tmp_path / "executor.py"
    executor_path.write_text("executor\n", encoding="utf-8")
    metadata = cache.build_metadata(
        {"T": 64, "seed": 7},
        7,
        executor_path,
        producer_torch_version="2.test",
    )
    cache_dir = tmp_path / "cache"
    with pytest.raises(cache.ReferenceCacheError, match="missing"):
        cache.CacheReader(cache_dir, metadata)

    manifest_path = _write_test_cache(cache, cache_dir, metadata)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["producer_torch_version"] = "stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(cache.ReferenceCacheError, match="stale"):
        cache.CacheReader(cache_dir, metadata)

    manifest_path.unlink()
    with cache.CacheWriter(cache_dir, metadata, overwrite=True) as writer:
        writer.write_shard("inputs", {"seed": 7, "q": torch.ones(2)})
        writer.write_shard("cpu_fp64", (torch.ones(1),) + (None,) * 11)
        writer.write_shard("cpu_same_precision", (torch.ones(1),) + (None,) * 11)
        manifest_path = writer.commit()
    reader = cache.CacheReader(cache_dir, metadata)
    shard = manifest_path.parent / reader.shards["cpu_fp64"]["file"]
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(cache.ReferenceCacheError, match="checksum"):
        reader.load_shard("cpu_fp64")


def test_cache_catalog_records_exact_case_ids_and_count(monkeypatch, tmp_path):
    cache = _load_cache_module(monkeypatch)
    source = tmp_path / "cases.json"
    source.write_text("[]\n", encoding="utf-8")
    catalog = cache.build_catalog(
        source,
        "atk-json:v1",
        [
            {"case_id": 250, **_catalog_cache_entry("a" * 64)},
            {"case_id": 297, **_catalog_cache_entry("b" * 64)},
        ],
        producer_torch_version="2.test",
        **_catalog_producer_identity(),
    )
    cache.write_catalog(tmp_path / "cache", catalog)
    cache.validate_catalog(tmp_path / "cache", catalog)
    pinned = cache.PinnedCatalog(tmp_path / "cache", catalog["catalog_key"])
    pinned.validate_expected(catalog)
    assert catalog["case_ids"] == [250, 297]
    assert catalog["case_count"] == 2
    assert catalog["cache_entry_count"] == 2
    assert catalog["catalog_format_version"] == cache.CATALOG_FORMAT_VERSION
    assert catalog["producer_torch_version"] == "2.test"
    assert catalog["producer_executor_sha256"] == "d" * 64
    assert catalog["producer_golden_executor_sha256"] == "e" * 64
    assert catalog["producer_benchmark_executor_sha256"] == "f" * 64
    assert catalog["entries"][0]["cache_entries"] == [
        {
            "variant": "default",
            **_catalog_cache_entry("a" * 64),
        }
    ]

    changed = cache.build_catalog(
        source,
        "atk-json:v1",
        [{"case_id": 250, **_catalog_cache_entry("a" * 64)}],
        producer_torch_version="2.test",
        **_catalog_producer_identity(),
    )
    with pytest.raises(cache.ReferenceCacheError, match="exact case ids/count"):
        cache.validate_catalog(tmp_path / "cache", changed)
    with pytest.raises(cache.ReferenceCacheError, match="exact source"):
        pinned.validate_expected(changed)

    with_adapter_v1 = cache.build_catalog(
        source,
        "adapter:materialize",
        [{"case_id": 250, **_catalog_cache_entry("a" * 64)}],
        adapter_sha256="1" * 64,
        variant_materializer_schema="variants.v1",
        producer_torch_version="2.test",
        **_catalog_producer_identity(),
    )
    with_adapter_v2 = cache.build_catalog(
        source,
        "adapter:materialize",
        [{"case_id": 250, **_catalog_cache_entry("a" * 64)}],
        adapter_sha256="2" * 64,
        variant_materializer_schema="variants.v1",
        producer_torch_version="2.test",
        **_catalog_producer_identity(),
    )
    assert with_adapter_v1["catalog_key"] != with_adapter_v2["catalog_key"]


def test_current_pr297_cache_source_is_exactly_the_48_case_subset(monkeypatch):
    builder = _load_builder(monkeypatch)
    cases = builder._load_cases(ATK_CASE_PATH, set(), None)
    assert len(cases) == 48
    assert [case_id for case_id, _ in cases] == list(range(250, 298))


def test_legacy_stress_uses_only_validated_cached_inputs():
    stress = (
        ROOT / "test/chunk_kda_fwd/stress_legacy_cached_cases.py"
    ).read_text(encoding="utf-8")
    assert "PinnedCatalog" in stress
    assert "catalog.reader_for(" in stress
    assert "for shard_name in reader.required_shards" in stress
    assert "reader.validate_shard_file(shard_name)" in stress
    assert 'reader.load_shard("inputs")' in stress
    assert "_prepared_inputs_from_cpu" in stress
    assert "_select_cached_input_payload" in stress
    assert "_prepare_inputs" not in stress
    assert "compare_outputs_bitwise" in stress
    assert "torch.npu.synchronize()" in stress
    assert "torch.isfinite(output).all().item()" in stress
    assert "default=100" in stress


def test_future_canonical_adapter_requires_explicit_id_and_spec_records(
    monkeypatch, tmp_path
):
    builder = _load_builder(monkeypatch)
    adapter_module = types.ModuleType("test_chunk_kda_adapter")
    adapter_module.materialize = lambda _path: [
        {"id": 1001, "spec": {"seed": 7, "tags": "accuracy,varlen"}}
    ]
    monkeypatch.setitem(sys.modules, adapter_module.__name__, adapter_module)
    source = tmp_path / "canonical.json"
    source.write_text("{}\n", encoding="utf-8")

    cases = builder._load_cases(
        source, set(), "test_chunk_kda_adapter:materialize"
    )
    assert cases == [(1001, {"seed": 7, "tags": "accuracy,varlen"})]


def test_builder_loads_the_exact_176_canonical_accuracy_specs(monkeypatch):
    builder = _load_builder(monkeypatch)
    monkeypatch.syspath_prepend(str(CANONICAL_ADAPTER_PATH.parent))

    cases = builder._load_cases(
        CANONICAL_MANIFEST_PATH,
        set(),
        "canonical_case_adapter:materialize",
    )

    assert len(cases) == 176
    assert [case_id for case_id, _ in cases] == [
        *range(1001, 1097),
        *range(2001, 2081),
    ]
    assert cases[0][1]["design_id"] == "KDA-FWD-P001"
    assert cases[-1][1]["design_id"] == "KDA-FWD-G080"

    catalog = builder.build_catalog(
        CANONICAL_MANIFEST_PATH,
        "canonical_case_adapter:materialize",
        [
            {
                "case_id": case_id,
                **_catalog_cache_entry(f"{case_id:064x}"),
            }
            for case_id, _ in cases
        ],
        producer_torch_version="2.test",
        **_catalog_producer_identity(),
    )
    assert catalog["case_count"] == 176
    assert catalog["case_ids"] == [
        *range(1001, 1097),
        *range(2001, 2081),
    ]


def test_cached_inputs_round_trip_and_fp64_promotion(monkeypatch):
    executor = _load_executor(monkeypatch)
    inputs = executor._PreparedInputs(
        q=torch.ones(2, dtype=torch.bfloat16),
        k=torch.ones(2, dtype=torch.bfloat16),
        v=torch.ones(2, dtype=torch.float16),
        g=torch.ones(2, dtype=torch.float32),
        beta=torch.ones(2, dtype=torch.bfloat16),
        A_log=torch.ones(1, dtype=torch.float32),
        dt_bias=None,
        initial_state=torch.ones(1, dtype=torch.float32),
        cu_seqlens=[0, 2],
        chunk_indices=[0, 0],
        seed=23,
    )
    payload = executor._prepared_inputs_to_cpu(inputs)
    restored = executor._prepared_inputs_from_cpu(
        payload, torch.device("cpu"), high_precision=True
    )

    for name in executor._PREPARED_TENSOR_NAMES:
        value = getattr(restored, name)
        if value is not None:
            assert value.dtype == torch.float64
    assert restored.cu_seqlens == [0, 2]
    assert restored.chunk_indices == [0, 0]
    assert restored.seed == 23


def test_traceable_accuracy_variant_materializes_distinct_executable_inputs(
    monkeypatch,
):
    executor = _load_executor(monkeypatch)
    adapter = _load_canonical_adapter()
    base = _canonical_spec("KDA-FWD-G002")
    variants = {
        item["variant"]: item["spec"]
        for item in adapter.materialize_cache_variants(base)
    }
    marker = torch.empty(0, device="cpu")
    random_inputs = executor._prepare_inputs(variants["random"], marker, marker)
    traceable_inputs = executor._prepare_inputs(
        variants["traceable_metamorphic"], marker, marker
    )
    random_q = executor._layout_to_bsnd(random_inputs.q, base["layout"])
    traceable_q = executor._layout_to_bsnd(traceable_inputs.q, base["layout"])

    assert not torch.equal(random_inputs.q, traceable_inputs.q)
    assert torch.equal(random_q[:, 1, :, :], traceable_q[:, 1, :, :])
    assert traceable_q[0, 0, :, 0].unique().numel() == base["H"]
    assert traceable_q[0, 64, :, 0].unique().numel() == base["H"]


def test_gate_parameter_dtypes_are_explicit_and_default_to_fp32(monkeypatch):
    executor = _load_executor(monkeypatch)
    marker = torch.empty(0, device="cpu")
    bf16_spec = _canonical_spec("KDA-FWD-P036")
    inputs = executor._prepare_inputs(bf16_spec, marker, marker)
    promoted = executor._prepare_inputs(
        bf16_spec, marker, marker, high_precision=True
    )

    assert inputs.A_log.dtype == torch.bfloat16
    assert inputs.dt_bias.dtype == torch.bfloat16
    assert promoted.A_log.dtype == torch.float64
    assert promoted.dt_bias.dtype == torch.float64

    legacy_spec = dict(bf16_spec)
    legacy_spec.pop("a_log_dtype")
    legacy_spec.pop("dt_bias_dtype")
    legacy_inputs = executor._prepare_inputs(legacy_spec, marker, marker)
    assert legacy_inputs.A_log.dtype == torch.float32
    assert legacy_inputs.dt_bias.dtype == torch.float32


def test_structured_input_variants_and_noncontiguous_views_are_executable(monkeypatch):
    executor = _load_executor(monkeypatch)
    marker = torch.empty(0, device="cpu")

    storage_inputs = executor._prepare_inputs(
        _canonical_spec("KDA-FWD-P024"), marker, marker
    )
    assert all(
        not getattr(storage_inputs, name).is_contiguous()
        for name in ("q", "k", "v", "g", "beta")
    )

    distinct_a_log = executor._prepare_inputs(
        _canonical_spec("KDA-FWD-G039"), marker, marker
    ).A_log
    assert distinct_a_log.dtype == torch.bfloat16
    assert torch.unique(distinct_a_log).numel() == 96

    pulse = executor._prepare_inputs(
        _canonical_spec("KDA-FWD-G043"), marker, marker
    ).initial_state
    assert torch.count_nonzero(pulse) == 1
    assert pulse[0, 2, 0, 0] == 1


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_cpu_references_export_rank4_varlen_h_without_a_batch_axis(
    monkeypatch, layout
):
    executor = _load_executor(monkeypatch)
    marker = torch.empty(0, device="cpu")
    spec = _canonical_spec("KDA-FWD-G067")
    spec.update(
        {
            "layout": layout,
            "H": 1,
            "HV": 2,
            "T": 16,
            "K": 16,
            "V": 16,
            "chunk_size": 64,
            "scale": 0.25,
            "cu_seqlens": "0,8,16",
            "explicit_chunk_indices": False,
        }
    )
    inputs = executor._prepare_inputs(spec, marker, marker)

    golden_h = executor._reference_impl(inputs, spec)[10]
    benchmark_h = executor._reference_model_parallel(inputs, spec)[10]
    assert golden_h.shape == (2, 2, 16, 16)
    assert benchmark_h.shape == (2, 2, 16, 16)


def test_cpu_reference_dense_h_rank_contract_is_unchanged(monkeypatch):
    executor = _load_executor(monkeypatch)
    marker = torch.empty(0, device="cpu")
    base = _canonical_spec("KDA-FWD-G067")
    base.update(
        {
            "H": 1,
            "HV": 2,
            "T": 16,
            "K": 16,
            "V": 16,
            "chunk_size": 64,
            "scale": 0.25,
            "cu_seqlens": "",
            "explicit_chunk_indices": False,
        }
    )

    rank4_spec = dict(base, layout="BNSD", B=2)
    rank4_inputs = executor._prepare_inputs(rank4_spec, marker, marker)
    assert executor._reference_impl(rank4_inputs, rank4_spec)[10].shape == (
        2,
        1,
        2,
        16,
        16,
    )

    rank3_spec = dict(base, layout="TND", B=1)
    rank3_inputs = executor._prepare_inputs(rank3_spec, marker, marker)
    assert executor._reference_model_parallel(rank3_inputs, rank3_spec)[10].shape == (
        1,
        2,
        16,
        16,
    )


def test_cpu_roles_replace_gpu_roles_without_removing_legacy_gpu(monkeypatch):
    executor = _load_executor(monkeypatch)
    assert executor._reference_role("cpu", True) == "cpu_fp64"
    assert executor._reference_role("cpu", False) == "cpu_same_precision"
    assert executor._reference_role("gpu", True) == "gpu_fp64"
    assert executor._reference_role("gpu", False) == "gpu_triton_same_precision"
    assert executor._reference_role("npu", False) == "npu_dut"


def test_accuracy_lt_rejects_fixed_persistent_cache(monkeypatch):
    executor = _load_executor(monkeypatch)
    with pytest.raises(executor.ReferenceCacheError, match="accuracy_lt"):
        executor._validate_persistent_cache_task("readonly", {"accuracy_lt"})
    executor._validate_persistent_cache_task("off", {"accuracy_lt"})


def test_canonical_executor_rejects_runtime_input_generation(monkeypatch):
    executor = _load_executor(monkeypatch)
    api = types.SimpleNamespace(
        spec=None,
        randomize_values=False,
        runtime_case_id=3001,
        persistent_cache_mode="off",
        high_precision=False,
        device="npu",
        is_benchmark_task=False,
        cache_reader=None,
        execution_device=None,
    )
    input_data = types.SimpleNamespace(
        kwargs={
            "case_spec": json.dumps(
                {
                    "tags": "run,negative,canonical_300",
                    "seed": 7,
                }
            ),
            "low_precision_marker": torch.empty(0),
            "fp32_marker": torch.empty(0),
        }
    )

    with pytest.raises(executor.ReferenceCacheError, match="prebuilt readonly"):
        executor.ChunkKdaFwdApi.init_by_input_data(api, input_data)


def test_offline_builder_writes_all_three_readonly_shards(monkeypatch, tmp_path):
    executor = _load_executor(monkeypatch)
    inputs = executor._PreparedInputs(
        q=torch.ones(2, dtype=torch.bfloat16),
        k=torch.ones(2, dtype=torch.bfloat16),
        v=torch.ones(2, dtype=torch.bfloat16),
        g=torch.ones(2, dtype=torch.float32),
        beta=torch.ones(2, dtype=torch.bfloat16),
        A_log=None,
        dt_bias=None,
        initial_state=None,
        cu_seqlens=[0, 2],
        chunk_indices=None,
        seed=7,
    )
    outputs = (torch.ones(2),) + (None,) * 11
    monkeypatch.setattr(executor, "_prepare_inputs", lambda *_args, **_kwargs: inputs)
    monkeypatch.setattr(executor, "_reference_impl", lambda *_args: outputs)
    monkeypatch.setattr(executor, "_reference_model_parallel", lambda *_args: outputs)
    spec = {"seed": 7, "output_final_state": False, "disable_recompute": False}

    executor.build_persistent_reference_cache(spec, tmp_path)
    reader = executor.CacheReader(
        tmp_path, executor._persistent_cache_metadata(spec, 7)
    )
    reader.validate_all()
    assert reader.load_shard("inputs")["seed"] == 7
    assert reader.load_shard("cpu_fp64")[0].shape == (2,)
    assert reader.load_shard("cpu_same_precision")[0].shape == (2,)


def test_non_accuracy_builder_writes_one_deduplicated_input_bundle(monkeypatch, tmp_path):
    executor = _load_executor(monkeypatch)
    inputs = executor._PreparedInputs(
        q=torch.ones(2, dtype=torch.bfloat16),
        k=torch.ones(2, dtype=torch.bfloat16),
        v=torch.ones(2, dtype=torch.bfloat16),
        g=torch.ones(2, dtype=torch.float32),
        beta=torch.ones(2, dtype=torch.bfloat16),
        A_log=None,
        dt_bias=None,
        initial_state=None,
        cu_seqlens=None,
        chunk_indices=None,
        seed=7,
    )
    monkeypatch.setattr(executor, "_prepare_inputs", lambda *_args, **_kwargs: inputs)
    spec = _canonical_execution_spec("KDA-FWD-S004")
    spec["seed"] = 7

    executor.build_persistent_reference_cache(
        spec, tmp_path, include_references=False
    )
    metadata = executor.build_chunk_kda_metadata(
        spec,
        7,
        executor._EXECUTOR_PATH,
        include_references=False,
    )
    reader = executor.CacheReader(tmp_path, metadata)
    reader.validate_all()
    bundle = reader.load_shard("inputs")

    assert set(reader.shards) == {"inputs"}
    assert bundle["schema"] == "chunk_kda_fwd.canonical_input_variants.v1"
    assert bundle["aliases"] == {
        "all_outputs": "all_outputs",
        "hidden_outputs": "all_outputs",
    }
    assert set(bundle["variants"]) == {"all_outputs"}


def test_negative_executor_dispatches_the_requested_real_route(monkeypatch):
    executor = _load_executor(monkeypatch)
    api = types.SimpleNamespace(
        spec={"tags": "run,negative,canonical_300", "route": "ascendc"},
        device="npu",
        inputs=object(),
    )
    marker = object()
    monkeypatch.setattr(executor, "_run_negative_ascendc", lambda *_args: marker)
    monkeypatch.setattr(
        executor,
        "_run_negative_aclnn",
        lambda *_args: pytest.fail("aclnn route must not be called"),
    )

    assert executor.ChunkKdaFwdApi.__call__(api, None) is marker

    api.spec["route"] = "aclnn"
    monkeypatch.setattr(executor, "_run_negative_aclnn", lambda *_args: marker)
    monkeypatch.setattr(
        executor,
        "_run_negative_ascendc",
        lambda *_args: pytest.fail("public route must not be called"),
    )
    assert executor.ChunkKdaFwdApi.__call__(api, None) is marker


def _negative_values():
    return {
        "q": object(),
        "k": object(),
        "v": object(),
        "g": object(),
        "beta": object(),
        "A_log": object(),
        "dt_bias": object(),
        "initial_state": None,
        "cu": None,
        "indices": None,
        "layout": "INVALID",
        "chunk_size": 64,
        "lower_bound": -5.0,
    }


def test_public_negative_route_validates_exception_type_and_message(monkeypatch):
    executor = _load_executor(monkeypatch)
    ascendc = types.ModuleType("fla_npu.ops.ascendc")

    def fail(*_args, **_kwargs):
        raise RuntimeError("layout must be uppercase and one of BSND, BNSD, TND, NTD")

    ascendc.chunk_kda_fwd = fail
    monkeypatch.setitem(sys.modules, "fla_npu", types.ModuleType("fla_npu"))
    monkeypatch.setitem(sys.modules, "fla_npu.ops", types.ModuleType("fla_npu.ops"))
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", ascendc)
    monkeypatch.setattr(executor, "_raw_output_tensors", lambda *_args: [])
    monkeypatch.setattr(executor, "_mutate_raw", lambda *_args: _negative_values())
    spec = {
        "scale": 1.0,
        "output_final_state": False,
        "safe_gate": True,
        "use_gate_in_kernel": True,
        "disable_recompute": True,
        "return_intermediate_states": True,
        "state_v_first": False,
        "expected_code_name": "RuntimeError",
        "expected_return_code": "RuntimeError",
        "expected_message": "layout must be uppercase",
    }

    with pytest.raises(RuntimeError, match=r"RuntimeError\(RuntimeError\)"):
        executor._run_negative_ascendc(object(), spec)

    bad = dict(spec, expected_message="different message")
    with pytest.raises(RuntimeError, match="message did not"):
        executor._run_negative_ascendc(object(), bad)


def test_aclnn_negative_route_validates_numeric_code_and_recent_message(monkeypatch):
    executor = _load_executor(monkeypatch)
    ctypes_module = types.ModuleType("fla_npu.ops.ascendc._aclnn_ctypes")
    ctypes_module._GET_WORKSPACE_ARGTYPES = ()

    def fail(*_args, **_kwargs):
        raise RuntimeError("aclnnChunkKdaFwdGetWorkspaceSize aclnnStatus=161002")

    ctypes_module._call_aclnn = fail
    monkeypatch.setitem(sys.modules, "fla_npu", types.ModuleType("fla_npu"))
    monkeypatch.setitem(sys.modules, "fla_npu.ops", types.ModuleType("fla_npu.ops"))
    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", types.ModuleType("fla_npu.ops.ascendc"))
    monkeypatch.setitem(
        sys.modules, "fla_npu.ops.ascendc._aclnn_ctypes", ctypes_module
    )
    monkeypatch.setattr(executor, "_raw_output_tensors", lambda *_args: [None] * 11)
    monkeypatch.setattr(executor, "_mutate_raw", lambda *_args: _negative_values())
    monkeypatch.setattr(
        executor,
        "_recent_aclnn_error",
        lambda: "chunkSize must be 64 or 128",
    )
    spec = {
        "scale": 1.0,
        "safe_gate": True,
        "use_gate_in_kernel": True,
        "state_v_first": False,
        "expected_code_name": "ACLNN_ERR_PARAM_INVALID",
        "expected_return_code": 161002,
        "expected_message": "chunkSize must be 64 or 128",
    }

    with pytest.raises(RuntimeError, match=r"ACLNN_ERR_PARAM_INVALID\(161002\)"):
        executor._run_negative_aclnn(object(), spec)

    bad_code = dict(spec, expected_return_code=161001)
    with pytest.raises(RuntimeError, match="returned 161002, expected 161001"):
        executor._run_negative_aclnn(object(), bad_code)


def test_triton_triangular_normalization_clears_only_unwritten_packed_regions(
    monkeypatch,
):
    executor = _load_executor(monkeypatch)
    total_t, chunk_size = 10, 4
    base = torch.arange(total_t * chunk_size, dtype=torch.float32).reshape(
        1, total_t, chunk_size
    )
    local_rows = torch.tensor([0, 1, 2, 3, 0, 1, 0, 1, 2, 3])
    valid = torch.arange(chunk_size).unsqueeze(0) <= local_rows.unsqueeze(1)
    dirty = base.masked_fill(~valid.unsqueeze(0), float("nan"))
    inputs = executor._PreparedInputs(
        q=torch.empty(0),
        k=torch.empty(0),
        v=torch.empty(0),
        g=torch.empty(0),
        beta=torch.empty(0),
        A_log=None,
        dt_bias=None,
        initial_state=None,
        cu_seqlens=[0, 6, 10],
        chunk_indices=None,
        seed=0,
    )

    actual = executor._zero_undefined_triton_triangular_regions(
        _outputs(dirty, dirty.clone()),
        inputs,
        {"T": total_t, "chunk_size": chunk_size},
    )

    for index in (3, 4):
        assert torch.equal(actual[index][0][valid], base[0][valid])
        assert torch.count_nonzero(actual[index][0][~valid]) == 0
        assert torch.isfinite(actual[index]).all()


def test_triton_triangular_normalization_preserves_nonfinite_valid_values(monkeypatch):
    executor = _load_executor(monkeypatch)
    tensor = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
    tensor[0, 0, 1, 1] = float("nan")
    inputs = executor._PreparedInputs(
        q=torch.empty(0),
        k=torch.empty(0),
        v=torch.empty(0),
        g=torch.empty(0),
        beta=torch.empty(0),
        A_log=None,
        dt_bias=None,
        initial_state=None,
        cu_seqlens=None,
        chunk_indices=None,
        seed=0,
    )

    actual = executor._zero_undefined_triton_triangular_regions(
        _outputs(tensor, tensor.clone()),
        inputs,
        {"T": 3, "chunk_size": 4},
    )

    assert torch.isnan(actual[3][0, 0, 1, 1])
    assert torch.isnan(actual[4][0, 0, 1, 1])
