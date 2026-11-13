#!/usr/bin/env python3
"""Record resolved experimental inputs and audit an isolated OPP installation."""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

VENDOR = "fla_npu_experimental_transformer"
MANIFEST = "experimental-build-manifest.json"
# Installed explicitly by custom_build.cmake from Catlass/OPBASE, not from the
# experimental common snapshot. Keep these namespaces separate in the audit.
SDK_COMMON_ROOTS = {"catlass", "tla", "atvoss", "op_kernel"}


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"Experimental input escapes its tree: {path} -> {resolved}") from exc
    return resolved


def _names(path: Path) -> list[str]:
    names = sorted(set(path.read_text(encoding="utf-8").split()))
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in {"all", "ALL"}:
            raise ValueError(f"Invalid resolved experimental operator: {name}")
    return names


def _hashes(path: Path, root: Path, *, operator=False) -> dict[str, str]:
    result = {}
    for entry in sorted(path.rglob("*")):
        relative = entry.relative_to(path)
        # Provenance covers build inputs, not development docs/examples/tests.
        # The CMake tree audit still rejects escaping links throughout the tree.
        if operator and relative.parts[0] not in {"CMakeLists.txt", "op_host", "op_kernel", "adapter"}:
            continue
        real = _inside(entry, root)
        if real.is_file() and "__pycache__" not in entry.parts:
            result[entry.relative_to(path).as_posix()] = hashlib.sha256(real.read_bytes()).hexdigest()
    return result


def create_manifest(repo: Path, selected_file: Path, resolved_file: Path,
                    soc: str, arch_directories: str) -> dict:
    repo = repo.resolve()
    experimental = repo / "experimental"
    if experimental.resolve() != experimental:
        raise ValueError("Experimental root must not redirect to another source tree")
    operators = experimental / "ops"
    if operators.resolve() != operators:
        raise ValueError("Experimental ops root must not redirect to another source tree")
    selected, resolved = _names(selected_file), _names(resolved_file)
    if not selected or not resolved or not set(selected).issubset(resolved):
        raise ValueError("Experimental build must contain its nonempty selected operator set")
    arches = sorted(set(part for part in re.split(r"[,;]", arch_directories) if part))
    if not arches or any(not re.fullmatch(r"arch[A-Za-z0-9_]+", arch) for arch in arches):
        raise ValueError("Record the nonempty architecture directory selection supplied by CMake")
    records = []
    for name in resolved:
        operator = operators / name
        _inside(operator / "CMakeLists.txt", experimental)
        definitions = sorted((operator / "op_host").glob("*_def.cpp"))
        if len(definitions) != 1:
            raise ValueError(f"{name}: declare exactly one OpDef file for package identity")
        definition = definitions[0].read_text(encoding="utf-8")
        op_types = re.findall(r"\bOP_ADD\s*\(\s*(\w+)\s*\)", definition)
        if len(op_types) != 1:
            raise ValueError(f"{name}: cannot determine one registered OpType")
        supported = re.findall(r'AddConfig\s*\(\s*"([^"]+)"', definition)
        requested_soc = [part for part in re.split(r"[,;]", soc) if part]
        if not requested_soc or set(requested_soc) - set(supported):
            raise ValueError(f"{name}: unsupported SoC {requested_soc}; declared {supported}")
        entry = operator / "op_kernel" / f"{name}.cpp"
        if not entry.is_file():
            raise ValueError(f"{name}: kernel entry must match the operator name: {entry}")
        symbols = set()
        for header in (operator / "op_host/op_api").glob("aclnn_*.h"):
            symbols.update(re.findall(r"\b(aclnn\w+)\s*\(", header.read_text(encoding="utf-8")))
        if not symbols:
            raise ValueError(f"{name}: no ACLNN export declarations found")
        records.append({
            "name": name, "source": operator.relative_to(repo).as_posix(),
            "op_type": op_types[0], "op_file": name,
            "proto_stem": definitions[0].stem.removesuffix("_def"),
            "aclnn_symbols": sorted(symbols), "supported_soc": sorted(set(supported)),
            "source_sha256": _hashes(operator, experimental, operator=True),
        })
    common = experimental / "_common"
    required = [line.strip() for line in (common / "manifest.txt").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith(("#", ";"))]
    for item in required:
        if not _inside(common / item, common).is_file():
            raise ValueError(f"Common requirement is not a file: {item}")
    identity_file = repo / ".source-identity.json"
    if identity_file.is_file():
        identity = json.loads(identity_file.read_text(encoding="utf-8"))
    else:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True,
                                             stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True))
            identity = {"source_commit": commit, "source_dirty": dirty}
        except (subprocess.CalledProcessError, FileNotFoundError):
            identity = {"source_commit": os.environ.get("FLA_SOURCE_COMMIT", "unknown"), "source_dirty": True}
    return {
        "schema_version": 2, "mode": "experimental", "vendor": VENDOR,
        "source_commit": identity["source_commit"], "source_dirty": identity.get("source_dirty", True),
        "source_snapshot_sha256": identity.get("source_snapshot_sha256"),
        "soc": [part for part in re.split(r"[,;]", soc) if part],
        "arch_directories": arches,
        "selected": selected, "operators": records,
        "common": {"source": "experimental/_common", "required": required,
                   "sdk_roots": sorted(SDK_COMMON_ROOTS),
                   "source_sha256": _hashes(common, common)},
    }


def _equal(actual, expected, description):
    actual, expected = set(actual), set(expected)
    if actual != expected:
        raise ValueError(f"{description}: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")


def _json_object(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Expected nonempty JSON object: {path}")
    return data


def _nonempty_list(data: dict, key: str, description: str) -> list[dict]:
    values = data.get(key) if isinstance(data, dict) else None
    if not isinstance(values, list) or not values or any(not isinstance(v, dict) for v in values):
        raise ValueError(f"{description}: expected nonempty {key}")
    return values


def _audit_kernel_artifacts(tbe: Path, soc: str, op: dict, aggregate: dict) -> None:
    name = op["op_file"]
    kernel_dir = tbe / "kernel" / soc / name
    files = {p.relative_to(kernel_dir).as_posix(): p for p in kernel_dir.rglob("*") if p.is_file()}
    objects = {name for name in files if name.endswith(".o")}
    metadata = {name for name in files if name.endswith(".json")}
    if not objects or not metadata:
        raise ValueError(f"{name}: kernel directory must contain nonempty .o/.json artifacts")
    _equal(files, objects | metadata, f"{name} kernel artifact files")
    _equal(metadata, {str(Path(name).with_suffix(".json")).replace("\\", "/") for name in objects},
           f"{name} kernel object/metadata pairs")
    for filename, artifact in files.items():
        if not artifact.stat().st_size:
            raise ValueError(f"{name}: empty kernel artifact: {filename}")
    for filename in metadata:
        details = _json_object(files[filename])
        binary = files[str(Path(filename).with_suffix(".o")).replace("\\", "/")]
        if details.get("binFileName") != binary.stem or details.get("binFileSuffix") != ".o":
            raise ValueError(f"{name}: kernel metadata names a different binary: {filename}")
        if details.get("sha256") != hashlib.sha256(binary.read_bytes()).hexdigest():
            raise ValueError(f"{name}: kernel object SHA256 differs from metadata: {binary.name}")
    expected_json = {f"{soc}/{name}/{filename}" for filename in metadata}
    expected_objects = {f"{soc}/{name}/{filename}" for filename in objects}
    config = _json_object(tbe / "kernel/config" / soc / f"{name}.json")
    entries = _nonempty_list(config, "binList", name)
    json_refs = []
    for entry in entries:
        info = entry.get("binInfo")
        reference = info.get("jsonFilePath") if isinstance(info, dict) else None
        if not isinstance(reference, str):
            raise ValueError(f"{name}: kernel config is missing binInfo.jsonFilePath")
        json_refs.append(reference)
    _equal(json_refs, expected_json, f"{name} kernel config JSON references")
    entries = _nonempty_list(aggregate[op["op_type"]], "binaryList", op["op_type"])
    for field, expected in (("binPath", expected_objects), ("jsonPath", expected_json)):
        references = [entry.get(field) for entry in entries]
        if any(not isinstance(reference, str) for reference in references):
            raise ValueError(f"{name}: binary info config is missing {field}")
        _equal(references, expected, f"{name} binary info {field} references")
    for entry in entries:
        if entry["binPath"].removesuffix(".o") != entry["jsonPath"].removesuffix(".json"):
            raise ValueError(f"{name}: binary info config pairs different object and JSON files")


def _audit_operator_sources(impl: Path, op: dict, arches: set[str]) -> None:
    expected = {}
    for source, digest in op["source_sha256"].items():
        if not source.startswith("op_kernel/"):
            continue
        relative = source.removeprefix("op_kernel/")
        parts = relative.split("/")
        # filter_copy_files filters only top-level directories beginning with
        # arch. A root file named arch*.hpp remains a required source file.
        if len(parts) > 1 and parts[0].startswith("arch") and parts[0] not in arches:
            continue
        expected[relative] = digest
    directory = impl / "ascendc" / op["op_file"]
    actual = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*") if p.is_file()}
    _equal(actual, expected, f"{op['name']} packaged kernel source files")
    for name, digest in expected.items():
        if hashlib.sha256(actual[name].read_bytes()).hexdigest() != digest:
            raise ValueError(f"Kernel source does not match resolved experimental input: {actual[name]}")


def _audit_common_sources(impl: Path, common: dict) -> list[str]:
    directory = impl / "ascendc/common"
    expected = common["source_sha256"]
    if any(name.split("/")[0] in SDK_COMMON_ROOTS for name in expected):
        raise ValueError("Experimental common inputs overlap reserved SDK common namespaces")
    actual = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*")
              if p.is_file() and p.relative_to(directory).parts[0] not in SDK_COMMON_ROOTS}
    _equal(actual, expected, "Experimental common source files")
    for name, digest in expected.items():
        if hashlib.sha256(actual[name].read_bytes()).hexdigest() != digest:
            raise ValueError(f"Common source does not match experimental snapshot: {name}")
    return sorted(p.name for p in directory.iterdir() if p.is_dir() and p.name in SDK_COMMON_ROOTS)


def _audit_dynamic_adapters(directory: Path, expected: set[str]) -> None:
    actual = {p.stem for p in directory.glob("*.py")}
    optional = {name + "_apt" for name in expected}
    _equal(actual & expected, expected, "Dynamic adapters (base adapters required)")
    _equal(actual - expected - optional, set(), "Unexpected dynamic adapters")


def audit_package(install_root: Path, soc: str, expected_manifest: Path | None = None) -> dict:
    vendors = install_root / "vendors"
    _equal([p.name for p in vendors.iterdir() if p.is_dir()], [VENDOR], "Vendor set")
    vendor = vendors / VENDOR
    for path in vendor.rglob("*"):
        _inside(path, vendor)
    manifest_path = vendor / MANIFEST
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("mode") != "experimental" or data.get("vendor") != VENDOR:
        raise ValueError("Package identity is not experimental")
    if data.get("schema_version") != 2 or not data.get("arch_directories"):
        raise ValueError("Package manifest must record schema 2 and its CMake architecture selection")
    if expected_manifest and data != json.loads(expected_manifest.read_text(encoding="utf-8")):
        raise ValueError("Installed manifest differs from this build's resolved input manifest")
    if soc not in data["soc"]:
        raise ValueError(f"Requested audit SoC {soc} is absent from package manifest")
    ops = data["operators"]
    tbe = vendor / "op_impl/ai_core/tbe"
    impl = tbe / f"{VENDOR}_impl"
    expected_files = {op["op_file"] for op in ops}
    _equal([p.name for p in (tbe / "kernel" / soc).iterdir()], expected_files, "Kernel directories")
    _audit_dynamic_adapters(impl / "dynamic", expected_files)
    _equal([p.stem for p in (tbe / "kernel/config" / soc).glob("*.json")],
           expected_files | {"binary_info_config"}, "Kernel config files")
    aggregate = _json_object(tbe / "kernel/config" / soc / "binary_info_config.json")
    _equal(aggregate, [op["op_type"] for op in ops], "Binary info operator types")
    ops_info = _json_object(tbe / "config" / soc / f"aic-{soc}-ops-info.json")
    _equal(ops_info, [op["op_type"] for op in ops], "Ops-info operator types")
    _equal([p.name for p in (impl / "ascendc").iterdir() if p.is_dir()], expected_files | {"common"}, "Packaged source directories")
    _equal([p.name for p in (vendor / "op_proto/inc").glob("*_proto.h")],
           [op["proto_stem"] + "_proto.h" for op in ops], "Proto headers")
    aggregate_header = "aclnn_ops_transformer_fla_npu_experimental.h"
    _equal([p.name for p in (vendor / "op_api/include/aclnnop").glob("*.h") if p.name != aggregate_header],
           ["aclnn_" + op["op_file"] + ".h" for op in ops], "ACLNN headers")
    for op in ops:
        _audit_operator_sources(impl, op, set(data["arch_directories"]))
        _audit_kernel_artifacts(tbe, soc, op, aggregate)
        if ops_info[op["op_type"]].get("opFile", {}).get("value") != op["op_file"]:
            raise ValueError(f"{op['name']}: ops-info references a different operator source")
    sdk_roots = _audit_common_sources(impl, data["common"])
    library = vendor / "op_api/lib/libcust_opapi.so"
    output = subprocess.check_output(["nm", "-D", "--defined-only", str(library)], text=True)
    exports = {line.split()[-1] for line in output.splitlines() if line.split() and line.split()[-1].startswith("aclnn")}
    _equal(exports, {symbol for op in ops for symbol in op["aclnn_symbols"]}, "ACLNN exports")
    helper = Path(__file__).resolve().parents[2] / "torch_custom/fla_npu/fla_npu/_op_abi.py"
    spec = importlib.util.spec_from_file_location("experimental_audit_abi", helper)
    abi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abi)
    try:
        contracts = abi.validated_contracts(vendor, require_manifest=True)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    _equal(set(contracts), exports, "ABI manifest exports")

    return {"mode": "experimental", "soc": soc, "operators": sorted(expected_files),
            "vendor": str(vendor.resolve()), "source_commit": data["source_commit"],
            "source_dirty": data["source_dirty"],
            "source_snapshot_sha256": data.get("source_snapshot_sha256"),
            "sdk_common_roots_outside_source_snapshot": sdk_roots,
            "arch_directories": data["arch_directories"], "runtime_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--repo", type=Path, required=True)
    generate.add_argument("--selected", type=Path, required=True)
    generate.add_argument("--resolved", type=Path, required=True)
    generate.add_argument("--soc", required=True)
    generate.add_argument("--arch-directories", required=True)
    generate.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--install-root", type=Path, required=True)
    audit.add_argument("--soc", required=True)
    audit.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            data = create_manifest(args.repo, args.selected, args.resolved, args.soc, args.arch_directories)
            args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            print(json.dumps(audit_package(args.install_root, args.soc, args.manifest), indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"[experimental] {exc}\n")


if __name__ == "__main__":
    main()
