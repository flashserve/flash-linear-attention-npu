"""Build and verify public and launcher-private two-phase OPP contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path


TENSOR = "tensor"
INT_ARRAY = "int_array"
ACL_SCALAR = "acl_scalar"

_BASE_TYPES = {
    "aclTensor": "aclTensor", "aclIntArray": "aclIntArray",
    "aclScalar": "aclScalar", "aclOpExecutor": "executor",
    "aclrtStream": "stream", "void": "void", "char": "char",
    "double": "double", "float": "float", "bool": "bool",
    "uint64_t": "uint64", "int64_t": "int64",
    "uint32_t": "uint32", "int32_t": "int32",
}
_POINTER_KINDS = {"aclTensor*": TENSOR, "aclIntArray*": INT_ARRAY,
                  "aclScalar*": ACL_SCALAR, "char*": "char_ptr"}
_FUNCTION = re.compile(r"\baclnnStatus\s+([A-Za-z_]\w*)\s*\(([^;{}]*)\)\s*;", re.S | re.ASCII)
_PRIVATE_STAMP = "fla_private_abi_sha256_"
_WORKSPACE_TAIL = ("uint64*", "executor**")
_EXECUTE_SIGNATURE = ("void*", "uint64", "executor*", "stream")
MANIFEST_NAME = "abi_manifest.json"
LIBRARY = "op_api/lib/libcust_opapi.so"
INCLUDE = "op_api/include/aclnnop"
# Explicit, source-audited legacy contract. The public header contains only
# l0op::SolveTri; the C entry definitions live in op_api/aclnn_solve_tri.cpp.
KNOWN_CONTRACTS = {
    "aclnnSolveTriGetWorkspaceSize":
        (TENSOR, INT_ARRAY, INT_ARRAY, "char_ptr", TENSOR, *_WORKSPACE_TAIL),
    "aclnnSolveTri": _EXECUTE_SIGNATURE,
}
# This tensor-input variant intentionally shares the same phase-two entry.
EXECUTE_ALIASES = {
    "aclnnCausalConv1dTensor": "aclnnCausalConv1d",
    "aclnnCausalConv1dBwdTensor": "aclnnCausalConv1dBwd",
}


def strip_comments(text: str) -> str:
    """Remove C/C++ comments without treating comment punctuation as syntax."""

    out: list[str] = []
    index, size = 0, len(text)
    quote = ""
    while index < size:
        char = text[index]
        if quote:
            out.append(char)
            if char == "\\" and index + 1 < size:
                out.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = ""
            index += 1
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            index += 1
            continue
        if char == "/" and text.startswith("//", index):
            while index < size and text[index] != "\n":
                index += 1
            continue
        if char == "/" and text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = size if end < 0 else end + 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def split_params(text: str) -> list[str]:
    """Split a C parameter list on top-level commas."""

    text = strip_comments(text)
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "([{<":
            depth += 1
        elif char in ")]}>":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return [part for part in parts if part]


def parameter_kind(param: str) -> str:
    """Parse the supported C parameter grammar, retaining every pointer level.

    cv qualifiers and argument names do not change the C calling convention.
    Arrays, references, function pointers and unknown typedefs are rejected,
    rather than guessed from a substring (e.g. double versus double*).
    """
    normalized = re.sub(r"\b(?:const|volatile|restrict|__restrict__)\b", " ", param)
    match = re.fullmatch(r"\s*(\w+)\s*((?:\*\s*)*)(?:[A-Za-z_]\w*)?\s*", normalized)
    if match is None or match[1] not in _BASE_TYPES:
        raise RuntimeError(f"Unsupported aclnn parameter declaration: {param!r}")
    kind = _BASE_TYPES[match[1]] + "*" * match[2].count("*")
    return _POINTER_KINDS.get(kind, kind)


def public_prototypes(include_dir: Path, wanted: set[str] | None = None) -> dict[str, tuple[str, ...]]:
    return header_prototypes(sorted(include_dir.glob("aclnn_*.h")), wanted, public=True)


def header_prototypes(headers, wanted=None, *, public=False):
    signatures = {}
    for header in headers:
        text = strip_comments(header.read_text(encoding="utf-8"))
        for match in _FUNCTION.finditer(text):
            symbol = match[1]
            if public and not symbol.startswith("aclnn"):
                continue
            if wanted is not None and symbol not in wanted:
                continue
            signature = tuple(parameter_kind(p) for p in split_params(match[2]))
            if symbol in signatures and signatures[symbol] != signature:
                raise RuntimeError(f"Conflicting prototypes for {symbol}")
            signatures[symbol] = signature
    return signatures


def private_headers(declaration: Path) -> dict[str, Path]:
    """Each entry binds one launcher base name to its own source header."""
    entries = json.loads(declaration.read_text(encoding="utf-8"))
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"Expected private ABI entry-to-header mapping in {declaration}")
    result = {}
    for entry, filename in entries.items():
        if (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry)
                or entry.endswith("GetWorkspaceSize") or entry.startswith(_PRIVATE_STAMP)
                or not isinstance(filename, str) or Path(filename).is_absolute()):
            raise ValueError(f"Invalid private ABI declaration for {entry!r} in {declaration}")
        header = (declaration.parent / filename).resolve()
        if not header.is_relative_to(declaration.parent.resolve()) or not header.is_file():
            raise ValueError(f"Missing or external private ABI header: {header}")
        result[entry] = header
    return result


def private_contracts(declarations) -> dict[str, tuple[str, ...]]:
    contracts = {}
    for declaration in declarations:
        for entry, header in private_headers(declaration).items():
            pair = {entry, entry + "GetWorkspaceSize"}
            signatures = header_prototypes([header], pair)
            if signatures.keys() != pair:
                raise ValueError(f"{header}: missing private prototype(s) {sorted(pair - signatures.keys())}")
            if pair.intersection(contracts):
                raise ValueError(f"Duplicate private ABI entry {entry} in {declaration}")
            _validate_exports(pair, signatures)
            contracts.update(signatures)
    return contracts


def private_stamp(contracts) -> str:
    # A symbol in the library independently binds the complete private inventory
    # and its types. Readers can verify an installed OPP without source headers.
    payload = json.dumps(contracts, sort_keys=True, separators=(",", ":")).encode("ascii")
    return _PRIVATE_STAMP + hashlib.sha256(payload).hexdigest()


def parse_aclnn_signatures(include_dir: Path, wanted: set[str] | None = None) -> dict[str, tuple[str, ...]]:
    """Read ``symbol -> parameter kinds`` from public GetWorkspaceSize headers."""

    symbols = None if wanted is None else {s + "GetWorkspaceSize" for s in wanted}
    signatures = {}
    for symbol, params in public_prototypes(include_dir, symbols).items():
        if not symbol.endswith("GetWorkspaceSize"):
            continue
        if params[-2:] != _WORKSPACE_TAIL:
            raise RuntimeError(f"Invalid workspace/executor parameters for {symbol}: {params}")
        signatures[symbol.removesuffix("GetWorkspaceSize")] = params[:-2]
    return signatures


def vendor_aclnn_signatures(vendor_dir: Path) -> dict[str, tuple[str, ...]]:
    return parse_aclnn_signatures(vendor_dir / "op_api" / "include" / "aclnnop")


def exported_symbols(library: Path, wanted=()) -> set[str]:
    """Build-time only: inspect ELF without executing any vendor code."""
    result = subprocess.run(["nm", "-D", "--defined-only", "--format=posix", str(library)],
                            check=True, capture_output=True, text=True)
    names = (line.split()[0].split("@", 1)[0] for line in result.stdout.splitlines() if line.strip())
    return {name for name in names if name.startswith(("aclnn", _PRIVATE_STAMP)) or name in wanted}


def elf_exported_symbols(library: Path, wanted=()) -> set[str]:
    """Read defined public ACLNN symbols without loading ELF or invoking nm.

    The dynamic symbol table is independent of the JSON contract list, so
    deleting a complete interface from that list cannot hide an ABI conflict.
    Both ELF classes and byte orders are accepted; invalid tables fail closed.
    """
    extra_names = {name.encode("ascii") for name in wanted}
    stamp_prefix = _PRIVATE_STAMP.encode("ascii")
    data = library.read_bytes()

    def fail():
        raise RuntimeError(f"Invalid ELF dynamic symbol table: {library}")

    def region(offset, size):
        if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
            fail()
        return data[offset:offset + size]

    if len(data) < 16 or data[:4] != b"\x7fELF" or data[4] not in (1, 2) or data[5] not in (1, 2):
        fail()
    order = "<" if data[5] == 1 else ">"
    is64 = data[4] == 2
    header_fmt = order + ("HHIQQQIHHHHHH" if is64 else "HHIIIIIHHHHHH")
    section_fmt = order + ("IIQQQQIIQQ" if is64 else "IIIIIIIIII")
    symbol_fmt = order + ("IBBHQQ" if is64 else "IIIBBH")
    header = struct.unpack(header_fmt, region(16, struct.calcsize(header_fmt)))
    section_offset, section_size, section_count = header[5], header[10], header[11]
    if not section_offset or section_size < struct.calcsize(section_fmt):
        fail()

    def section(index):
        return struct.unpack(section_fmt, region(
            section_offset + index * section_size, struct.calcsize(section_fmt)))

    if not section_count:  # ELF extended section numbering.
        section_count = section(0)[5]
    if not section_count or section_count > (len(data) - section_offset) // section_size:
        fail()
    exports = set()
    found_dynamic_table = False
    for index in range(section_count):
        entry = section(index)
        if entry[1] != 11:  # SHT_DYNSYM
            continue
        found_dynamic_table = True
        offset, size, link, stride = entry[4], entry[5], entry[6], entry[9]
        if stride < struct.calcsize(symbol_fmt) or size % stride or link >= section_count:
            fail()
        region(offset, size)
        strings_section = section(link)
        if strings_section[1] != 3:  # SHT_STRTAB
            fail()
        strings = region(strings_section[4], strings_section[5])
        for position in range(offset, offset + size, stride):
            symbol = struct.unpack(symbol_fmt, region(position, struct.calcsize(symbol_fmt)))
            name_offset = symbol[0]
            info, other, defined = symbol[1:4] if is64 else symbol[3:6]
            if defined == 0 or info >> 4 not in (1, 2) or other & 3 not in (0, 3):
                continue
            if name_offset >= len(strings):
                fail()
            end = strings.find(b"\0", name_offset)
            if end < 0:
                fail()
            name = strings[name_offset:end]
            if name.startswith((b"aclnn", stamp_prefix)) or name in extra_names:
                try:
                    exports.add(name.decode("ascii").split("@", 1)[0])
                except UnicodeDecodeError:
                    fail()
    if not found_dynamic_table:
        fail()
    return exports


def _file_hashes(vendor: Path, *, private=False) -> dict[str, str]:
    paths = [vendor / LIBRARY, *sorted((vendor / INCLUDE).glob("aclnn_*.h"))]
    if len(paths) < 2 and not private:
        raise RuntimeError(f"Missing public aclnn headers under {vendor / INCLUDE}")
    hashes = {}
    for path in paths:
        if not path.is_file() or not path.resolve().is_relative_to(vendor.resolve()):
            raise RuntimeError(f"Missing or external ABI input: {path}")
        hashes[path.relative_to(vendor).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _contracts(vendor: Path) -> dict[str, tuple[str, ...]]:
    contracts = public_prototypes(vendor / INCLUDE)
    # Apply only the explicitly audited gap, and only to its own existing header.
    if (vendor / INCLUDE / "aclnn_solve_tri.h").is_file():
        for symbol, signature in KNOWN_CONTRACTS.items():
            contracts.setdefault(symbol, signature)
    return contracts


def _validate_exports(exports: set[str], contracts: dict[str, tuple[str, ...]]) -> None:
    if not exports:
        raise RuntimeError("No public aclnn exports to validate")
    missing = exports - contracts.keys()
    if missing:
        raise RuntimeError(f"Missing public aclnn contracts: {sorted(missing)}")
    workspaces = {s for s in exports if s.endswith("GetWorkspaceSize")}
    executions = set()
    for symbol in workspaces:
        name = symbol.removesuffix("GetWorkspaceSize")
        execute = EXECUTE_ALIASES.get(name, name)
        if execute not in exports:
            raise RuntimeError(f"Missing phase-two export {execute} for {symbol}")
        if contracts[symbol][-2:] != _WORKSPACE_TAIL or contracts[execute] != _EXECUTE_SIGNATURE:
            raise RuntimeError(f"Invalid two-phase aclnn contract: {symbol}, {execute}")
        executions.add(execute)
    if exports != workspaces | executions:
        raise RuntimeError(f"Unpaired aclnn exports: {sorted(exports - workspaces - executions)}")


def _check_stamp(exports, private):
    stamps = {s for s in exports if s.startswith(_PRIVATE_STAMP)}
    if stamps != ({private_stamp(private)} if private else set()):
        raise RuntimeError("Private ABI declarations do not match the compiled library")
    return exports - stamps


def write_manifest(vendor: Path, declarations=()) -> Path:
    vendor = vendor.resolve()
    private = private_contracts(declarations)
    hashes = _file_hashes(vendor, private=bool(private))
    contracts = _contracts(vendor)
    exports = _check_stamp(exported_symbols(vendor / LIBRARY, private), private)
    if contracts.keys() & private.keys():
        raise RuntimeError("A private ABI entry also has a public prototype")
    if not private.keys() <= exports:
        raise RuntimeError(f"Missing private ABI exports: {sorted(private.keys() - exports)}")
    _validate_exports(exports, {**contracts, **private})
    data = {"schema_version": 1, "files": hashes,
            "contracts": {s: list(contracts[s]) for s in sorted(exports - private.keys())}}
    if private:
        data["private_contracts"] = private
    target = vendor / MANIFEST_NAME
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def validated_contract_groups(vendor: Path, *, require_manifest: bool):
    """Return verified (public, private) contracts, including old public-only OPPs."""
    path = vendor / MANIFEST_NAME
    if not path.is_file():
        if require_manifest:
            raise RuntimeError(f"Missing {path}; rebuild this overlay with ABI manifest support")
        # Existing base wheels predate the manifest. Never relax this for overlays.
        contracts = _contracts(vendor)
        if not contracts:
            raise RuntimeError(f"No base public aclnn contracts under {vendor}")
        exports = elf_exported_symbols(vendor / LIBRARY)
        _validate_exports(exports, contracts)
        return {symbol: contracts[symbol] for symbol in exports}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid ABI manifest object: {path}")
    version = data.get("schema_version")
    if version != 1:
        raise RuntimeError(f"Unsupported OPP ABI manifest version {version!r}: {path}")
    private = data.get("private_contracts", {})
    if (not isinstance(private, dict)
            or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol)
                   or not isinstance(sig, list) or not all(isinstance(k, str) for k in sig)
                   for symbol, sig in private.items())):
        raise RuntimeError(f"Invalid private ABI contracts in {path}")
    private = {s: tuple(sig) for s, sig in private.items()}
    if data.get("files") != _file_hashes(vendor, private=bool(private)):
        raise RuntimeError(f"Stale or invalid ABI manifest: {path} (library/header hashes differ)")
    actual = _contracts(vendor)
    recorded = data.get("contracts")
    if not isinstance(recorded, dict) or any(
        not isinstance(v, list) or not all(isinstance(kind, str) for kind in v)
        for v in recorded.values()
    ):
        raise RuntimeError(f"Invalid ABI contracts in {path}")
    contracts = {s: tuple(v) for s, v in recorded.items()}
    if private.keys() & (actual.keys() | contracts.keys()):
        raise RuntimeError(f"Public/private ABI collision: {path}")
    public = contracts
    contracts = {**public, **private}
    _validate_exports(set(contracts), contracts)
    exports = _check_stamp(elf_exported_symbols(vendor / LIBRARY, private), private)
    if set(contracts) != exports:
        raise RuntimeError(
            f"Incomplete ABI manifest: {path}; "
            f"missing exports={sorted(exports - contracts.keys())}, "
            f"unexported contracts={sorted(contracts.keys() - exports)}"
        )
    if any(actual.get(s) != sig for s, sig in public.items()):
        raise RuntimeError(f"ABI manifest disagrees with public headers: {path}")
    return public, private


def validated_contracts(vendor: Path, *, require_manifest: bool) -> dict[str, tuple[str, ...]]:
    public, private = validated_contract_groups(vendor, require_manifest=require_manifest)
    return {**public, **private}


def compare_overlay_to_base(overlay_dir: Path, base_dir: Path) -> list[str]:
    """Validate an overlay and reject changes to contracts shared with the base.

    Additional interfaces are allowed after manifest and two-phase validation.
    They do not replace any base entry, nor add Python/Stable adapter support.
    """

    return compare_overlays_to_base([overlay_dir], base_dir)


def compare_overlays_to_base(overlay_dirs: list[Path], base_dir: Path) -> list[str]:
    """Allow additions, but require one ABI per symbol across the active stack."""
    seen = {}
    verified = {}
    private_names = set()
    problems = []
    for vendor in [base_dir, *overlay_dirs]:
        try:
            public, private = validated_contract_groups(vendor, require_manifest=vendor != base_dir)
        except (OSError, ValueError, RuntimeError) as exc:
            problems.append(f"{vendor.name}: {exc}")
            continue
        contracts = {**public, **private}
        verified[vendor] = contracts
        private_names.update(private)
        for symbol, signature in sorted(contracts.items()):
            if symbol in seen:
                previous, owner = seen[symbol]
                if previous != signature:
                    problems.append(
                        f"{symbol}: {vendor.name}={list(signature)} {owner}={list(previous)}"
                    )
            else:
                seen[symbol] = (signature, vendor.name)
    # A legacy library may physically export an Inner name without a contract.
    # Only a higher-priority such library can shadow a verified provider.
    needed = set(verified.get(base_dir, {})) & private_names
    for vendor in reversed(overlay_dirs):
        if vendor not in verified:
            continue
        unknown = needed - verified[vendor].keys()
        if unknown:
            try:
                shadowed = elf_exported_symbols(vendor / LIBRARY, unknown) & unknown
            except (OSError, ValueError, RuntimeError) as exc:
                problems.append(f"{vendor.name}: {exc}")
            else:
                if shadowed:
                    problems.append(f"{vendor.name}: undeclared private ABI exports {sorted(shadowed)}")
        needed.update(private_names & verified[vendor].keys())
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a vendor ABI manifest before packaging")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-manifest", type=Path)
    action.add_argument("--write-stamp-source", type=Path)
    action.add_argument("--list-private-headers", type=Path)
    parser.add_argument("--private-declarations", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    if args.list_private_headers:
        print(";".join(str(p) for p in private_headers(args.list_private_headers).values()))
    elif args.write_stamp_source:
        stamp = private_stamp(private_contracts(args.private_declarations))
        args.write_stamp_source.write_text(
            f'extern "C" __attribute__((visibility("default"))) void {stamp}() {{}}\n',
            encoding="utf-8")
    else:
        print(write_manifest(args.write_manifest, args.private_declarations))
