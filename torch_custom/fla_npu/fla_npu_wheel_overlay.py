"""Wheel layer storage and resolution. Safe to import before torch and CANN.

Only the explicit manager installs layers. Importing this module never executes
wheel code, loads shared libraries, or changes an installed base file.
"""
from __future__ import annotations

import ast
import base64
from contextlib import contextmanager
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
import zipfile

MANIFEST = "layer_manifest.json"
BASE = "fla_npu_transformer"
NAME = "fla_npu_overlay_transformer"
LIBRARY = "op_api/lib/libcust_opapi.so"
PACKAGE = "fla_npu"
LAUNCHER = "libfla_npu_stable.so"
STAMP = "ops/ascendc/_stable_hash.py"
OVERLAY_PREFIX = "fla_npu_overlay_"


def configure_layout(*, package, base_vendor, overlay_vendor, launcher, stamp, overlay_prefix):
    """Configure a private module instance before use; defaults preserve fla_npu.

    This describes payload paths, not a package identity or compatibility gate.
    Callers selecting another layout must load their own module instance.
    """
    global PACKAGE, BASE, NAME, LAUNCHER, STAMP, OVERLAY_PREFIX
    PACKAGE, BASE, NAME = package, base_vendor, overlay_vendor
    LAUNCHER, STAMP, OVERLAY_PREFIX = launcher, stamp, overlay_prefix


def package_dir():
    return Path(__file__).resolve().parent / PACKAGE


def storage(package):
    return package / "opp/wheel_layers"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def locked(package):
    root = storage(package)
    root.mkdir(parents=True, exist_ok=True)
    with (package / "opp/.overlay.lock").open("a") as handle:
        if os.name == "posix":
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(handle, fcntl.LOCK_UN)


def state(package):
    path = storage(package) / "state.json"
    data = _json(path) if path.is_file() else {"active": None, "pending": None}
    if not isinstance(data, dict) or data.get("version") not in (None, 2):
        raise ValueError("Invalid wheel overlay state; use --remove-overlay to recover")
    for kind in ("active", "pending"):
        if data.get(kind) is not None:
            layer_path(package, data[kind])
    return data


def layer_path(package, identity):
    if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{32}", identity):
        raise RuntimeError("Invalid wheel layer identity")
    root = storage(package).resolve()
    path = root / identity
    if path.is_symlink() or path.resolve().parent != root:
        raise RuntimeError("Wheel layer escapes its storage directory")
    return path


def enabled():
    selection = os.environ.get("FLA_NPU_OVERLAY_VENDOR")
    return selection is None or NAME in (part.strip() for part in selection.split(os.pathsep))


def active(package):
    identity = state(package).get("active") if enabled() else None
    if not identity:
        return None
    root = layer_path(package, identity)
    component_paths(package, root)
    return root


def components(package, snapshot):
    """Read immutable component selections; accept the original whole-wheel layout."""
    path = snapshot / "components.json"
    if not path.is_file():
        record = snapshot / "installation.json"
        if record.is_file() and _json(record).get("component_protocol") == 1:
            raise RuntimeError("Overlay component selections are missing; use --remove-overlay to recover")
        # Previously published wheel layers kept both components in one payload.
        if not (snapshot / PACKAGE / MANIFEST).is_file():
            raise RuntimeError("Overlay snapshot is damaged; use --remove-overlay to recover")
        return {"adapter": snapshot.name,
                "opp": snapshot.name if _vendor(snapshot / PACKAGE) else None}
    data = _json(path)
    if (not isinstance(data, dict) or data.get("version") != 1
            or set(data) != {"version", "adapter", "opp"}):
        raise RuntimeError("Invalid overlay component selections")
    result = {kind: data[kind] for kind in ("adapter", "opp")}
    for identity in result.values():
        if identity is not None:
            layer_path(package, identity)
    return result


def component_paths(package, snapshot):
    selected = components(package, snapshot)
    adapter = layer_path(package, selected["adapter"]) if selected["adapter"] else None
    opp = layer_path(package, selected["opp"]) if selected["opp"] else None
    if adapter and not (adapter / PACKAGE / MANIFEST).is_file():
        raise RuntimeError("Selected adapter is missing; use --remove-overlay to recover")
    vendor = _vendor(opp / PACKAGE) if opp else None
    if opp and vendor is None:
        raise RuntimeError("Selected OPP is missing; use --remove-overlay to recover")
    return adapter, vendor


def environment_vendors(package):
    """Return the active component order, or None for an unmanaged installation.

    Pending selections never configure CANN. Disabling a managed composition
    selects base, including after adopting an older vendor-directory overlay.
    """
    current = state(package)
    if not (current.get("active") or current.get("pending") or current.get("version") == 2):
        return None
    snapshot = active(package)
    overlay = component_paths(package, snapshot)[1] if snapshot else None
    return [p for p in (overlay, _vendor(package)) if p is not None]


def manifest(package):
    data = _json(package / MANIFEST)
    if not isinstance(data, dict) or data.get("protocol") != 1:
        raise RuntimeError("Wheel lacks layer protocol 1; rebuild its Python/Stable adapter package")
    exports, operators = data.get("exports"), data.get("operators")
    if (not isinstance(exports, list) or not exports or not all(isinstance(v, str) for v in exports)
            or len(exports) != len(set(exports))
            or not isinstance(operators, dict) or set(exports) - operators.keys()):
        raise RuntimeError("Invalid wheel layer operator inventory")
    for name, value in operators.items():
        if (not re.fullmatch(r"npu_[a-z0-9_]+", name) or not isinstance(value, dict)
                or not isinstance(value.get("requires"), dict)):
            raise RuntimeError("Invalid wheel layer operator contract")
        for symbol, signature in value["requires"].items():
            if not re.fullmatch(r"aclnn\w+", symbol) or not isinstance(signature, list) or not all(isinstance(v, str) for v in signature):
                raise RuntimeError("Invalid wheel layer ACLNN contract")
    assignments = {}
    for node in ast.parse((package / STAMP).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = ast.literal_eval(node.value)
    if assignments.get("LAYER_PROTOCOL") != 1 or assignments.get("SOURCE_HASH") != data.get("source_hash"):
        raise RuntimeError("Wheel launcher metadata does not match its layer manifest")
    if not (package / LAUNCHER).is_file():
        raise RuntimeError("Wheel has no Stable launcher")
    return data


def abi_module(package):
    spec = importlib.util.spec_from_file_location("_wheel_layer_abi", package / "_op_abi.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _vendor(package):
    directory = package / "opp/vendors" / BASE
    if not directory.resolve().is_relative_to(package.resolve()):
        raise RuntimeError("OPP vendor escapes its package")
    library = directory / LIBRARY
    if not library.resolve().is_relative_to(package.resolve()):
        raise RuntimeError("OPP library escapes its package")
    return directory if library.is_file() else None


def resolve(package, layer):
    """Return explicit adapter bindings and CANN order, or readiness errors.

    A custom symbol has one effective CANN implementation. Updated adapters
    must match it, including base adapters reached by an OPP-only update.
    """
    base = manifest(package)
    adapter, overlay_vendor = component_paths(package, layer)
    update = manifest(adapter / PACKAGE) if adapter else {"operators": {}, "exports": []}
    if adapter and base.get("tier") != update.get("tier"):
        raise RuntimeError("Wheel layer and base target different NPU products")
    abi = abi_module(package)
    base_vendor = _vendor(package)
    vendors = [p for p in (overlay_vendor, base_vendor) if p is not None]
    contracts = {str(p): abi.validated_contracts(p, require_manifest=p == overlay_vendor) for p in vendors}
    exports = set(update["exports"])
    errors, binding, affected = [], {}, set()
    if not vendors:
        errors.append("No usable OPP vendor; provide a matching --opp-source")
    for owner, data, names in (("overlay", update, exports),
                               ("base", base, set(base["operators"]) - exports)):
        required = {}
        for name in sorted(names):
            for symbol, expected in data["operators"][name]["requires"].items():
                chosen = next((p for p in vendors if symbol in contracts[str(p)]), None)
                if owner == "base" and (overlay_vendor is None or chosen != overlay_vendor):
                    # This update does not change the existing binding. Do not
                    # turn installation into an unrelated full-base ABI audit.
                    continue
                if owner == "base":
                    affected.add(name)
                if symbol in required and required[symbol] != expected:
                    errors.append(f"{owner}/{name}: conflicting requirement for {symbol}")
                required[symbol] = expected
                if chosen is None:
                    if owner == "overlay":
                        errors.append(f"{owner}/{name}: missing OPP symbol {symbol}")
                    continue  # A skeleton base may have unavailable, uncovered ops.
                if tuple(expected) != contracts[str(chosen)][symbol]:
                    errors.append(f"{owner}/{name}: {symbol} mismatches {chosen.name}; include its matching adapter/OPP in the update")
        # All reachable adapters use the same validated CANN ownership order.
        # The binding is explicit and immutable, never read from a later env edit.
        binding[owner] = [str(p / LIBRARY) for p in vendors]
    return {"vendors": [str(p) for p in vendors], "bindings": binding,
            "exports": sorted(exports), "errors": sorted(set(errors)),
            "layer": str(adapter) if adapter else None,
            "components": components(package, layer),
            "affected_base_operators": sorted(affected)}


def runtime_plan(package):
    layer = active(package)
    if layer is None:
        return None
    plan = resolve(package, layer)
    if plan["errors"]:
        raise RuntimeError("Active wheel overlay is no longer compatible:\n" + "\n".join(plan["errors"]))
    plan["snapshot"] = str(layer)
    return plan


def _payload_name(name):
    path = PurePosixPath(name)
    if "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe wheel member: {name}")
    parts = path.parts
    if len(parts) > 2 and parts[0].endswith(".data") and parts[1] in ("purelib", "platlib"):
        parts = parts[2:]
    return PurePosixPath(*parts)


def extract_wheel(wheel, destination):
    """Apply wheel purelib/platlib mapping without executing .pth/install code."""
    with zipfile.ZipFile(wheel) as archive:
        validate_wheel_environment(archive)
        records = [n for n in archive.namelist() if n.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise RuntimeError("Expected one wheel RECORD")
        record = {row[0]: row[1:] for row in csv.reader(io.StringIO(archive.read(records[0]).decode()))}
        seen = set()
        for item in archive.infolist():
            mapped = _payload_name(item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise RuntimeError("Wheel symlinks are unsupported")
            if item.is_dir() or not mapped.parts or mapped.parts[0] != PACKAGE:
                continue
            if mapped in seen:
                raise RuntimeError(f"Duplicate wheel payload: {mapped}")
            seen.add(mapped)
            data = archive.read(item)
            expected = record.get(item.filename)
            if not expected or len(expected) != 2 or not expected[0].startswith("sha256="):
                raise RuntimeError(f"Missing SHA256 RECORD entry: {item.filename}")
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            if expected != ["sha256=" + digest, str(len(data))]:
                raise RuntimeError(f"Wheel RECORD mismatch: {item.filename}")
            target = destination.joinpath(*mapped.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    manifest(destination / PACKAGE)


def validate_wheel_environment(archive):
    from email.parser import Parser
    import importlib.metadata
    import sys
    try:
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        from packaging.tags import parse_tag, sys_tags
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
        from pip._vendor.packaging.specifiers import SpecifierSet
        from pip._vendor.packaging.tags import parse_tag, sys_tags
    wheels = [n for n in archive.namelist() if n.endswith(".dist-info/WHEEL")]
    if len(wheels) != 1:
        raise RuntimeError("Expected one WHEEL metadata file")
    fields = Parser().parsestr(archive.read(wheels[0]).decode())
    tags = set().union(*(parse_tag(t) for t in fields.get_all("Tag", [])))
    if not tags.intersection(sys_tags()):
        raise RuntimeError("Wheel platform/Python tags do not match this environment")
    metadata = Parser().parsestr(archive.read(wheels[0].removesuffix("WHEEL") + "METADATA").decode())
    required_python = metadata.get("Requires-Python")
    if required_python and not SpecifierSet(required_python).contains(".".join(map(str, sys.version_info[:3]))):
        raise RuntimeError("Wheel requires a different Python version")
    for value in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(value)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Install wheel dependency first: {requirement}") from exc
        if requirement.url or not requirement.specifier.contains(version, prereleases=True):
            raise RuntimeError(f"Wheel dependency {requirement} does not match installed {version}")


def _attach_opp(source, destination):
    source = source.resolve()
    with tempfile.TemporaryDirectory(prefix="fla-layer-opp-") as temporary:
        if source.is_file() and source.suffix == ".run":
            subprocess.run(["bash", str(source), "--install-path=" + temporary, "--quiet"], check=True)
            source = Path(temporary) / "vendors"
        candidates = ([source] if (source / LIBRARY).is_file() else
                      [p for p in source.iterdir() if p.is_dir() and (p / LIBRARY).is_file()])
        if len(candidates) != 1:
            raise RuntimeError("--opp-source must contain exactly one OPP vendor")
        target = destination / PACKAGE / "opp/vendors" / BASE
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(candidates[0], target)


def _adopt_vendor_overlay(package, previous, created):
    """Copy a pre-component run overlay without changing its live directory."""
    if previous.get("active") or previous.get("pending") or previous.get("version") == 2:
        return previous
    candidates = sorted((package / "opp/vendors").glob(OVERLAY_PREFIX + "*"))
    if not candidates:
        return previous
    if len(candidates) != 1:
        raise RuntimeError("Multiple legacy vendor overlays exist; remove extras before composing components")
    source = candidates[0]
    if (not re.fullmatch(re.escape(OVERLAY_PREFIX) + r"[A-Za-z0-9][A-Za-z0-9_.-]*", source.name)
            or source.is_symlink() or not source.is_dir()):
        raise RuntimeError("Invalid legacy overlay vendor")
    identity = uuid.uuid4().hex
    target = layer_path(package, identity)
    target.mkdir()
    created.append(target)
    _attach_opp(source, target)
    _atomic_json(target / "installation.json", {"component_protocol": 1, "source": str(source)})
    _atomic_json(target / "components.json", {"version": 1, "adapter": None, "opp": identity})
    plan = resolve(package, target)
    if plan["errors"]:
        raise RuntimeError("Existing vendor overlay is incompatible: " + "; ".join(plan["errors"]))
    return dict(previous, active=identity)


def install(package, source, *, opp_source=None, force=False):
    """Replace only components supplied by this artifact; last installation wins.

    Pending is the latest requested combination and active is the last runnable
    one. Component payloads are immutable and shared by snapshots. ``force`` is
    accepted for existing callers; component replacement no longer requires it.
    """
    manifest(package)  # Requires the new controller, not a complete base OPP.
    source = Path(source)
    is_wheel = source.suffix == ".whl"
    if opp_source is not None and not is_wheel:
        raise RuntimeError("--opp-source requires a wheel --source")
    with locked(package):
        previous = state(package)
        created = []
        try:
            previous = _adopt_vendor_overlay(package, previous, created)
            latest = previous.get("pending") or previous.get("active")
            selected = (components(package, layer_path(package, latest)) if latest
                        else {"adapter": None, "opp": None})
            previous_opp = (layer_path(package, selected["opp"]) / "fla_npu/opp/vendors" / BASE
                            if selected["opp"] else None)
            identity = uuid.uuid4().hex
            target = layer_path(package, identity)
            target.mkdir()
            created.append(target)
            if is_wheel:
                extract_wheel(source, target)
                selected["adapter"] = identity
                if opp_source is not None:
                    _attach_opp(opp_source, target)
                if _vendor(target / PACKAGE):
                    selected["opp"] = identity
            else:
                _attach_opp(source, target)
                selected["opp"] = identity
            _atomic_json(target / "components.json", {"version": 1, **selected})
            plan = resolve(package, target)
            base_opp = _vendor(package)
            installed_opp = _vendor(target / "fla_npu")
            _atomic_json(target / "installation.json", {"component_protocol": 1, "source": str(source.resolve()),
                         "opp_source": str(Path(opp_source).resolve()) if opp_source else None,
                         "errors": plan["errors"]})
            current = dict(previous, version=2)
            if plan["errors"]:
                current["pending"] = identity
            else:
                current.update(active=identity, pending=None)
            _atomic_json(storage(package) / "state.json", current)
        except BaseException:
            # A signal may arrive immediately after os.replace committed the
            # pointer. Never remove a payload that the state already references.
            try:
                current = state(package)
                committed = any(p.name in (current.get("active"), current.get("pending")) for p in created)
            except (OSError, ValueError, RuntimeError):
                committed = True  # Preserve evidence if commit status is unknown.
            if not committed:
                for path in reversed(created):
                    shutil.rmtree(path)
            raise
        # Never delete the old active payload during installation: running
        # processes may still map it. Explicit removal reclaims retained layers.
        return {"state": "pending" if plan["errors"] else "active", "path": str(target),
                "operators": plan["exports"], "components": plan["components"],
                "affected_base_operators": plan["affected_base_operators"], "errors": plan["errors"],
                "installation": {
                    "artifact": "wheel" if is_wheel else "run",
                    "base_opp": str(base_opp) if base_opp else None,
                    "previous_overlay_opp": str(previous_opp) if previous_opp else None,
                    "previous_state": "pending" if previous.get("pending") else "active",
                    "installed_opp": str(installed_opp) if installed_opp else None,
                    "selection": "overlay" if enabled() and current.get("active") else "base",
                }}


def remove(package):
    with locked(package):
        _atomic_json(storage(package) / "state.json", {"version": 2, "active": None, "pending": None})
        for child in storage(package).iterdir():
            if re.fullmatch(r"[a-f0-9]{32}", child.name):
                path = layer_path(package, child.name)
                if path.is_dir():
                    shutil.rmtree(path)
        # A pre-component vendor may have been adopted. Do not resurrect it on
        # removal, including recovery from an unreadable state.json.
        vendors = package / "opp/vendors"
        for path in vendors.glob(OVERLAY_PREFIX + "*"):
            if (re.fullmatch(re.escape(OVERLAY_PREFIX) + r"[A-Za-z0-9][A-Za-z0-9_.-]*", path.name)
                    and path.is_dir() and not path.is_symlink()
                    and path.resolve().parent == vendors.resolve()):
                shutil.rmtree(path)


def status(package):
    result = []
    try:
        current = state(package)
    except (OSError, ValueError, RuntimeError) as exc:
        return [{"state": "damaged", "errors": [str(exc)]}]
    for kind in ("active", "pending"):
        identity = current.get(kind)
        if identity:
            path = layer_path(package, identity)
            try:
                plan = resolve(package, path)
                result.append({"state": kind, "path": str(path), "operators": plan["exports"],
                               "components": plan["components"],
                               "affected_base_operators": plan["affected_base_operators"],
                               "errors": plan["errors"]})
            except (OSError, ValueError, RuntimeError) as exc:
                result.append({"state": kind, "path": str(path), "errors": [str(exc)]})
    return result


def shell_environment(package):
    """The manual source path uses the same selection as the .pth hook."""
    import shlex
    vendors = environment_vendors(package)
    if vendors is None:
        return ""
    root = (package / "opp").resolve()
    external = []
    for part in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep):
        if part and not Path(part).resolve().is_relative_to(root):
            external.append(part)
    values = {"ASCEND_CUSTOM_OPP_PATH": os.pathsep.join([str(p) for p in vendors] + external),
              "FLA_NPU_OP_API_LIBS": os.pathsep.join(str(p / LIBRARY) for p in vendors)}
    return "\n".join("export " + key + "=" + shlex.quote(value) for key, value in values.items())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare wheel layer paths for manual source")
    parser.add_argument("--shell-env", type=Path, required=True)
    print(shell_environment(parser.parse_args().shell_env.resolve()))
