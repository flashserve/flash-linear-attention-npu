#!/usr/bin/env python3
"""Build an isolated Stable ABI development wheel from a verified experimental OPP."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
STABLE = ROOT / "torch_custom/fla_npu"
VENDOR = "fla_npu_experimental_transformer"
PACKAGE = "fla_npu_experimental"
LIBRARY = "libfla_npu_experimental_stable.so"
WRAPPERS = {"fused_recurrent_rwkv8": ("_rwkv8", "fused_recurrent_rwkv8")}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def selection(manifest):
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if data.get("mode") != "experimental" or data.get("vendor") != VENDOR:
        raise ValueError("Expected an experimental input manifest")
    records = data.get("operators", [])
    names = [op["name"] for op in records]
    if not names or len(names) != len(set(names)):
        raise ValueError("Experimental adapter selection must be nonempty and unique")
    identifiers = [name.replace("-", "_").replace(".", "_") for name in names]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Experimental names collide after Python identifier normalization")
    for record, name in zip(records, names):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError(f"Unsupported adapter identity: {name}")
        path = ROOT / "experimental/ops" / name / "adapter" / f"stable_{name}.cpp"
        if not path.is_file() or not path.resolve().is_relative_to(ROOT / "experimental"):
            raise ValueError(f"Missing or external experimental adapter: {path}")
        relative = "adapter/" + path.name
        if record.get("source_sha256", {}).get(relative) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Adapter differs from the OPP source snapshot: {name}; rebuild the run package")
    return data


def prepare(data, directory):
    adapters = directory / "adapters"
    adapters.mkdir(parents=True, exist_ok=True)
    # Each invocation owns a fresh temporary directory, so no stale adapters survive.
    for op in data["operators"]:
        name = op["name"]
        shutil.copy2(ROOT / "experimental/ops" / name / "adapter" / f"stable_{name}.cpp", adapters)
    return adapters


def check(data, adapters, vendor):
    abi = module("experimental_build_abi", STABLE / "fla_npu/_op_abi.py")
    abi.validated_contracts(vendor, require_manifest=True)
    checker = module("experimental_adapter_check", STABLE / "tools/op_abi_validate.py")
    calls = checker.adapter_calls(adapters)
    expected = {s for op in data["operators"] for s in op["aclnn_symbols"] if not s.endswith("GetWorkspaceSize")}
    if set(calls) != expected:
        raise ValueError(f"Adapter coverage differs from manifest: {set(calls)} != {expected}")
    subprocess.run([sys.executable, str(STABLE / "tools/op_abi_validate.py"),
                    "--src-dir", str(adapters), "--ops", ",".join(op["name"] for op in data["operators"]),
                    "--opp-include", str(vendor / "op_api/include/aclnnop")], check=True)


def registration_text(data):
    names = [op["name"] for op in data["operators"]]
    identifiers = [name.replace("-", "_").replace(".", "_") for name in names]
    text = (ROOT / "experimental/torch_custom/registration.cpp.in").read_text(encoding="utf-8")
    fields = {
        "ADAPTERS": '\n'.join(f'#include "adapters/stable_{name}.cpp"' for name in names),
        "SCHEMAS": '\n'.join(f'  if (active_ops.count("npu_{name}")) m.def(kSchema_{name});' for name in identifiers),
        "IMPLEMENTATIONS": '\n'.join(
            f'  if (active_ops.count("npu_{name}")) m.impl("npu_{name}", '
            f'&fla_npu_stable::stable::boxed_adapter<run_npu_{name}>);' for name in identifiers),
        "EXPORTS": ', '.join(json.dumps("npu_" + name) for name in identifiers),
        "BINDINGS": '\n'.join(
            f'    if (active_ops.count("npu_{name}")) bind_experimental_symbols(paths, {{'
            + ', '.join(json.dumps(s) for s in sorted(op.get("aclnn_symbols", []))) + '});'
            for name, op in zip(identifiers, data["operators"])),
    }
    for name, value in fields.items():
        text = text.replace("@" + name + "@", value)
    return text


def registration(data, adapters, path):
    path.write_text(registration_text(data), encoding="utf-8")


def fingerprint(data):
    paths = [STABLE / "csrc/src/runtime.cpp", Path(__file__), STABLE / "csrc/build_stable.py", STABLE / "fla_npu/_op_abi.py"]
    paths += list((STABLE / "csrc/include").rglob("*.h"))
    paths += list((ROOT / "experimental/torch_custom").glob("*.*"))
    paths += [STABLE / "fla_npu_wheel_overlay.py"]
    paths += list((ROOT / "experimental/_python" / PACKAGE).glob("*.py"))
    paths += [ROOT / "experimental/ops" / op["name"] / "adapter" / f"stable_{op['name']}.cpp" for op in data["operators"]]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for p in sorted(paths)}
    hashes["experimental/_framework/generated_ops.cpp"] = hashlib.sha256(registration_text(data).encode()).hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def wheel_version(package):
    """Identify the assembled payload, independently of the launcher ABI stamp."""
    paths = [p for p in package.rglob("*") if p.is_file()]
    paths.append(package.parent / "fla_npu_experimental_manage.py")
    hashes = {p.relative_to(package.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return "0.1.0.dev0+" + digest[:12]


def write_layer_metadata(package, data, vendor, stamp, tier):
    abi = module("experimental_layer_abi", STABLE / "fla_npu/_op_abi.py")
    contracts = abi.validated_contracts(vendor, require_manifest=True)
    operators = {}
    for op in data["operators"]:
        name = "npu_" + op["name"].replace("-", "_").replace(".", "_")
        operators[name] = {"requires": {s: list(contracts[s]) for s in op["aclnn_symbols"]},
                           "dispatch": [name]}
    manifest = {"protocol": 1, "tier": tier, "source_hash": stamp,
                "exports": sorted(operators), "operators": operators}
    (package / "layer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (package / "_stable_hash.py").write_text(
        f"LAYER_PROTOCOL = 1\nSOURCE_HASH = {stamp!r}\n", encoding="utf-8")


def build_wheel(data, install_root, output, compiler):
    audit = module("experimental_package_audit", ROOT / "experimental/scripts/experimental_manifest.py")
    if len(data["soc"]) != 1:
        raise ValueError("Build one SoC per experimental development wheel")
    audit.audit_package(install_root, data["soc"][0])
    vendor = install_root / "vendors" / VENDOR
    if json.loads((vendor / "experimental-build-manifest.json").read_text()) != data:
        raise ValueError("Installed OPP differs from the requested build manifest")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fla-experimental-build-") as tmp:
        stage = Path(tmp)
        package = stage / PACKAGE
        shutil.copytree(ROOT / "experimental/_python" / PACKAGE, package,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so", "_build.json"))
        selected = {op["name"] for op in data["operators"]}
        wrappers = WRAPPERS
        if selected - wrappers.keys():
            raise ValueError(f"Missing Python wrapper inventory: {sorted(selected - wrappers.keys())}")
        for name, (filename, _) in wrappers.items():
            if name not in selected:
                (package / (filename + ".py")).unlink(missing_ok=True)
        (package / "_exports.py").write_text(
            "EXPORTS = " + repr({wrappers[name][1]: wrappers[name][0] for name in sorted(selected)}) + "\n",
            encoding="utf-8")
        shutil.copytree(install_root, package / "opp", symlinks=True)
        shutil.copy2(STABLE / "fla_npu/_op_abi.py", package / "_op_abi.py")
        adapters = prepare(data, stage)
        check(data, adapters, vendor)
        unit = stage / "experimental_ops.cpp"
        registration(data, adapters, unit)
        stamp, hashes = fingerprint(data)
        builder = module("experimental_private_builder", ROOT / "experimental/torch_custom/build_stable.py")
        builder.compile_launcher([unit, STABLE / "csrc/src/runtime.cpp"], package / LIBRARY, stamp,
            compiler=compiler, extra_flags=["-I" + str(ROOT / "experimental/torch_custom"),
            "-Dfla_npu_stable=fla_npu_experimental_runtime",
            "-Dfla_npu_stable_overlay_order_version=fla_npu_experimental_overlay_order_version"])
        # Keep the same ELF and C ABI floor as main; never silently relax it.
        subprocess.run([sys.executable, str(STABLE / "tools/stable_abi_audit.py"),
                        "--lib", str(package / LIBRARY), "--source", str(unit),
                        *map(str, adapters.glob("*.cpp"))], check=True)
        build = {"source_hash": stamp, "source_files": hashes, "operators": data["operators"],
                 "launcher_sha256": hashlib.sha256((package / LIBRARY).read_bytes()).hexdigest(),
                 "source_commit": data["source_commit"], "source_dirty": data["source_dirty"]}
        (package / "_build.json").write_text(json.dumps(build, indent=2), encoding="utf-8")
        artifacts = module("experimental_artifacts", ROOT / "scripts/fla_npu_artifacts.py")
        tier = artifacts.get_tier(data["soc"][0])
        write_layer_metadata(package, data, vendor, stamp, tier)
        shutil.copy2(STABLE / "fla_npu_wheel_overlay.py", package / "_wheel_overlay.py")
        shutil.copy2(ROOT / "experimental/_python/fla_npu_experimental_manage.py", stage)
        dist = "flash-linear-attention-npu-experimental-" + tier
        version = wheel_version(package)
        (stage / "setup.py").write_text(
            'from setuptools import setup, Distribution\n'
            'from wheel.bdist_wheel import bdist_wheel\n'
            'class Binary(Distribution):\n    def has_ext_modules(self): return True\n'
            'class Wheel(bdist_wheel):\n'
            '    def get_tag(self): return ("py3", "none", super().get_tag()[2])\n'
            f'setup(name={dist!r}, version={version!r}, packages=[{PACKAGE!r}], '
            f'package_data={{{PACKAGE!r}: ["**/*"]}}, py_modules=["fla_npu_experimental_manage"], distclass=Binary, '
            'cmdclass={"bdist_wheel": Wheel}, install_requires=["torch>=2.7.1", "torch_npu>=2.7.1"])\n', encoding="utf-8")
        subprocess.run([sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(output.resolve())], cwd=stage, check=True)


def package_options():
    """Resolve the same last-wins selection passed to build.sh, before compiling."""
    artifacts = module("experimental_artifacts", ROOT / "scripts/fla_npu_artifacts.py")
    extra = shlex.split(os.getenv("FLA_NPU_BUILD_ARGS", ""))
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--soc", "-c", "--compute-unit", default=artifacts.get_soc())
    parser.add_argument("--ops", "-n", "--op-name", default=os.getenv("FLA_NPU_OPS", ""))
    options, _ = parser.parse_known_args(extra)
    soc = options.soc.lower()
    if soc not in {"ascend910b", "ascend910_93", "ascend950"}:
        raise ValueError(f"Experimental wheel requires one supported SoC, got {soc!r}")
    checker = module("experimental_build_ops", ROOT / "experimental/scripts/check_build_ops.py")
    supported = checker.discover_supported_ops(ROOT)
    if not checker.common.validate_ops_filter(options.ops, supported=supported):
        raise ValueError("Invalid experimental operator selection")
    names = supported if checker.common.is_build_all(options.ops) else checker.common.parse_ops_filter(options.ops)
    if not names:
        raise ValueError("No experimental operators found")
    for name in names:
        adapter = ROOT / "experimental/ops" / name / "adapter" / f"stable_{name}.cpp"
        if not adapter.is_file() or not adapter.resolve().is_relative_to(ROOT / "experimental"):
            raise ValueError(f"Missing or external experimental adapter: {adapter}")
        if name not in WRAPPERS or not (ROOT / "experimental/_python" / PACKAGE / (WRAPPERS[name][0] + ".py")).is_file():
            raise ValueError(f"Missing experimental Python wrapper: {name}")
    return soc, names, extra


def build_package(output, compiler):
    output = output.resolve()
    if any(output.is_relative_to(ROOT / name) for name in ("build", "build_out", "output")):
        raise ValueError("Wheel output must be outside build/, build_out/ and output/, which build.sh cleans; use dist/")
    soc, names, extra = package_options()
    env = {**os.environ, "FLA_NPU_SOC": soc, "FLA_NPU_DISABLE_PTH": "1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
    print(f"[fla-npu build] FLA_NPU_SOC={soc}; experimental operators={','.join(names)}", flush=True)
    subprocess.run([sys.executable, ROOT / "scripts/check_npu_env.py", "--build-only"], cwd=ROOT, env=env, check=True)
    dependencies = module("experimental_dependencies", ROOT / "experimental/_tests/check_environment.py")
    if dependencies.optional_atk()["ATK"] is None:
        print("[experimental][WARNING] ATK is not installed; optional ATK validation is unavailable.", flush=True)
    run = ROOT / "build_out" / ("fla_npu_experimental_linux-" + os.uname().machine + ".run")
    manifest = ROOT / "build/experimental-build-manifest.json"
    # Invalidate only expected inputs so an early-exit build cannot reuse old products.
    run.unlink(missing_ok=True)
    manifest.unlink(missing_ok=True)
    command = ["bash", "build.sh", "--pkg", "--experimental", f"--soc={soc}",
               "--ops=" + ",".join(names), *extra]
    print("[fla-npu build] START " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    print("[fla-npu build] DONE: build.sh", flush=True)
    if not run.is_file() or not manifest.is_file():
        raise RuntimeError("Build did not produce the experimental run package and manifest")
    data = selection(manifest)
    with tempfile.TemporaryDirectory(prefix="fla-experimental-opp-") as tmp:
        install = Path(tmp) / "opp"
        print(f"[fla-npu build] START --install-path={install}", flush=True)
        subprocess.run(["bash", str(run), "--install-path=" + str(install), "--quiet"], cwd=ROOT, env=env, check=True)
        print("[fla-npu build] DONE: --install-path", flush=True)
        print("[fla-npu build] START bdist_wheel", flush=True)
        build_wheel(data, install, output, compiler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["check", "wheel", "package"])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--build-dir", type=Path, default=ROOT / "build")
    p.add_argument("--install-root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--compiler", default=os.getenv("CXX", "g++"))
    a = p.parse_args()
    if a.command == "package":
        build_package(a.output or ROOT / "dist", a.compiler)
        return
    if a.manifest is None:
        p.error(f"{a.command} requires --manifest")
    data = selection(a.manifest)
    if a.command == "wheel":
        if a.install_root is None:
            p.error("wheel requires --install-root")
        build_wheel(data, a.install_root.resolve(), a.output or ROOT / "build_out/experimental-wheel", a.compiler)
    else:
        candidates = list(a.build_dir.glob(f"_CPack_Packages/**/packages/vendors/{VENDOR}"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one staged experimental vendor, found {candidates}")
        with tempfile.TemporaryDirectory() as tmp:
            check(data, prepare(data, Path(tmp)), candidates[0])


if __name__ == "__main__":
    main()
