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
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
STABLE = ROOT / "torch_custom/fla_npu"
VENDOR = "fla_npu_experimental_transformer"
PACKAGE = "fla_npu_experimental"
LIBRARY = "libfla_npu_experimental_stable.so"


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
        path = ROOT / "experimental" / name / "adapter" / f"stable_{name}.cpp"
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
        shutil.copy2(ROOT / "experimental" / name / "adapter" / f"stable_{name}.cpp", adapters)
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
    text = '#include <torch/csrc/stable/library.h>\n'
    text += '\n'.join(f'#include "adapters/stable_{name}.cpp"' for name in names)
    text += '''
extern "C" __attribute__((visibility("default")))
const char* fla_npu_experimental_source_hash() { return FLA_STABLE_SOURCE_HASH; }
extern "C" __attribute__((visibility("default")))
int32_t fla_npu_experimental_queue_enqueue_available() {
  return fla_npu_stable::Runtime::instance().enqueue_enabled() ? 1 : 0;
}
STABLE_TORCH_LIBRARY(fla_npu_experimental_stable, m) {
'''
    identifiers = [name.replace("-", "_").replace(".", "_") for name in names]
    text += '\n'.join(f'  m.def(kSchema_{name});' for name in identifiers) + '\n}\n'
    text += 'STABLE_TORCH_LIBRARY_IMPL(fla_npu_experimental_stable, CompositeExplicitAutograd, m) {\n'
    text += '\n'.join(f'  m.impl("npu_{name}", &fla_npu_stable::stable::boxed_adapter<run_npu_{name}>);' for name in identifiers)
    return text + '\n}\n'


def registration(data, adapters, path):
    path.write_text(registration_text(data), encoding="utf-8")


def fingerprint(data):
    paths = [STABLE / "csrc/src/runtime.cpp", Path(__file__), STABLE / "csrc/build_stable.py", STABLE / "fla_npu/_op_abi.py"]
    paths += list((STABLE / "csrc/include").rglob("*.h"))
    paths += list((ROOT / "experimental/_python" / PACKAGE).glob("*.py"))
    paths += [ROOT / "experimental" / op["name"] / "adapter" / f"stable_{op['name']}.cpp" for op in data["operators"]]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for p in sorted(paths)}
    hashes["experimental/_framework/generated_ops.cpp"] = hashlib.sha256(registration_text(data).encode()).hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes


def build_wheel(data, install_root, output, compiler):
    audit = module("experimental_package_audit", ROOT / "scripts/experimental_manifest.py")
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
        shutil.copytree(install_root, package / "opp", symlinks=True)
        shutil.copy2(STABLE / "fla_npu/_op_abi.py", package / "_op_abi.py")
        adapters = prepare(data, stage)
        check(data, adapters, vendor)
        unit = stage / "experimental_ops.cpp"
        registration(data, adapters, unit)
        stamp, hashes = fingerprint(data)
        owned = ":" + ":".join(sorted(s for op in data["operators"] for s in op["aclnn_symbols"])) + ":"
        builder = module("experimental_shared_builder", STABLE / "csrc/build_stable.py")
        builder.compile_launcher([unit, STABLE / "csrc/src/runtime.cpp"], package / LIBRARY, stamp,
            compiler=compiler, extra_flags=["-DFLA_STABLE_EXPERIMENTAL", "-Dfla_npu_stable=fla_npu_experimental_runtime",
            "-Dfla_npu_stable_overlay_order_version=fla_npu_experimental_overlay_order_version",
            '-DFLA_EXPERIMENTAL_SYMBOLS="' + owned + '"'])
        # Keep the same ELF and C ABI floor as main; never silently relax it.
        subprocess.run([sys.executable, str(STABLE / "tools/stable_abi_audit.py"),
                        "--lib", str(package / LIBRARY), "--source", str(unit),
                        *map(str, adapters.glob("*.cpp"))], check=True)
        build = {"source_hash": stamp, "source_files": hashes, "operators": data["operators"],
                 "launcher_sha256": hashlib.sha256((package / LIBRARY).read_bytes()).hexdigest(),
                 "source_commit": data["source_commit"], "source_dirty": data["source_dirty"],
                 "python_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in package.glob("*.py")}}
        (package / "_build.json").write_text(json.dumps(build, indent=2), encoding="utf-8")
        artifacts = module("experimental_artifacts", ROOT / "scripts/fla_npu_artifacts.py")
        dist = "flash-linear-attention-npu-experimental-" + artifacts.get_tier(data["soc"][0])
        (stage / "setup.py").write_text(
            'from setuptools import setup, Distribution\n'
            'from wheel.bdist_wheel import bdist_wheel\n'
            'class Binary(Distribution):\n    def has_ext_modules(self): return True\n'
            'class Wheel(bdist_wheel):\n'
            '    def get_tag(self): return ("py3", "none", super().get_tag()[2])\n'
            f'setup(name={dist!r}, version={"0.1.0.dev0+" + stamp[:12]!r}, packages=[{PACKAGE!r}], '
            f'package_data={{{PACKAGE!r}: ["**/*"]}}, distclass=Binary, '
            'cmdclass={"bdist_wheel": Wheel}, install_requires=["torch>=2.7.1", "torch_npu>=2.7.1"])\n', encoding="utf-8")
        subprocess.run([sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(output.resolve())], cwd=stage, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["check", "wheel"])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, default=ROOT / "build")
    p.add_argument("--install-root", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "build_out/experimental-wheel")
    p.add_argument("--compiler", default=os.getenv("CXX", "g++"))
    a = p.parse_args()
    data = selection(a.manifest)
    if a.command == "wheel":
        if a.install_root is None:
            p.error("wheel requires --install-root")
        build_wheel(data, a.install_root.resolve(), a.output, a.compiler)
    else:
        candidates = list(a.build_dir.glob(f"_CPack_Packages/**/packages/vendors/{VENDOR}"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one staged experimental vendor, found {candidates}")
        with tempfile.TemporaryDirectory() as tmp:
            check(data, prepare(data, Path(tmp)), candidates[0])


if __name__ == "__main__":
    main()
