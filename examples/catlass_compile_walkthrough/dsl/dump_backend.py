#!/usr/bin/env python3
"""Re-run the recorded CATLASS 2.2.0 Cube backend with LLVM IR retention."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, help="build/dsl/artifacts.json")
    args = parser.parse_args()
    info = json.loads(args.artifacts.read_text(encoding="utf-8"))
    directory = args.artifacts.parent.resolve()
    cache = Path(info["cache_dir"])
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest["arch_scope"] != "aic.c310":
        raise ValueError("this inspection recipe is for the tutorial's Ascend 950 Cube kernel")
    bitcodes = manifest["helper_bitcode_files"]
    if not bitcodes or not all(Path(path).is_file() for path in bitcodes):
        raise FileNotFoundError("run run_dsl.sh to prepare the matching helper bitcode")
    temporary = directory / "backend-temps"
    temporary.mkdir(parents=True, exist_ok=True)
    output = directory / "recompiled.o"
    # These options match CATLASS 2.2.0's Cube compilation in execution.py.
    # --save-temps is the only additional diagnostic option.
    command = [
        manifest["hivmc"], str(directory / "02-hivm.mlir"),
        "--target=Ascend950PR_9589", "--disable-ffts", "--enable-hivm-compile=False",
        "--link-aicore-bitcode=" + ",".join(bitcodes),
        "--append-bisheng-options=-mllvm --cce-vf-auto-sync=fused -mllvm -cce-enable-dynamic-misched=true",
        "--save-temps=" + str(temporary), "-o", str(output),
    ]
    (directory / "backend-command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    subprocess.run(command, check=True, timeout=60)
    original = Path(info["kernel_binary_path"])
    report = {
        "llvm_ir": str(temporary / "recompiled.ll"),
        "original_sha256": sha256(original),
        "recompiled_sha256": sha256(output),
    }
    if not Path(report["llvm_ir"]).is_file():
        raise FileNotFoundError(report["llvm_ir"])
    report["binary_identical"] = report["original_sha256"] == report["recompiled_sha256"]
    (directory / "backend-check.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["binary_identical"]:
        raise RuntimeError("diagnostic binary differs; do not claim it matches the executed kernel")


if __name__ == "__main__":
    main()
