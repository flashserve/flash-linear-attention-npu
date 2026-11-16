"""Record the libraries and installed SolveTri kernel assets of a Linux worker.

Usage: python performance_provenance.py --pid WORKER_PID --output worker.json
The record captures installed kernel files; profiling evidence is still needed to
confirm which kernel was executed. Capture each ATK task's worker separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture(pid: int) -> dict:
    process = Path("/proc") / str(pid)
    env = dict(item.split("=", 1) for item in (process / "environ").read_bytes().decode().split("\0")
               if "=" in item)
    home = env.get("ASCEND_HOME_PATH")
    vendor_paths = [Path(value) for value in env.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if value]
    if not home or not vendor_paths:
        raise ValueError("Worker must declare ASCEND_HOME_PATH and ASCEND_CUSTOM_OPP_PATH")
    mapped = {line.split()[-1] for line in (process / "maps").read_text().splitlines()
              if line.split()[-1].startswith("/")}
    libraries = {name: _sha256(Path(name)) for name in sorted(mapped)
                 if name.endswith(("libcust_opapi.so", "libfla_npu_stable.so")) and Path(name).is_file()}
    if not any(Path(name).name == "libcust_opapi.so" for name in libraries):
        raise ValueError("Worker has not loaded libcust_opapi.so")
    assets = {}
    for vendor in vendor_paths:
        root = vendor / "op_impl" / "ai_core" / "tbe" / "kernel"
        for path in sorted(root.glob("*/solve_tri/*")):
            if path.is_file() and path.suffix in {".o", ".json"}:
                assets[str(path)] = _sha256(path)
    if not any(name.endswith(".o") for name in assets):
        raise ValueError("No installed SolveTri kernel object under ASCEND_CUSTOM_OPP_PATH")
    return {"pid": pid, "hostnames": sorted({socket.gethostname(), socket.getfqdn()}),
            "environment": {"ASCEND_HOME_PATH": home, "ASCEND_CUSTOM_OPP_PATH": os.pathsep.join(map(str, vendor_paths))},
            "mapped_files_sha256": libraries, "kernel_assets_sha256": assets}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = capture(args.pid)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, UnicodeError) as error:
        parser.exit(1, f"provenance capture failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
