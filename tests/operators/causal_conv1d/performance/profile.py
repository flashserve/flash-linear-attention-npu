#!/usr/bin/env python3
"""Profile JSON performance cases for causal_conv1d with msopprof."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.operators.causal_conv1d.common.case_matrix import case_ids


RUNNER = ROOT / "tests/operators/causal_conv1d/accuracy/backend.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="outputs/causal_conv1d_msopprof")
    args = parser.parse_args()
    ids = case_ids(tag="performance", route="ascendc")
    if not ids:
        raise RuntimeError("no performance case is defined")
    env = os.environ.copy()
    env.update({
        "FLA_NPU_OPERATOR": "causal_conv1d",
        "FLA_NPU_CASE_MANIFEST": str(ROOT / "tests/op_cases/causal_conv1d.json"),
        "FLA_NPU_CASE_IDS": ",".join(ids),
    })
    application = f"{shlex.quote(sys.executable)} {shlex.quote(str(RUNNER))}"
    command = [
        "msopprof", f"--application={application}", f"--output={args.output}",
        "--aic-metrics=BasicInfo", "--launch-count=20", "--warm-up=5", "--kill=off",
    ]
    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return
    subprocess.run(command, cwd=RUNNER.parent, env=env, check=True)


if __name__ == "__main__":
    main()
