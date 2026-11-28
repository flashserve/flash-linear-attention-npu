"""Install a CPU build of the chosen torch release and a compatible torch_npu."""

import argparse
import re
import subprocess
import sys

from stable_build_policy import RUNTIME_REQUIREMENTS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("torch_version", help="public torch version, e.g. 2.12.0")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9][0-9A-Za-z.!_-]*", args.torch_version):
        parser.error("supply a single public torch version without a build suffix")
    return subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--extra-index-url", "https://download.pytorch.org/whl/cpu",
        *RUNTIME_REQUIREMENTS, f"torch=={args.torch_version}+cpu",
    ]).returncode


if __name__ == "__main__":
    sys.exit(main())
