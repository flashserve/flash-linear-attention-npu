"""Manage embedded OPP without importing fla_npu or loading vendor libraries.

Usage: python -m fla_npu_opp_manage --status / --overlay / --remove-overlay
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> int:
    # Resolve from this installed module, never from a CWD or environment override.
    path = Path(__file__).resolve().parent / "fla_npu" / "install_opp.py"
    spec = importlib.util.spec_from_file_location("_fla_npu_opp_installer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the OPP management module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
