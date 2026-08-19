#!/usr/bin/env python3
"""Forward to the torch_custom precision test for ChunkScaledDotKkt."""

from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        target = parent / "torch_custom" / "fla_npu" / "test" / "test_npu_chunk_scaled_dot_kkt.py"
        if target.exists():
            return parent
    raise RuntimeError("Cannot locate torch_custom/fla_npu/test/test_npu_chunk_scaled_dot_kkt.py")


repo_root = _find_repo_root()
sys.path.insert(0, str(repo_root / "torch_custom" / "fla_npu" / "test"))

from test_npu_chunk_scaled_dot_kkt import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
