#!/usr/bin/env python3
"""Compatibility entry forwarding to the canonical ChunkScaledDotKkt backend."""

from __future__ import annotations

import sys
from pathlib import Path


ACCURACY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ACCURACY_DIR))

from backend import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
