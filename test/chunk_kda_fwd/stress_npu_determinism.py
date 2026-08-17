#!/usr/bin/env python3
"""Compatibility entrypoint for cached canonical NPU determinism stress."""

from __future__ import annotations

import sys

from canonical_execution_runner import main


if __name__ == "__main__":
    sys.argv.insert(1, "stress")
    raise SystemExit(main())
