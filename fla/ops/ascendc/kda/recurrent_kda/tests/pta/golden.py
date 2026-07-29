"""Compatibility wrapper for the shared recurrent KDA CPU reference."""

from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[8]
sys.path.insert(0, str(ROOT))

from tests.reference.recurrent_kda_reference import recurrent_kda_reference as recurrent_kda_golden  # noqa: E402
