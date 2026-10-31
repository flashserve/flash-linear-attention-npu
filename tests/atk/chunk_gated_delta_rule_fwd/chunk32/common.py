"""Path helper imported by ``gpu_triton_fwd``."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_fla_on_path(fla_root: str | Path | None = None) -> Path | None:
    """Put an explicit flash-linear-attention checkout on ``sys.path``.

    An installed ``fla`` package needs no checkout. ``FLA_ROOT`` or ``fla_root``
    is used only when the GPU node does not have that package installed.
    """

    if fla_root is None:
        fla_root = os.environ.get("FLA_ROOT")
    if not fla_root:
        return None
    root = Path(fla_root).expanduser().resolve()
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)
    return root
