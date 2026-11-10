"""Add compile-time entry markers to op_build-generated ACLNN source files.

Called after generation, before compilation; never edits CANN's generator or
headers. Hand-written ACLNN sources carry the same macro explicitly.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ENTRY = re.compile(r"\baclnnStatus\s+(aclnn\w+)\s*\([^;{}]*\)\s*\{")


def instrument(source: str) -> str:
    matches = list(ENTRY.finditer(source))
    if not matches:
        raise ValueError("No ACLNN function definitions found in generated source")
    for match in reversed(matches):
        if source[match.end():].lstrip().startswith("FLA_ACLNN_TRACE();"):
            continue
        source = source[:match.end()] + "\n    FLA_ACLNN_TRACE();" + source[match.end():]
    if '#include "fla_aclnn_trace.h"' not in source:
        source = '#include "fla_aclnn_trace.h"\n' + source
    return source


def main() -> None:
    for argument in sys.argv[1:]:
        path = Path(argument)
        if path.name.startswith("aclnn") and path.suffix == ".cpp":
            source = path.read_text(encoding="utf-8")
            updated = instrument(source)
            if updated != source:
                path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
