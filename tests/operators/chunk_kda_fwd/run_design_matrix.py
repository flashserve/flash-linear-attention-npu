#!/usr/bin/env python3
"""Print a read-only execution plan for the Chapter 21 KDA matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.operators.chunk_kda_fwd.common.design_matrix import (  # noqa: E402
    EXPECTED_KIND_COUNTS,
    SUPPORTED_SOCS,
    materialize_tasks,
    summarize_tasks,
    validate_design_matrix,
)


def _parse_csv(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the canonical 300-case ChunkKdaFwd design matrix. "
            "This command never launches ATK, msopprof, or sanitizer."
        )
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--kind",
        type=_parse_csv,
        default=(),
        help="comma-separated accuracy,run,msopprof,stress,sanitizer filter",
    )
    parser.add_argument("--soc", choices=SUPPORTED_SOCS)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    unknown_kinds = set(args.kind).difference(EXPECTED_KIND_COUNTS)
    if unknown_kinds:
        parser.error("unknown execution kinds: {}".format(", ".join(sorted(unknown_kinds))))

    try:
        validation = validate_design_matrix()
        tasks = materialize_tasks(case_ids=args.case_id, kinds=args.kind, soc=args.soc)
    except ValueError as error:
        parser.error(str(error))
    payload = {
        "mode": "read_only_plan",
        "validation": validation,
        "summary": summarize_tasks(tasks),
        "tasks": [] if args.summary else tasks,
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=args.indent, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
