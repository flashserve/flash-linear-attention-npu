#!/usr/bin/env python3
"""Create and update the structured per-platform NPU CI stage report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


SCHEMA = "npu-ci-stage-report-v1"
STAGES = (
    "environment-contracts",
    "opp-package",
    "standalone-layout",
    "torch-adapter",
    "gdr-example-st",
)
STAGE_STATUSES = {"not_run", "running", "success", "failure", "skipped"}


def _metadata() -> dict[str, str]:
    return {
        "platform": os.environ.get("CI_ACCURACY_PLATFORM", ""),
        "soc": os.environ.get("CI_SOC", ""),
        "head_sha": os.environ.get("CI_ACCURACY_HEAD_SHA", ""),
        "run_id": os.environ.get("CI_ACCURACY_RUN_ID", ""),
        "run_attempt": os.environ.get("CI_ACCURACY_RUN_ATTEMPT", ""),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    _update_timing_table(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _update_timing_table(payload: dict[str, Any]) -> None:
    filename = os.environ.get("CI_STAGE_TIMING_FILE")
    if not filename:
        return
    import fcntl  # The persistent table is written on Linux CI runners.

    path = Path(filename)
    metadata = payload["metadata"]
    keys = ("run_id", "run_attempt", "platform")
    identity = tuple(metadata[key] for key in keys)
    if not all(identity):
        raise ValueError("persistent stage timings require run_id, run_attempt and platform")
    fields = [*keys, "head_sha", "soc", "stage", "status", "exit_code",
              "started_at", "finished_at", "duration_seconds"]
    # Lock a separate inode because the CSV is atomically replaced. Each run
    # updates only its own rows, preserving concurrent runs and all history.
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != fields:
                    raise ValueError(f"unexpected stage timing table header: {path}")
                rows = [row for row in reader if tuple(row[key] for key in keys) != identity]
        for stage, item in payload["stages"].items():
            rows.append({field: {**metadata, **item, "stage": stage}.get(field, "")
                         for field in fields})
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                             dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fchmod(stream.fileno(), 0o644)
                os.fsync(stream.fileno())
            temporary.replace(path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def initialize(path: Path) -> None:
    payload = {
        "schema": SCHEMA,
        "complete": False,
        "metadata": _metadata(),
        "status": "running",
        "stages": {
            stage: {"status": "not_run", "exit_code": None, "reason": ""}
            for stage in STAGES
        },
    }
    _write(path, payload)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or not isinstance(payload.get("stages"), dict):
        raise ValueError(f"invalid NPU CI stage report: {path}")
    return payload


def update(path: Path, stage: str, status: str, exit_code: int | None, reason: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unsupported NPU CI stage: {stage}")
    if status not in STAGE_STATUSES:
        raise ValueError(f"unsupported NPU CI stage status: {status}")
    if status == "success" and exit_code != 0:
        raise ValueError("a successful stage must have exit code 0")
    if status == "failure" and (exit_code is None or exit_code == 0):
        raise ValueError("a failed stage must have a non-zero exit code")
    if status in {"not_run", "running", "skipped"} and exit_code is not None:
        raise ValueError(f"stage status {status} must not have an exit code")

    payload = _read(path)
    item = payload["stages"][stage]
    timestamp = datetime.now(timezone.utc).isoformat()
    now_ns = time.monotonic_ns()
    # Environment bootstrap and the stage dispatcher both report running.
    if status == "running" and item.get("started_at_monotonic_ns") is None:
        item.update(started_at=timestamp, started_at_monotonic_ns=now_ns)
    if status in {"success", "failure"}:
        item["finished_at"] = timestamp
        start_ns = item.get("started_at_monotonic_ns")
        item["duration_seconds"] = round((now_ns - start_ns) / 1e9, 3) if start_ns is not None else None
    item.update({
        "status": status,
        "exit_code": exit_code,
        "reason": reason,
    })
    _write(path, payload)


def finalize(path: Path) -> int:
    payload = _read(path)
    failed = any(item.get("status") == "failure" for item in payload["stages"].values())
    payload["complete"] = True
    payload["status"] = "failure" if failed else "success"
    _write(path, payload)
    return 1 if failed else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init")

    update_parser = subparsers.add_parser("update")
    update_parser.add_argument("--stage", required=True, choices=STAGES)
    update_parser.add_argument("--status", required=True, choices=sorted(STAGE_STATUSES))
    update_parser.add_argument("--exit-code", type=int)
    update_parser.add_argument("--reason", default="")

    subparsers.add_parser("finalize")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "init":
        initialize(args.output)
        return 0
    if args.command == "update":
        update(args.output, args.stage, args.status, args.exit_code, args.reason)
        return 0
    return finalize(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
