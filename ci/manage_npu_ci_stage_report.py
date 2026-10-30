#!/usr/bin/env python3
"""Create and update the structured per-platform NPU CI stage report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any


SCHEMA = "npu-ci-stage-report-v1"
STAGES = (
    "environment-contracts",
    "opp-package",
    "standalone-layout",
    "torch-adapter",
    "gdr-example-st",
    "scoped-overlay",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    if os.name == "posix":
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _now() -> tuple[str, int]:
    return datetime.now(timezone.utc).isoformat(), time.monotonic_ns()


def _persist(path: Path, payload: dict[str, Any], event: dict[str, Any]) -> None:
    archive_name = os.environ.get("CI_TIMING_ARCHIVE_DIR")
    if archive_name:
        archive = Path(archive_name)
        if not archive.is_dir():
            raise ValueError(f"CI timing archive directory does not exist: {archive}")
        record = {"metadata": payload["metadata"], **event}
        encoded = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(archive / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, encoded)
            os.fsync(fd)
        finally:
            os.close(fd)
        _write(archive / "stages.json", payload)
    _write(path, payload)


def initialize(path: Path) -> None:
    timestamp, monotonic_ns = _now()
    payload = {
        "schema": SCHEMA,
        "complete": False,
        "metadata": _metadata(),
        "status": "running",
        "stages": {
            stage: {"status": "not_run", "exit_code": None, "reason": "",
                    "started_at": None, "finished_at": None, "duration_seconds": None,
                    "started_at_monotonic_ns": None}
            for stage in STAGES
        },
    }
    _persist(path, payload, {"kind": "report", "status": "initialized",
                             "timestamp": timestamp, "monotonic_ns": monotonic_ns})


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
    timestamp, monotonic_ns = _now()
    previous = payload["stages"][stage]
    started_at = previous.get("started_at")
    started_ns = previous.get("started_at_monotonic_ns")
    if status == "running" and not started_ns:
        started_at, started_ns = timestamp, monotonic_ns
    finished_at = timestamp if status in {"success", "failure"} else None
    duration_seconds = None
    if finished_at and started_ns is not None:
        duration_seconds = round((monotonic_ns - started_ns) / 1_000_000_000, 3)
    payload["stages"][stage] = {
        "status": status,
        "exit_code": exit_code,
        "reason": reason,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "started_at_monotonic_ns": started_ns,
    }
    _persist(path, payload, {"kind": "stage", "stage": stage, "status": status,
                             "exit_code": exit_code, "reason": reason,
                             "timestamp": timestamp, "monotonic_ns": monotonic_ns,
                             "duration_seconds": duration_seconds})


def finalize(path: Path) -> int:
    payload = _read(path)
    timestamp, monotonic_ns = _now()
    failed = any(item.get("status") == "failure" for item in payload["stages"].values())
    payload["complete"] = True
    payload["status"] = "failure" if failed else "success"
    _persist(path, payload, {"kind": "report", "status": payload["status"],
                             "timestamp": timestamp, "monotonic_ns": monotonic_ns})
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

