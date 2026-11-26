#!/usr/bin/env python3
"""Verify the complete daily matrix through the public Huawei mirror."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from daily_plan import validate_plan
from npu_platforms import ARCH_TO_WHEEL, SOC_TO_PACKAGE
from daily_version import version_for, wheel_name, wheel_identity


DEFAULT_BASE = "https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly"
HREF_RE = re.compile(r"href=[\"']([^\"']+)[\"']", re.IGNORECASE)


def public_project(package: str) -> str:
    return package.replace("_", "-")


def fetch(url: str, timeout: int = 30,
          byte_limit: int | None = None) -> tuple[int, bytes]:
    headers = {"User-Agent": "fla-npu-daily-public-verifier"}
    if byte_limit is not None:
        headers["Range"] = f"bytes=0-{max(0, byte_limit - 1)}"
    request = urllib.request.Request(
        url,
        headers=headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read() if byte_limit is None else response.read(byte_limit)
            return int(response.status), body
    except urllib.error.HTTPError as exc:
        return int(exc.code), b""
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, b""


def directory_files(base: str, package: str, latest: bool = False) -> tuple[int, set[str]]:
    url = f"{base.rstrip('/')}/{public_project(package)}/" + ("latest/" if latest else "")
    status, body = fetch(url)
    if status != 200:
        return status, set()
    text = body.decode("utf-8", errors="replace")
    names: set[str] = set()
    for href in HREF_RE.findall(text):
        value = urllib.parse.unquote(html.unescape(href))
        name = Path(urllib.parse.urlsplit(value).path).name
        if name.endswith(".whl"):
            names.add(name)
    return status, names


def expected_rows(plan: dict) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for branch in validate_plan(plan):
        for soc, package in SOC_TO_PACKAGE.items():
            for arch, wheel_arch in ARCH_TO_WHEEL.items():
                rows.append({
                    "branch": branch["branch"],
                    "sha": branch["sha"],
                    "date": plan["date"],
                    "version": version_for(branch["branch"], plan["date"], branch["sha"]),
                    "soc": soc,
                    "package": package,
                    "arch": arch,
                    "wheel_arch": wheel_arch,
                })
    return rows


def match_filename(row: dict[str, str], names: set[str]) -> str:
    expected = wheel_name(row['package'], row['branch'], row['date'], row['sha'], row['wheel_arch'])
    return expected if expected in names else ""


def check_once(plan: dict, base: str) -> list[dict[str, object]]:
    inventory: dict[str, tuple[int, set[str]]] = {}
    results: list[dict[str, object]] = []
    for row in [dict(row, location=location) for row in expected_rows(plan)
                for location in ("history", "latest")]:
        package = row["package"]
        latest = row["location"] == "latest"
        key = (package, latest)
        if key not in inventory:
            inventory[key] = directory_files(base, package, latest)
        directory_status, names = inventory[key]
        filename = match_filename(row, names)
        # A formally released branch stops daily, but its last successful wheel
        # can remain in latest. Only require this active branch/CPU slot to be unique.
        slot_names = {name for name in names
                      if (identity := wheel_identity(name, package)) is not None
                      and (identity[0], identity[2]) == (row['branch'], row['wheel_arch'])}
        latest_exact = not latest or slot_names == {filename}
        file_status = 0
        if filename:
            url = (f"{base.rstrip('/')}/{public_project(package)}/"
                   + ("latest/" if latest else "") +
                   f"{urllib.parse.quote(filename, safe='+._-')}")
            file_status, _ = fetch(url, byte_limit=1)
        results.append({
            **row,
            "directory_status": directory_status,
            "filename": filename,
            "file_status": file_status,
            "latest_exact": latest_exact,
            "visible": bool(filename and file_status in (200, 206) and latest_exact),
        })
    return results


def write_report(path: Path | None, plan: dict, results: list[dict[str, object]],
                 status: str, attempts: int) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "fla-npu-daily-public-verify-v1",
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan_date": plan.get("date"),
        "status": status,
        "attempts": attempts,
        "expected": len(results),
        "visible": sum(bool(row["visible"]) for row in results),
        "results": results,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="检查 daily 各分支 × ARM/x86 × 全 NPU 平台公开可访问")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--timeout", type=int, default=21600,
                        help="等待公开镜像同步的最长秒数，默认 6 小时")
    parser.add_argument("--interval", type=int, default=300,
                        help="重试间隔秒数，默认 5 分钟")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if args.timeout < 0 or args.interval <= 0:
        parser.error("--timeout 必须 >=0，--interval 必须 >0")
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        validate_plan(plan)
    except Exception as exc:  # noqa: BLE001
        print(f"[public-verify][ERROR] plan 无效: {exc}", file=sys.stderr)
        return 2

    deadline = time.monotonic() + args.timeout
    attempts = 0
    while True:
        attempts += 1
        results = check_once(plan, args.base_url)
        visible = sum(bool(row["visible"]) for row in results)
        print(f"[public-verify] attempt={attempts} visible={visible}/{len(results)}",
              flush=True)
        if visible == len(results):
            write_report(args.report, plan, results, "success", attempts)
            print("[public-verify] SUCCESS：daily 全矩阵已可从公开镜像访问")
            return 0
        missing = [
            f"{row['branch']}/{row['soc']}/{row['arch']}/{row['location']}"
            for row in results if not row["visible"]
        ]
        print("[public-verify] pending: " + ", ".join(missing), flush=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            write_report(args.report, plan, results, "failed", attempts)
            print("[public-verify][ERROR] 等待超时，公开镜像仍缺少 daily 组合",
                  file=sys.stderr)
            return 1
        time.sleep(min(args.interval, remaining))


if __name__ == "__main__":
    raise SystemExit(main())
