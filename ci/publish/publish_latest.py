"""Keep the latest successful daily per branch/CPU, preserving history."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import time
import urllib.parse

from daily_plan import atomic_write, parse_iso, validate_plan
from daily_version import wheel_identity, wheel_name
from npu_platforms import ARCH_TO_WHEEL
from verify_daily_public import DEFAULT_BASE, expected_rows, fetch, public_project

BASE = Path(__file__).resolve().parent
PREFIX = "pypi/packages/ascend/repos/pypi/nightly"


def check_plan(plan: dict, arch: str, state_path: Path) -> None:
    validate_plan(plan)
    current_cutoff = parse_iso(plan.get('cutoff_utc', ''))
    if current_cutoff is None:
        raise RuntimeError('latest 计划缺少明确截点')
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        previous_cutoff = parse_iso(previous.get('cutoff_utc', ''))
        if previous_cutoff and previous_cutoff > current_cutoff:
            raise RuntimeError('此计划早于已发布的 latest，拒绝回退；请生成当前最新计划')


def stale_keys(remote: dict, prefix: str, rows: list[dict]) -> list[str]:
    base = prefix.rstrip('/') + '/'
    targets = {(row['branch'], row['wheel_arch']): row for row in rows}
    stale = []
    for key in remote:
        if not key.startswith(base) or '/' in key[len(base):]:
            continue
        name = key[len(base):]
        identity = wheel_identity(name, rows[0]['package'])
        if identity is None:
            continue
        branch, date, arch = identity
        row = targets.get((branch, arch))
        if row is None:
            continue
        if date > row['date']:
            raise RuntimeError(f'latest 已存在更晚日期，拒绝回退/清理: {name}')
        expected = wheel_name(row['package'], branch, row['date'], row['sha'], arch)
        if name != expected:
            stale.append(key)
    return stale


def promote_one(client, bucket: str, row: dict) -> list[str] | None:
    """A failed/missing wheel never authorizes deletion of its old slot."""
    from obs_publish import list_all, delete_one, is_ok
    package = row['package']
    prefix = f'{PREFIX}/{public_project(package)}/latest'
    name = wheel_name(package, row['branch'], row['date'], row['sha'], row['wheel_arch'])
    url = f'{DEFAULT_BASE}/{public_project(package)}/latest/{urllib.parse.quote(name)}'
    status, _ = fetch(url, byte_limit=1)
    if status not in (200, 206):
        return None
    status, message, remote = list_all(client, bucket, prefix + '/')
    if not is_ok(status):
        raise RuntimeError(f'latest 列举失败: HTTP{status}: {message}')
    if prefix + '/' + name not in remote:
        return None
    # Verify the archival copy before retiring the old latest wheel.
    # History is append-only here, including any earlier filename spellings.
    history_prefix = f'{PREFIX}/{public_project(package)}'
    history_url = f'{DEFAULT_BASE}/{public_project(package)}/{urllib.parse.quote(name)}'
    history_status, _ = fetch(history_url, byte_limit=1)
    if history_status not in (200, 206):
        return None
    status, message, history = list_all(client, bucket, history_prefix + '/')
    if not is_ok(status):
        raise RuntimeError(f'history 列举失败: HTTP{status}: {message}')
    if history_prefix + '/' + name not in history:
        return None
    candidates = stale_keys(remote, prefix, [row])
    for key in candidates:
        if not key.startswith(prefix + '/') or '/' in key[len(prefix) + 1:]:
            raise RuntimeError(f'拒绝清理 latest 目录以外的对象: {key}')
    if len(candidates) > 200:
        raise RuntimeError('latest 单组合待清理对象超过200个，拒绝执行')
    for key in candidates:
        result = delete_one(client, bucket, key, 3)
        if result.status != 'deleted':
            raise RuntimeError(f'latest 清理失败: {key}: {result.error}')
        print(f'[latest] removed replaced wheel: {key}', flush=True)
    return candidates


def uploaded_this_run(row: dict, started: float) -> bool:
    """Do not promote a pre-existing object before this run has uploaded it."""
    out = BASE / 'out/daily' / row['branch']
    name = wheel_name(row['package'], row['branch'], row['date'], row['sha'], row['wheel_arch'])
    for suffix in ('', '-latest'):
        reports = [out / f"obs-daily-{row['package']}-{row['arch']}-incremental{suffix}.json",
                   out / f"obs-daily-{row['package']}{suffix}.json"]
        key = f"{PREFIX}/{public_project(row['package'])}/" + ('latest/' if suffix else '') + name
        matched = False
        for report in reports:
            try:
                if report.stat().st_mtime < started:
                    continue
                data = json.loads(report.read_text())
                matched |= any(item.get('key') == key and item.get('status') in ('uploaded', 'skipped')
                               for item in data.get('results', []))
            except (OSError, ValueError):
                continue
        if not matched:
            return False
    return True


def main() -> int:
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--arch', choices=tuple(ARCH_TO_WHEEL), required=True)
    parser.add_argument('--check-plan', action='store_true')
    parser.add_argument('--timeout', type=int, default=21600)
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--done-file', type=Path, help='构建结束后写入返回码；失败时跳过没有新上传的组合')
    args = parser.parse_args()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    state_path = BASE / 'plans' / f'latest-state-{args.arch}.json'
    check_plan(plan, args.arch, state_path)
    if args.check_plan:
        return 0
    lock = (BASE / f'.latest-finalize-{args.arch}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    check_plan(plan, args.arch, state_path)
    rows = [row for row in expected_rows(plan) if row['arch'] == args.arch]
    deadline = time.monotonic() + args.timeout
    from obs_publish import build_client, list_all, is_ok, quiet_obs_logger
    quiet_obs_logger()
    client = build_client(os.getenv('FLA_PKG_OBS_ENDPOINT', 'https://obs.cn-north-4.myhuaweicloud.com'),
                          os.environ['OBS_ACCESS_KEY'], os.environ['OBS_SECRET_KEY'])
    bucket = os.getenv('FLA_PKG_OBS_BUCKET', 'ascend-artifcat-packages')
    deleted = []
    promoted = []
    failed = []
    pending = rows.copy()
    try:
        while pending:
            for row in pending.copy():
                fresh_upload = not args.done_file or uploaded_this_run(row, started)
                removed = promote_one(client, bucket, row) if fresh_upload else None
                label = f"{row['branch']}/{row['soc']}/{args.arch}"
                if removed is not None:
                    deleted.extend(removed)
                    promoted.append(label)
                    pending.remove(row)
                    state = dict(plan, arch=args.arch, deleted=deleted, promoted=promoted,
                                 finalized_at=dt.datetime.now(dt.timezone.utc).isoformat())
                    atomic_write(state_path, state)
                    print(f'[latest] slot promoted: {label}; old files removed={len(removed)}', flush=True)
                elif args.done_file and args.done_file.exists():
                    prefix = f"{PREFIX}/{public_project(row['package'])}/latest/"
                    status, message, remote = list_all(client, bucket, prefix)
                    if not is_ok(status):
                        raise RuntimeError(f'latest 列举失败: HTTP{status}: {message}')
                    expected = prefix + wheel_name(row['package'], row['branch'], row['date'], row['sha'], row['wheel_arch'])
                    if not fresh_upload or expected not in remote:
                        print(f'[latest] no new uploaded wheel: {label}; KEEP previous daily', flush=True)
                        failed.append(label)
                        pending.remove(row)
            print(f'[latest] promoted={len(promoted)}/{len(rows)} pending={len(pending)} kept-old={len(failed)}', flush=True)
            if not pending:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('latest 部分新包尚未公网可读；已成功组合完成清理，其余组合保留旧包')
            time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
    finally:
        client.close()
    state = dict(plan, arch=args.arch, deleted=deleted, promoted=promoted, kept_old=failed,
                 finalized_at=dt.datetime.now(dt.timezone.utc).isoformat())
    if promoted:
        atomic_write(state_path, state)
    atomic_write(BASE / 'logs' / f'{args.plan.stem}-{args.arch}-latest.json', state)
    print(f'[latest] {"PARTIAL" if failed else "SUCCESS"} arch={args.arch} slots={len(promoted)}/{len(rows)} deleted={len(deleted)}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'[latest][ERROR] {exc}', flush=True)
        raise SystemExit(1)
