#!/usr/bin/env python3
"""在独立进程组中执行命令，并在超时后完整回收该进程组。"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Optional


_TERMINATE_SIGNAL = getattr(signal, "SIGTERM", 15)
_KILL_SIGNAL = getattr(signal, "SIGKILL", 9)


class _ForwardedSignal(BaseException):
    def __init__(self, signal_number: int) -> None:
        super().__init__(signal_number)
        self.signal_number = signal_number


def _group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(process_group_id: int, signal_number: int) -> None:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        pass


def _wait_group_exit(
    process: subprocess.Popen[bytes], process_group_id: int, timeout: float
) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        # 先回收直接子进程，避免它的僵尸状态让进程组被误判为仍在运行。
        process.poll()
        if not _group_exists(process_group_id):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _terminate_group(
    process: subprocess.Popen[bytes], process_group_id: int, grace_seconds: float
) -> bool:
    _signal_group(process_group_id, _TERMINATE_SIGNAL)
    if not _wait_group_exit(process, process_group_id, grace_seconds):
        # 即使直接子进程已经响应 TERM 退出，仍对存活的 worker 所在组发送 KILL。
        _signal_group(process_group_id, _KILL_SIGNAL)
        if not _wait_group_exit(process, process_group_id, 2.0):
            return False
    return process.poll() is not None


def _restore_signal_handlers(
    handlers: dict[int, signal.Handlers]
) -> None:
    for signal_number, handler in handlers.items():
        signal.signal(signal_number, handler)


def _ignore_signals(signal_numbers: tuple[int, ...]) -> None:
    for signal_number in signal_numbers:
        signal.signal(signal_number, signal.SIG_IGN)


def _report_cleanup_failure() -> None:
    print("无法完整回收待执行命令的进程组", file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("--term-grace-seconds", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.deadline_seconds <= 0:
        parser.error("--deadline-seconds 必须大于 0")
    if args.term_grace_seconds < 0:
        parser.error("--term-grace-seconds 不能小于 0")
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("缺少待执行命令")
    return args


def main() -> int:
    args = _parse_args()
    if os.name != "posix":
        print("进程组 hard deadline 仅支持 POSIX 系统", file=sys.stderr)
        return 2

    handled_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {
        signal_number: signal.getsignal(signal_number)
        for signal_number in handled_signals
    }
    process: Optional[subprocess.Popen[bytes]] = None
    process_group_id = 0
    pending_signal: Optional[int] = None

    def forward_signal(signal_number: int, _frame: object) -> None:
        nonlocal pending_signal
        # 首个外部信号负责进入清理；清理期间忽略重复信号，避免异常重入。
        pending_signal = signal_number
        _ignore_signals(handled_signals)
        if process_group_id > 0:
            raise _ForwardedSignal(signal_number)

    for signal_number in handled_signals:
        signal.signal(signal_number, forward_signal)

    try:
        if pending_signal is not None:
            return 128 + pending_signal
        try:
            process = subprocess.Popen(args.command, start_new_session=True)
        except FileNotFoundError:
            print("待执行命令不存在", file=sys.stderr)
            return 127
        except OSError as error:
            print(f"无法启动待执行命令：{error}", file=sys.stderr)
            return 126

        process_group_id = process.pid
        if pending_signal is not None:
            raise _ForwardedSignal(pending_signal)

        try:
            return_code = process.wait(timeout=args.deadline_seconds)
        except subprocess.TimeoutExpired:
            _ignore_signals(handled_signals)
            cleanup_ok = _terminate_group(
                process, process_group_id, args.term_grace_seconds
            )
            if not cleanup_ok:
                _report_cleanup_failure()
                return 125
            print(
                f"进程级 hard deadline 已触发：{args.deadline_seconds:g}s",
                file=sys.stderr,
            )
            return 124

        if _group_exists(process_group_id):
            _ignore_signals(handled_signals)
            if not _terminate_group(
                process, process_group_id, args.term_grace_seconds
            ):
                _report_cleanup_failure()
                return 125
            print(
                "待执行命令退出后仍有进程组成员存活，已强制回收",
                file=sys.stderr,
            )
            return 125
        if return_code < 0:
            return 128 - return_code
        return return_code
    except _ForwardedSignal as forwarded:
        if process is None:
            return 128 + forwarded.signal_number
        if not _terminate_group(
            process, process_group_id, args.term_grace_seconds
        ):
            _report_cleanup_failure()
            return 125
        return 128 + forwarded.signal_number
    except BaseException:
        _ignore_signals(handled_signals)
        if process is not None:
            if not _terminate_group(
                process, process_group_id, args.term_grace_seconds
            ):
                _report_cleanup_failure()
        raise
    finally:
        _restore_signal_handlers(previous_handlers)


if __name__ == "__main__":
    raise SystemExit(main())
