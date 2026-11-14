#!/usr/bin/env python3
"""Build CI package artifacts on the current runner.

The package workflow checks out the PR merge commit before invoking this
script. This module deliberately has one job: build the requested SOC
packages in the local runner container and leave them under ci/pkg/out so
the workflow can upload them as Actions artifacts.

The command is intentionally CI-only. It does not clone another repository,
connect to another machine, or publish a GitHub Release.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


SUPPORTED_SOCS = ("ascend910b", "ascend910_93", "ascend950")
SUPPORTED_ARCHES = ("arm", "x86")
DEFAULT_DOCKER_IMAGE = "fla-npu-ci:9.1.0-910b"
WHEEL_GLOB = "flash_linear_attention_npu*-*.whl"


def log(message: str) -> None:
    print(f"[pkg][{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}",
          flush=True)


def env_or(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def parse_list(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    """Parse a comma/space separated CI input while preserving its order."""

    result: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[,\s]+", value.strip()):
        if not token:
            continue
        token = token.lower()
        if token == "all":
            tokens = allowed
        elif token in allowed:
            tokens = (token,)
        else:
            raise SystemExit(
                f"不支持的 {label} 值: {token}；可选值: {', '.join(allowed)}"
            )
        for item in tokens:
            if item not in seen:
                seen.add(item)
                result.append(item)
    return result


def current_architecture() -> str:
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64", "armv8l"}:
        return "arm"
    if machine in {"x86_64", "amd64"}:
        return "x86"
    raise RuntimeError(
        f"无法识别 runner 架构 {machine!r}；仅支持 aarch64/arm64 或 x86_64/amd64"
    )


def git_output(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def clean_build_state(repo_root: Path) -> None:
    """Remove generated state so each SOC starts from a complete build."""

    for name in ("build", "third_party"):
        path = repo_root / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def sanitize_ref_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return value.strip("-") or "unknown"


def resolve_metadata(repo_root: Path, head_sha: str) -> dict[str, str]:
    ref = env_or("FLA_PKG_REF_NAME", "")
    if not ref:
        ref = git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if not ref or ref == "HEAD":
        ref = "unknown"

    commit = re.sub(r"[^A-Za-z0-9]", "", head_sha)[:7]
    if not commit:
        commit = re.sub(
            r"[^A-Za-z0-9]",
            "",
            env_or("FLA_PKG_COMMIT_ID", ""),
        )[:7] or "unknown"

    timestamp = env_or("FLA_PKG_DATETIME", "")
    if not timestamp:
        now = datetime.now(timezone(timedelta(hours=8)))
        timestamp = now.strftime("%Y%m%d-%H%M%S")

    return {
        "ref": sanitize_ref_name(ref),
        "datetime": timestamp,
        "pr_id": env_or("FLA_PKG_PR_ID", "0"),
        "commit": commit,
    }


def artifact_name(meta: dict[str, str], soc: str, arch: str) -> str:
    return (
        f"Release-{meta['ref']}-{meta['datetime']}-PR{meta['pr_id']}"
        f"-{soc}-{arch}-{meta['commit']}"
    )


class BuildCleanupError(RuntimeError):
    """The child process group could not be proven to have stopped."""


def _process_group_alive(pgid: int) -> bool:
    """Return whether a live process remains in a POSIX process group."""

    if sys.platform.startswith("linux"):
        proc_root = Path("/proc")
        try:
            entries = proc_root.iterdir()
        except OSError:
            return False
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False

    try:
        os.killpg(pgid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _terminate_process_group(
    process: subprocess.Popen,
    *,
    grace_seconds: float = 5.0,
) -> None:
    """Stop a command and any descendants before a retry or next SOC."""

    if os.name != "posix":
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=grace_seconds)
        return

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        if not _process_group_alive(process.pid):
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            process.poll()
            if not _process_group_alive(process.pid):
                return
            time.sleep(0.05)
    raise BuildCleanupError(
        f"编译进程组 {process.pid} 仍在运行，拒绝继续处理"
    )


def run_logged(
    argv: list[str],
    log_file: Path,
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    stop: threading.Event | None = None,
) -> int:
    """Run a command while streaming the same output to CI and a log file."""

    printable = " ".join(shlex.quote(item) for item in argv)
    print(f"$ {printable}", flush=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen | None = None
    watcher_done = threading.Event()
    cancellation_errors: list[Exception] = []
    watcher: threading.Thread | None = None
    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            bufsize=1,
            start_new_session=(os.name == "posix"),
        )
        if stop is not None:
            def watch_stop() -> None:
                while not watcher_done.wait(0.1):
                    if stop.is_set():
                        try:
                            _terminate_process_group(process)
                        except Exception as exc:
                            cancellation_errors.append(exc)
                        return

            watcher = threading.Thread(
                target=watch_stop,
                name="fla-package-stop-watcher",
                daemon=True,
            )
            watcher.start()

        assert process.stdout is not None
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {printable}\n")
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(f"    {line}", end="", flush=True)
        returncode = process.wait()
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"[exit={returncode}]\n")
        return returncode
    except KeyboardInterrupt:
        if process is not None:
            _terminate_process_group(process)
        raise
    finally:
        watcher_done.set()
        if watcher is not None:
            watcher.join()
        if cancellation_errors:
            raise BuildCleanupError("编译进程组清理失败") from cancellation_errors[0]
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            # The leader can exit while descendants still hold the mounted
            # repository. Do not start the next SOC until they are gone.
            try:
                _terminate_process_group(process)
            except BuildCleanupError:
                raise
            except Exception as exc:
                raise BuildCleanupError("编译进程组清理失败") from exc


def _docker_remove(container: str) -> None:
    result = subprocess.run(
        ["docker", "rm", "-f", container],
        capture_output=True,
        text=True,
    )
    if result.returncode and "No such container" not in (result.stderr or ""):
        raise BuildCleanupError(
            f"无法清理 Docker 容器 {container}: "
            f"{(result.stderr or result.stdout).strip()[-500:]}"
        )


def build_local(
    repo_root: Path,
    soc: str,
    wheel_dir: Path,
    log_file: Path,
    jobs: int = 16,
) -> list[Path]:
    """Build one SOC wheel in the local runner container."""

    if jobs < 1:
        raise ValueError("编译并发数必须是正整数")
    image = env_or("FLA_PKG_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE)
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    if inspect.returncode:
        raise RuntimeError(
            f"Docker 镜像 {image} 不存在；请先在 runner 上准备 CI 镜像"
        )

    clean_build_state(repo_root)
    if wheel_dir.exists():
        shutil.rmtree(wheel_dir)
    wheel_dir.mkdir(parents=True, exist_ok=True)

    container = f"fla-pkg-{soc}-{os.getpid()}-{time.time_ns()}"
    command = f"""
set -euo pipefail
if [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
  echo '[CI][ERROR] 未找到 CANN set_env.sh' >&2
  exit 1
fi
cd /workspace/repo
# Package builds always include all operators, regardless of runner/image filters.
unset FLA_NPU_OPS FLA_NPU_BUILD_ARGS
export CMAKE_BUILD_PARALLEL_LEVEL={jobs} MAKEFLAGS=-j{jobs} MAX_JOBS={jobs}
echo '[pkg] Full wheel build: all operators; parallel jobs: {jobs}'
exec python3 scripts/build_wheel.py --wheel-dir /workspace/pkg-out --build-args='--ops=all -j{jobs}'
"""
    argv = [
        "docker",
        "run",
        "--rm",
        "--name",
        container,
        "--network",
        "host",
        "-v",
        f"{repo_root}:/workspace/repo",
        "-v",
        f"{wheel_dir}:/workspace/pkg-out",
        "-e",
        f"FLA_NPU_SOC={soc}",
        "-e",
        "TORCH_DEVICE_BACKEND_AUTOLOAD=0",
        "-e",
        "HOME=/root",
    ]
    for name in (
        "FLA_NPU_BRANCH_NAME",
        "FLA_NPU_COMMIT_ID",
    ):
        value = os.getenv(name, "").strip()
        if value:
            argv.extend(["-e", f"{name}={value}"])
    argv.extend([image, "bash", "-lc", command])

    returncode: int | None = None
    try:
        returncode = run_logged(argv, log_file)
    except KeyboardInterrupt:
        _docker_remove(container)
        raise
    finally:
        if returncode not in (0, None):
            _docker_remove(container)

    if returncode != 0:
        raise RuntimeError(
            f"本地 {soc} 编译失败 (exit={returncode})，详见 {log_file}"
        )

    wheels = sorted(wheel_dir.glob(WHEEL_GLOB))
    if not wheels:
        raise RuntimeError(
            f"{soc} 编译完成但未生成 wheel，详见 {log_file}"
        )
    return wheels


# Keep the old function name for callers that import the build entry point.
build_arm = build_local


class Remote:
    """Compatibility process controller used only by old process tests.

    This class has no network implementation. Production code never creates
    it; the small adapter lets older tests exercise process-group cleanup while
    the package builder remains entirely local.
    """

    def exec(self, script: str, *, timeout: int | None = None):
        raise RuntimeError("本地 Package 编译不支持外部命令执行")

    def check(self, script: str, *, timeout: int | None = None) -> str:
        result = self.exec(script, timeout=timeout)
        if result.returncode:
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"本地进程控制命令失败: {detail[-1000:]}")
        return result.stdout

    def cancel_remote_process_group(self, out_dir: str) -> None:
        """Compatibility hook; production builds use Docker cleanup."""
        # Legacy contract calls cancel_remote_process_group(out_dir).

        script = f"""
for attempt in $(seq 1 30); do
    pgid=$(cat {shlex.quote(out_dir)}/pgid 2>/dev/null || true)
    [[ -n "$pgid" ]] && break
    [[ -f {shlex.quote(out_dir)}/pid ]] || exit 0
    sleep 0.1
done
case "$pgid" in
  ''|*[!0-9]*|0|1) echo "invalid process group" >&2; exit 1 ;;
esac
if [[ -r /proc/$pgid/stat ]]; then
    recorded=$(cat {shlex.quote(out_dir)}/start_time 2>/dev/null)
    actual=$(awk '{{print $22}}' /proc/$pgid/stat)
    [[ -n "$recorded" && "$recorded" == "$actual" ]] || {{
        echo "process identity changed" >&2
        exit 1
    }}
fi
group_alive() {{
    ps -eo pgid=,stat= | awk -v wanted="$pgid" '$1 == wanted && $2 !~ /^[ZX]/ {{ alive=1 }} END {{ exit !alive }}'
}}
for sig in INT TERM KILL; do
    group_alive || exit 0
    kill -"$sig" -- -"$pgid" 2>/dev/null || true
    for attempt in $(seq 1 30); do
        group_alive || exit 0
        sleep 0.1
    done
done
echo "process group is still alive" >&2
exit 1
"""
        result = self.exec(script, timeout=60)
        if result.returncode:
            detail = (result.stderr or result.stdout or "").strip()
            raise BuildCleanupError(
                f"进程组清理失败: {detail[-1000:]}"
            )


def _remote_build_startup(out_dir: str, build_timeout: int) -> str:
    """Legacy local process fixture; no production path calls this helper."""

    inner = (
        "awk '{print $22}' /proc/$$/stat > start_time; "
        "echo $$ > pgid; "
        f"timeout --foreground {build_timeout} bash build.sh; "
        "rc=$?; echo $rc > status"
    )
    return (
        f"cd {shlex.quote(out_dir)}\n"
        f"nohup setsid bash -c {shlex.quote(inner)} > build.log 2>&1 "
        "< /dev/null &\n"
        "echo $! > pid\n"
        "echo BOOT:$(awk '/^btime/{print $2}' /proc/stat 2>/dev/null || "
        "echo unknown)\n"
    )


def copy_artifacts(wheels: list[Path], target: Path) -> list[str]:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for wheel in wheels:
        destination = target / wheel.name
        shutil.copy2(wheel, destination)
        names.append(destination.name)
    return names


def write_manifest(
    path: Path,
    *,
    head_sha: str,
    meta: dict[str, str],
    socs: list[str],
    arches: list[str],
    host_arch: str,
    results: list[dict],
) -> None:
    payload = {
        "schema_version": 2,
        "head_sha": head_sha,
        "ref": meta["ref"],
        "datetime": meta["datetime"],
        "pr_id": meta["pr_id"],
        "commit": meta["commit"],
        "socs": socs,
        "arches": arches,
        "runner_arch": host_arch,
        "results": results,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    repo_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="在当前 CI runner 上完整编译 A2/A3/A5 wheel 并生成 artifact"
    )
    parser.add_argument(
        "--repo-root",
        default=str(repo_default),
        help="已由 package workflow 检出的仓库目录",
    )
    parser.add_argument(
        "--soc",
        default=",".join(SUPPORTED_SOCS),
        help=f"SOC 列表，默认: {','.join(SUPPORTED_SOCS)}",
    )
    parser.add_argument(
        "--arch",
        default="arm",
        help="架构列表；必须与当前 runner 架构一致，默认 arm",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=16,
        help="每个出包任务的编译并发数，默认 16",
    )
    parser.add_argument(
        "--head-sha",
        default="",
        help="package workflow 锁定的 PR 合并 commit SHA",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="artifact 输出目录，默认 <repo>/ci/pkg/out",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出支持的 SOC/架构组合后退出",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs 必须是正整数")

    if args.list:
        for soc in SUPPORTED_SOCS:
            for arch in SUPPORTED_ARCHES:
                print(f"{soc} / {arch}")
        return 0

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        raise SystemExit(f"代码目录不存在: {repo_root}")
    if not (repo_root / "scripts" / "build_wheel.py").is_file():
        raise SystemExit(f"找不到 scripts/build_wheel.py: {repo_root}")

    socs = parse_list(args.soc, SUPPORTED_SOCS, "SOC")
    arches = parse_list(args.arch, SUPPORTED_ARCHES, "架构")
    if not socs or not arches:
        raise SystemExit("--soc/--arch 解析结果不能为空")

    host_arch = current_architecture()
    unsupported = [arch for arch in arches if arch != host_arch]
    if unsupported:
        raise SystemExit(
            f"当前 runner 是 {host_arch} 架构，不能在本机生成 "
            f"{','.join(unsupported)} 包；请使用 /package arch={host_arch}"
        )

    head_sha = (
        args.head_sha.strip()
        or os.getenv("FLA_PKG_COMMIT_ID", "").strip()
        or git_output(repo_root, "rev-parse", "HEAD")
    )
    if not head_sha:
        raise SystemExit("无法确定 PR 合并 commit，请提供 --head-sha")
    if args.head_sha and not re.fullmatch(r"[0-9a-fA-F]{40}", head_sha):
        raise SystemExit("--head-sha 必须是 40 位 commit SHA")

    checked_out = git_output(repo_root, "rev-parse", "HEAD")
    if args.head_sha and checked_out and checked_out.lower() != head_sha.lower():
        raise SystemExit(
            "检出的代码与 package workflow 锁定的 commit 不一致："
            f"expected={head_sha} actual={checked_out}"
        )
    if args.head_sha and not checked_out:
        # actions/checkout has already fetched the requested SHA. Some
        # self-hosted workspaces temporarily reject a second git query; that
        # local read failure must not turn a valid checkout into a false SHA
        # mismatch or make the builder contact GitHub.
        log("[警告] 无法从 checkout 工作区读取 HEAD，使用 workflow 锁定的 SHA")
        checked_out = head_sha

    meta = resolve_metadata(repo_root, head_sha)
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else repo_root / "ci" / "pkg" / "out"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("wheels", "artifacts", "logs"):
        path = out_dir / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    log(
        f"代码: {checked_out or head_sha}；runner 架构: {host_arch}；"
        f"SOC: {','.join(socs)}；输出: {out_dir}"
    )
    log(f"本地全量编译所有算子，并发数: {args.jobs}；结果上传为 artifact")

    results: list[dict] = []
    retries = int(env_or("FLA_PKG_BUILD_RETRIES", "1"))
    if retries < 0:
        raise SystemExit("FLA_PKG_BUILD_RETRIES 不能为负数")

    for soc in socs:
        arch = host_arch
        name = artifact_name(meta, soc, arch)
        wheel_dir = out_dir / "wheels" / f"{soc}-{arch}"
        log_file = out_dir / "logs" / f"{soc}-{arch}.log"
        started = time.monotonic()
        record = {
            "soc": soc,
            "arch": arch,
            "jobs": args.jobs,
            "artifact": name,
            "status": "failed",
            "wheels": [],
            "seconds": 0.0,
            "error": "",
        }
        try:
            wheels: list[Path] = []
            for attempt in range(retries + 1):
                try:
                    wheels = build_local(repo_root, soc, wheel_dir, log_file, jobs=args.jobs)
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    if attempt >= retries:
                        raise
                    log(
                        f"[{soc}/{arch}] 编译失败: {exc}; "
                        f"清理后重试 {attempt + 1}/{retries}"
                    )
            record["wheels"] = copy_artifacts(
                wheels,
                out_dir / "artifacts" / name,
            )
            record["status"] = "success"
            log(f"[{soc}/{arch}] 编译成功: {len(wheels)} 个 wheel")
        except KeyboardInterrupt:
            record["error"] = "用户中断"
            results.append(record)
            write_manifest(
                out_dir / "manifest.json",
                head_sha=head_sha,
                meta=meta,
                socs=socs,
                arches=arches,
                host_arch=host_arch,
                results=results,
            )
            return 130
        except Exception as exc:
            record["error"] = str(exc)
            log(f"[{soc}/{arch}] 编译失败: {record['error']}")
        finally:
            record["seconds"] = round(time.monotonic() - started, 1)
            results.append(record)
            write_manifest(
                out_dir / "manifest.json",
                head_sha=head_sha,
                meta=meta,
                socs=socs,
                arches=arches,
                host_arch=host_arch,
                results=results,
            )

    successful = [item for item in results if item["status"] == "success"]
    (out_dir / "artifact-name.txt").write_text(
        "\n".join(item["artifact"] for item in successful) + "\n",
        encoding="utf-8",
    )

    print("\n========== Package 编译汇总 ==========")
    for item in results:
        status = "成功" if item["status"] == "success" else "失败"
        print(
            f"[{status}] {item['soc']:<14} {item['arch']:<4} "
            f"{item['seconds']:>7.1f}s {item['artifact']}"
        )
        if item["error"]:
            print(f"        错误: {item['error'][:500]}")
        for wheel in item["wheels"]:
            print(f"        - {wheel}")
    print(f"manifest: {out_dir / 'manifest.json'}")

    failed = len(results) - len(successful)
    if failed:
        log(f"{failed} 个 SOC 编译失败")
        return 1
    log(f"{len(successful)} 个 SOC 全部编译成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
