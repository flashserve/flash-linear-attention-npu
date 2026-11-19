#!/usr/bin/env python3
"""Prepare CI Git dependencies at the revisions declared by this checkout.

Image seeds are an optimization, not the dependency version authority. Never
modify them or fall back to an old checkout when fetching a pinned revision fails.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time


GIT_TIMEOUT_SECONDS = 120
GIT_CLEANUP_GRACE_SECONDS = 5


class GitCleanupError(RuntimeError):
    """Git descendants could not be stopped; retrying would be unsafe."""


@dataclass(frozen=True)
class Dependency:
    name: str
    repository: str
    revision: str
    marker: str = ""


def cmake_literal(text, variable):
    match = re.search(
        rf'^\s*set\(\s*{re.escape(variable)}\s+"?([^"\s()]+)"?\s*'
        rf'(?=\)|CACHE\s+STRING\b)',
        text, re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot read literal {variable} from dependency CMake file")
    return match.group(1)


def cmake_repository(text, variable=None):
    if variable and re.search(rf'^\s*set\(\s*{re.escape(variable)}\b', text, re.MULTILINE):
        return cmake_literal(text, variable)
    repository = re.search(r'^\s*GIT_REPOSITORY\s+"?([^"\s()]+)"?\s*$',
                           text, re.MULTILINE)
    if not repository:
        raise ValueError("Cannot read dependency GIT_REPOSITORY")
    return repository.group(1)


def read_dependencies(repo):
    catlass = (repo / "cmake/third_party/catlass.cmake").read_text(encoding="utf-8")
    opbase = (repo / "cmake/third_party/opbase.cmake").read_text(encoding="utf-8")
    dependencies = (
        Dependency("catlass", cmake_repository(catlass, "CATLASS_VENDOR_GIT_REPOSITORY"),
                   cmake_literal(catlass, "CATLASS_VENDOR_GIT_TAG"),
                   "include/catlass/catlass.hpp"),
        Dependency("opbase", cmake_repository(opbase), cmake_literal(opbase, "OPBASE_TAG_ID")),
    )
    for dependency in dependencies:
        if not re.fullmatch(r"[0-9a-f]{40}", dependency.revision):
            raise ValueError(f"{dependency.name}: expected a full pinned Git commit SHA")
    return dependencies


def _git_group_alive(pgid):
    if sys.platform.startswith("linux"):
        # Orphaned zombies cannot hold files or download data. Container PID 1
        # may not reap them promptly, so killpg(pgid, 0) alone is insufficient.
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                return True
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _stop_git_processes(process):
    """Stop the whole POSIX group, even if its Git leader already exited."""
    try:
        if os.name != "posix":
            # CI runs on Linux. Keep the previous direct-process behavior on
            # other platforms, where POSIX process groups are unavailable.
            if process.poll() is None:
                process.kill()
            process.wait(timeout=GIT_CLEANUP_GRACE_SECONDS)
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                process.wait(timeout=GIT_CLEANUP_GRACE_SECONDS)
                return
            deadline = time.monotonic() + GIT_CLEANUP_GRACE_SECONDS
            while time.monotonic() < deadline:
                process.poll()  # Reap our child; descendants may outlive it.
                if not _git_group_alive(process.pid):
                    process.wait(timeout=GIT_CLEANUP_GRACE_SECONDS)
                    return
                time.sleep(0.05)
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
        raise GitCleanupError(f"Cannot confirm cleanup of Git process group {process.pid}") from error
    raise GitCleanupError(f"Git process group {process.pid} is still running; refusing to retry")


def run_git(path, *args, check=True):
    # Explicit paths avoid Git discovering the parent source repository. Optional
    # locks are disabled so even status checks never refresh a read-only seed index.
    command = ["git", f"--git-dir={path / '.git'}", f"--work-tree={path}", *args]
    process = subprocess.Popen(
        command,
        env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0"),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=(os.name == "posix"))
    try:
        stdout, stderr = process.communicate(timeout=GIT_TIMEOUT_SECONDS)
    finally:
        # Also clean up on interruption or when a failed leader leaves helpers
        # behind. Cleanup errors are non-retryable and override fetch timeouts.
        try:
            _stop_git_processes(process)
        finally:
            process.stdout.close()
            process.stderr.close()
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result


def matches(path, dependency):
    if path.is_symlink() or not (path / ".git").is_dir():
        return False
    head = run_git(path, "rev-parse", "HEAD", check=False)
    if head.returncode or head.stdout.strip() != dependency.revision:
        return False
    status = run_git(path, "status", "--porcelain", "--untracked-files=all", check=False)
    return (status.returncode == 0 and not status.stdout.strip()
            and (not dependency.marker or (path / dependency.marker).is_file()))


def fetch(path, dependency):
    run_git(path, "init", "-q")
    run_git(path, "remote", "add", "origin", dependency.repository)
    for attempt in range(1, 4):
        try:
            run_git(path, "fetch", "--no-tags", "--depth=1", "origin", dependency.revision)
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 3:
                raise
            print(f"[CI][WARN] {dependency.name}: fetch attempt {attempt} failed; retrying",
                  flush=True)
    run_git(path, "checkout", "--detach", "-q", "FETCH_HEAD")


def prepare(dependency, cache, seed):
    destination = cache / dependency.name
    if matches(destination, dependency):
        print(f"[CI] {dependency.name}: reuse cache at {dependency.revision}", flush=True)
        return

    # Build a complete replacement first. A failed download must stop CI without
    # replacing an existing checkout with partial files or continuing on old code.
    with tempfile.TemporaryDirectory(prefix=f".{dependency.name}-", dir=cache) as temporary:
        staged = Path(temporary) / dependency.name
        source = seed / dependency.name if seed else None
        if source and matches(source, dependency):
            print(f"[CI] {dependency.name}: seed matches {dependency.revision}", flush=True)
            shutil.copytree(source, staged, symlinks=True)
        else:
            print(f"[CI] {dependency.name}: no matching cache/seed; fetching "
                  f"{dependency.revision}", flush=True)
            staged.mkdir()
            fetch(staged, dependency)
        if not matches(staged, dependency):
            raise RuntimeError(f"{dependency.name}: revision/content verification failed "
                               f"for {dependency.revision}")
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        elif destination.exists():
            shutil.rmtree(destination)
        staged.rename(destination)
    print(f"[CI] {dependency.name}: verified {dependency.revision}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seed-dir", type=Path)
    args = parser.parse_args()
    dependencies = read_dependencies(args.repo_dir)
    cache = args.cache_dir.resolve()
    seed = args.seed_dir.resolve() if args.seed_dir else None
    # A seed is read-only input and must not overlap any managed cache directory.
    if seed and (cache == seed or cache in seed.parents or seed in cache.parents):
        raise ValueError("Dependency cache and image seed directories must be separate")
    cache.mkdir(parents=True, exist_ok=True)
    for dependency in dependencies:
        prepare(dependency, cache, seed)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"[CI][ERROR] Dependency preparation failed: {error}", file=sys.stderr)
        sys.exit(1)
