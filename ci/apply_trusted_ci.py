#!/usr/bin/env python3
"""Replace merged PR CI scripts while retaining its Example/ST case data."""

import argparse
import json
import os
from pathlib import Path
import shutil
import stat


def read_pr_cases(repo):
    """Read only a regular case file, without following untrusted PR symlinks."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    repo_fd = os.open(repo, directory_flags)
    try:
        try:
            ci_fd = os.open("ci", directory_flags, dir_fd=repo_fd)
        except FileNotFoundError:
            return None
        try:
            try:
                case_fd = os.open("example_st_cases.json", os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                                  dir_fd=ci_fd)
            except FileNotFoundError:
                return None
            with os.fdopen(case_fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("PR Example/ST case path must be a regular file")
                data = stream.read()
        finally:
            os.close(ci_fd)
    finally:
        os.close(repo_fd)
    json.loads(data.decode("utf-8"))
    return data


def apply_trusted_ci(repo, trusted_ci):
    repo = Path(repo).absolute()
    trusted_ci = Path(trusted_ci).absolute()
    source = trusted_ci / "ci"
    if repo == trusted_ci or repo in trusted_ci.parents or trusted_ci in repo.parents:
        raise ValueError("PR and trusted CI checkouts must be separate directories")
    if not source.is_dir() or source.is_symlink():
        raise ValueError("Trusted checkout does not contain a regular ci directory")
    cases = read_pr_cases(repo)
    destination = repo / "ci"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, symlinks=True)
    if cases is not None:
        case_path = destination / "example_st_cases.json"
        # Replace any trusted case entry instead of following a possible link.
        case_path.unlink(missing_ok=True)
        case_path.write_bytes(cases)
    print("[CI] Applied workflow-pinned trusted CI; retained merged PR Example/ST case data.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--trusted-ci", type=Path, required=True)
    args = parser.parse_args()
    apply_trusted_ci(args.repo, args.trusted_ci)


if __name__ == "__main__":
    main()
