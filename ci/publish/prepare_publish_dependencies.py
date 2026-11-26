#!/usr/bin/env python3
"""Prepare the entire third_party input set once, before wheel compilation."""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile

import ci_git_dependencies as compat

REPORT = "dependency-preparation.json"


def load_git_helper(repo):
    path = repo / "ci/prepare_ci_dependencies.py"
    if not path.is_file():
        return compat
    spec = importlib.util.spec_from_file_location("source_ci_dependencies", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git_requirements(repo, helper):
    text = (repo / "cmake/third_party/opbase.cmake").read_text()
    if "OPBASE_LEGACY_TAG_ID" not in text:
        return helper.read_dependencies(repo)
    catlass = (repo / "cmake/third_party/catlass.cmake").read_text()
    selected = helper.cmake_literal(text, "OPBASE_LEGACY_TAG_ID")
    toolkit = Path(os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    header = toolkit / (platform.machine() + "-linux/pkg_inc/op_common/atvoss/elewise/elewise_tiling.h")
    if header.is_file() and "int64_t GetBlockDim() const;" in header.read_text():
        selected = helper.cmake_literal(text, "OPBASE_CONST_API_TAG_ID")
    return (
        helper.Dependency("catlass", helper.cmake_repository(catlass, "CATLASS_VENDOR_GIT_REPOSITORY"),
                          helper.cmake_literal(catlass, "CATLASS_VENDOR_GIT_TAG"),
                          "include/catlass/catlass.hpp"),
        helper.Dependency("opbase", helper.cmake_repository(text), selected),
    )


def literal_assignment(path, name):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Cannot read {name} from {path}")


def archive_requirements(repo):
    # The checkout owns the list and URLs, including pybind11 and offline-bundle inputs.
    urls = literal_assignment(repo / "scripts/tools/third_lib_download.py", "my_urls")
    archives = {}
    for url in urls:
        if url.endswith(".git"):
            continue
        name = Path(urllib.parse.urlparse(url).path).name
        if not name or not (name.endswith(".tar.gz") or name.endswith(".zip")):
            raise ValueError(f"Unsupported archive declaration: {url}")
        if name in archives and archives[name]["url"] != url:
            raise ValueError(f"Conflicting archive URLs: {name}")
        archives[name] = {"name": name, "url": url}
    # Verify the preparation list agrees with the build's CMake download declarations.
    for path in (repo / "cmake/third_party").glob("*.cmake"):
        text = path.read_text(encoding="utf-8")
        for url in re.findall(r'https://gitcode.com/cann-src-third-party/[^"\s)]+', text):
            name = Path(urllib.parse.urlparse(url).path).name
            if name not in archives or archives[name]["url"] != url:
                raise ValueError(f"{path.name}: archive declaration disagrees with third_lib_download.py: {url}")
            match = re.search(r"URL_HASH\s+SHA256=([0-9a-fA-F]{64})", text)
            if match:
                archives[name]["expected_sha256"] = match.group(1).lower()
    required = literal_assignment(repo / "scripts/tools/prepare_offline_bundle.py", "_ARCHIVES")
    if set(required) - archives.keys():
        raise ValueError(f"Missing offline bundle inputs: {set(required) - archives.keys()}")
    return list(archives.values())


def validate_archive(path, expected=None):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise ValueError(f"Archive SHA256 mismatch: {path.name}")
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist() or archive.testzip() is not None:
                raise ValueError(f"Invalid ZIP: {path.name}")
    else:
        # Consume every file so truncated gzip data is rejected before the build.
        with tarfile.open(path, "r:gz") as archive:
            members = 0
            for member in archive:
                members += 1
                if member.isfile():
                    with archive.extractfile(member) as stream:
                        while stream.read(1024 * 1024):
                            pass
            if not members:
                raise ValueError(f"Empty archive: {path.name}")
    return digest


def read_evidence(cache):
    try:
        return json.loads((cache / REPORT).read_text())
    except (OSError, ValueError):
        return {}


def prepare_archive(spec, destination, caches):
    for cache in caches:
        previous = next((r for r in read_evidence(cache).get("archives", [])
                         if r.get("url") == spec["url"]), {})
        expected = spec.get("expected_sha256") or previous.get("sha256")
        candidate = cache / "pkg" / spec["name"]
        # An unrecorded filename, such as include.zip, cannot identify its version.
        if not expected or not candidate.is_file():
            continue
        try:
            digest = validate_archive(candidate, expected)
        except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile, EOFError):
            continue
        shutil.copyfile(candidate, destination)
        return dict(spec, sha256=digest, size=destination.stat().st_size, source="cache")
    for attempt in range(3):
        try:
            request = urllib.request.Request(spec["url"], headers={
                "Cache-Control": "no-cache", "User-Agent": "fla-npu-publish"})
            with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            digest = validate_archive(destination, spec.get("expected_sha256"))
            return dict(spec, sha256=digest, size=destination.stat().st_size, source="download")
        except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile, EOFError):
            destination.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(3)


@contextmanager
def preparation_lock(repo):
    import fcntl
    with (repo / ".publish-dependencies.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def prepare(repo, output, seed, fresh=False):
    repo = repo.resolve()
    cache = repo / "third_party"
    if cache.is_symlink():
        raise ValueError("third_party must be a private directory, not a shared symlink")
    seed = seed.resolve() if seed else None
    if seed and (seed == cache or seed in cache.parents or cache in seed.parents):
        raise ValueError("Image seed must be separate from the build workspace")
    helper = load_git_helper(repo)
    dependencies = git_requirements(repo, helper)
    archives = archive_requirements(repo)
    for dep in dependencies:
        if not re.fullmatch(r"[0-9a-f]{40}", dep.revision):
            raise ValueError(f"{dep.name}: expected a full commit SHA")
    with preparation_lock(repo), tempfile.TemporaryDirectory(prefix=".publish-deps-", dir=repo) as temporary:
        temporary = Path(temporary)
        staged = temporary / "prepared"
        (staged / "pkg").mkdir(parents=True)
        caches = [] if fresh else [cache] + ([seed] if seed else [])
        evidence = {"mode": "fresh-download-no-image-cache" if fresh else "validated-cache",
                    "archives": [], "git": []}
        for dep in dependencies:
            matching = next((c for c in caches if helper.matches(c / dep.name, dep)), None)
            helper.prepare(dep, staged, matching)
            evidence["git"].append({"name": dep.name, "repository": dep.repository,
                                    "fetched_commits": [dep.revision], "head": dep.revision,
                                    "source": "cache" if matching else "download"})
        for spec in archives:
            record = prepare_archive(spec, staged / "pkg" / spec["name"], caches)
            evidence["archives"].append(record)
            print(f"[dependencies] {spec['name']}: {record['source']} {record['sha256']}", flush=True)
        result = subprocess.run(["git", "--git-dir=" + str(repo / ".git"),
                                 "--work-tree=" + str(repo), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        evidence["source_sha"] = result.stdout.strip()
        serialized = json.dumps(evidence, indent=2) + "\n"
        (staged / REPORT).write_text(serialized)
        if fresh:
            (staged / "fresh-download-evidence.json").write_text(serialized)
        # Publish the complete set only after every dependency is ready.
        previous = temporary / "previous"
        if cache.exists():
            cache.rename(previous)
        try:
            staged.rename(cache)
        except BaseException:
            if previous.exists():
                previous.rename(cache)
            raise
        output.mkdir(parents=True, exist_ok=True)
        (output / REPORT).write_text(serialized)
        if fresh:
            (output / "fresh-download-evidence.json").write_text(serialized)
        else:
            (output / "fresh-download-evidence.json").unlink(missing_ok=True)
        print(f"[dependencies] READY: {len(dependencies)} Git + {len(archives)} archives", flush=True)
        return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/workspace/repo"))
    parser.add_argument("--output", type=Path, default=Path("/workspace/pkg-out"))
    parser.add_argument("--seed", type=Path, default=Path("/opt/fla-ci/third_party"))
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)
    prepare(args.repo, args.output, args.seed, args.fresh)


if __name__ == "__main__":
    main()
