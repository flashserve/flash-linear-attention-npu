#!/usr/bin/env python3
"""Prepare a full wheel, or a branch-matched Daily base with a safe OPP delta.

Inputs: CI_BUILD_KIND=full|incremental|torch|base, CI_BUILD_OPS, CI_BASE_REF,
CI_SOC, CI_JOBS, CI_PACKAGE_CACHE, CI_UPDATE_PACKAGE=true|false.
Outputs: CI_PACKAGE_INFO_FILE and CI_PACKAGE_ENV_FILE. --update-package only
refreshes the compatible Daily cache; it never installs or compiles a package.
No test process should import fla_npu before the generated env has been sourced.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile

try:
    from packaging.tags import sys_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import Version
except ImportError:  # pip is already required in every CI build image.
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename
    from pip._vendor.packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[1]
DAILY_INDEX = "https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/"
TIERS = {"ascend910b": "a2", "ascend910_93": "a3", "ascend950": "a5"}
OVERLAY = "fla_npu_overlay_transformer"
VENDOR = "fla_npu_transformer"
DEFAULT_PACKAGE_CACHE = "/home/huangjunzhe/ci/pkg-cache"
BEIJING = timezone(timedelta(hours=8))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shared_directory(path: Path):
    """Keep a setgid group on newly created cache directories only."""
    try:
        path.mkdir(mode=0o2775)
    except FileExistsError:
        return
    if os.name != "nt":
        path.chmod(0o2775)


@contextmanager
def package_cache_lock(path: Path):
    """A persistent lock inode serializes readers, daily refresh and publication."""
    shared_directory(path.parent)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o664)
        created = True
        if os.name != "nt":
            os.fchmod(descriptor, 0o664)
    except FileExistsError:
        descriptor = os.open(path, os.O_RDWR)
        created = False
    with os.fdopen(descriptor, "r+b") as handle:
        if os.name == "nt":
            import msvcrt
            if created:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def branch_version(base_ref: str) -> str:
    """Use the Daily publisher's distinct main/maintenance version namespaces."""
    if base_ref == "main":
        return "0"
    if re.fullmatch(r"v\d+\.\d+\.\d+", base_ref):
        return base_ref[1:]
    raise ValueError(
        f"No Daily base mapping for target branch {base_ref!r}. "
        "Publish a matching main/vX.Y.Z Daily base, or request /run-npu-ci full."
    )


def validate_daily_version(version: Version, base_ref: str) -> None:
    prefix = branch_version(base_ref)
    expected_release = tuple(int(value) for value in prefix.split("."))
    if version.release != expected_release or not version.is_devrelease:
        raise ValueError(
            f"Daily version {version} does not belong to target branch {base_ref}; "
            "select its matching Daily build or request /run-npu-ci full."
        )
    if base_ref == "main" and not (version.local or "").startswith("main."):
        raise ValueError(f"Main Daily base must carry +main.<SHA>, got {version}")
    if base_ref != "main" and (version.local or "").startswith("main."):
        raise ValueError(f"Main Daily base cannot be used for {base_ref}: {version}")


def installed_wheel_paths(archive: zipfile.ZipFile, stem: str) -> set[str]:
    """Map only this wheel's root and library schemes to installed package paths."""
    library_prefixes = (f"{stem}.data/purelib/", f"{stem}.data/platlib/")
    paths = set()
    for entry in archive.infolist():
        # ZipInfo normalizes Windows separators and strips NUL suffixes from
        # filename; validate the original ZIP name before trusting that view.
        original = entry.orig_filename
        archive_path = original[:-1] if original.endswith("/") else original
        components = archive_path.split("/")
        if ("\\" in original or "\0" in original or re.match(r"^[A-Za-z]:", archive_path)
                or any(part in {"", ".", ".."} for part in components)):
            raise ValueError(f"Daily base has an invalid ZIP path: {entry.filename}")
        if (len(components) >= 3 and components[0].endswith(".data")
                and components[1] in {"purelib", "platlib"} and components[2] == "fla_npu"
                and components[0] != f"{stem}.data"):
            raise ValueError(f"Daily base has a foreign data directory for fla_npu: {entry.filename}")
        if entry.is_dir():
            continue
        relative = entry.filename
        for prefix in library_prefixes:
            if relative.startswith(prefix):
                relative = relative[len(prefix):]
                break
        if not relative.startswith("fla_npu/"):
            continue
        if relative in paths:
            raise ValueError(f"Daily base has conflicting installed package entries: {relative}")
        paths.add(relative)
    return paths


def validate_base_wheel(wheel: Path, *, package: str, base_ref: str, soc: str) -> str:
    name, version, _build, tags = parse_wheel_filename(wheel.name)
    if canonicalize_name(name) != canonicalize_name(package):
        raise ValueError(f"Expected {package} for {soc}, downloaded {wheel.name}")
    cpu = platform.machine().lower()
    cpu = {"amd64": "x86_64", "arm64": "aarch64"}.get(cpu, cpu)
    compatible = tags.intersection(sys_tags())
    if not any(tag.platform.endswith("_" + cpu) for tag in compatible):
        raise ValueError(
            f"Daily base {wheel.name} is incompatible with Python {sys.version_info.major}."
            f"{sys.version_info.minor} / CPU {platform.machine()}; publish a matching wheel."
        )
    validate_daily_version(version, base_ref)
    with zipfile.ZipFile(wheel) as archive:
        if archive.testzip() is not None:
            raise ValueError("Daily base wheel has corrupt ZIP contents; download or publish it again.")
        metadata_paths = [entry.filename for entry in archive.infolist()
                          if not entry.is_dir() and entry.filename.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1 or not re.fullmatch(r"[^/]+\.dist-info/METADATA", metadata_paths[0]):
            raise ValueError("Daily base must contain exactly one top-level wheel METADATA")
        stem = metadata_paths[0][:-len(".dist-info/METADATA")]
        distribution, separator, metadata_version = stem.rpartition("-")
        if (not separator or canonicalize_name(distribution) != canonicalize_name(package)
                or Version(metadata_version) != version):
            raise ValueError("Daily filename and METADATA directory disagree; republish the Daily wheel.")
        from email.parser import Parser
        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode("utf-8"))
        if (canonicalize_name(metadata.get("Name", "")) != canonicalize_name(package)
                or Version(metadata.get("Version", "0")) != version):
            raise ValueError("Daily filename and package METADATA disagree; republish the Daily wheel.")
        names = installed_wheel_paths(archive, stem)
        prefix = f"fla_npu/opp/vendors/{VENDOR}/"
        required = {"fla_npu/libfla_npu_stable.so", prefix + "op_api/lib/libcust_opapi.so"}
        missing = sorted(required - names)
        for section in ("kernel", "config"):
            if not any(name.startswith(prefix + f"op_impl/ai_core/tbe/{section}/{soc}/")
                       for name in names):
                missing.append(f"{soc} {section}")
        if missing:
            raise ValueError(f"Daily base is incomplete or has the wrong SOC: {', '.join(missing)}")
    return str(version)


def installed_package() -> Path:
    # find_spec on a top-level package does not import it or initialize CANN.
    spec = importlib.util.find_spec("fla_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("Installed fla_npu package cannot be found after pip install")
    return Path(spec.origin).resolve().parent


class PackagePreparer:
    def __init__(self, repo: Path = REPO_ROOT, environ=None):
        self.repo = repo
        self.env = dict(os.environ if environ is None else environ)
        self.kind = self.env.get("CI_BUILD_KIND") or "full"
        self.ops = self.env.get("CI_BUILD_OPS", "").strip()
        self.base_ref = self.env.get("CI_BASE_REF", "").strip()
        self.soc = self.env.get("CI_SOC") or "ascend910b"
        jobs = self.env.get("CI_JOBS") or "16"
        if self.kind not in {"full", "incremental", "torch", "base"}:
            raise ValueError(f"Unknown CI_BUILD_KIND: {self.kind!r}")
        if self.soc not in TIERS:
            raise ValueError(f"Unsupported CI_SOC: {self.soc!r}")
        if not jobs.isdigit() or int(jobs) < 1:
            raise ValueError("CI_JOBS must be a positive integer")
        self.jobs = str(min(16, int(jobs)))
        if self.kind == "incremental":
            if not re.fullmatch(r"[a-z][a-z0-9_]*(,[a-z][a-z0-9_]*)*", self.ops):
                raise ValueError("Incremental compilation requires nonempty comma-separated operator names")
        elif self.ops:
            raise ValueError(f"CI_BUILD_OPS must be empty for {self.kind} preparation")
        self.package = f"flash-linear-attention-npu-{TIERS[self.soc]}"
        refresh = (self.env.get("CI_UPDATE_PACKAGE") or "false").strip().lower()
        if refresh not in {"true", "false"}:
            raise ValueError("CI_UPDATE_PACKAGE must be true or false")
        self.refresh_package = refresh == "true"
        self.package_cache = self.path(self.env.get("CI_PACKAGE_CACHE") or DEFAULT_PACKAGE_CACHE)
        self.base_cache_info = {}
        self.info_file = self.path(self.env.get("CI_PACKAGE_INFO_FILE") or ".ci-tmp/npu-ci-package.json")
        self.env_file = self.path(self.env.get("CI_PACKAGE_ENV_FILE") or ".ci-tmp/npu-ci-package-env.sh")
        self.env.update(FLA_NPU_SOC=self.soc, FLA_NPU_BUILD_ARGS=f"-j{self.jobs}",
                        CMAKE_BUILD_PARALLEL_LEVEL=self.jobs, MAX_JOBS=self.jobs,
                        FLA_NPU_OVERLAY_VENDOR="")
        self.env.pop("FLA_NPU_OPS", None)
        if self.base_ref:
            self.env["FLA_NPU_BRANCH_NAME"] = self.base_ref

    def path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.repo / path

    def run(self, args, *, cwd=None):
        print("[CI][PACKAGE] " + shlex.join([str(arg) for arg in args]), flush=True)
        subprocess.run([str(arg) for arg in args], cwd=cwd or self.repo, env=self.env, check=True)

    def pip_install(self, wheel: Path):
        self.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir",
                  "--no-deps", wheel])

    def one_wheel(self, directory: Path) -> Path:
        wheels = sorted(directory.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected exactly one new wheel under {directory}, found {len(wheels)}")
        return wheels[0]

    def daily_source(self) -> tuple[str, str, str]:
        prefix = branch_version(self.base_ref)
        version = self.env.get("CI_BASE_WHEEL_VERSION", "").strip()
        if version:
            validate_daily_version(Version(version), self.base_ref)
        index = (self.env.get("CI_BASE_WHEEL_INDEX_URL") or DAILY_INDEX).rstrip("/") + "/"
        if not index.startswith("https://"):
            raise ValueError("CI_BASE_WHEEL_INDEX_URL must use HTTPS")
        url = f"{index}{self.package}/latest/"
        return prefix, version, url

    def download_base(self, directory: Path) -> tuple[Path, str, str]:
        prefix, version, url = self.daily_source()
        # No index fallback: a missing branch/SOC/architecture must never pick PyPI
        # or a different release line. pip chooses the current Python/CPU tags.
        self.run([sys.executable, "-m", "pip", "download", "--no-index", "--find-links", url,
                  "--only-binary=:all:", "--pre", "--no-cache-dir", "--no-deps", "--dest", directory,
                  f"{self.package}=={version or prefix + '.*'}"])
        wheel = self.one_wheel(directory)
        selected_version = validate_base_wheel(wheel, package=self.package,
                                               base_ref=self.base_ref, soc=self.soc)
        return wheel, selected_version, url

    def cached_base(self, directory: Path, *, force: bool = False) -> tuple[Path, str, str]:
        """Copy a verified Daily into this run; refresh each Beijing day once.

        Downloaded generations are immutable. Holding the lock until the copy
        finishes lets publication prune old generations without racing installers.
        A refresh failure propagates, even when an older manifest still exists.
        """
        _prefix, requested_version, url = self.daily_source()
        identity = {
            "schema": 1, "base_ref": self.base_ref, "soc": self.soc,
            "package": self.package, "url": url, "requested_version": requested_version,
            "cpu": {"amd64": "x86_64", "arm64": "aarch64"}.get(
                platform.machine().lower(), platform.machine().lower()),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "python_tags": [str(tag) for tag in sys_tags()],
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.package_cache.parent.mkdir(parents=True, exist_ok=True)
        shared_directory(self.package_cache)
        cache = self.package_cache / key
        manifest = cache / "manifest.json"
        requested_at = utc_now()
        force = force or self.refresh_package
        with package_cache_lock(cache / ".lock"):
            today = utc_now().astimezone(BEIJING).date().isoformat()
            cached = None
            try:
                metadata = json.loads(manifest.read_text(encoding="utf-8"))
                filename, digest = metadata["wheel"], metadata["sha256"]
                if (metadata["identity"] != identity or not isinstance(filename, str)
                        or Path(filename).name != filename or "/" in filename or "\\" in filename
                        or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                    raise ValueError("Daily cache identity or filename is invalid")
                refreshed_at = datetime.fromisoformat(metadata["refreshed_at"])
                started_at = datetime.fromisoformat(metadata["refresh_started_at"])
                if (refreshed_at.tzinfo is None or started_at.tzinfo is None
                        or started_at > refreshed_at or refreshed_at > utc_now()):
                    raise ValueError("Daily cache refresh timestamp is invalid")
                if started_at.astimezone(BEIJING).date().isoformat() != metadata["refreshed_date"]:
                    raise ValueError("Daily cache refresh date disagrees with its timestamp")
                wheel = cache / "wheels" / digest / filename
                if not wheel.is_file() or sha256_file(wheel) != digest:
                    raise ValueError("Daily cache wheel is missing or its SHA256 changed")
                version = validate_base_wheel(wheel, package=self.package,
                                              base_ref=self.base_ref, soc=self.soc)
                if version != metadata["version"] or (requested_version and version != requested_version):
                    raise ValueError("Daily cache version disagrees with its metadata")
                if (metadata["refreshed_date"] == today
                        and (not force or refreshed_at >= requested_at)):
                    cached = (wheel, version, metadata)
            except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
                # Missing or damaged cache is repaired by a complete new download.
                # No invalid or stale cache can become a successful fallback.
                pass
            cache_hit = cached is not None
            if cached is None:
                print(f"[CI][PACKAGE] Refresh Daily cache for {self.base_ref}/{self.soc} ({today})",
                      flush=True)
                with tempfile.TemporaryDirectory(prefix=".download-", dir=cache) as tmp:
                    started_at = utc_now()
                    wheel, version, _url = self.download_base(Path(tmp))
                    digest = sha256_file(wheel)
                    destination = cache / "wheels" / digest / wheel.name
                    shared_directory(cache / "wheels")
                    shared_directory(destination.parent)
                    if os.name != "nt":
                        wheel.chmod(0o664)
                    wheel.replace(destination)
                    refreshed_at = utc_now()
                    metadata = {
                        "identity": identity, "wheel": destination.name, "version": version,
                        "sha256": digest, "refreshed_at": refreshed_at.isoformat(),
                        "refresh_started_at": started_at.isoformat(),
                        "refreshed_date": started_at.astimezone(BEIJING).date().isoformat(),
                    }
                    temporary_manifest = Path(tmp) / "manifest.json"
                    temporary_manifest.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n",
                                                  encoding="utf-8")
                    if os.name != "nt":
                        temporary_manifest.chmod(0o664)
                    temporary_manifest.replace(manifest)
                    cached = (destination, version, metadata)
            wheel, version, metadata = cached
            directory.mkdir(parents=True, exist_ok=True)
            copied = directory / wheel.name
            shutil.copyfile(wheel, copied)
            self.base_cache_info = {
                "path": str(cache), "hit": cache_hit, "sha256": metadata["sha256"],
                "refreshed_date": metadata["refreshed_date"],
                "refreshed_at": metadata["refreshed_at"],
            }
            # Only per-run copies are installed; obsolete generations cannot be
            # in use after obtaining this lock. Keep the current generation only.
            for generation in (cache / "wheels").iterdir():
                if generation.name != metadata["sha256"] and generation.is_dir():
                    shutil.rmtree(generation)
            return copied, version, url

    def update_package(self):
        """Refresh the compatible cache without pip install, build or NPU access."""
        self.info_file.parent.mkdir(parents=True, exist_ok=True)
        self.info_file.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="ci-package-update-", dir=self.info_file.parent) as tmp:
            wheel, version, url = self.cached_base(Path(tmp), force=True)
            info = {"kind": "update", "base_ref": self.base_ref, "soc": self.soc,
                    "base_version": version, "base_url": url, "wheel": wheel.name,
                    "cache": self.base_cache_info}
            self.info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[CI][PACKAGE] Updated Daily cache: {json.dumps(info, ensure_ascii=False)}", flush=True)

    def build_incremental(self, staging: Path):
        self.run(["bash", "ci/prepare_ci_cache.sh"])
        self.run(["bash", "build.sh", "--pkg", f"--soc={self.soc}", "--vendor_name=fla_npu",
                  f"--ops={self.ops}", f"-j{self.jobs}"])
        # build_out is cleaned by build.sh; require a single fresh run package.
        runs = sorted((self.repo / "build_out").glob("fla_npu_linux-*.run"))
        if len(runs) != 1:
            raise RuntimeError(f"Expected one incremental run package, found {len(runs)}")
        self.run(["bash", runs[0], f"--install-path={staging}", "--quiet"])
        # The existing installer checks overlay/base ABI and preserves every
        # untouched base entry. A partial .run --install would truncate the base.
        self.run([sys.executable, "-m", "fla_npu_opp_manage", "--overlay", "--source",
                  staging / "vendors", "--force"])
        self.env["FLA_NPU_OVERLAY_VENDOR"] = OVERLAY

    def build_torch(self, work: Path):
        package_dir = installed_package()
        vendor = package_dir / "opp/vendors" / VENDOR
        includes = [vendor / "op_api/include/aclnnop"]
        cann = Path(self.env.get("ASCEND_HOME_PATH") or self.env.get("ASCEND_TOOLKIT_HOME")
                    or "/usr/local/Ascend/ascend-toolkit/latest")
        if (cann / "include/aclnnop").is_dir():
            includes.append(cann / "include/aclnnop")
        includes.extend(sorted(cann.glob("*/include/aclnnop")))
        self.run([sys.executable, "torch_custom/fla_npu/tools/op_abi_validate.py", "--opp-include", *includes])
        backup = work / "base-opp"
        shutil.copytree(package_dir / "opp", backup, symlinks=True)
        runtime = self.repo / "torch_custom/fla_npu"
        # Standalone build.sh requires a .run and overwrites OPP; use setup.py
        # to rebuild only the runtime, then retain the complete downloaded base.
        self.run([sys.executable, "setup.py", "bdist_wheel", "--dist-dir", work / "torch-wheel"], cwd=runtime)
        self.pip_install(self.one_wheel(work / "torch-wheel"))
        package_dir = installed_package()
        shutil.rmtree(package_dir / "opp")
        shutil.copytree(backup, package_dir / "opp", symlinks=True)
        self.run([sys.executable, "scripts/package/ops_transformer/scripts/finalize_wheel_opp.py",
                  "--package-dir", package_dir])

    def prepare(self):
        self.env_file.parent.mkdir(parents=True, exist_ok=True)
        self.env_file.unlink(missing_ok=True)
        self.info_file.unlink(missing_ok=True)
        info = {"kind": self.kind, "ops": self.ops.split(",") if self.ops else [],
                "base_ref": self.base_ref, "soc": self.soc, "cpu": platform.machine(),
                "python": platform.python_version(), "jobs": int(self.jobs)}
        with tempfile.TemporaryDirectory(prefix="ci-package-", dir=self.env_file.parent) as tmp:
            work = Path(tmp)
            if self.kind == "full":
                self.run(["bash", "ci/prepare_ci_cache.sh"])
                self.run([sys.executable, "scripts/build_wheel.py", "--wheel-dir", work / "full-wheel"])
                wheel = self.one_wheel(work / "full-wheel")
                self.pip_install(wheel)
            else:
                wheel, version, url = self.cached_base(work / "base-wheel")
                info.update(base_version=version, base_url=url, cache=self.base_cache_info)
                self.pip_install(wheel)
                if self.kind == "incremental":
                    self.build_incremental(work / "delta-opp")
                elif self.kind == "torch":
                    self.build_torch(work)
            info["wheel"] = wheel.name
            package_dir = installed_package()
            env_path = package_dir / "opp/vendors" / VENDOR / "bin/set_env.bash"
            opapi = env_path.parents[1] / "op_api/lib/libcust_opapi.so"
            if not env_path.is_file() or not opapi.is_file():
                raise RuntimeError("Installed package lacks its complete base OPP or bin/set_env.bash")
            exports = {"FLA_NPU_ENV": str(env_path), "FLA_NPU_OPP_PATH": str(package_dir / "opp"),
                       "FLA_NPU_OVERLAY_VENDOR": self.env["FLA_NPU_OVERLAY_VENDOR"]}
            info.update(package_dir=str(package_dir), environment_file=str(self.env_file))
            self.info_file.parent.mkdir(parents=True, exist_ok=True)
            self.info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            env_tmp = self.env_file.with_suffix(".tmp")
            env_tmp.write_text("# Generated after package installation completed successfully.\n" + "".join(
                f"export {key}={shlex.quote(value)}\n" for key, value in exports.items()), encoding="utf-8")
            env_tmp.replace(self.env_file)
        print(f"[CI][PACKAGE] Prepared {self.kind}: {json.dumps(info, ensure_ascii=False)}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-package", action="store_true",
                        help="Refresh only the matching Daily cache; do not install or compile")
    args = parser.parse_args(argv)
    try:
        preparer = PackagePreparer()
        if args.update_package:
            preparer.update_package()
        else:
            preparer.prepare()
    except (ValueError, RuntimeError, OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        print(f"[CI][ERROR] Package preparation failed: {exc}", file=sys.stderr)
        print("[CI][FIX] Check the matching Daily base/build log and ABI compatibility; "
              "use /run-npu-ci full to rebuild a complete matching wheel.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
