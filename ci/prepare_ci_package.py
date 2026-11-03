#!/usr/bin/env python3
"""Prepare a full wheel, or a branch-matched Daily base with a safe OPP delta.

Inputs: CI_BUILD_KIND=full|incremental|torch|base, CI_BUILD_OPS, CI_BASE_REF,
CI_SOC, CI_JOBS. Outputs: CI_PACKAGE_INFO_FILE and CI_PACKAGE_ENV_FILE.
No test process should import fla_npu before the generated env has been sourced.
"""
from __future__ import annotations

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
        names = set(archive.namelist())
        prefix = f"fla_npu/opp/vendors/{VENDOR}/"
        required = {"fla_npu/libfla_npu_stable.so", prefix + "op_api/lib/libcust_opapi.so"}
        missing = sorted(required - names)
        for section in ("kernel", "config"):
            if not any(name.startswith(prefix + f"op_impl/ai_core/tbe/{section}/{soc}/")
                       for name in names):
                missing.append(f"{soc} {section}")
        if missing:
            raise ValueError(f"Daily base is incomplete or has the wrong SOC: {', '.join(missing)}")
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("Daily base must contain exactly one wheel METADATA")
        from email.parser import Parser
        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode("utf-8"))
        if (canonicalize_name(metadata.get("Name", "")) != canonicalize_name(package)
                or Version(metadata.get("Version", "0")) != version):
            raise ValueError("Daily filename and package METADATA disagree; republish the Daily wheel.")
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

    def download_base(self, directory: Path) -> tuple[Path, str, str]:
        prefix = branch_version(self.base_ref)
        version = self.env.get("CI_BASE_WHEEL_VERSION", "").strip()
        if version:
            validate_daily_version(Version(version), self.base_ref)
        index = (self.env.get("CI_BASE_WHEEL_INDEX_URL") or DAILY_INDEX).rstrip("/") + "/"
        if not index.startswith("https://"):
            raise ValueError("CI_BASE_WHEEL_INDEX_URL must use HTTPS")
        url = f"{index}{self.package}/latest/"
        # No index fallback: a missing branch/SOC/architecture must never pick PyPI
        # or a different release line. pip chooses the current Python/CPU tags.
        self.run([sys.executable, "-m", "pip", "download", "--no-index", "--find-links", url,
                  "--only-binary=:all:", "--pre", "--no-cache-dir", "--no-deps", "--dest", directory,
                  f"{self.package}=={version or prefix + '.*'}"])
        wheel = self.one_wheel(directory)
        selected_version = validate_base_wheel(wheel, package=self.package,
                                               base_ref=self.base_ref, soc=self.soc)
        return wheel, selected_version, url

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
                wheel, version, url = self.download_base(work / "base-wheel")
                info.update(base_version=version, base_url=url)
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


def main() -> int:
    try:
        PackagePreparer().prepare()
    except (ValueError, RuntimeError, OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        print(f"[CI][ERROR] Package preparation failed: {exc}", file=sys.stderr)
        print("[CI][FIX] Check the matching Daily base/build log and ABI compatibility; "
              "use /run-npu-ci full to rebuild a complete matching wheel.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
