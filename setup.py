import glob
import importlib
from importlib import metadata as importlib_metadata
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import re
import time
from pathlib import Path

from packaging.version import InvalidVersion, Version
from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py as _build_py

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except Exception:
    try:
        from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
    except Exception:
        _bdist_wheel = None


REPO_ROOT = Path(__file__).resolve().parent
TORCH_EXTENSION_DIR = REPO_ROOT / "torch_custom" / "fla_npu"
FLA_NPU_PACKAGE_DIR = TORCH_EXTENSION_DIR / "fla_npu"
TRITON_CORE_PACKAGE = "fla_npu.ops.triton.triton_core"
TRITON_CORE_SOURCE = REPO_ROOT / "fla" / "ops" / "triton" / "triton_core"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fla_npu_artifacts import get_package_version, get_wheel_build_tag  # noqa: E402


DEFAULT_SOC = "ascend910b"
DEFAULT_VENDOR_NAME = "fla_npu"
MIN_PYTHON = (3, 9)
MIN_TORCH = "2.6.0"
MIN_TRITON_ASCEND = "3.2.0"
MIN_TRITON_ASCEND_A5 = "3.2.1"
TORCH_NPU_GDN_FIX_MINIMUMS = {
    "2.7.1": "2.7.1.post5",
    "2.8.0": "2.8.0.post5",
    "2.9.0": "2.9.0.post3",
    "2.10.0": "2.10.0.post2",
    "2.11.0": "2.11.0rc3",
    "2.12.0": "2.12.0rc1",
}
MIN_TORCH_NPU_FUTURE_FIX_FAMILY = "2.13.0"
TORCH_NPU_GDN_FIX_RELEASE_URL = (
    "https://gitcode.com/Ascend/pytorch/releases?"
    "presetConfig={%22tags%22:229,%22release%22:122}"
)

TORCHNPUGEN_MODULES = (
    "torchnpugen.gen_op_plugin_functions",
    "torchnpugen.gen_derivatives",
    "torchnpugen.gen_op_backend",
    "torchnpugen.gen_backend_stubs",
    "torchnpugen.struct.gen_struct_opapi",
)


def _read_requirements():
    requirements = REPO_ROOT / "requirements.txt"
    if not requirements.exists():
        return []
    deps = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            deps.append(line)
    return deps


def _packages():
    packages = find_packages(
        where=str(TORCH_EXTENSION_DIR),
        include=["fla_npu", "fla_npu.*"],
    )
    if not TRITON_CORE_SOURCE.is_dir():
        raise FileNotFoundError(
            f"Triton source directory is missing: {TRITON_CORE_SOURCE}"
        )
    if TRITON_CORE_PACKAGE not in packages:
        packages.append(TRITON_CORE_PACKAGE)
    return packages


def _package_dir():
    return {
        "fla_npu": str(FLA_NPU_PACKAGE_DIR.relative_to(REPO_ROOT)),
        TRITON_CORE_PACKAGE: str(TRITON_CORE_SOURCE.relative_to(REPO_ROOT)),
    }


def _env_flag(name):
    return os.getenv(name, "FALSE").upper() in {"1", "TRUE", "YES", "ON"}


def _external_build_ops():
    raw_ops = os.getenv("FLA_NPU_OPS", "").strip()
    if not raw_ops:
        return []

    ops = []
    seen = set()
    for raw_op in raw_ops.split(","):
        op = raw_op.strip()
        if not op or not re.fullmatch(r"[a-z0-9_]+", op):
            raise RuntimeError(
                "FLA_NPU_OPS must be a comma-separated list of lowercase snake_case operator names"
            )
        if op not in seen:
            seen.add(op)
            ops.append(op)
    return ops


def _external_build_args():
    raw_args = os.getenv("FLA_NPU_BUILD_ARGS_JSON", "").strip()
    if not raw_args:
        return []
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise RuntimeError("FLA_NPU_BUILD_ARGS_JSON must contain a JSON string array") from exc
    if not isinstance(args, list) or not all(isinstance(arg, str) and arg for arg in args):
        raise RuntimeError("FLA_NPU_BUILD_ARGS_JSON must contain a JSON string array")

    reserved = ("--ops", "--soc", "--vendor_name", "--pkg")
    if any(arg.startswith(reserved) for arg in args):
        raise RuntimeError(
            "FLA_NPU_BUILD_ARGS_JSON cannot override --ops, --soc, --vendor_name, or --pkg"
        )
    return args


def _run(cmd, cwd, env=None):
    printable = " ".join(str(part) for part in cmd)
    print(f"[fla-npu build] START {printable}", flush=True)
    start = time.monotonic()
    try:
        subprocess.run(cmd, cwd=str(cwd), check=True, env=env)
    except subprocess.CalledProcessError:
        elapsed = time.monotonic() - start
        print(f"[fla-npu build] FAILED after {elapsed:.1f}s: {printable}", flush=True)
        raise
    elapsed = time.monotonic() - start
    print(f"[fla-npu build] DONE in {elapsed:.1f}s: {printable}", flush=True)


def _check_min_version(failures, name, actual, minimum):
    if not actual:
        failures.append(f"{name} version is unknown")
        return
    actual_version = _version_obj(actual)
    minimum_version = _version_obj(minimum)
    if actual_version is None:
        failures.append(f"{name} has an unsupported version string: {actual}")
    elif minimum_version is not None and actual_version < minimum_version:
        failures.append(f"{name}>={minimum} is required, got {actual}")


def _version_obj(value):
    try:
        return Version(value.split("+", 1)[0])
    except InvalidVersion:
        return None


def _version_key(value):
    parts = re.findall(r"\d+", value.split("+", 1)[0])
    if not parts:
        return None
    nums = [int(part) for part in parts[:3]]
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def _check_torch_npu_gdn_fix(failures, actual):
    actual_version = _version_obj(actual)
    if actual_version is None:
        failures.append(f"torch_npu has an unsupported version string: {actual}")
        return

    minimum = TORCH_NPU_GDN_FIX_MINIMUMS.get(actual_version.base_version)
    if minimum and actual_version >= Version(minimum):
        return

    if actual_version >= Version(MIN_TORCH_NPU_FUTURE_FIX_FAMILY):
        return

    if minimum is None:
        return

    requirements = ", ".join(
        f"{family}>={minimum}" for family, minimum in TORCH_NPU_GDN_FIX_MINIMUMS.items()
    )
    failures.append(
        "torch_npu must come from an Ascend PyTorch release that contains the "
        "GDN aclnn_extension stream fix. Packages from releases before "
        "v26.1.0-beta.1, such as v26.0.0-pytorch2.x, are rejected. "
        f"Expected one of: {requirements}, or torch_npu>={MIN_TORCH_NPU_FUTURE_FIX_FAMILY} "
        f"from {TORCH_NPU_GDN_FIX_RELEASE_URL}; got {actual}."
    )


def _check_triton_ascend_a5_compat(failures, actual):
    soc = os.getenv("FLA_NPU_SOC", DEFAULT_SOC)
    if soc != "ascend950":
        return
    actual_version = _version_obj(actual)
    if actual_version is None or actual_version < Version(MIN_TRITON_ASCEND_A5):
        failures.append(
            f"triton-ascend>={MIN_TRITON_ASCEND_A5} is required for FLA_NPU_SOC={soc}; got {actual}. "
            "triton-ascend 3.2.0 can crash on the A5 Triton runtime."
        )


def _distribution_version(name):
    try:
        return importlib_metadata.version(name)
    except Exception:
        return ""


def _detect_cann_version():
    candidates = []
    for env_name in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH"):
        value = os.getenv(env_name)
        if not value:
            continue
        path = Path(value)
        candidates.extend(
            [
                path / "version.info",
                path / "ascend_toolkit_install.info",
                path.parent / "version.info",
                path.parent / "ascend_toolkit_install.info",
            ]
        )

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines():
                stripped = line.strip()
                lower = stripped.lower()
                if "version" in lower and "=" in stripped:
                    return stripped
                if lower.startswith("version"):
                    return stripped
        except OSError:
            continue
    return "<unknown>"


def _check_build_environment():
    failures = []
    build_legacy_extension = _env_flag("FLA_NPU_BUILD_LEGACY_EXTENSION")

    if shutil.which("bash") is None:
        failures.append("bash is required")
    else:
        print(f"[fla-npu build][OK] bash: {shutil.which('bash')}")

    if not (os.getenv("ASCEND_HOME_PATH") or os.getenv("ASCEND_OPP_PATH")):
        failures.append(
            "ASCEND_HOME_PATH or ASCEND_OPP_PATH must be set. "
            "Please source the CANN set_env.sh before running pip install."
        )
    else:
        print(f"[fla-npu build][OK] ASCEND_HOME_PATH={os.getenv('ASCEND_HOME_PATH') or '<unset>'}")
        print(f"[fla-npu build][OK] ASCEND_OPP_PATH={os.getenv('ASCEND_OPP_PATH') or '<unset>'}")
        print(f"[fla-npu build][OK] CANN version: {_detect_cann_version()}")

    if sys.version_info < MIN_PYTHON:
        failures.append(
            f"Python>={MIN_PYTHON[0]}.{MIN_PYTHON[1]} is required, "
            f"got {sys.version_info.major}.{sys.version_info.minor}"
        )
    else:
        print(f"[fla-npu build][OK] Python: {sys.version.split()[0]}")

    if build_legacy_extension:
        torch = None
        torch_npu = None
        for module in ("torch", "torch_npu"):
            try:
                loaded = importlib.import_module(module)
                print(f"[fla-npu build][OK] {module}: {getattr(loaded, '__version__', '<unknown>')}")
                if module == "torch":
                    torch = loaded
                else:
                    torch_npu = loaded
            except Exception as exc:
                failures.append(f"{module}: {exc}")

        if torch is not None:
            _check_min_version(failures, "torch", getattr(torch, "__version__", ""), MIN_TORCH)
        if torch_npu is not None:
            _check_torch_npu_gdn_fix(failures, getattr(torch_npu, "__version__", ""))

        triton_ascend_version = _distribution_version("triton-ascend")
        try:
            triton = importlib.import_module("triton")
            print(f"[fla-npu build][OK] triton: {getattr(triton, '__file__', '<unknown>')}")
        except Exception as exc:
            failures.append(f"triton: {exc}")

        if triton_ascend_version:
            print(f"[fla-npu build][OK] triton-ascend: {triton_ascend_version}")
            _check_min_version(failures, "triton-ascend", triton_ascend_version, MIN_TRITON_ASCEND)
            _check_triton_ascend_a5_compat(failures, triton_ascend_version)
        else:
            failures.append("triton-ascend distribution was not found")
    else:
        print("[fla-npu build][OK] Skipping torch/torch_npu build-time checks for Python-only wheel")

    if build_legacy_extension and not _env_flag("FLA_NPU_SKIP_TORCH_GEN"):
        for module in TORCHNPUGEN_MODULES:
            try:
                spec = importlib.util.find_spec(module)
            except Exception as exc:
                failures.append(f"{module}: {exc}")
                continue
            if spec is None:
                failures.append(f"{module}: not found")
            else:
                print(f"[fla-npu build][OK] {module}: {spec.origin or 'namespace package'}")

    print(f"[fla-npu build][OK] FLA_NPU_SOC={os.getenv('FLA_NPU_SOC', DEFAULT_SOC)}")
    print(f"[fla-npu build][OK] FLA NPU vendor={DEFAULT_VENDOR_NAME}")

    if failures:
        raise RuntimeError(
            "Build environment check failed:\n  - "
            + "\n  - ".join(failures)
            + "\nInstall matching CANN and, when FLA_NPU_BUILD_LEGACY_EXTENSION=1, "
              "torch, torch_npu, torchnpugen and triton-ascend first, "
              "then run `pip install --no-build-isolation .`."
        )


def _find_single_run_package():
    run_files = sorted((REPO_ROOT / "build_out").glob("fla-npu-*.run"))
    if not run_files:
        raise RuntimeError("No fla-npu-*.run package found in build_out")
    if len(run_files) > 1:
        raise RuntimeError(
            "Multiple fla-npu-*.run packages found in build_out: "
            + ", ".join(str(path) for path in run_files)
        )
    return run_files[0]


def _install_run_package(run_file, install_path):
    run_file = Path(run_file)
    try:
        run_file.chmod(run_file.stat().st_mode | stat.S_IXUSR)
    except OSError:
        pass

    cmd = [str(run_file), "--quiet", f"--install-path={Path(install_path).resolve()}"]
    tmp_dir = Path(install_path).resolve().parent / ".run-installer-tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_dir)
    try:
        _run(cmd, REPO_ROOT, env=env)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_run_package():
    soc = os.getenv("FLA_NPU_SOC", DEFAULT_SOC)
    build_out = REPO_ROOT / "build_out"
    if build_out.exists():
        shutil.rmtree(build_out)
    cmd = [
        "bash",
        "build.sh",
        f"--soc={soc}",
        "--pkg",
        f"--vendor_name={DEFAULT_VENDOR_NAME}",
    ]
    ops = _external_build_ops()
    if ops:
        cmd.append(f"--ops={','.join(ops)}")
    cmd.extend(_external_build_args())
    _run(cmd, REPO_ROOT)

    return _find_single_run_package()


def _vendor_dir_name(vendor_name=None):
    vendor_name = vendor_name or DEFAULT_VENDOR_NAME
    return vendor_name if vendor_name.endswith("_transformer") else f"{vendor_name}_transformer"


def _find_staged_vendor_dir(opp_root):
    vendors_root = Path(opp_root) / "vendors"
    expected = vendors_root / _vendor_dir_name()
    if expected.exists():
        return expected

    vendor_dirs = [path for path in vendors_root.iterdir() if path.is_dir()] if vendors_root.exists() else []
    if len(vendor_dirs) == 1:
        return vendor_dirs[0]
    raise RuntimeError(
        "Unable to find staged custom OPP vendor directory under "
        f"{vendors_root}. Expected {_vendor_dir_name()}."
    )


def _rewrite_set_env(vendor_dir):
    vendor_dir = Path(vendor_dir)
    bin_dir = vendor_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    set_env = bin_dir / "set_env.bash"
    set_env.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                '_FLA_NPU_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                '_FLA_NPU_VENDOR_DIR="$(cd "${_FLA_NPU_SCRIPT_DIR}/.." && pwd)"',
                '_FLA_NPU_OPP_ROOT="$(cd "${_FLA_NPU_VENDOR_DIR}/../.." && pwd)"',
                "_fla_npu_prepend_path() {",
                '    local name="$1"',
                '    local value="$2"',
                "    local current=\"\"",
                '    if [[ -v "${name}" ]]; then',
                '        current="${!name}"',
                "    fi",
                '    case ":${current}:" in',
                '        *":${value}:"*) return 0 ;;',
                "    esac",
                '    printf -v "${name}" "%s" "${value}${current:+:${current}}"',
                '    export "${name}"',
                "}",
                '_fla_npu_prepend_path ASCEND_CUSTOM_OPP_PATH "${_FLA_NPU_VENDOR_DIR}"',
                '_fla_npu_prepend_path ASCEND_CUSTOM_OPP_PATH "${_FLA_NPU_OPP_ROOT}"',
                '_fla_npu_prepend_path LD_LIBRARY_PATH "${_FLA_NPU_VENDOR_DIR}/op_api/lib"',
                'export FLA_NPU_OPP_PATH="${_FLA_NPU_OPP_ROOT}"',
                'export FLA_NPU_OP_API_LIB="${_FLA_NPU_VENDOR_DIR}/op_api/lib/libcust_opapi.so"',
                "unset -f _fla_npu_prepend_path",
                "unset _FLA_NPU_SCRIPT_DIR _FLA_NPU_VENDOR_DIR _FLA_NPU_OPP_ROOT",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_vendors_config(vendor_dir):
    vendor_dir = Path(vendor_dir)
    config_file = vendor_dir.parent / "config.ini"
    config_file.write_text(f"load_priority={vendor_dir.name}\n", encoding="utf-8")


def _has_glob(root, pattern):
    return any(Path(root).glob(pattern))


def _has_any_glob(root, patterns):
    return any(_has_glob(root, pattern) for pattern in patterns)


def _validate_staged_opp(vendor_dir):
    vendor_dir = Path(vendor_dir)
    conflicting_opapi = vendor_dir / "op_api" / "lib" / "libopapi.so"
    if conflicting_opapi.exists() or conflicting_opapi.is_symlink():
        raise RuntimeError(
            "Embedded OPP must not contain libopapi.so because it shadows the "
            f"CANN runtime library: {conflicting_opapi}"
        )

    required_groups = {
        "custom op_api library": ("op_api/lib/libcust_opapi.so",),
        "host/proto shared library": (
            "op_impl/ai_core/tbe/op_host/lib/linux/*/libophost*.so",
            "op_impl/ai_core/tbe/op_tiling/lib/linux/*/*.so",
            "op_proto/lib/linux/*/*.so",
            "op_proto/es/lib/linux/*/*.so",
        ),
        "AI Core kernel object": ("op_impl/ai_core/tbe/kernel/*/*/*.o",),
        "binary info config": (
            "op_impl/ai_core/tbe/config/*/binary_info_config.json",
            "op_impl/ai_core/tbe/kernel/config/*/binary_info_config.json",
        ),
    }
    missing = [
        description
        for description, patterns in required_groups.items()
        if not _has_any_glob(vendor_dir, patterns)
    ]
    if missing:
        raise RuntimeError(
            f"Embedded OPP under {vendor_dir} is incomplete; missing "
            + ", ".join(missing)
        )

    aicpu_config = vendor_dir / "op_impl" / "cpu" / "config"
    aicpu_impl = vendor_dir / "op_impl" / "cpu" / "aicpu_kernel" / "impl"
    if aicpu_config.exists() and not _has_glob(aicpu_config, "*aicpu*.json"):
        raise RuntimeError(f"Embedded OPP AICPU config is incomplete under {aicpu_config}")
    if aicpu_impl.exists() and not _has_glob(aicpu_impl, "*.so"):
        raise RuntimeError(f"Embedded OPP AICPU impl is incomplete under {aicpu_impl}")


def _stage_run_package(run_file, opp_root):
    opp_root = Path(opp_root).resolve()
    if opp_root.exists():
        shutil.rmtree(opp_root)
    opp_root.mkdir(parents=True, exist_ok=True)

    _install_run_package(run_file, opp_root)
    vendor_dir = _find_staged_vendor_dir(opp_root)
    _write_vendors_config(vendor_dir)
    _rewrite_set_env(vendor_dir)

    op_api_lib = vendor_dir / "op_api" / "lib" / "libcust_opapi.so"
    if not op_api_lib.exists():
        raise RuntimeError(f"Embedded OPP is missing {op_api_lib}")
    op_api_alias = op_api_lib.with_name("libopapi.so")
    if op_api_alias.exists() or op_api_alias.is_symlink():
        op_api_alias.unlink()
        print(
            "[fla-npu build] Removed conflicting custom libopapi.so alias; "
            "CANN must provide libopapi.so"
        )
    _validate_staged_opp(vendor_dir)
    print(f"[fla-npu build] Embedded OPP staged at {vendor_dir}")


def _cleanup_build_residuals():
    """Remove setuptools egg-info artifacts from the repo root.

    Building a wheel from the source root leaves ``*.egg-info`` directories
    behind. When the repo root is on ``sys.path`` (e.g. the CANN set_env.sh
    appends a trailing ``:`` to PYTHONPATH, which makes Python add the current
    directory), those stale metadata dirs shadow the real installed
    distribution: ``pip show`` reports the source root as Location and
    ``pip install --force-reinstall`` fails to uninstall the previous build.
    """
    for egg_info in glob.glob(str(REPO_ROOT / "*.egg-info")):
        shutil.rmtree(egg_info, ignore_errors=True)


def _build_torch_extension_inplace():
    for so_file in FLA_NPU_PACKAGE_DIR.glob("custom_aclnn_extension_lib*.so"):
        so_file.unlink()

    if not _env_flag("FLA_NPU_BUILD_LEGACY_EXTENSION"):
        print("[fla-npu build] Skipping legacy PyTorch C++ extension build for Python-only wheel")
        return

    if not _env_flag("FLA_NPU_SKIP_TORCH_GEN"):
        _run(["bash", "gen.sh", "npu_custom.yaml"], TORCH_EXTENSION_DIR)

    build_dir = TORCH_EXTENSION_DIR / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

    _run([sys.executable, "setup.py", "build_ext", "--force", "--inplace"], TORCH_EXTENSION_DIR)

    so_files = sorted(FLA_NPU_PACKAGE_DIR.glob("custom_aclnn_extension_lib*.so"))
    if not so_files:
        raise RuntimeError(
            "custom_aclnn_extension_lib*.so was not produced under "
            f"{FLA_NPU_PACKAGE_DIR}"
        )


_EXTERNAL_BUILD_DONE = False
_RUN_PACKAGE = None


class FlaNpuBuildPy(_build_py):
    def run(self):
        global _EXTERNAL_BUILD_DONE, _RUN_PACKAGE
        if not _EXTERNAL_BUILD_DONE:
            _check_build_environment()
            _RUN_PACKAGE = _build_run_package()
            _build_torch_extension_inplace()
            _EXTERNAL_BUILD_DONE = True

        built_package_dir = Path(self.build_lib) / "fla_npu"
        if built_package_dir.exists():
            shutil.rmtree(built_package_dir)
        super().run()
        run_package = _RUN_PACKAGE or _find_single_run_package()
        _stage_run_package(run_package, Path(self.build_lib) / "fla_npu" / "opp")


class BinaryDistribution(Distribution):
    def is_pure(self):
        return True

    def has_ext_modules(self):
        return False


CMDCLASS = {"build_py": FlaNpuBuildPy}

if _bdist_wheel is not None:
    class FlaNpuBdistWheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = True
            build_tag = get_wheel_build_tag(REPO_ROOT)
            if build_tag:
                self.build_number = build_tag

        def run(self):
            super().run()
            _cleanup_build_residuals()

    CMDCLASS["bdist_wheel"] = FlaNpuBdistWheel


setup(
    name="flash-linear-attention-npu",
    version=get_package_version(REPO_ROOT),
    description="High-performance linear attention operators for Ascend NPU",
    long_description=(REPO_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=_packages(),
    package_dir=_package_dir(),
    package_data={"fla_npu": ["opp/**/*"]},
    include_package_data=True,
    license_files=[
        "LICENSE",
        "NOTICE",
        "LICENSES/BSD-3-Clause.txt",
        "LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt",
    ],
    distclass=BinaryDistribution,
    cmdclass=CMDCLASS,
    python_requires=">=3.9",
    install_requires=_read_requirements(),
)
