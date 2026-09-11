"""Import-time version guards for tiered PyPI wheels.

The guard is inert for legacy/local builds that lack the generated
``_build_meta.py`` / ``_compat.py`` files; only tiered PyPI wheels
(flash-linear-attention-npu-a2/a3/a5) get version checks.

Notes:
- Chip-tier detection via device-name mapping was intentionally NOT included:
  device-name -> tier normalization is not deterministic across hardware
  generations (a3 has not been calibrated on real hardware). Revisit only
  after device names are collected on every supported chip.
- Installed package versions are read with importlib.metadata (no import).
- Version tables mirror scripts/npu_compat.py through the build-generated
  ``_compat.py`` (single source of truth in the repo).
- Version comparison is dependency-free (numeric tuple compare).
- Advisory only: every check emits a RuntimeWarning and import continues, so
  installing a tiered wheel never blocks an environment that used to import
  (this module adds no new hard failures versus legacy/source installs).
- The release-validated matrix (VALIDATED_COMBOS) is NOT enforced here either;
  an environment above the documented minimums is usable even when it was not
  part of the release test matrix (see scripts/check_npu_env.py / README).
"""

from __future__ import annotations

import importlib.metadata
import os
import pathlib
import re
import warnings

_PKG_DIR = pathlib.Path(__file__).resolve().parent


_PRERELEASE_ORDER = {"dev": 0, "a": 1, "b": 2, "rc": 3}


def _num(value: str) -> tuple:
    """PEP 440-ish ordering key, e.g. '2.7.1.post5' -> (2,7,1,5,5),
    '2.12.0rc1' -> (2,12,0,3,1), '2.12.0' -> (2,12,0,4,0).

    Final releases sort after rc/beta/dev and before post releases, so a final
    build satisfies a minimum like 2.12.0rc1.
    """
    base = value.split("+", 1)[0].split("-", 1)[0]
    match = re.match(r"^(\d+(?:\.\d+)*)(.*)$", base)
    release = tuple(int(part) for part in match.group(1).split(".")) if match else ()
    tail = (match.group(2) if match else "").lower().lstrip(".")
    if not tail:
        return release + (4, 0)  # final
    suffix = re.match(r"^(dev|a|b|rc|post)(\d*)$", tail)
    if not suffix:
        return release + (4, 0)
    order = _PRERELEASE_ORDER.get(suffix.group(1), 5)  # post -> 5
    number = int(suffix.group(2) or 0)
    return release + (order, number)


def _load_module_value(filename: str, key: str):
    path = _PKG_DIR / filename
    try:
        namespace: dict = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        return namespace.get(key)
    except Exception:
        return None


def _compat() -> dict:
    return {
        "MIN_CANN": _load_module_value("_compat.py", "MIN_CANN"),
        "MIN_TORCH": _load_module_value("_compat.py", "MIN_TORCH"),
        "TORCH_NPU_GDN_FIX_MINIMUMS": _load_module_value(
            "_compat.py", "TORCH_NPU_GDN_FIX_MINIMUMS"
        )
        or {},
    }


def detect_cann_version() -> str | None:
    """Best-effort CANN toolkit version, mirroring check_npu_env.py ordering.

    The OPP install dir's version.info is authoritative (e.g. Version=9.1.0);
    driver-version-like values are skipped.
    """
    candidates: list[str] = []
    for env_name in ("ASCEND_OPP_PATH", "ASCEND_HOME_PATH"):
        base = os.getenv(env_name)
        if not base:
            continue
        path = os.path.abspath(base)
        candidates.extend(
            [
                os.path.join(path, "version.info"),
                os.path.join(os.path.dirname(path), "version.info"),
                os.path.join(path, "ascend_toolkit_install.info"),
                os.path.join(os.path.dirname(path), "ascend_toolkit_install.info"),
            ]
        )
    for candidate in candidates:
        try:
            with open(candidate, encoding="utf-8", errors="ignore") as file:
                lines = file.read().splitlines()
        except OSError:
            continue
        for line in lines:
            if "driver" in line.lower():
                continue
            key, _, value = line.partition("=")
            if key.strip().lower() != "version":
                continue
            version = value.strip().strip('"')
            if version:
                return version
    return None


def _dist_version(dist_name: str) -> str | None:
    for candidate in (dist_name, dist_name.replace("-", "_"), dist_name.replace("_", "-")):
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _check_versions(compat: dict) -> None:
    min_cann = compat.get("MIN_CANN")
    min_torch = compat.get("MIN_TORCH")
    if not min_cann or not min_torch:
        return  # legacy wheel without generated _compat.py
    table = compat.get("TORCH_NPU_GDN_FIX_MINIMUMS") or {}

    cann = detect_cann_version()
    if cann and _num(cann) < _num(min_cann):
        warnings.warn(
            f"fla_npu: detected CANN {cann}, below the recommended minimum "
            f"{min_cann}; import continues, but operators may fail. See the "
            "README install guide.",
            RuntimeWarning,
            stacklevel=2,
        )

    torch_version = _dist_version("torch")
    if torch_version is None:
        return  # torch not installed yet; OPP import alone stays torch-free
    torch_nums = _num(torch_version)
    if torch_nums < _num(min_torch):
        warnings.warn(
            f"fla_npu: detected torch {torch_version}, below the recommended "
            f"minimum {min_torch}; import continues, but operators may fail. "
            "See the README for a supported torch/torch_npu combination.",
            RuntimeWarning,
            stacklevel=2,
        )

    torch_npu_version = _dist_version("torch-npu")
    if torch_npu_version:
        key = ".".join(str(part) for part in torch_nums[:3])
        minimum = table.get(key)
        if minimum and _num(torch_npu_version) < _num(minimum):
            warnings.warn(
                f"fla_npu: detected torch_npu {torch_npu_version}, below the "
                f"recommended minimum {minimum} for torch {torch_version} "
                "(missing GDN fixes); import continues, but those operators "
                "may fail. See the README for the supported version matrix.",
                RuntimeWarning,
                stacklevel=2,
            )


def run_guards() -> None:
    """Entry point called on ``import fla_npu`` before OPP loading.

    Never raises: version problems are reported as RuntimeWarnings so the
    import path stays compatible with packages that skipped these checks.
    """
    try:
        _check_versions(_compat())
    except Exception as exc:
        warnings.warn(f"fla_npu version guard skipped: {exc}", RuntimeWarning)
