"""Collect public environment details for fla_npu issue reports.

The script intentionally reports only versions and hit/miss status. It does not
print usernames, hostnames, IP addresses, absolute paths, or raw environment
variables.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from importlib import metadata
except ImportError:  # Python 3.7 compatibility for environment collection only.
    try:
        import importlib_metadata as metadata
    except Exception:
        metadata = None


PUBLIC_FIELDS = (
    "CANN complete version",
    "Python",
    "PyTorch",
    "torch_npu",
    "fla_npu",
    "fla_npu commit",
    "SOC",
    "device model",
    "custom OPP env",
    "fla_npu runtime readiness",
)

VENDOR_NAMES = ("fla_npu_transformer", "fla_npu", "custom_transformer", "custom")

FIELD_VALUE_COMMENTS = {
    ("fla_npu commit", "<unknown>"): "当前执行目录不是 git checkout，或安装包元数据未携带 commit。",
    ("custom OPP env", "enabled"): "环境里启用了 ASCEND_CUSTOM_OPP_PATH，通常说明 source 过 custom OPP。",
    ("custom OPP env", "disabled"): "环境里未启用 ASCEND_CUSTOM_OPP_PATH；如果一体化 wheel 内置 OPP 可用，也可能不影响运行。",
}


def _clean(value: object) -> str:
    if value is None:
        return "<unknown>"
    text = str(value).strip()
    return text if text else "<unknown>"


def _safe_version(module_name: str, dist_names: tuple[str, ...] = (), include_dist_name: bool = False) -> str:
    if metadata is not None:
        for dist_name in dist_names:
            try:
                version = metadata.version(dist_name)
                return f"{dist_name}=={version}" if include_dist_name else version
            except Exception:
                pass

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return f"not available ({exc.__class__.__name__})"

    version = _clean(getattr(module, "__version__", "<unknown>"))
    return f"{module_name}=={version}" if include_dist_name else version


def _read_first_version_line(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            lower = stripped.lower()
            if lower.startswith(("version=", "version:")):
                return stripped
            if "version" in lower and "=" in stripped:
                return stripped
    except OSError:
        pass
    return ""


def _cann_version() -> str:
    candidates: list[Path] = []
    for env_name in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH"):
        value = os.environ.get(env_name)
        if not value:
            continue
        root = Path(value).expanduser()
        candidates.extend(
            [
                root / "version.info",
                root / "ascend_toolkit_install.info",
                root.parent / "version.info",
                root.parent / "ascend_toolkit_install.info",
            ]
        )

    seen: set[Path] = set()
    versions: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        version = _read_first_version_line(candidate)
        if version and version not in versions:
            versions.append(version)
    return "; ".join(versions) if versions else "<unknown>"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "<unknown>"


def _soc_from_model(device_model: str) -> str:
    model = device_model.lower()
    if "910b" in model:
        return "ascend910b"
    if "910_93" in model or "91093" in model:
        return "ascend910_93"
    if "950" in model:
        return "ascend950"
    return ""


def _soc(device_model: str = "") -> str:
    for env_name in ("FLA_NPU_SOC", "ASCEND_CHIP_TYPE", "SOC_VERSION"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    inferred = _soc_from_model(device_model)
    if inferred:
        return inferred
    return "<unknown>"


def _run_npu_smi_json() -> object | None:
    for cmd in (
        ["npu-smi", "info", "-t", "board", "--json"],
        ["npu-smi", "info", "--json"],
    ):
        try:
            output = subprocess.check_output(
                cmd,
                encoding="utf-8",
                errors="ignore",
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except Exception:
            continue
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            continue
    return None


def _is_public_model_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    blocked = ("host", "user", "account", "ip", "address", "serial", "sn", "uuid", "bus", "path", "logic_id")
    if any(part in normalized for part in blocked):
        return False
    allowed = (
        "chip_name",
        "chip_type",
        "chip_model",
        "product_name",
        "product_type",
        "product_model",
        "npu_name",
        "device_name",
        "device_model",
        "model",
    )
    return normalized in allowed


def _collect_model_names(value: object) -> list[str]:
    names: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and _is_public_model_key(key):
                text = str(item).strip()
                if text and not re.search(r"(/|\\|@|\b\d{1,3}(?:\.\d{1,3}){3}\b)", text):
                    names.append(text)
            names.extend(_collect_model_names(item))
    elif isinstance(value, list):
        for item in value:
            names.extend(_collect_model_names(item))
    return list(dict.fromkeys(names))


def _device_model() -> str:
    try:
        output = subprocess.check_output(
            ["npu-smi", "info"],
            encoding="utf-8",
            errors="ignore",
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except Exception:
        return "<unknown>"

    names: list[str] = []
    for line in output.splitlines():
        columns = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(columns) >= 2 and columns[0].isdigit():
            model = columns[1].split()[0] if columns[1].split() else ""
            if re.fullmatch(r"(Ascend[ A-Za-z0-9_-]+|910B[0-9A-Za-z_-]*|910_[0-9A-Za-z_-]+|950[0-9A-Za-z_-]*)", model):
                if model not in names:
                    names.append(model)
            continue

        if columns:
            fields = columns[0].split()
            if len(fields) >= 2 and fields[0].isdigit():
                model = fields[1]
                if re.fullmatch(r"(Ascend[ A-Za-z0-9_-]+|910B[0-9A-Za-z_-]*|910_[0-9A-Za-z_-]+|950[0-9A-Za-z_-]*)", model):
                    if model not in names:
                        names.append(model)
                continue

        if not re.search(r"(Name|Chip|Product|Model)", line, re.I):
            continue
        for token in re.findall(r"(Ascend[ A-Za-z0-9_-]+|910B[0-9A-Za-z_-]*|910_[0-9A-Za-z_-]+|950[0-9A-Za-z_-]*)", line):
            token = token.strip(" |:")
            if token and token not in names:
                names.append(token)
    if names:
        return ", ".join(names[:8])

    data = _run_npu_smi_json()
    if data is not None:
        names = _collect_model_names(data)
        if names:
            return ", ".join(names[:8])

    return "<unknown>"


def _resolve_fla_npu_vendor_dir() -> Path | None:
    try:
        fla_npu = importlib.import_module("fla_npu")
    except Exception:
        return None

    resolver = getattr(fla_npu, "_resolve_vendor_dir", None)
    if resolver is None:
        return None

    try:
        return Path(resolver())
    except Exception:
        return None


def _candidate_opp_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("FLA_NPU_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH", "ASCEND_OPP_PATH"):
        for part in os.environ.get(env_name, "").split(os.pathsep):
            if part:
                roots.append(Path(part).expanduser())

    try:
        fla_npu = importlib.import_module("fla_npu")
    except Exception:
        fla_npu = None
    module_file = getattr(fla_npu, "__file__", None)
    if module_file:
        roots.append(Path(module_file).resolve().parent / "opp")

    return list(dict.fromkeys(roots))


def _has_opp(root: Path) -> bool:
    if any((root / "vendors" / vendor_name).exists() for vendor_name in VENDOR_NAMES):
        return True

    return (root / "op_api" / "lib").exists() and any(
        (root / child).exists() for child in ("op_impl", "framework", "op_host")
    )


def _has_op_api(root: Path) -> bool:
    configured_op_api = os.environ.get("FLA_NPU_OP_API_LIB", "").strip()
    if configured_op_api and Path(configured_op_api).is_file():
        return True

    candidates = [
        root / "op_api" / "lib" / "libcust_opapi.so",
        root / "op_api" / "lib" / "libopapi.so",
    ]

    vendors = root / "vendors"
    if vendors.exists():
        for vendor_name in VENDOR_NAMES:
            vendor = vendors / vendor_name
            candidates.extend(
                [
                    vendor / "op_api" / "lib" / "libcust_opapi.so",
                    vendor / "op_api" / "lib" / "libopapi.so",
                ]
            )

    return any(path.exists() for path in candidates)


def _hit_status(predicate) -> str:
    try:
        vendor_dir = _resolve_fla_npu_vendor_dir()
        if vendor_dir is not None and predicate(vendor_dir):
            return "found"

        roots = _candidate_opp_roots()
        if any(predicate(root) for root in roots):
            return "found"
    except Exception:
        return "unknown"
    return "missing"


def _custom_opp_status() -> str:
    return "enabled" if os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").strip() else "disabled"


def _runtime_readiness(opp_status: str, op_api_status: str) -> str:
    if opp_status == "found" and op_api_status == "found":
        return "ready"
    if "unknown" in (opp_status, op_api_status):
        return "unknown"
    return "not ready"


def _runtime_readiness_comment(readiness: str, opp_status: str, op_api_status: str) -> str:
    if readiness == "ready":
        return "已同时命中 fla_npu OPP 和 fla_npu op_api。"
    if readiness == "unknown":
        return f"fla_npu OPP/op_api 状态未知（OPP={opp_status}, op_api={op_api_status}），请检查安装和环境。"
    return f"未同时命中 fla_npu OPP/op_api（OPP={opp_status}, op_api={op_api_status}），自定义算子加载可能失败。"


def collect() -> dict[str, str]:
    device_model = _device_model()
    opp_status = _hit_status(_has_opp)
    op_api_status = _hit_status(_has_op_api)
    readiness = _runtime_readiness(opp_status, op_api_status)
    return {
        "CANN complete version": _cann_version(),
        "Python": sys.version.split()[0],
        "PyTorch": _safe_version("torch", ("torch",)),
        "torch_npu": _safe_version("torch_npu", ("torch-npu", "torch_npu")),
        "fla_npu": _safe_version(
            "fla_npu",
            ("flash-linear-attention-npu", "fla-npu", "fla_npu"),
            include_dist_name=True,
        ),
        "fla_npu commit": _git_commit(),
        "SOC": _soc(device_model),
        "device model": device_model,
        "custom OPP env": _custom_opp_status(),
        "fla_npu runtime readiness": readiness,
        "_fla_npu runtime readiness comment": _runtime_readiness_comment(readiness, opp_status, op_api_status),
    }


def main() -> int:
    info = collect()
    for key in PUBLIC_FIELDS:
        value = info.get(key, "<unknown>")
        comment = info.get(f"_{key} comment") or FIELD_VALUE_COMMENTS.get((key, value))
        suffix = f"  # {comment}" if comment else ""
        print(f"{key}: {value}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
