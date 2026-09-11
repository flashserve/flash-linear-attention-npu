#!/usr/bin/env python3
"""校验 ChunkKdaFwdPrepare ATK 分片及完整矩阵覆盖。"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path


OP_NAME = "chunk_kda_fwd_prepare"
HOST_KEY_RE = re.compile(
    r"ChunkKdaFwdPrepare tiling:.*?outputMode=(\d+).*?tilingKey=(\d+)"
)
LAUNCH_KEY_RE = re.compile(
    r"OpName:\[[^\]]*ChunkKdaFwdPrepare[^\]]*\]\s+"
    r"Tiling Key:\s*(\d+)"
)
KERNEL_NAME_PATTERN = r"(?:chunk_kda_fwd_prepare|ChunkKdaFwdPrepare)"
SHARD_DIR_RE = re.compile(r"^shard_(0|[1-9][0-9]*)_(0|[1-9][0-9]*)$")
KERNEL_BIN_NAME_RE = re.compile(
    rf"^(?P<op>{KERNEL_NAME_PATTERN})_(?P<compile_hash>[0-9a-fA-F]{{32}})$"
)
SANITIZER_TOOLS = frozenset(
    {"memcheck", "racecheck", "initcheck", "synccheck"}
)
SCOPE_CONTRACTS = {
    "accuracy": {
        "case_count": 200,
        "loop_nums": 1,
        "gm_init_mode": "off",
        "single_process_mode": "off",
        "tools": frozenset({""}),
        "require_tiling_log": False,
    },
    "determinism": {
        "case_count": 432,
        "loop_nums": 50,
        "gm_init_mode": "not_applicable",
        "single_process_mode": "off",
        "tools": frozenset({""}),
        "require_tiling_log": True,
    },
    "mssanitizer": {
        "case_count": 432,
        "loop_nums": 1,
        "gm_init_mode": "not_applicable",
        "single_process_mode": "off",
        "tools": SANITIZER_TOOLS,
        "require_tiling_log": True,
    },
}


def _load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"用例文件必须是非空列表：{path}")
    return cases


def _case_spec(case: dict) -> dict:
    for item in case.get("inputs", ()):
        if item.get("name") == "case_spec":
            value = item.get("range_values")
            return json.loads(value) if isinstance(value, str) else value
    raise ValueError(f"case {case.get('id')} 缺少 case_spec")


def _case_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_hash(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"证据文件不存在或为空：{path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key_digest(pairs: list[tuple[int, int]]) -> str:
    payload = "".join(f"{case_id},{key}\n" for case_id, key in pairs)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _integer_digest(values: set[int]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _string_digest(values: set[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _validate_scope_contract(args: argparse.Namespace, case_count: int) -> None:
    contract = SCOPE_CONTRACTS[args.scope]
    if not 1 <= args.timeout <= 60:
        raise ValueError(
            f"{args.scope} timeout 必须在 1..60 秒内，实际为 {args.timeout}"
        )
    if case_count != contract["case_count"]:
        raise ValueError(
            f"{args.scope} 正式矩阵必须为 {contract['case_count']} 条，"
            f"实际为 {case_count} 条"
        )
    if args.loop_nums != contract["loop_nums"]:
        raise ValueError(
            f"{args.scope} loop_nums 必须为 {contract['loop_nums']}，"
            f"实际为 {args.loop_nums}"
        )
    if args.gm_init_mode != contract["gm_init_mode"]:
        raise ValueError(
            f"{args.scope} gm_init_mode 必须为 {contract['gm_init_mode']}，"
            f"实际为 {args.gm_init_mode}"
        )
    if args.single_process_mode != contract["single_process_mode"]:
        raise ValueError(
            f"{args.scope} single_process_mode 必须为 "
            f"{contract['single_process_mode']}，实际为 {args.single_process_mode}"
        )
    if args.tool not in contract["tools"]:
        raise ValueError(f"{args.scope} sanitizer tool 非法：{args.tool!r}")
    require_tiling_log = bool(getattr(args, "require_tiling_log", False))
    if require_tiling_log != contract["require_tiling_log"]:
        expected = "开启" if contract["require_tiling_log"] else "关闭"
        raise ValueError(f"{args.scope} require_tiling_log 必须{expected}")


def _manifest_kernel_map(manifest: dict) -> dict[str, int]:
    if manifest.get("schema") != "kda-prepare-runtime/v2":
        raise ValueError("runtime manifest schema 不是 v2")
    binaries = manifest.get("kernel_binaries")
    if not isinstance(binaries, list) or not binaries:
        raise ValueError("runtime manifest 缺少 kernel_binaries")

    kernel_map: dict[str, int] = {}
    for binary in binaries:
        bin_file_name = str(binary.get("bin_file_name", ""))
        match = KERNEL_BIN_NAME_RE.fullmatch(bin_file_name)
        if match is None:
            raise ValueError(f"runtime manifest 的 binFileName 非法：{bin_file_name}")
        if binary.get("compile_hash") != match.group("compile_hash"):
            raise ValueError(f"runtime manifest 的编译 hash 不一致：{bin_file_name}")
        if binary.get("metadata_kernel_name") != bin_file_name:
            raise ValueError(f"runtime manifest 的 kernelName 不一致：{bin_file_name}")
        if binary.get("metadata_bin_sha256") != binary.get("bin_file_sha256"):
            raise ValueError(f"runtime manifest 的对象 SHA256 不一致：{bin_file_name}")

        kernels = binary.get("kernels")
        if not isinstance(kernels, list) or not kernels:
            raise ValueError(f"runtime manifest 的 kernelList 为空：{bin_file_name}")
        for item in kernels:
            tiling_key = int(item["tiling_key"])
            kernel_name = str(item.get("kernel_name", ""))
            if kernel_name != f"{bin_file_name}_{tiling_key}":
                raise ValueError(
                    f"runtime manifest 的完整 kernelName 不一致：{kernel_name}"
                )
            if kernel_name in kernel_map:
                raise ValueError(
                    f"runtime manifest 的 kernelName 重复：{kernel_name}"
                )
            kernel_map[kernel_name] = tiling_key

    if manifest.get("kernel_name_count") != len(kernel_map):
        raise ValueError("runtime manifest 的 kernelName 数量不一致")
    if manifest.get("kernel_name_sha256") != _string_digest(set(kernel_map)):
        raise ValueError("runtime manifest 的 kernelName 摘要不一致")
    return kernel_map


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _load_runtime_manifest(
    path: Path,
    case_file: Path,
    require_sanitizer: bool,
    require_complete_key_set: bool,
) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"runtime manifest 无法读取：{path}：{error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"runtime manifest 必须是 JSON object：{path}")

    kernel_map = _manifest_kernel_map(manifest)
    binaries = manifest["kernel_binaries"]
    compiled_keys = set(kernel_map.values())
    if manifest.get("platform") not in {
        "ascend910b",
        "ascend910_93",
        "ascend950",
    }:
        raise ValueError("runtime manifest 的 platform 非法")
    if not _is_sha256(manifest.get("runtime_sha256")):
        raise ValueError("runtime manifest 的 runtime_sha256 非法")
    if not _is_sha256(manifest.get("op_api_sha256")):
        raise ValueError("runtime manifest 的 op_api_sha256 非法")
    if manifest.get("object_count") != len(binaries):
        raise ValueError("runtime manifest 的对象数量不一致")
    if manifest.get("metadata_count") != len(binaries):
        raise ValueError("runtime manifest 的 metadata 数量不一致")
    if manifest.get("compiled_tiling_key_count") != len(compiled_keys):
        raise ValueError("runtime manifest 的 TilingKey 数量不一致")
    if manifest.get("compiled_tiling_key_sha256") != _integer_digest(
        compiled_keys
    ):
        raise ValueError("runtime manifest 的 TilingKey 摘要不一致")
    if manifest.get("complete_key_set_required") is not require_complete_key_set:
        raise ValueError("runtime manifest 的完整 TilingKey 集合要求不一致")
    if manifest.get("sanitizer_required") is not require_sanitizer:
        raise ValueError("runtime manifest 的 sanitizer 要求不一致")
    expected_sanitizer_objects = len(binaries) if require_sanitizer else 0
    if manifest.get("sanitizer_object_count") != expected_sanitizer_objects:
        raise ValueError("runtime manifest 的 sanitizer 对象数量不一致")

    for binary in binaries:
        for field in (
            "metadata_sha256",
            "metadata_bin_sha256",
            "bin_file_sha256",
        ):
            if not _is_sha256(binary.get(field)):
                raise ValueError(f"runtime manifest 的 {field} 非法")

    wrapper_hashes = manifest.get("python_wrapper_sha256")
    expected_wrappers = {
        "__init__.py",
        "ops/ascendc/__init__.py",
        "ops/ascendc/_aclnn_ctypes.py",
        "ops/ascendc/_runtime.py",
    }
    if (
        not isinstance(wrapper_hashes, dict)
        or set(wrapper_hashes) != expected_wrappers
    ):
        raise ValueError("runtime manifest 的 Python 包装器集合不闭合")
    if any(not _is_sha256(value) for value in wrapper_hashes.values()):
        raise ValueError("runtime manifest 的 Python 包装器摘要非法")
    artifact_hashes = manifest.get("test_artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("runtime manifest 缺少测试程序摘要")
    if any(not _is_sha256(value) for value in artifact_hashes.values()):
        raise ValueError("runtime manifest 的测试程序摘要非法")

    cases = _load_cases(case_file)
    _, expected_pairs = _expected_slice(cases, 0, len(cases))
    expected_keys = {key for _, key in expected_pairs}
    missing_keys = expected_keys - compiled_keys
    extra_keys = compiled_keys - expected_keys
    if missing_keys or (require_complete_key_set and extra_keys):
        raise ValueError(
            "runtime manifest 的编译 key 与冻结矩阵不一致："
            f"missing={sorted(missing_keys)}, extra={sorted(extra_keys)}"
        )
    return manifest


def _candidate_opp_roots() -> list[Path]:
    roots: list[Path] = []
    op_api_lib = os.environ.get("FLA_NPU_OP_API_LIB", "").strip()
    if op_api_lib:
        lib_path = Path(op_api_lib).expanduser().resolve()
        if not lib_path.is_file():
            raise ValueError(f"FLA_NPU_OP_API_LIB 不存在：{lib_path}")
        roots.append(lib_path.parents[2])
    for name in ("FLA_NPU_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH"):
        for value in os.environ.get(name, "").split(os.pathsep):
            if value:
                roots.append(Path(value).expanduser())

    spec = importlib.util.find_spec("fla_npu")
    if spec is not None and spec.submodule_search_locations:
        package_root = Path(next(iter(spec.submodule_search_locations)))
        roots.append(package_root / "opp")

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _find_runtime_kernel_dir(soc: str) -> Path:
    relative = Path("op_impl/ai_core/tbe/kernel")
    for root in _candidate_opp_roots():
        vendor_roots = [root / "vendors/fla_npu_transformer", root]
        matches: set[Path] = set()
        for vendor_root in vendor_roots:
            kernel_root = vendor_root / relative
            if not kernel_root.is_dir():
                continue
            platforms = [kernel_root / soc] if soc != "auto" else kernel_root.iterdir()
            for platform in platforms:
                candidate = platform / OP_NAME
                if candidate.is_dir():
                    matches.add(candidate.resolve())
        if matches:
            if len(matches) != 1:
                raise ValueError(
                    f"当前 OPP 中找到多个 {OP_NAME} kernel 目录，请显式指定 SoC"
                )
            return matches.pop()
    raise ValueError(f"当前运行环境中找不到 {OP_NAME} kernel 目录")


def _runtime_op_api_lib() -> Path:
    value = os.environ.get("FLA_NPU_OP_API_LIB", "").strip()
    if value:
        path = Path(value).expanduser().resolve()
        if path.is_file():
            return path
        raise ValueError(f"FLA_NPU_OP_API_LIB 不存在：{path}")

    for root in _candidate_opp_roots():
        for candidate in (
            root / "op_api/lib/libcust_opapi.so",
            root / "vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so",
        ):
            if candidate.is_file():
                return candidate.resolve()
    raise ValueError("当前运行环境中找不到 libcust_opapi.so")


def _runtime_manifest(
    case_file: Path,
    soc: str,
    require_sanitizer: bool,
    require_complete_key_set: bool = False,
    test_artifacts: tuple[Path, ...] = (),
) -> dict:
    kernel_dir = _find_runtime_kernel_dir(soc)
    op_api_lib = _runtime_op_api_lib()
    vendor_root = op_api_lib.parents[2]
    try:
        kernel_dir.relative_to(vendor_root)
    except ValueError as error:
        raise ValueError("op_api 与 kernel 不属于同一运行时 OPP") from error
    object_files = sorted(kernel_dir.glob("*.o"))
    metadata_files = sorted(kernel_dir.glob("*.json"))
    if not object_files or not metadata_files:
        raise ValueError(f"kernel 目录缺少 .o 或 metadata：{kernel_dir}")

    compiled_keys: set[int] = set()
    compiled_kernel_names: set[str] = set()
    referenced_objects: set[str] = set()
    kernel_binaries = []
    for path in metadata_files:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        bin_file_name = str(metadata.get("binFileName", ""))
        bin_file_suffix = str(metadata.get("binFileSuffix", ".o"))
        bin_name_match = KERNEL_BIN_NAME_RE.fullmatch(bin_file_name)
        if bin_name_match is None:
            raise ValueError(f"kernel metadata 的 binFileName 非法：{path.name}")
        if path.stem != bin_file_name:
            raise ValueError(
                f"kernel metadata 文件名与 binFileName 不一致：{path.name}"
            )
        metadata_kernel_name = str(metadata.get("kernelName", ""))
        if metadata_kernel_name != bin_file_name:
            raise ValueError(
                f"kernel metadata 顶层 kernelName 与 binFileName 不一致：{path.name}"
            )

        object_name = f"{bin_file_name}{bin_file_suffix}"
        object_path = kernel_dir / object_name
        referenced_objects.add(object_name)
        object_sha256 = _file_hash(object_path)
        metadata_bin_sha256 = str(metadata.get("sha256", "")).lower()
        if metadata_bin_sha256 != object_sha256:
            raise ValueError(
                f"kernel metadata 的 sha256 与对象不一致：{object_name}"
            )

        kernel_list = metadata.get("kernelList")
        if not isinstance(kernel_list, list) or not kernel_list:
            raise ValueError(f"kernel metadata 的 kernelList 为空：{path.name}")
        kernels = []
        for item in kernel_list:
            tiling_key = int(item["tilingKey"])
            kernel_name = str(item.get("kernelName", ""))
            if kernel_name != f"{bin_file_name}_{tiling_key}":
                raise ValueError(
                    "kernelList 完整 kernelName 与 binFileName/TilingKey 不一致："
                    f"{kernel_name}, {bin_file_name}, {tiling_key}"
                )
            if kernel_name in compiled_kernel_names:
                raise ValueError(f"kernelName 重复：{kernel_name}")
            compiled_kernel_names.add(kernel_name)
            compiled_keys.add(tiling_key)
            kernels.append(
                {"kernel_name": kernel_name, "tiling_key": tiling_key}
            )
        kernel_binaries.append(
            {
                "metadata_file": path.name,
                "metadata_sha256": _file_hash(path),
                "metadata_kernel_name": metadata_kernel_name,
                "metadata_bin_sha256": metadata_bin_sha256,
                "bin_file_name": bin_file_name,
                "bin_file_suffix": bin_file_suffix,
                "bin_file_sha256": object_sha256,
                "compile_hash": bin_name_match.group("compile_hash"),
                "kernels": kernels,
            }
        )
    if not compiled_keys:
        raise ValueError("kernel metadata 中没有 TilingKey")
    actual_objects = {path.name for path in object_files}
    if actual_objects != referenced_objects:
        raise ValueError(
            "kernel metadata 引用的对象集合不闭合："
            f"missing={sorted(referenced_objects - actual_objects)}, "
            f"extra={sorted(actual_objects - referenced_objects)}"
        )

    cases = _load_cases(case_file)
    _, expected_pairs = _expected_slice(cases, 0, len(cases))
    expected_keys = {key for _, key in expected_pairs}
    missing_keys = expected_keys - compiled_keys
    extra_keys = compiled_keys - expected_keys
    if missing_keys or (require_complete_key_set and extra_keys):
        raise ValueError(
            "安装包编译 key 与冻结矩阵不一致："
            f"missing={sorted(missing_keys)}, extra={sorted(extra_keys)}"
        )

    digest = hashlib.sha256()
    for path in object_files + metadata_files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(b"op_api/lib/libcust_opapi.so\0")
    digest.update(op_api_lib.read_bytes())
    digest.update(b"\0")

    test_artifact_hashes = {}
    for path in test_artifacts:
        resolved = path.resolve()
        test_artifact_hashes[resolved.name] = _file_hash(resolved)
        digest.update(f"test/{resolved.name}\0".encode("utf-8"))
        digest.update(resolved.read_bytes())
        digest.update(b"\0")

    wrapper_hashes = {}
    package_spec = importlib.util.find_spec("fla_npu")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise ValueError("找不到实际加载的 fla_npu Python 包")
    package_root = Path(next(iter(package_spec.submodule_search_locations))).resolve()
    for relative in (
        "__init__.py",
        "ops/ascendc/__init__.py",
        "ops/ascendc/_aclnn_ctypes.py",
        "ops/ascendc/_runtime.py",
    ):
        path = package_root / relative
        wrapper_hashes[relative] = _file_hash(path)
        digest.update(f"python/{relative}\0".encode("utf-8"))
        digest.update(path.read_bytes())
        digest.update(b"\0")

    sanitizer_objects = 0
    if require_sanitizer:
        for path in object_files:
            result = subprocess.run(
                ("nm", str(path)),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
            if result.returncode != 0 or "sanitizer" not in result.stdout.lower():
                raise ValueError(f"kernel 对象缺少 sanitizer 符号：{path.name}")
            sanitizer_objects += 1

    return {
        "schema": "kda-prepare-runtime/v2",
        "platform": kernel_dir.parent.name,
        "runtime_sha256": digest.hexdigest(),
        "op_api_sha256": _file_hash(op_api_lib),
        "test_artifact_sha256": test_artifact_hashes,
        "python_wrapper_sha256": wrapper_hashes,
        "object_count": len(object_files),
        "metadata_count": len(metadata_files),
        "compiled_tiling_key_count": len(compiled_keys),
        "compiled_tiling_key_sha256": _integer_digest(compiled_keys),
        "kernel_name_count": len(compiled_kernel_names),
        "kernel_name_sha256": _string_digest(compiled_kernel_names),
        "kernel_binaries": kernel_binaries,
        "complete_key_set_required": require_complete_key_set,
        "sanitizer_required": require_sanitizer,
        "sanitizer_object_count": sanitizer_objects,
    }


def _verify_current_runtime(args: argparse.Namespace) -> dict:
    require_sanitizer = args.scope == "mssanitizer"
    require_complete_key_set = args.scope != "accuracy"
    existing = _load_runtime_manifest(
        args.runtime_manifest,
        args.case_file,
        require_sanitizer,
        require_complete_key_set,
    )
    actual = _runtime_manifest(
        args.case_file,
        args.soc,
        require_sanitizer,
        require_complete_key_set,
        tuple(args.test_artifact),
    )
    if existing != actual:
        raise ValueError(
            "runtime manifest 与当前实际 OPP、包装器或测试程序不一致"
        )
    return existing


def verify_runtime(args: argparse.Namespace) -> int:
    manifest = _runtime_manifest(
        args.case_file,
        args.soc,
        args.require_sanitizer,
        args.require_complete_key_set,
        tuple(args.test_artifact),
    )
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                "运行时 kernel 已变化，拒绝在原矩阵目录续跑："
                f"existing={existing}, actual={manifest}"
            )
    else:
        args.output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _find_report(root: Path, scope: str) -> Path:
    patterns = {
        "accuracy": (
            f"accuracy/atk_output/atk_{OP_NAME}_*/report/*.xlsx",
            f"accuracy/atk_{OP_NAME}_*/report/*.xlsx",
        ),
        "determinism": (
            f"determinism/atk_output/atk_{OP_NAME}_*/report/*.xlsx",
            f"determinism/atk_{OP_NAME}_*/report/*.xlsx",
            f"atk_{OP_NAME}_*/report/*.xlsx",
        ),
        "mssanitizer": (
            f"mssanitizer/atk_output/atk_{OP_NAME}_*/report/*.xlsx",
            f"mssanitizer/atk_{OP_NAME}_*/report/*.xlsx",
            f"atk_{OP_NAME}_*/report/*.xlsx",
        ),
    }
    paths: list[Path] = []
    for pattern in patterns[scope]:
        paths.extend(Path(value) for value in glob.glob(str(root / pattern)))
    if not paths:
        raise ValueError(f"{root} 下没有找到 {scope} 报告")
    return max(set(paths), key=lambda path: path.stat().st_mtime_ns)


def _report_case_ids(report: Path, scope: str) -> tuple[list[int], int]:
    try:
        import openpyxl
    except ImportError as error:
        raise RuntimeError("读取 ATK 报告需要安装 openpyxl") from error

    workbook = openpyxl.load_workbook(report, read_only=True, data_only=True)
    try:
        if "statistic" not in workbook.sheetnames:
            raise ValueError(f"报告缺少 statistic sheet：{report.name}")
        if "summary" not in workbook.sheetnames:
            raise ValueError(f"报告缺少 summary sheet：{report.name}")

        statistic_rows = list(workbook["statistic"].iter_rows(values_only=True))
        if not statistic_rows:
            raise ValueError(f"statistic sheet 为空：{report.name}")
        statistic_header = {
            str(value).strip(): index
            for index, value in enumerate(statistic_rows[0])
            if value is not None
        }
        if "编号" not in statistic_header or "运行结果" not in statistic_header:
            raise ValueError(f"statistic sheet 缺少编号或运行结果列：{report.name}")

        case_ids = []
        memory_columns = [
            index
            for name, index in statistic_header.items()
            if name == "内存检测通过" or name.endswith("_内存检测通过")
        ]
        accuracy_columns = [
            index
            for name, index in statistic_header.items()
            if name == "精度通过" or name.endswith("_精度通过")
        ]
        determinism_columns = [
            index
            for name, index in statistic_header.items()
            if name == "确定性计算结果" or name.endswith("_确定性计算结果")
        ]
        if scope == "mssanitizer" and not memory_columns:
            raise ValueError(f"statistic sheet 缺少内存检测通过列：{report.name}")
        if scope == "accuracy" and not accuracy_columns:
            raise ValueError(f"statistic sheet 缺少精度通过列：{report.name}")
        if scope == "determinism" and not determinism_columns:
            raise ValueError(f"statistic sheet 缺少确定性计算结果列：{report.name}")

        id_column = statistic_header["编号"]
        run_column = statistic_header["运行结果"]
        for row in statistic_rows[1:]:
            case_id = row[id_column] if id_column < len(row) else None
            if case_id is None:
                continue
            case_ids.append(int(case_id))
            run_result = row[run_column] if run_column < len(row) else None
            if str(run_result).strip().upper() != "SUCCESS":
                raise ValueError(f"case {case_id} 运行结果不是 SUCCESS")
            checked_columns = memory_columns if scope == "mssanitizer" else []
            if scope == "accuracy":
                checked_columns = accuracy_columns
            elif scope == "determinism":
                checked_columns = determinism_columns
            checked_values = [
                row[column] if column < len(row) else None
                for column in checked_columns
            ]
            active_values = [
                value for value in checked_values if value not in (None, "")
            ]
            if checked_columns and (
                not active_values
                or any(
                    value is not True and str(value).strip().lower() != "true"
                    for value in active_values
                )
            ):
                raise ValueError(f"case {case_id} 的检查结果不是全部 True")

        summary_rows = list(workbook["summary"].iter_rows(values_only=True))
        if not summary_rows:
            raise ValueError(f"summary sheet 为空：{report.name}")
        summary_header = {
            str(value).strip(): index
            for index, value in enumerate(summary_rows[0])
            if value is not None
        }
        required = {"总用例数"}
        if scope in ("accuracy", "determinism"):
            required.update(
                {"执行失败用例个数", "通过用例个数", "精度是否达标"}
            )
            if scope == "determinism":
                required.add("确定性计算是否达标")
        else:
            required.update({"内存检测通过率", "内存检测是否达标"})
        missing = sorted(required - summary_header.keys())
        if missing:
            raise ValueError(f"summary sheet 缺少列 {missing}：{report.name}")

        total = 0
        passed = 0
        for row in summary_rows[1:]:
            total_value = row[summary_header["总用例数"]]
            if total_value is None:
                continue
            row_total = int(total_value)
            total += row_total
            if scope in ("accuracy", "determinism"):
                failed = int(row[summary_header["执行失败用例个数"]])
                row_passed = int(row[summary_header["通过用例个数"]])
                if failed != 0 or row_passed != row_total:
                    raise ValueError(f"报告存在执行或检查失败：{report.name}")
                if str(row[summary_header["精度是否达标"]]).strip() not in {
                    "Pass",
                    "-",
                }:
                    raise ValueError(f"精度达标列失败：{report.name}")
                if scope == "determinism" and str(
                    row[summary_header["确定性计算是否达标"]]
                ).strip() != "Pass":
                    raise ValueError(f"确定性达标列失败：{report.name}")
                passed += row_passed
            else:
                rate = float(
                    str(row[summary_header["内存检测通过率"]]).strip().rstrip("%")
                )
                status = str(row[summary_header["内存检测是否达标"]]).strip()
                if rate != 100.0 or status != "Pass":
                    raise ValueError(f"内存检测未全部通过：{report.name}")
                passed += row_total
        if total == 0 or passed != total:
            raise ValueError(
                f"报告通过数不匹配：total={total}, passed={passed}"
            )
        return case_ids, total
    finally:
        workbook.close()


def _logged_keys(log_path: Path) -> tuple[set[int], set[int]]:
    host_keys: set[int] = set()
    launch_keys: set[int] = set()
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            host_match = HOST_KEY_RE.search(line)
            if host_match:
                output_mode = int(host_match.group(1))
                tiling_key = int(host_match.group(2))
                encoded_output_mode = (tiling_key >> 23) & 0x3
                if output_mode != encoded_output_mode:
                    raise ValueError(
                        "Host 日志中的 outputMode 与 TilingKey 编码不一致："
                        f"mode={output_mode}, key={tiling_key}"
                    )
                host_keys.add(tiling_key)
            launch_match = LAUNCH_KEY_RE.search(line)
            if launch_match:
                launch_keys.add(int(launch_match.group(1)))
    return host_keys, launch_keys


def _sanitizer_evidence(
    console_log: Path,
    sanitizer_log: Path,
    tool: str,
    expected_keys: set[int],
    runtime_manifest: dict,
) -> tuple[int, set[int], set[str], str, str]:
    if tool not in SANITIZER_TOOLS:
        raise ValueError(f"非法 sanitizer 工具：{tool}")

    start_re = re.compile(
        rf"Start\s+{re.escape(tool)}\s+sanitizer\s+on\s+kernel\s+"
        rf"(?P<kernel>{KERNEL_NAME_PATTERN}(?:_[A-Za-z0-9]+)+)"
        rf"(?=$|[^A-Za-z0-9_])",
        re.IGNORECASE,
    )
    inactive_re = re.compile(
        rf"No\s+active\s+sanitizer\s+tool\s+on\s+kernel[^\r\n]*"
        rf"{KERNEL_NAME_PATTERN}",
        re.IGNORECASE,
    )
    contents = []
    for path in (console_log, sanitizer_log):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"sanitizer 证据文件不存在或为空：{path}")
        contents.append(path.read_text(encoding="utf-8", errors="replace"))
    evidence = "\n".join(contents)
    if inactive_re.search(evidence):
        raise ValueError(f"{tool} 日志显示目标 kernel 未加载 sanitizer 插桩")
    matched_kernel_names = start_re.findall(evidence)
    manifest_kernel_map = _manifest_kernel_map(runtime_manifest)
    unexpected_kernel_names = set(matched_kernel_names) - set(manifest_kernel_map)
    if unexpected_kernel_names:
        raise ValueError(
            f"{tool} 启动了 runtime manifest 之外的 kernel："
            f"{sorted(unexpected_kernel_names)}"
        )
    started_kernel_names = set(matched_kernel_names)
    started_keys = {
        manifest_kernel_map[name] for name in started_kernel_names
    }
    if started_keys != expected_keys:
        raise ValueError(
            f"{tool} 启动 TilingKey 不闭合："
            f"missing={sorted(expected_keys - started_keys)}, "
            f"extra={sorted(started_keys - expected_keys)}"
        )
    return (
        len(matched_kernel_names),
        started_keys,
        started_kernel_names,
        _file_hash(console_log),
        _file_hash(sanitizer_log),
    )


def _expected_slice(
    cases: list[dict], start: int, end: int
) -> tuple[list[int], list[tuple[int, int]]]:
    if not (0 <= start < end <= len(cases)):
        raise ValueError(f"非法分片范围 {start}:{end}，总用例数 {len(cases)}")
    selected = cases[start:end]
    ids = [int(case["id"]) for case in selected]
    if ids != list(range(start, end)):
        raise ValueError(f"冻结用例 ID 与位置不一致：{ids[:3]}...{ids[-3:]}")
    pairs = []
    for case in selected:
        spec = _case_spec(case)
        if "expected_tiling_key" in spec:
            pairs.append((int(case["id"]), int(spec["expected_tiling_key"])))
    return ids, pairs


def _build_shard_summary(args: argparse.Namespace) -> dict:
    cases = _load_cases(args.case_file)
    _validate_scope_contract(args, len(cases))
    runtime_manifest = _load_runtime_manifest(
        args.runtime_manifest,
        args.case_file,
        args.scope == "mssanitizer",
        args.scope != "accuracy",
    )
    shard_root = args.shard_root.resolve()
    if args.shard_root.is_symlink() or not shard_root.is_dir():
        raise ValueError(f"分片目录不存在、不是目录或为符号链接：{args.shard_root}")
    if args.console_log.resolve() != shard_root / "console.log":
        raise ValueError("console 日志必须是分片目录下的 console.log")
    if args.summary.resolve() != shard_root / "summary.json":
        raise ValueError("summary 必须是分片目录下的 summary.json")

    expected_ids, expected_pairs = _expected_slice(cases, args.start, args.end)
    if args.end - args.start != 1:
        raise ValueError("正式矩阵每个分片必须只包含一条 case")
    report = _find_report(args.shard_root, args.scope)
    try:
        report.resolve().relative_to(shard_root)
    except ValueError as error:
        raise ValueError(f"ATK 报告越出分片目录：{report}") from error
    if report.is_symlink():
        raise ValueError(f"ATK 报告不能是符号链接：{report}")
    report_ids, report_total = _report_case_ids(report, args.scope)
    if sorted(report_ids) != expected_ids or len(set(report_ids)) != len(report_ids):
        raise ValueError(
            f"报告 case ID 不闭合：expected={expected_ids}, actual={sorted(report_ids)}"
        )
    if report_total != len(expected_ids):
        raise ValueError(
            f"报告总数不匹配：expected={len(expected_ids)}, actual={report_total}"
        )

    expected_keys = {key for _, key in expected_pairs}
    host_keys: set[int] = set()
    launch_keys: set[int] = set()
    if args.require_tiling_log:
        host_keys, launch_keys = _logged_keys(args.console_log)
        if host_keys != expected_keys:
            raise ValueError(
                f"Host TilingKey 不匹配：missing={sorted(expected_keys - host_keys)}, "
                f"extra={sorted(host_keys - expected_keys)}"
            )
        if launch_keys != expected_keys:
            raise ValueError(
                f"Launch TilingKey 不匹配：missing={sorted(expected_keys - launch_keys)}, "
                f"extra={sorted(launch_keys - expected_keys)}"
            )

    sanitizer_start_count = 0
    sanitizer_started_keys: set[int] = set()
    sanitizer_started_kernel_names: set[str] = set()
    sanitizer_log_sha256 = ""
    console_log_sha256 = _file_hash(args.console_log)
    if args.scope == "mssanitizer":
        sanitizer_log = args.shard_root / f"{args.tool}.log"
        (
            sanitizer_start_count,
            sanitizer_started_keys,
            sanitizer_started_kernel_names,
            console_log_sha256,
            sanitizer_log_sha256,
        ) = _sanitizer_evidence(
            args.console_log,
            sanitizer_log,
            args.tool,
            expected_keys,
            runtime_manifest,
        )

    return {
        "schema": "kda-prepare-atk-shard/v2",
        "scope": args.scope,
        "tool": args.tool,
        "timeout_seconds": args.timeout,
        "loop_nums": args.loop_nums,
        "gm_init_mode": args.gm_init_mode,
        "single_process_mode": args.single_process_mode,
        "require_tiling_log": args.require_tiling_log,
        "runtime_manifest_sha256": _file_hash(args.runtime_manifest),
        "start": args.start,
        "end": args.end,
        "expected_cases": len(expected_ids),
        "case_ids": expected_ids,
        "case_json_sha256": _case_hash(args.case_file),
        "expected_tiling_keys": sorted(expected_keys),
        "host_tiling_keys": sorted(host_keys),
        "launch_tiling_keys": sorted(launch_keys),
        "key_pair_sha256": _key_digest(expected_pairs),
        "report": str(report.relative_to(args.shard_root)),
        "report_sha256": _file_hash(report),
        "console_log_sha256": console_log_sha256,
        "sanitizer_log_sha256": sanitizer_log_sha256,
        "sanitizer_start_count": sanitizer_start_count,
        "sanitizer_started_tiling_keys": sorted(sanitizer_started_keys),
        "sanitizer_started_kernel_names": sorted(
            sanitizer_started_kernel_names
        ),
        "passed": True,
    }


def _read_summary(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"分片摘要无法读取：{path}：{error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"分片摘要必须是 JSON object：{path}")
    return value


def _verify_or_write_summary(args: argparse.Namespace, summary: dict) -> None:
    if args.summary.exists():
        if _read_summary(args.summary) != summary:
            raise ValueError(f"分片摘要与当前原始证据不一致：{args.summary}")
        return
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def summarize_shard(args: argparse.Namespace) -> int:
    summary = _build_shard_summary(args)
    _verify_or_write_summary(args, summary)
    print(
        f"分片 {args.start}:{args.end} 校验通过："
        f"cases={summary['expected_cases']}, "
        f"keys={len(summary['expected_tiling_keys'])}"
    )
    return 0


def _matrix_shards(matrix_root: Path, case_count: int) -> list[tuple[Path, int, int]]:
    root = matrix_root.resolve()
    if matrix_root.is_symlink() or not root.is_dir():
        raise ValueError(f"矩阵目录不存在、不是目录或为符号链接：{matrix_root}")
    shards: list[tuple[Path, int, int]] = []
    for path in root.iterdir():
        if not path.name.startswith("shard_"):
            continue
        match = SHARD_DIR_RE.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_dir():
            raise ValueError(f"非法分片目录：{path.name}")
        start, end = (int(value) for value in match.groups())
        if not 0 <= start < end <= case_count:
            raise ValueError(f"分片目录范围非法：{path.name}")
        if end - start != 1:
            raise ValueError(f"正式矩阵分片必须为单 case：{path.name}")
        shards.append((path, start, end))
    if not shards:
        raise ValueError("没有找到可聚合的分片目录")

    shards.sort(key=lambda item: (item[1], item[2]))
    cursor = 0
    for path, start, end in shards:
        if start != cursor:
            raise ValueError(
                f"分片范围不连续或重叠：期望从 {cursor} 开始，实际为 {path.name}"
            )
        cursor = end
    if cursor != case_count:
        raise ValueError(
            f"分片范围未覆盖完整矩阵：覆盖到 {cursor}，应为 {case_count}"
        )
    return shards


def _build_matrix_summary(args: argparse.Namespace) -> dict:
    cases = _load_cases(args.case_file)
    _validate_scope_contract(args, len(cases))
    runtime_manifest = _verify_current_runtime(args)
    summaries = []
    for shard_root, start, end in _matrix_shards(args.matrix_root, len(cases)):
        summary_path = shard_root / "summary.json"
        if not summary_path.is_file():
            raise ValueError(f"完整分片缺少 summary.json：{shard_root}")
        shard_args = argparse.Namespace(
            case_file=args.case_file,
            scope=args.scope,
            tool=args.tool,
            runtime_manifest=args.runtime_manifest,
            timeout=args.timeout,
            loop_nums=args.loop_nums,
            gm_init_mode=args.gm_init_mode,
            single_process_mode=args.single_process_mode,
            require_tiling_log=args.require_tiling_log,
            shard_root=shard_root,
            console_log=shard_root / "console.log",
            summary=summary_path,
            start=start,
            end=end,
        )
        rebuilt = _build_shard_summary(shard_args)
        if _read_summary(summary_path) != rebuilt:
            raise ValueError(
                f"分片摘要与当前原始证据不一致：{summary_path}"
            )
        summaries.append(rebuilt)

    case_hash = _case_hash(args.case_file)
    case_ids: list[int] = []
    expected_keys: set[int] = set()
    host_keys: set[int] = set()
    launch_keys: set[int] = set()
    sanitizer_start_count = 0
    sanitizer_started_keys: set[int] = set()
    sanitizer_started_kernel_names: set[str] = set()
    runtime_manifest_hash = _file_hash(args.runtime_manifest)
    manifest_kernel_map: dict[str, int] = {}
    if args.scope == "mssanitizer":
        manifest_kernel_map = _manifest_kernel_map(runtime_manifest)
    for summary in summaries:
        if summary.get("schema") != "kda-prepare-atk-shard/v2":
            raise ValueError("分片摘要 schema 不是 v2")
        if (
            summary.get("scope") != args.scope
            or summary.get("tool") != args.tool
            or summary.get("timeout_seconds") != args.timeout
            or summary.get("loop_nums") != args.loop_nums
            or summary.get("gm_init_mode") != args.gm_init_mode
            or summary.get("single_process_mode") != args.single_process_mode
            or summary.get("require_tiling_log") != args.require_tiling_log
        ):
            raise ValueError(
                "分片 scope/tool/timeout/loop_nums/gm/tiling-log 不一致"
            )
        if summary.get("runtime_manifest_sha256") != runtime_manifest_hash:
            raise ValueError("分片运行时 kernel 指纹不一致")
        if summary.get("case_json_sha256") != case_hash or not summary.get("passed"):
            raise ValueError("分片用例哈希不一致或未通过")
        case_ids.extend(int(value) for value in summary["case_ids"])
        expected_keys.update(int(value) for value in summary["expected_tiling_keys"])
        host_keys.update(int(value) for value in summary["host_tiling_keys"])
        launch_keys.update(int(value) for value in summary["launch_tiling_keys"])
        if args.scope == "mssanitizer":
            shard_start_count = int(summary.get("sanitizer_start_count", 0))
            if shard_start_count <= 0:
                raise ValueError("内存分片缺少 sanitizer 启动证据")
            sanitizer_start_count += shard_start_count
            shard_started_keys = {
                int(value)
                for value in summary.get("sanitizer_started_tiling_keys", ())
            }
            if shard_started_keys != {
                int(value) for value in summary["expected_tiling_keys"]
            }:
                raise ValueError("内存分片的 sanitizer 启动 TilingKey 不闭合")
            shard_started_kernel_names = {
                str(value)
                for value in summary.get(
                    "sanitizer_started_kernel_names", ()
                )
            }
            unexpected_kernel_names = (
                shard_started_kernel_names - set(manifest_kernel_map)
            )
            if unexpected_kernel_names:
                raise ValueError(
                    "内存分片命中 runtime manifest 之外的 kernel："
                    f"{sorted(unexpected_kernel_names)}"
                )
            if {
                manifest_kernel_map[name]
                for name in shard_started_kernel_names
            } != shard_started_keys:
                raise ValueError(
                    "内存分片的完整 kernelName 与 TilingKey 不一致"
                )
            if len(shard_started_kernel_names) != len(shard_started_keys):
                raise ValueError(
                    "内存分片同一 TilingKey 命中了多个 kernel 身份"
                )
            sanitizer_started_keys.update(shard_started_keys)
            sanitizer_started_kernel_names.update(
                shard_started_kernel_names
            )

    expected_ids = list(range(len(cases)))
    duplicate_ids = sorted(
        case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
    )
    if sorted(case_ids) != expected_ids or duplicate_ids:
        raise ValueError(
            f"矩阵 case ID 不闭合：missing={sorted(set(expected_ids) - set(case_ids))}, "
            f"extra={sorted(set(case_ids) - set(expected_ids))}, "
            f"duplicates={duplicate_ids}"
        )

    _, full_pairs = _expected_slice(cases, 0, len(cases))
    full_expected_keys = {key for _, key in full_pairs}
    if args.require_tiling_log:
        if len(full_expected_keys) != len(cases):
            raise ValueError(
                "冻结矩阵不是一条 case 对应一个唯一 TilingKey："
                f"cases={len(cases)}, keys={len(full_expected_keys)}"
            )
        if host_keys != full_expected_keys or launch_keys != full_expected_keys:
            raise ValueError("完整矩阵的 expected/host/launch TilingKey 集合不一致")
        if args.scope == "mssanitizer" and sanitizer_started_keys != full_expected_keys:
            raise ValueError("完整矩阵的 sanitizer 启动 TilingKey 集合不一致")
        if args.scope == "mssanitizer" and len(
            sanitizer_started_kernel_names
        ) != len(full_expected_keys):
            raise ValueError("完整矩阵的 sanitizer kernel 身份数量不一致")

    return {
        "schema": "kda-prepare-atk-matrix/v2",
        "scope": args.scope,
        "tool": args.tool,
        "timeout_seconds": args.timeout,
        "loop_nums": args.loop_nums,
        "gm_init_mode": args.gm_init_mode,
        "single_process_mode": args.single_process_mode,
        "require_tiling_log": args.require_tiling_log,
        "runtime_manifest_sha256": runtime_manifest_hash,
        "expected_cases": len(cases),
        "observed_cases": len(case_ids),
        "shard_count": len(summaries),
        "case_json_sha256": case_hash,
        "expected_tiling_key_count": len(full_expected_keys),
        "host_tiling_key_count": len(host_keys),
        "launch_tiling_key_count": len(launch_keys),
        "sanitizer_start_count": sanitizer_start_count,
        "sanitizer_started_tiling_key_count": len(sanitizer_started_keys),
        "sanitizer_started_kernel_name_count": len(
            sanitizer_started_kernel_names
        ),
        "sanitizer_started_kernel_name_sha256": _string_digest(
            sanitizer_started_kernel_names
        ),
        "key_pair_sha256": _key_digest(full_pairs),
        "complete": True,
        "passed": True,
    }


def _verify_or_write_matrix_summary(path: Path, summary: dict) -> None:
    if path.exists():
        if _read_summary(path) != summary:
            raise ValueError(f"矩阵汇总与当前原始证据不一致：{path}")
        return
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def aggregate_matrix(args: argparse.Namespace) -> int:
    result = _build_matrix_summary(args)
    _verify_or_write_matrix_summary(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def verify_sanitizer_suite(args: argparse.Namespace) -> int:
    cases = _load_cases(args.case_file)
    _, expected_pairs = _expected_slice(cases, 0, len(cases))
    expected_keys = {key for _, key in expected_pairs}
    if len(cases) != 432 or len(expected_keys) != 432:
        raise ValueError(
            "sanitizer suite 要求 432 条 case 对应 432 个唯一 TilingKey"
        )
    expected_tools = {"memcheck", "racecheck", "initcheck", "synccheck"}
    if len(args.aggregate) != len(expected_tools):
        raise ValueError("必须提供四种 sanitizer 的 aggregate_summary.json")

    results: dict[str, dict] = {}
    runtime_hashes: set[str] = set()
    kernel_name_hashes: set[str] = set()
    for path in args.aggregate:
        resolved_path = path.resolve()
        if (
            path.is_symlink()
            or not resolved_path.is_file()
            or resolved_path.name != "aggregate_summary.json"
        ):
            raise ValueError(f"sanitizer 汇总路径非法：{path}")
        matrix_root = resolved_path.parent
        runtime_manifest = matrix_root / "runtime_manifest.json"
        existing = _read_summary(resolved_path)
        tool = str(existing.get("tool", ""))
        if tool in results:
            raise ValueError(f"sanitizer 汇总重复：{tool}")
        matrix_args = argparse.Namespace(
            case_file=args.case_file,
            scope="mssanitizer",
            tool=tool,
            runtime_manifest=runtime_manifest,
            timeout=int(existing.get("timeout_seconds", 0)),
            loop_nums=1,
            gm_init_mode="not_applicable",
            single_process_mode="off",
            require_tiling_log=True,
            matrix_root=matrix_root,
            soc=args.soc,
            test_artifact=args.test_artifact,
        )
        summary = _build_matrix_summary(matrix_args)
        if existing != summary:
            raise ValueError(
                f"sanitizer 汇总与当前原始证据不一致：{resolved_path}"
            )
        if (
            summary.get("schema") != "kda-prepare-atk-matrix/v2"
            or summary.get("scope") != "mssanitizer"
            or not summary.get("complete")
            or not summary.get("passed")
        ):
            raise ValueError(f"sanitizer 汇总未通过或 schema 错误：{path}")
        if summary.get("case_json_sha256") != _case_hash(args.case_file):
            raise ValueError(f"sanitizer 汇总的 case 哈希不一致：{path}")
        if (
            summary.get("expected_cases") != len(cases)
            or summary.get("observed_cases") != len(cases)
            or summary.get("expected_tiling_key_count") != len(expected_keys)
            or summary.get("host_tiling_key_count") != len(expected_keys)
            or summary.get("launch_tiling_key_count") != len(expected_keys)
            or summary.get("sanitizer_started_tiling_key_count")
            != len(expected_keys)
            or summary.get("sanitizer_started_kernel_name_count")
            != len(expected_keys)
            or not summary.get("sanitizer_started_kernel_name_sha256")
            or summary.get("key_pair_sha256") != _key_digest(expected_pairs)
        ):
            raise ValueError(f"sanitizer 汇总的 case/key 覆盖不闭合：{path}")
        if not 1 <= int(summary.get("timeout_seconds", 0)) <= 60:
            raise ValueError(f"sanitizer 超时配置不在 1..60 秒内：{path}")
        if (
            summary.get("loop_nums") != 1
            or summary.get("gm_init_mode") != "not_applicable"
            or summary.get("single_process_mode") != "off"
            or summary.get("require_tiling_log") is not True
        ):
            raise ValueError(f"sanitizer 汇总不符合正式 scope 合同：{path}")
        runtime_hashes.add(str(summary.get("runtime_manifest_sha256", "")))
        kernel_name_hashes.add(
            str(summary.get("sanitizer_started_kernel_name_sha256", ""))
        )
        results[tool] = summary

    if set(results) != expected_tools:
        raise ValueError(
            f"sanitizer 工具不闭合：missing={sorted(expected_tools - set(results))}, "
            f"extra={sorted(set(results) - expected_tools)}"
        )
    if len(runtime_hashes) != 1 or "" in runtime_hashes:
        raise ValueError("四种 sanitizer 未使用同一份 runtime kernel")
    if len(kernel_name_hashes) != 1 or "" in kernel_name_hashes:
        raise ValueError("四种 sanitizer 命中的完整 kernel 身份不一致")

    suite = {
        "schema": "kda-prepare-atk-sanitizer-suite/v1",
        "tools": sorted(results),
        "case_json_sha256": _case_hash(args.case_file),
        "runtime_manifest_sha256": runtime_hashes.pop(),
        "sanitizer_started_kernel_name_sha256": kernel_name_hashes.pop(),
        "expected_cases_per_tool": len(cases),
        "observed_cases_total": len(cases) * len(expected_tools),
        "tiling_key_count_per_tool": len(expected_keys),
        "complete": True,
        "passed": True,
    }
    _verify_or_write_matrix_summary(args.output, suite)
    print(json.dumps(suite, ensure_ascii=False, indent=2))
    return 0


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=("accuracy", "determinism", "mssanitizer"), required=True
    )
    parser.add_argument("--tool", default="")
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--loop-nums", type=int, required=True)
    parser.add_argument("--gm-init-mode", required=True)
    parser.add_argument(
        "--single-process-mode", choices=("on", "off"), required=True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    runtime = subparsers.add_parser("runtime", help="冻结实际加载的 kernel 指纹")
    runtime.add_argument("--case-file", type=Path, required=True)
    runtime.add_argument("--soc", default="auto")
    runtime.add_argument("--require-sanitizer", action="store_true")
    runtime.add_argument("--require-complete-key-set", action="store_true")
    runtime.add_argument(
        "--test-artifact", type=Path, action="append", required=True
    )
    runtime.add_argument("--output", type=Path, required=True)
    runtime.set_defaults(func=verify_runtime)

    shard = subparsers.add_parser("shard", help="校验一个已完成的 ATK 分片")
    _common_parser(shard)
    shard.add_argument("--shard-root", type=Path, required=True)
    shard.add_argument("--console-log", type=Path, required=True)
    shard.add_argument("--summary", type=Path, required=True)
    shard.add_argument("--start", type=int, required=True)
    shard.add_argument("--end", type=int, required=True)
    shard.add_argument("--require-tiling-log", action="store_true")
    shard.set_defaults(func=summarize_shard)

    aggregate = subparsers.add_parser("aggregate", help="聚合并闭环校验完整矩阵")
    _common_parser(aggregate)
    aggregate.add_argument("--matrix-root", type=Path, required=True)
    aggregate.add_argument(
        "--soc",
        choices=("ascend910b", "ascend910_93", "ascend950"),
        required=True,
    )
    aggregate.add_argument(
        "--test-artifact", type=Path, action="append", required=True
    )
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--require-tiling-log", action="store_true")
    aggregate.set_defaults(func=aggregate_matrix)

    suite = subparsers.add_parser(
        "sanitizer-suite", help="校验四种 sanitizer 的全量矩阵汇总"
    )
    suite.add_argument("--case-file", type=Path, required=True)
    suite.add_argument(
        "--soc",
        choices=("ascend910b", "ascend910_93", "ascend950"),
        required=True,
    )
    suite.add_argument(
        "--test-artifact", type=Path, action="append", required=True
    )
    suite.add_argument("--aggregate", type=Path, action="append", required=True)
    suite.add_argument("--output", type=Path, required=True)
    suite.set_defaults(func=verify_sanitizer_suite)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
