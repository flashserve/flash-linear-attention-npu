#!/usr/bin/env python3
"""ATK 测试结果检查器。

解析 ATK 生成的 xlsx 报告，检查 accuracy / performance / determinism / mssanitizer，
统计 fail 数量。公共 ``--type all`` 保持原有语义，只包含 accuracy、determinism
和 mssanitizer；performance 由 runner 在对应阶段显式检查。

用法:
  python3 common/check_atk_result.py --type accuracy --output-root ./atk_output --op chunk_bwd_dqkwg
  python3 common/check_atk_result.py --type all --output-root ./atk_output --op chunk_bwd_dqkwg
  python3 common/check_atk_result.py --type accuracy --output-root ./atk_output --op chunk_kda_fwd_prepare --require-accuracy-pass

退出码: 0=全部通过, 1=存在失败, 2=检查器自身错误(如缺依赖)
"""

import argparse
import glob
import json
import math
import os
import re
import sys

try:
    import openpyxl
except ImportError:
    print("[ATK结果检查] 错误: 需要 openpyxl，请 pip install openpyxl", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# 报告查找
# ---------------------------------------------------------------------------

def _find_xlsx_files(output_root, pattern, *, min_mtime_ns=None):
    """返回按修改时间排序的 xlsx 文件列表（旧→新）。

    ``min_mtime_ns`` 用于把报告绑定到当前执行批次；不传时保持公共
    checker 对历史算子的兼容行为。
    """
    full_pattern = os.path.join(output_root, pattern, "report", "*.xlsx")
    files = []
    for path in glob.glob(full_pattern):
        if min_mtime_ns is not None:
            try:
                if os.stat(path).st_mtime_ns < int(min_mtime_ns):
                    continue
            except OSError:
                continue
        files.append(path)
    return sorted(files, key=os.path.getmtime)


def _find_accuracy_reports(output_root, op):
    """accuracy 报告在 accuracy/atk_output/atk_<op>_* 下。"""
    return _find_xlsx_files(output_root, f"accuracy/atk_output/atk_{op}_*")


def _find_stage_reports(output_root, op, stage, *, min_mtime_ns=None):
    """查找分阶段输出目录，并兼容旧版直接写在输出根目录的报告。"""
    files = []
    for pattern in (
        f"{stage}/atk_output/atk_{op}_*",
        f"{stage}/atk_{op}_*",
        f"atk_{op}_*",
    ):
        files.extend(
            _find_xlsx_files(output_root, pattern, min_mtime_ns=min_mtime_ns)
        )
    return sorted(set(files), key=os.path.getmtime)


# ---------------------------------------------------------------------------
# 报告解析
# ---------------------------------------------------------------------------

def _parse_summary(xlsx_path):
    """解析 xlsx 的 summary sheet，返回 (header_list, data_rows)。

    header_list: 第一行表头列表
    data_rows: 剩余行列表，每行为单元格值列表
    找不到 summary sheet 或文件损坏时返回 (None, None)。
    """
    try:
        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    except Exception:
        return None, None
    if "summary" not in wb.sheetnames:
        wb.close()
        return None, None
    ws = wb["summary"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return None, None
    header = [str(c) if c is not None else "" for c in rows[0]]
    data = [[c for c in r] for r in rows[1:]]
    return header, data


def _is_mssanitizer_report(header):
    """mssanitizer 报告的表头包含 '内存检测' 列。"""
    return any("内存检测" in h for h in header)


def _classify_report(header):
    """根据表头分类报告类型。

    返回 'accuracy' | 'mssanitizer' | 'unknown'
    accuracy 和 determinism 的表头结构相同（含 精度是否达标），无法仅凭表头区分，
    需要调用方根据所在目录判断。
    """
    if _is_mssanitizer_report(header):
        return "mssanitizer"
    if any("精度是否达标" in h or "通过率" in h for h in header):
        return "accuracy"
    return "unknown"


def _extract_summary_row(header, data_rows, *, require_accuracy_pass=False):
    """从 summary 行中提取关键指标。

    返回 dict: {total, exec_pass, exec_fail, check_pass, check_fail,
               pass_rate,达标}
    各字段不存在时为 None。
    """
    def _col_idx(name_part):
        for i, h in enumerate(header):
            if name_part in h:
                return i
        return None

    total_idx = _col_idx("总用例数")
    exec_pass_idx = _col_idx("执行成功用例个数")
    exec_fail_idx = _col_idx("执行失败用例个数")
    check_pass_idx = _col_idx("通过用例个数")
    err_match_idx = _col_idx("错误信息匹配用例个数")
    rate_idx = _col_idx("通过率")
    mss_rate_idx = _col_idx("内存检测通过率")
    # 收集所有"达标"列索引
    # accuracy: 精度是否达标
    # determinism: 精度是否达标(值为"-") + 确定性计算是否达标(实际结果)
    # mssanitizer: 内存检测是否达标
    pass_indices = [i for i, h in enumerate(header) if "达标" in h]

    # 汇总所有 node 行
    total = exec_pass = exec_fail = check_pass = check_fail = 0
    pass_rate = mss_pass_rate = None
    all_pass = True
    for row in data_rows:
        if not row or all(c is None for c in row):
            continue
        if total_idx is not None and row[total_idx] is not None:
            try:
                total += int(row[total_idx])
            except (ValueError, TypeError):
                pass
        if exec_pass_idx is not None and row[exec_pass_idx] is not None:
            try:
                exec_pass += int(row[exec_pass_idx])
            except (ValueError, TypeError):
                pass
        if exec_fail_idx is not None and row[exec_fail_idx] is not None:
            try:
                exec_fail += int(row[exec_fail_idx])
            except (ValueError, TypeError):
                pass
        if check_pass_idx is not None and row[check_pass_idx] is not None:
            try:
                check_pass += int(row[check_pass_idx])
            except (ValueError, TypeError):
                pass
        if err_match_idx is not None and row[err_match_idx] is not None:
            try:
                check_fail += int(row[err_match_idx])
            except (ValueError, TypeError):
                pass
        if rate_idx is not None and row[rate_idx] is not None:
            pass_rate = row[rate_idx]
        if mss_rate_idx is not None and row[mss_rate_idx] is not None:
            mss_pass_rate = row[mss_rate_idx]
        # 检查所有达标列：若某列值为 "Failed" 则整体失败。
        # 其他算子允许用 "-" 表示不适用；Prepare 正式精度则必须明确为 Pass。
        for pi in pass_indices:
            if pi < len(row) and row[pi] is not None:
                val = str(row[pi]).strip()
                if val == "Failed":
                    all_pass = False
                if (
                    require_accuracy_pass
                    and "精度是否达标" in header[pi]
                    and val != "Pass"
                ):
                    all_pass = False
            elif (
                require_accuracy_pass
                and pi < len(header)
                and "精度是否达标" in header[pi]
            ):
                all_pass = False

    # mssanitizer 没有 exec_fail，用 total - (pass count) 估算
    if exec_fail == 0 and total > 0 and check_pass == 0:
        # mssanitizer 情况：total 个用例，达标判断看 "内存检测是否达标"
        exec_pass = total
        check_pass = total if all_pass else 0
        check_fail = 0 if all_pass else total

    return {
        "total": total,
        "exec_pass": exec_pass,
        "exec_fail": exec_fail,
        "check_pass": check_pass,
        "check_fail": check_fail,
        "pass_rate": pass_rate,
        "mss_pass_rate": mss_pass_rate,
        "all_pass": all_pass,
    }


def _strict_accuracy_summary_valid(header, data_rows):
    """严格校验 Prepare accuracy summary 的字段和逐 node 计数。"""
    def _col_idx(name_part):
        for i, value in enumerate(header):
            if name_part in value:
                return i
        return None

    required = {
        "总用例数": _col_idx("总用例数"),
        "执行成功用例个数": _col_idx("执行成功用例个数"),
        "执行失败用例个数": _col_idx("执行失败用例个数"),
        "通过用例个数": _col_idx("通过用例个数"),
        "精度是否达标": _col_idx("精度是否达标"),
    }
    if any(index is None for index in required.values()):
        return False

    saw_row = False
    for row in data_rows:
        if not row or all(value is None for value in row):
            continue
        saw_row = True

        def _count(name):
            index = required[name]
            if index >= len(row) or row[index] is None:
                raise ValueError
            value = row[index]
            if isinstance(value, bool):
                raise ValueError
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            text = str(value).strip()
            if not text.isdigit():
                raise ValueError
            return int(text)

        try:
            total = _count("总用例数")
            exec_pass = _count("执行成功用例个数")
            exec_fail = _count("执行失败用例个数")
            check_pass = _count("通过用例个数")
        except (TypeError, ValueError):
            return False
        if (
            total <= 0
            or exec_pass != total
            or exec_fail != 0
            or check_pass != total
        ):
            return False
        status_index = required["精度是否达标"]
        if status_index >= len(row) or str(row[status_index]).strip() != "Pass":
            return False
    return saw_row


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def check_accuracy(output_root, op, *, require_accuracy_pass=False):
    """检查 accuracy 报告。"""
    files = _find_accuracy_reports(output_root, op)
    if not files:
        return {"found": False, "total": 0, "pass": 0, "fail": 0, "all_pass": False,
                "xlsx": None, "detail": "未找到 accuracy 报告"}
    xlsx = files[-1]
    header, data = _parse_summary(xlsx)
    if header is None:
        return {"found": True, "total": 0, "pass": 0, "fail": 0, "all_pass": False,
                "xlsx": xlsx, "detail": "无法解析 summary sheet"}
    info = _extract_summary_row(
        header, data, require_accuracy_pass=require_accuracy_pass
    )
    if require_accuracy_pass and not any("精度是否达标" in item for item in header):
        info["all_pass"] = False
    if require_accuracy_pass and not _strict_accuracy_summary_valid(header, data):
        info["all_pass"] = False
    if info["total"] <= 0 or info["check_pass"] != info["total"]:
        info["all_pass"] = False
    return {"found": True, **info, "xlsx": xlsx,
            "detail": f"通过率={info['pass_rate']}"}


def _parse_case_id(raw_id):
    """将 JSON/xlsx 中的 case 编号规范化为非负整数。"""
    if isinstance(raw_id, bool):
        raise ValueError
    if isinstance(raw_id, int):
        case_id = raw_id
    elif isinstance(raw_id, float) and raw_id.is_integer():
        case_id = int(raw_id)
    elif isinstance(raw_id, str) and raw_id.strip().isdigit():
        case_id = int(raw_id.strip())
    else:
        raise ValueError
    if case_id < 0:
        raise ValueError
    return case_id


def _load_expected_case_ids(path):
    """读取性能用例 JSON 的精确 case id 集合。"""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            cases = json.load(handle)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"性能用例文件无法解析：{path}：{error}") from error
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"性能用例文件必须是非空列表：{path}")
    case_ids = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"性能用例文件包含非法 case id：{path}")
        try:
            case_id = _parse_case_id(case.get("id"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"性能用例文件包含非法 case id：{path}") from error
        case_ids.append(case_id)
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"性能用例文件包含重复 case id：{path}")
    return set(case_ids)


def _load_expected_performance_targets(path):
    """读取性能 JSON 中可选的 ``performance_target_us``。

    目标通常位于 ``inputs.case_spec.range_values`` 的 JSON 字符串中；同时
    接受顶层字段，便于公共 checker 兼容简化的测试 JSON。返回
    ``(case_ids, targets, errors)``，其中 targets 只包含声明了目标的 case。
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            cases = json.load(handle)
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"性能用例文件无法解析：{path}：{error}") from error
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"性能用例文件必须是非空列表：{path}")

    case_ids = []
    targets = {}
    errors = []
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"性能用例文件包含非法 case：{path}")
        try:
            case_id = _parse_case_id(case.get("id"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"性能用例文件包含非法 case id：{path}") from error
        case_ids.append(case_id)

        spec = case.get("case_spec")
        if spec is None:
            for item in case.get("inputs") or ():
                if isinstance(item, dict) and item.get("name") == "case_spec":
                    spec = item.get("range_values")
                    break
        if isinstance(spec, str):
            try:
                spec = json.loads(spec)
            except (TypeError, ValueError) as error:
                errors.append(f"case {case_id}: case_spec 无法解析")
                continue
        if spec is None:
            spec = case
        if not isinstance(spec, dict):
            errors.append(f"case {case_id}: case_spec 不是对象")
            continue

        raw_target = spec.get("performance_target_us")
        if raw_target is None:
            raw_target = case.get("performance_target_us")
        if raw_target is None:
            continue
        try:
            if isinstance(raw_target, bool):
                raise ValueError
            target = float(raw_target)
            if not math.isfinite(target) or target <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"case {case_id}: performance_target_us 非正有限数")
            continue
        targets[case_id] = target

    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"性能用例文件包含重复 case id：{path}")
    return set(case_ids), targets, errors


def _find_device_performance_column(header):
    """找到 NPU 设备耗时列；不把波动校验列误当成耗时。"""
    candidates = []
    for index, value in enumerate(header):
        text = str(value).strip()
        lowered = text.casefold().replace(" ", "")
        if "波动校验" in text:
            continue
        has_device = "device" in lowered or "设备" in lowered
        has_perf = "性能" in lowered or "performance" in lowered
        has_us = "us" in lowered or "μs" in lowered or "µs" in lowered
        if has_device and has_perf and has_us:
            # 优先 NPU DUT，避免同一报告同时出现 benchmark/其他设备列时取错。
            priority = (
                0
                if "npu_dut" in lowered
                else 1
                if "npu" in lowered
                else 2
            )
            candidates.append((priority, index))
    if not candidates:
        return None
    return min(candidates)[1]


_DURATION_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(us|μs|µs|ms)?$",
    re.IGNORECASE,
)


def _parse_duration_us(value):
    """解析设备耗时为微秒；缺失、非有限数或负数返回 None。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            duration = float(value)
        except (TypeError, ValueError):
            return None
    else:
        text = str(value).strip().replace(",", "")
        match = _DURATION_RE.fullmatch(text)
        if match is None:
            return None
        try:
            duration = float(match.group(1))
        except (TypeError, ValueError):
            return None
        if match.group(2) and match.group(2).casefold() == "ms":
            duration *= 1000.0
    if not math.isfinite(duration) or duration < 0:
        return None
    return duration


def _slice_expected_case_ids(case_file, start=None, end=None):
    """按 ATK 的半开区间选择性能期望 case，支持分段 smoke。"""
    expected_ids = _load_expected_case_ids(case_file)
    if start is None and end is None:
        return expected_ids
    if start is None or end is None:
        raise ValueError("性能期望 case 范围必须同时提供 start 和 end")
    if isinstance(start, bool) or isinstance(end, bool):
        raise ValueError("性能期望 case 范围必须是整数")
    try:
        start = int(start)
        end = int(end)
    except (TypeError, ValueError) as error:
        raise ValueError("性能期望 case 范围必须是整数") from error
    ordered_ids = sorted(expected_ids)
    if not (0 <= start < end <= len(ordered_ids)):
        raise ValueError(
            f"性能期望 case 范围非法：{start}:{end}，总用例数 {len(ordered_ids)}"
        )
    return set(ordered_ids[start:end])


def check_performance(
    output_root,
    op,
    *,
    expected_case_file=None,
    expected_case_start=None,
    expected_case_end=None,
    require_performance_stability=False,
    report_min_mtime_ns=None,
):
    """检查 performance_device 报告中的每条 case 执行结果。

    期望 JSON 声明性能目标时，目标 case 必须有可解析的 NPU 设备耗时，
    且耗时不得超过目标值；没有声明目标的公共算子保持原有兼容行为。
    """
    expected_ids = None
    target_by_id = {}
    target_errors = []
    if expected_case_file is None and (
        expected_case_start is not None or expected_case_end is not None
    ):
        return {
            "found": False,
            "total": 0,
            "pass": 0,
            "fail": 0,
            "all_pass": False,
            "xlsx": None,
            "detail": "性能期望 case 范围必须配合 --expected-case-file 使用",
        }
    if expected_case_file is not None:
        try:
            expected_ids = _slice_expected_case_ids(
                expected_case_file, expected_case_start, expected_case_end
            )
            _, all_targets, target_errors = _load_expected_performance_targets(
                expected_case_file
            )
            target_by_id = {
                case_id: target
                for case_id, target in all_targets.items()
                if case_id in expected_ids
            }
        except ValueError as error:
            return {
                "found": False,
                "total": 0,
                "pass": 0,
                "fail": 0,
                "all_pass": False,
                "xlsx": None,
                "detail": str(error),
            }
    if target_errors:
        return {
            "found": False,
            "total": 0,
            "pass": 0,
            "fail": 0,
            "all_pass": False,
            "xlsx": None,
            "detail": "性能目标声明非法：" + "；".join(target_errors),
        }
    files = _find_stage_reports(
        output_root, op, "perf", min_mtime_ns=report_min_mtime_ns
    )
    if not files:
        return {
            "found": False,
            "total": 0,
            "pass": 0,
            "fail": 0,
            "all_pass": False,
            "xlsx": None,
            "detail": (
                "未找到本次运行后的 performance 报告"
                if report_min_mtime_ns is not None
                else "未找到 performance 报告"
            ),
        }
    xlsx = files[-1]
    workbook = None
    summary_rows = None
    try:
        workbook = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
        if "statistic" not in workbook.sheetnames:
            return {
                "found": True,
                "total": 0,
                "pass": 0,
                "fail": 0,
                "all_pass": False,
                "xlsx": xlsx,
                "detail": "performance 报告缺少 statistic sheet",
            }
        rows = list(workbook["statistic"].iter_rows(values_only=True))
        if "summary" in workbook.sheetnames:
            summary_rows = list(workbook["summary"].iter_rows(values_only=True))
    except Exception as error:
        return {
            "found": True,
            "total": 0,
            "pass": 0,
            "fail": 0,
            "all_pass": False,
            "xlsx": xlsx,
            "detail": f"performance 报告无法读取：{error}",
        }
    finally:
        if workbook is not None:
            workbook.close()

    if not rows:
        return {
            "found": True,
            "total": 0,
            "pass": 0,
            "fail": 0,
            "all_pass": False,
            "xlsx": xlsx,
            "detail": "performance statistic sheet 为空",
        }
    header = [str(value).strip() if value is not None else "" for value in rows[0]]
    try:
        id_index = header.index("编号")
        result_index = header.index("运行结果")
    except ValueError:
        return {
            "found": True,
            "total": 0,
            "pass": 0,
            "fail": 0,
            "all_pass": False,
            "xlsx": xlsx,
            "detail": "performance statistic 缺少编号或运行结果列",
        }

    performance_index = _find_device_performance_column(header)
    target_column_missing = bool(target_by_id) and performance_index is None
    stability_indexes = [
        index
        for index, value in enumerate(header)
        if "性能波动校验结果" in value
    ]
    if require_performance_stability and not stability_indexes:
        return {
            "found": True,
            "total": 0,
            "pass": 0,
            "fail": 0,
            "all_pass": False,
            "xlsx": xlsx,
            "detail": "performance statistic 缺少性能波动校验结果列",
        }
    case_ids = []
    failed_ids = []
    target_failed_ids = []
    malformed_rows = False

    def mark_failed(case_id):
        if case_id not in failed_ids:
            failed_ids.append(case_id)

    for row in rows[1:]:
        if not row or all(value is None for value in row):
            continue
        if id_index >= len(row) or row[id_index] is None:
            malformed_rows = True
            continue
        try:
            case_id = _parse_case_id(row[id_index])
        except (TypeError, ValueError):
            return {
                "found": True,
                "total": 0,
                "pass": 0,
                "fail": 0,
                "all_pass": False,
                "xlsx": xlsx,
                "detail": "performance statistic 的 case 编号非法",
            }
        case_ids.append(case_id)
        result = str(row[result_index]).strip().upper() if result_index < len(row) else ""
        stable_values = [
            row[index] for index in stability_indexes if index < len(row)
        ]
        stable_invalid = require_performance_stability and (
            not stable_values
            or any(
                value is None
                or str(value).strip().lower()
                not in {"pass", "passed", "true", "1"}
                for value in stable_values
            )
        )
        target_invalid = False
        if case_id in target_by_id:
            measured = None
            if performance_index is not None and performance_index < len(row):
                measured = _parse_duration_us(row[performance_index])
            target_invalid = (
                target_column_missing
                or measured is None
                or measured > target_by_id[case_id]
            )
            if target_invalid:
                target_failed_ids.append(case_id)
        if result != "SUCCESS" or stable_invalid or target_invalid:
            mark_failed(case_id)
    total = len(case_ids)
    passed = total - len(failed_ids)
    duplicate_ids = len(case_ids) != len(set(case_ids))
    missing_ids = set()
    unexpected_ids = set()
    if expected_ids is not None:
        missing_ids = expected_ids - set(case_ids)
        unexpected_ids = set(case_ids) - expected_ids
    # 目标 case 若完全没有出现在报告中，也属于缺少测量值；显式列出，
    # 便于分段执行或报告截断时定位具体 case。
    for case_id in target_by_id:
        if case_id not in case_ids:
            target_failed_ids.append(case_id)
    summary_failed = False
    if summary_rows:
        summary_header = [
            str(value).strip() if value is not None else ""
            for value in summary_rows[0]
        ]
        status_indexes = [
            index
            for index, value in enumerate(summary_header)
            if "性能是否达标" in value
        ]
        rate_indexes = [
            index
            for index, value in enumerate(summary_header)
            if "性能通过率" in value
        ]
        for row in summary_rows[1:]:
            for index in status_indexes:
                value = row[index] if index < len(row) else None
                if value is None or str(value).strip().lower() not in {
                    "pass",
                    "passed",
                    "true",
                }:
                    summary_failed = True
            for index in rate_indexes:
                value = row[index] if index < len(row) else None
                if value is None:
                    summary_failed = True
                    continue
                try:
                    rate = float(str(value).strip().rstrip("%"))
                except (TypeError, ValueError):
                    summary_failed = True
                    continue
                if rate != 100.0:
                    summary_failed = True
    return {
        "found": True,
        "total": total,
        "pass": passed,
        "fail": len(failed_ids),
        "check_pass": passed,
        "all_pass": (
            total > 0
            and not failed_ids
            and not duplicate_ids
            and not malformed_rows
            and not summary_failed
            and (
                expected_ids is None
                or (set(case_ids) == expected_ids and total == len(expected_ids))
            )
        ),
        "xlsx": xlsx,
        "detail": (
            f"运行成功={passed}/{total}"
            if expected_ids is None
            else (
                f"运行成功={passed}/{total}，期望={len(expected_ids)}"
                f"，缺少={sorted(missing_ids)}，多出={sorted(unexpected_ids)}"
                + (
                    f"，目标未达标={sorted(set(target_failed_ids))}"
                    if target_by_id
                    else ""
                )
            )
        ),
        "target_case_count": len(target_by_id),
        "target_failed_ids": sorted(set(target_failed_ids)),
        "target_column": (
            header[performance_index] if performance_index is not None else None
        ),
        "report_min_mtime_ns": report_min_mtime_ns,
    }


def check_determinism(output_root, op):
    """检查 determinism 报告。"""
    files = _find_stage_reports(output_root, op, "determinism")
    # 从新到旧找第一个非 mssanitizer 报告
    for f in reversed(files):
        header, _ = _parse_summary(f)
        if header is not None and not _is_mssanitizer_report(header):
            header, data = _parse_summary(f)
            info = _extract_summary_row(header, data)
            return {"found": True, **info, "xlsx": f,
                    "detail": f"通过率={info['pass_rate']}"}
    return {"found": False, "total": 0, "pass": 0, "fail": 0, "all_pass": False,
            "xlsx": None, "detail": "未找到 determinism 报告"}


def check_mssanitizer(output_root, op):
    """检查 mssanitizer 报告。"""
    files = _find_stage_reports(output_root, op, "mssanitizer")
    for f in reversed(files):
        header, _ = _parse_summary(f)
        if header is not None and _is_mssanitizer_report(header):
            header, data = _parse_summary(f)
            info = _extract_summary_row(header, data)
            return {"found": True, **info, "xlsx": f,
                    "detail": f"内存检测通过率={info['mss_pass_rate']}"}
    return {"found": False, "total": 0, "pass": 0, "fail": 0, "all_pass": False,
            "xlsx": None, "detail": "未找到 mssanitizer 报告"}


CHECKERS = {
    "accuracy": check_accuracy,
    "performance": check_performance,
    "determinism": check_determinism,
    "mssanitizer": check_mssanitizer,
}

# ``--type all`` 是已有公共接口，保持原来的三项语义。性能报告格式和
# 验收口径由各算子单独声明，统一 runner 会在执行 performance 阶段后
# 显式调用该 checker，不把新要求扩散到其他算子的历史 ``all`` 流程。
DEFAULT_ALL_TYPES = ("accuracy", "determinism", "mssanitizer")

TYPE_LABELS = {
    "accuracy": "精度",
    "performance": "性能",
    "determinism": "确定性",
    "mssanitizer": "内存检测",
}


def main():
    parser = argparse.ArgumentParser(description="ATK 测试结果检查器")
    parser.add_argument("--type", required=True,
                        choices=["accuracy", "performance", "determinism", "mssanitizer", "all"],
                        help="检查的测试类型")
    parser.add_argument("--output-root", required=True,
                        help="ATK 输出根目录（如 ./atk_output）")
    parser.add_argument("--op", required=True, help="算子名（如 chunk_bwd_dqkwg）")
    parser.add_argument(
        "--require-accuracy-pass",
        action="store_true",
        help="accuracy summary 的精度是否达标列必须明确为 Pass",
    )
    parser.add_argument(
        "--expected-case-file",
        help="performance 报告必须覆盖的 case JSON；Prepare 性能阶段使用",
    )
    parser.add_argument(
        "--expected-case-start",
        type=int,
        help="性能期望 case 半开区间起点，配合 --expected-case-file 使用",
    )
    parser.add_argument(
        "--expected-case-end",
        type=int,
        help="性能期望 case 半开区间终点，配合 --expected-case-file 使用",
    )
    parser.add_argument(
        "--require-performance-stability",
        action="store_true",
        help="performance statistic 必须包含并通过性能波动校验结果列",
    )
    parser.add_argument(
        "--report-min-mtime-ns",
        type=int,
        help="只接受修改时间不早于该纳秒时间戳的 performance 报告",
    )
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_root)
    types = list(DEFAULT_ALL_TYPES) if args.type == "all" else [args.type]

    results = {}
    any_fail = False

    for t in types:
        if t == "accuracy":
            r = check_accuracy(
                output_root,
                args.op,
                require_accuracy_pass=args.require_accuracy_pass,
            )
        elif t == "performance":
            r = check_performance(
                output_root,
                args.op,
                expected_case_file=args.expected_case_file,
                expected_case_start=args.expected_case_start,
                expected_case_end=args.expected_case_end,
                require_performance_stability=args.require_performance_stability,
                report_min_mtime_ns=args.report_min_mtime_ns,
            )
        else:
            r = CHECKERS[t](output_root, args.op)
        results[t] = r
        label = TYPE_LABELS[t]
        if not r["found"]:
            print(f"[ATK结果检查] {label}: 未找到报告（失败）")
            any_fail = True
            continue
        status = "Pass" if r["all_pass"] else "Failed"
        if not r["all_pass"]:
            any_fail = True
        xlsx_name = os.path.basename(r.get("xlsx", "")) if r.get("xlsx") else ""
        # 实际失败数 = 总用例 - 通过用例 (check_fail 是"错误信息匹配"，不等价于未通过)
        actual_fail = r["total"] - r["check_pass"]
        print(f"[ATK结果检查] {label}: {status} "
              f"(总用例={r['total']}, 通过={r['check_pass']}, "
              f"失败={actual_fail}) [{xlsx_name}]")

    if args.type == "all":
        passed_types = sum(1 for t in types if results[t].get("all_pass"))
        fail_count = len(types) - passed_types
        print(
            f"[ATK结果检查] 汇总: {passed_types}/{len(types)} 通过, "
            f"{fail_count} 项失败"
        )

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
