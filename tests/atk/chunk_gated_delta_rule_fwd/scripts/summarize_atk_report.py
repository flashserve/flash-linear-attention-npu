#!/usr/bin/env python3
"""把 ATK Excel 的 summary 页转换成机器可读门禁结果。"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

PASS_WORDS = {"pass", "passed", "success", "successful", "true"}
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference.upper())
    if letters is None:
        return 0
    result = 0
    for letter in letters.group(0):
        result = result * 26 + ord(letter) - ord("A") + 1
    return result - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        data = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(data)
    values = []
    for item in root.findall(f"{{{MAIN_NS}}}si"):
        values.append("".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t")))
    return values


def _sheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relation_id = None
    for sheet in workbook.findall(f".//{{{MAIN_NS}}}sheet"):
        if sheet.attrib.get("name") == sheet_name:
            relation_id = sheet.attrib.get(f"{{{REL_NS}}}id")
            break
    if relation_id is None:
        raise ValueError(f"ATK Excel 缺少 {sheet_name} 工作表")
    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    target = None
    for relation in relationships.findall(f"{{{PKG_REL_NS}}}Relationship"):
        if relation.attrib.get("Id") == relation_id:
            target = relation.attrib.get("Target")
            break
    if not target:
        raise ValueError(f"{sheet_name} 工作表缺少 relationship")
    target = target.lstrip("/")
    return target if target.startswith("xl/") else f"xl/{target}"


def _xlsx_rows(path: Path, sheet_name: str) -> list[list[object]]:
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        sheet = ElementTree.fromstring(archive.read(_sheet_path(archive, sheet_name)))
    rows: list[list[object]] = []
    for row in sheet.findall(f".//{{{MAIN_NS}}}row"):
        values: dict[int, object] = {}
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            column = _column_index(cell.attrib.get("r", "A1"))
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                value = "".join(
                    node.text or "" for node in cell.iter(f"{{{MAIN_NS}}}t")
                )
            else:
                value_node = cell.find(f"{{{MAIN_NS}}}v")
                value = None if value_node is None else value_node.text
                if cell_type == "s" and value is not None:
                    value = shared[int(value)]
                elif cell_type == "b" and value is not None:
                    value = value == "1"
            values[column] = value
        if values:
            width = max(values) + 1
            rows.append([values.get(index) for index in range(width)])
    return rows


def _statistic_result(path: Path) -> dict[str, object]:
    rows = _xlsx_rows(path, "statistic")
    if not rows:
        return {
            "row_count": 0,
            "case_ids": [],
            "execution_failed_case_ids": [],
            "accuracy_failed_case_ids": [],
        }

    headers = [str(value or "") for value in rows[0]]
    id_index = headers.index("编号")
    execution_index = headers.index("运行结果")
    accuracy_indexes = [
        index for index, value in enumerate(headers) if value.endswith("精度通过")
    ]
    case_ids: list[int] = []
    execution_failed: list[int] = []
    accuracy_failed: list[int] = []
    for values in rows[1:]:
        if not values or all(value is None for value in values):
            continue
        padded = list(values) + [None] * max(0, len(headers) - len(values))
        case_id = int(padded[id_index])
        case_ids.append(case_id)
        if str(padded[execution_index] or "").strip().upper() != "SUCCESS":
            execution_failed.append(case_id)
        accuracy_values = [padded[index] for index in accuracy_indexes]
        effective_accuracy = [value for value in accuracy_values if value is not None]
        if effective_accuracy and not all(bool(value) for value in effective_accuracy):
            accuracy_failed.append(case_id)
    return {
        "row_count": len(case_ids),
        "case_ids": case_ids,
        "execution_failed_case_ids": execution_failed,
        "accuracy_failed_case_ids": accuracy_failed,
    }


def _number(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    args = parser.parse_args()

    rows = []
    for values in _xlsx_rows(args.report, "summary")[1:]:
        if not values or all(value is None for value in values):
            continue
        padded = list(values) + [None] * max(0, 8 - len(values))
        status = str(padded[7] or "").strip()
        rows.append(
            {
                "node": padded[0],
                "total_cases": _number(padded[1]),
                "execution_success": _number(padded[2]),
                "execution_failed": _number(padded[3]),
                "accuracy_passed": _number(padded[4]),
                "message_matched": _number(padded[5]),
                "pass_rate": _number(padded[6]),
                "status": status,
                "passed": status.lower() in PASS_WORDS,
            }
        )

    errors = []
    if not rows:
        errors.append("summary 无数据行")
    for row in rows:
        if row["total_cases"] != args.expected_cases:
            errors.append(
                f"{row['node']}: total_cases={row['total_cases']} != {args.expected_cases}"
            )
        if row["execution_success"] != args.expected_cases:
            errors.append(
                f"{row['node']}: execution_success={row['execution_success']} != {args.expected_cases}"
            )
        if row["accuracy_passed"] != args.expected_cases:
            errors.append(
                f"{row['node']}: accuracy_passed={row['accuracy_passed']} != {args.expected_cases}"
            )
        if not row["passed"]:
            errors.append(f"{row['node']}: status={row['status']!r}")

    statistic = _statistic_result(args.report)
    if statistic["row_count"] != args.expected_cases:
        errors.append(
            f"statistic: row_count={statistic['row_count']} != {args.expected_cases}"
        )
    result = {
        "schema": "gdn-core-double-benchmark-atk-summary/v1",
        "report": str(args.report),
        "expected_cases": args.expected_cases,
        "rows": rows,
        "statistic": statistic,
        "passed": not errors,
        "errors": errors,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not errors else 99


if __name__ == "__main__":
    raise SystemExit(main())
