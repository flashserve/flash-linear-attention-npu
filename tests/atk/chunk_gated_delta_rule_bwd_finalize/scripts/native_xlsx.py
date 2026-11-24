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
