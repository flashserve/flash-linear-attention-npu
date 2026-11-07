# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
# Adapted for flash-linear-attention-npu by Tianjin University.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import re
import logging
import json
from pathlib import Path
NEW_OPS_PATH = [
    "mc2",
    "attention",
    "ffn",
    "gmm",
    "moe",
]


class OperatorChangeInfo:
    def __init__(self, changed_operators=None, operator_file_map=None,
                 full_rebuild=False, deleted_operators=None):
        self.changed_operators = [] if changed_operators is None else changed_operators
        self.operator_file_map = {} if operator_file_map is None else operator_file_map
        self.full_rebuild = full_rebuild
        self.deleted_operators = [] if deleted_operators is None else deleted_operators


BlackList = {
        "moe_distribute_combine_shmem",
        "moe_distribute_dispatch_shmem",
        "rope_matrix",
        "all_gather_matmul_v2"
    }


def extract_operator_name(file_path, is_experimental):
    path_parts = file_path.lstrip('/').split('/')
    if is_experimental == "TRUE":
        if len(path_parts) < 2 or path_parts[0] != "experimental":
            return ""
        operator_name = path_parts[1]
        if operator_name.startswith("_"):
            return ""
        marker = Path("experimental", operator_name, "CMakeLists.txt")
        return operator_name if marker.is_file() else ""
    domain, operator_name = _get_domain_and_op(path_parts, is_experimental)
    if domain is None:
        return ""

    # 判断是否直接返回默认名称（黑名单、common、实验路径不存在、op_host缺失）
    if _should_return_default(domain, operator_name, path_parts, is_experimental):
        return _get_default_name(domain)

    # 非实验分支且 domain 不在 NEW_OPS_PATH 时，返回默认名称（空或 attention 特殊值）
    if is_experimental != "TRUE" and domain not in NEW_OPS_PATH:
        return _get_default_name(domain)

    # 其他情况返回 operator_name
    return operator_name


def _get_domain_and_op(path_parts, is_experimental):
    """从路径部分提取域和算子名"""
    if is_experimental == "TRUE":
        if len(path_parts) >= 3:
            return path_parts[1], path_parts[2]
    else:
        if len(path_parts) >= 2:
            return path_parts[0], path_parts[1]
    return None, None


def _should_return_default(domain, operator_name, path_parts, is_experimental):
    """检查是否应使用默认名称（而不是 operator_name）"""
    if operator_name in BlackList:
        return True

    # 检查是否是 common 或 experimental 路径不存在
    exp_path = f'experimental/{domain}/{operator_name}'
    if operator_name == "common" or not os.path.exists(exp_path):
        return True

    # 实验分支额外检查 op_host 目录
    if is_experimental == "TRUE":
        # 构造 parent 路径（原代码用 Path(*path_parts[:3])）
        if len(path_parts) >= 3:
            parent = Path(*path_parts[:3])
            target = parent / "op_host"
            if not (target.exists() and target.is_dir()):
                return True
    return False


def _get_default_name(domain):
    """根据域返回默认名称（目前只有 attention 特殊处理）"""
    if domain == 'attention':
        return "fused_infer_attention_score"
    return ""


def get_operator_info_from_ci(changed_file_info_from_ci, is_experimental):
    """
    get operator change info from ci, ci will write `git diff > /or_filelist.txt`
    :param changed_file_info_from_ci: git diff result file from ci
    :return: None or OperatorChangeInfo
    """
    if is_experimental == "TRUE":
        return _experimental_change_info(changed_file_info_from_ci)

    def is_skippable_file(line):
        ext = os.path.splitext(line)[-1].lower()
        return ext in (".md",)

    def process_line(line, operators_set, files_map):
        """处理单行：提取算子名并更新集合和映射"""
        line = line.strip()
        if is_skippable_file(line):
            return
        operator_name = extract_operator_name(line, is_experimental)
        if operator_name:
            operators_set.add(operator_name)
            if operator_name not in files_map:
                files_map[operator_name] = []
            files_map[operator_name].append(line)

    or_file_path = os.path.realpath(changed_file_info_from_ci)
    if not os.path.exists(or_file_path):
        logging.error("[ERROR] change file is not exist, can not get file change info in this pull request.")
        return None

    with open(or_file_path) as or_f:
        lines = or_f.readlines()
        changed_operators = set()
        operator_file_map = {}

        for line in lines:
            process_line(line, changed_operators, operator_file_map)

    return OperatorChangeInfo(changed_operators=list(changed_operators), operator_file_map=operator_file_map)


def _experimental_change_info(changed_file_info_from_ci):
    """Read name-only or tab-separated git --name-status changes.

    Rebuild the small experimental tree on every relevant code change. This
    includes reverse dependents without attempting to parse executable CMake
    dependency declarations in Python. Keep changed/deleted identities for
    artifact and registration cleanup checks.
    """
    change_file = Path(changed_file_info_from_ci)
    if not change_file.is_file():
        raise ValueError(f"Experimental changed-file list is missing: {change_file}")
    shared_prefixes = ("cmake/", "common/", "scripts/", "ci/",
                       ".github/workflows/", "torch_custom/")
    shared_files = {"build.sh", "CMakeLists.txt", "setup.py", "setup.cfg",
                    "pyproject.toml", "MANIFEST.in"}
    changed, deleted, files = set(), set(), {}
    full = False
    for line in change_file.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split("\t")
        if not fields[0]:
            continue
        status = fields[0] if len(fields) > 1 else ""
        paths = fields[1:] if status else fields
        for index, value in enumerate(paths):
            path = value.removeprefix("./")
            if path.endswith(".md"):
                continue
            if path in shared_files or path.startswith(shared_prefixes):
                full = True
                continue
            parts = path.split("/")
            if not parts or parts[0] != "experimental":
                continue
            full = True
            if len(parts) < 3 or parts[1].startswith("_"):
                continue
            name = parts[1]
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name):
                raise ValueError(f"Invalid experimental changed path: {path}")
            changed.add(name)
            files.setdefault(name, []).append(path)
            removed = status.startswith("D") or (status.startswith("R") and index == 0)
            if removed or not (Path("experimental") / name / "CMakeLists.txt").is_file():
                deleted.add(name)
    return OperatorChangeInfo(sorted(changed), files, full, sorted(deleted))


def get_change_ops_list(changed_file_info_from_ci, is_experimental):
    ops_change_info = get_operator_info_from_ci(changed_file_info_from_ci, is_experimental)
    if not ops_change_info:
        logging.info("[INFO] not found ops change info, run all c++.")
        return None

    if ops_change_info.full_rebuild:
        return "all"

    return ";".join(ops_change_info.changed_operators)


if __name__ == '__main__':
    try:
        if "--json" in sys.argv[3:]:
            info = get_operator_info_from_ci(sys.argv[1], sys.argv[2])
            print(json.dumps(vars(info) if info is not None else None, sort_keys=True))
        else:
            print(get_change_ops_list(sys.argv[1], sys.argv[2]))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
