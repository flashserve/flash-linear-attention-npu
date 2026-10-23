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
from pathlib import Path
# Keep the pre-existing parser contract for non-experimental callers.
NEW_OPS_PATH = ("mc2", "attention", "ffn", "gmm", "moe")
_EXPERIMENTAL_PREFIX = ("experimental",)


class OperatorChangeInfo:
    def __init__(self, changed_operators=None, operator_file_map=None):
        self.changed_operators = [] if changed_operators is None else changed_operators
        self.operator_file_map = {} if operator_file_map is None else operator_file_map


BlackList = {
        "moe_distribute_combine_shmem",
        "moe_distribute_dispatch_shmem",
        "rope_matrix",
        "all_gather_matmul_v2"
    }


def extract_operator_name(file_path, is_experimental):
    path_parts = file_path.lstrip('/').split('/')
    if is_experimental != "TRUE":
        return _extract_stable_operator_name(path_parts)
    if len(path_parts) < 2 or path_parts[0] != "experimental":
        return ""
    operator_name = path_parts[1]
    if operator_name.startswith("_"):
        return ""
    return operator_name if (Path(*_EXPERIMENTAL_PREFIX, operator_name) / "CMakeLists.txt").is_file() else ""


def _extract_stable_operator_name(path_parts):
    if len(path_parts) < 2:
        return ""
    domain, operator_name = path_parts[:2]
    default = "fused_infer_attention_score" if domain == "attention" else ""
    if (operator_name in BlackList or operator_name == "common" or
            not os.path.exists(f"experimental/{domain}/{operator_name}")):
        return default
    return operator_name if domain in NEW_OPS_PATH else default


def get_operator_info_from_ci(changed_file_info_from_ci, is_experimental):
    """
    get operator change info from ci, ci will write `git diff > /or_filelist.txt`
    :param changed_file_info_from_ci: git diff result file from ci
    :return: None or OperatorChangeInfo
    """
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


def get_change_ops_list(changed_file_info_from_ci, is_experimental):
    ops_change_info = get_operator_info_from_ci(changed_file_info_from_ci, is_experimental)
    if not ops_change_info:
        logging.info("[INFO] not found ops change info, run all c++.")
        return None

    return ";".join(ops_change_info.changed_operators)


if __name__ == '__main__':
    print(get_change_ops_list(sys.argv[1], sys.argv[2]))
