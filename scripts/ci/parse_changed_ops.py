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
# 本仓算子树的仓库相对根（见 docs/architecture/experimental 构建适配设计.md §4.1）：
#   稳定树 fla/ops/ascendc/<域>/<算子>/…
#   实验树 experimental/fla/ops/ascendc/<域>/<算子>/…    ← 同构镜像
_OPS_TREE_ROOT = ("fla", "ops", "ascendc")
_EXPERIMENTAL_PREFIX = ("experimental",)

# 受支持的域。本仓为 cp / gdn / kda（common 只放公共头，不是算子）
OP_DOMAINS = ("cp", "gdn", "kda")

# 非算子目录：即使层级对上，也不应当成算子
NON_OP_DIRS = ("common", "3rd", "tools", "tests", "internal_ops")

# 算子目录下的固定子目录。第一个出现的标记之前，整段就是算子目录本身。
# 用它定位而不是固定下标，是因为本仓算子层级不固定：
#   fla/ops/ascendc/kda/chunk_kda_fwd/…                                  算子在第 2 层
#   fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/…                      算子在第 3 层
#   fla/ops/ascendc/kda/chunk_kda_bwd/internal_ops/kda_gate_bwd_post/…   算子在第 4 层
_OP_SUBDIR_MARKERS = ("op_host", "op_kernel", "op_kernel_aicpu", "op_api",
                      "op_graph", "op_tiling", "tests", "examples")


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
    domain, operator_name, op_dir_parts = _locate_op_dir(path_parts, is_experimental)
    if domain is None:
        return ""

    # 判断是否直接返回默认名称（黑名单、域非法、非算子目录、目录不存在、op_host 缺失）
    if _should_return_default(domain, operator_name, op_dir_parts, is_experimental):
        return _get_default_name(domain)

    # 其他情况返回 operator_name
    return operator_name


def _tree_root(is_experimental):
    """本次判定的树根。"""
    if is_experimental == "TRUE":
        return _EXPERIMENTAL_PREFIX + _OPS_TREE_ROOT
    return _OPS_TREE_ROOT


def _locate_op_dir(path_parts, is_experimental):
    """定位变更文件所属的算子目录。

    返回 ``(domain, operator_name, op_dir_parts)``；路径不在本仓算子树下时返回
    ``(None, None, None)``。

    不按固定下标取域/算子，而是用 ``_OP_SUBDIR_MARKERS`` 定位算子目录的边界：
    第一个 ``op_host`` / ``op_kernel`` / ``tests`` … 之前的全部段落就是算子目录，
    其第 1 段是域、最后 1 段是算子名。
    """
    root = _tree_root(is_experimental)
    n = len(root)
    if len(path_parts) <= n or tuple(path_parts[:n]) != root:
        return None, None, None

    rest = list(path_parts[n:])
    cut = len(rest)
    for i, seg in enumerate(rest):
        if seg in _OP_SUBDIR_MARKERS:
            cut = i
            break
    op_dir = rest[:cut]

    # 变更文件可能直接落在算子目录里（如 <算子>/CMakeLists.txt）：
    # 从末尾逐段回退，直到剩下的是一个真实存在的目录。
    while op_dir and not Path(*(root + tuple(op_dir))).is_dir():
        op_dir = op_dir[:-1]
    if not op_dir:
        return None, None, None
    return op_dir[0], op_dir[-1], root + tuple(op_dir)


def _should_return_default(domain, operator_name, op_dir_parts, is_experimental):
    """检查是否应使用默认名称（而不是 operator_name）"""
    if operator_name in BlackList:
        return True

    # 域必须受支持；common / 3rd / tools / tests 等不是算子
    if domain not in OP_DOMAINS or operator_name in NON_OP_DIRS:
        return True

    # 该路径必须真实对应一个算子目录（op_dir_parts 相对当前工作目录，即仓库根）
    op_dir = Path(*op_dir_parts)
    if not op_dir.is_dir():
        return True

    # 实验分支额外要求实验树里确实有 op_host/
    if is_experimental == "TRUE":
        target = op_dir / "op_host"
        if not (target.exists() and target.is_dir()):
            return True
    return False


def _get_default_name(domain):
    """域的兜底算子名。

    参考仓在未命中白名单时会回退到某个「大盘算子」（如 fused_infer_attention_score）；
    本仓没有这个等价物，统一返回空串 —— 让调用方走「本次不编任何算子」，
    而不是把一个无关算子塞进编译范围。
    """
    return ""


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
