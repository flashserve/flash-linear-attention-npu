# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import sys
import re


black_list = ['moe_gather_v2',
              'moe_inplace_index_add',
              'moe_inplace_index_add_with_sorted',
              'moe_masked_scatter']
op_level_list = ['moe_token_permute_with_routing_map',
                 'moe_token_permute_with_routing_map_grad',
                 'moe_token_unpermute_with_routing_map']


def get_sh_files(gen_dir):
    """获取目录中所有 .sh 文件名（不包含路径）"""
    sh_files = []
    for item in os.listdir(gen_dir):
        item_path = os.path.join(gen_dir, item)
        if os.path.isfile(item_path) and item.lower().endswith('.sh'):
            sh_files.append(item)
    return sh_files


def parse_opname_from_filename(filename):
    """
    从文件名解析 op_name。
    要求格式：xxx-<opname>-<digits>.sh
    成功返回 op_name，失败返回 None
    """
    parts = filename.split('-')
    if len(parts) < 3:
        return None

    return parts[1]


def count_opnames(sh_filenames):
    """统计每个 op_name 出现的次数"""
    opname_to_count = {}
    for filename in sh_filenames:
        op_name = parse_opname_from_filename(filename)
        if op_name is not None:
            opname_to_count[op_name] = opname_to_count.get(op_name, 0) + 1
    opname_to_count_sorted = dict(sorted(opname_to_count.items()))
    return opname_to_count_sorted


def grouped(gen_path, soc, group_size):
    result: list[list[str]] = [[] for _ in range(group_size)]
    if not os.path.isdir(gen_path):
        return result
    sh_files = get_sh_files(gen_path)
    op_counts = count_opnames(sh_files)

    all_rows = []
    added_op_levels = set()
    special_task = ""
    for op_name, count in op_counts.items():
        op_name_real = op_name
        if soc == 'ascend950' and op_name.endswith('_apt'):
            op_name_real = op_name.replace('_apt', '')
        if op_name == 'allto_all_matmul_apt' and op_name.endswith('_apt'):
            op_name_real = op_name.replace('_apt', '')
        if op_name == 'matmul_allto_all_apt' and op_name.endswith('_apt'):
            op_name_real = op_name.replace('_apt', '')
        if op_name_real in black_list:
            continue
        for i in range(count):
            if op_name_real in op_level_list:
                if op_name_real in added_op_levels:
                    continue
                else:
                    added_op_levels.add(op_name_real)
                    special_task = special_task + str(op_name_real) + ","
            else:
                row_string = f"{op_name_real},{count}-{i}"
                all_rows.append(row_string)
    if len(special_task) != 0:
        special_task = special_task[:-1]
        all_rows.append(special_task)

    for idx, row in enumerate(all_rows):
        result[idx % group_size].append(row)

    return result


def grouped_back(gen_path, soc, group_size):
    result: list[list[str]] = [[] for _ in range(group_size)]
    if not os.path.isdir(gen_path):
        return result
    sh_files = get_sh_files(gen_path)
    op_counts = count_opnames(sh_files)

    added_op_levels = set()
    special_task_parts = []
    current_group_index = 0

    for op_name, count in op_counts.items():
        op_name_real = op_name
        if soc == 'ascend950' and op_name.endswith('_apt'):
            op_name_real = op_name.replace('_apt', '')
        elif op_name in ('allto_all_matmul_apt', 'matmul_allto_all_apt'):
            op_name_real = op_name.replace('_apt', '')

        if op_name_real in black_list:
            continue
        if op_name_real in op_level_list:
            if op_name_real not in added_op_levels:
                added_op_levels.add(op_name_real)
                special_task_parts.append(str(op_name_real))
            continue
        if count >= group_size:
            for i in range(group_size):
                row_string = f"{op_name_real},{group_size}-{i}"
                result[current_group_index].append(row_string)
                current_group_index = (current_group_index + 1) % group_size
        else:
            for i in range(count):
                row_string = f"{op_name_real},{count}-{i}"
                result[current_group_index].append(row_string)
                current_group_index = (current_group_index + 1) % group_size

    if special_task_parts:
        special_task = ','.join(special_task_parts)
        result[current_group_index].append(special_task)

    return result


def main(repository_path, soc, group_size=1):
    project_path = os.path.abspath(repository_path)
    gen_path = os.path.abspath(os.path.join(project_path, "build", "binary", soc, "gen"))
    if group_size > 1:
        op_data = grouped_back(gen_path, soc, group_size)
    else:
        op_data = grouped(gen_path, soc, group_size)
    return op_data