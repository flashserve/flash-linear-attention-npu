#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""合并待拆分算子的json文件。"""

import json
import sys
import os
from typing import List
import argparse


def load_json_file(file_path: str) -> dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {file_path} (original error: {e.msg})") from e


def save_json_file(file_path: str, data: dict) -> None:
    # 确保输出目录存在
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def merge_binlist(base_content: dict, update_content: dict) -> dict:
    # 获取两个文件的binList，不存在则为空列表
    binlist_base = base_content.get("binList", [])
    binlist_update = update_content.get("binList", [])

    #复制base内容并合并binList
    merged_content = base_content.copy()
    merged_content["binList"] = binlist_base + binlist_update
    return merged_content


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-file',
                        required=True,
                        help='Path to the base JSON file')
    parser.add_argument('--update-file',
                        required=True,
                        help='Path to the JSON file to be merged')
    parser.add_argument('--output-file',
                        required=True,
                        type=os.path.realpath,
                        help='Path to the output JSON file after merging')
    args = parser.parse_args(argv)
    return args


def main(argv: List[str]) -> bool:
    # 解析参数
    args = parse_args(argv)
    # 加载基础文件和待合并文件
    base_content = load_json_file(args.base_file)
    update_content = load_json_file(args.update_file)
    # 合并binList
    merged_result = merge_binlist(base_content, update_content)
    # 保存结果
    save_json_file(args.output_file, merged_result)
    return True


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)


