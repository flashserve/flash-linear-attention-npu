#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
获取修改文件应触发的soc版本.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import yaml
from parse_changed_files import Module
from parse_changed_files import Parser


class GetSocModules(Module):
    def __init__(self, name):
        super().__init__(name=name)

    def get_test_options(self, f: Path) -> List[str]:

        def check_option_arch(path_obj: Path, option: List[str]):
            if not option:
                return True
            if option[0] == "ascend310p":
                ascend310p_exclued_folders = ["arch32", "arch35",
                    "ascend910b", "ascend910_93", "ascend950"]
                return any(folder in path_obj.parts for folder in ascend310p_exclued_folders)
            if option[0] == "ascend910b":
                ascend910b_exclued_folders = ["arch20", "arch31", "arch35",
                    "ascend310p", "ascend310p"]
                return any(folder in path_obj.parts for folder in ascend910b_exclued_folders)
            if option[0] == "ascend950":
                ascend950_exclued_folders = ["arch20", "arch31", "arch32",
                    "ascend910b", "ascend910_93", "ascend310p"]
                return any(folder in path_obj.parts for folder in ascend950_exclued_folders)
            return False

        def is_excluded(e_f: Path, option: str):
            if check_option_arch(e_f, option):
                return True
            for e in self.src_exclude_files:
                try:
                    e_f.relative_to(e)
                    return True
                except ValueError:
                    continue
            return False

        related_socs: List[str] = []
        for s in self.src_files:
            if is_excluded(e_f=f, option=self.options):
                continue
            try:
                if f.relative_to(s):
                    # 当同一个修改文件需要触发多个 soc 时, 需要把这些 soc 全部添加
                    related_socs.extend(self.options)
            except ValueError:
                continue
        # 关联 socs 去重
        related_socs = list(set(related_socs))
        return related_socs


class ParserSoc(Parser):
    """
    规则文件、修改文件列表文件解析.
    """

    _Modules: List[GetSocModules] = []         # 保存规则文件(tests/test_soc_config.yaml)内设置的模块列表

    @staticmethod
    def main() -> str:
        # 脚本运行参数注册，包括配置文件和修改文件列表
        ps = argparse.ArgumentParser(description="Get socs according to changed files", epilog="End!")
        ps.add_argument("-c", "--classify", required=True, nargs=1, type=Path, help="tests/test_soc_config.yaml")
        ps.add_argument("-f", "--file", required=True, nargs=1, type=Path, help="changed files desc file.")
        # 子命令行，绑定一下获取soc的函数
        sub_ps = ps.add_subparsers(help="Sub-Command")
        p_soc = sub_ps.add_parser('get_related_soc', help="Get related soc.")
        p_soc.set_defaults(func=ParserSoc.get_related_soc)
        # 处理，先获取配置文件的信息，然后是修改文件列表
        args = ps.parse_args()
        logging.debug(args)
        if not ParserSoc.parse_classify_file(file=Path(args.classify[0])):
            return ""
        if not ParserSoc.parse_changed_file(file=Path(args.file[0])):
            return ""
        ParserSoc.print_details()
        rst_soc = args.func()
        return rst_soc

    @classmethod
    def get_related_soc(cls):
        def soc_test_list_append(soc, soc_test_option_lst):
            if soc not in soc_test_option_lst:
                soc_test_option_lst.append(soc)
        soc_test_option_lst: List[str] = []
        for p in cls._ChangedPaths:
            for m in cls._Modules:
                new_options = m.get_test_options(f=p)
                for soc in new_options:
                    soc_test_list_append(soc, soc_test_option_lst)
        if len(soc_test_option_lst) == 0:
            logging.info("lst is empty, trigger soc ascend910b.")
            return "ascend910b"
        soc_test_ut_str: str = ""
        for soc in soc_test_option_lst:
            if soc not in cls._UTExcludes:
                soc_test_ut_str += f"{soc},"
        soc_test_ut_str = f"{soc_test_ut_str}"
        soc_test_ut_str = soc_test_ut_str[:-1]
        logging.info(f"Trigger UTs of soc: {soc_test_ut_str}")
        return soc_test_ut_str

    @classmethod
    def get_ops_test_option_lst(cls) -> List[str]:
        def ops_test_list_append(opt, ops_test_option_lst):
            if opt not in ops_test_option_lst:
                ops_test_option_lst.append(opt)
        ops_test_option_lst: List[str] = []
        for p in cls._ChangedPaths:
            for m in cls._Modules:
                new_options = m.get_test_example_ops_test_options(f=p)
                for opt in new_options:
                    ops_test_list_append(opt, ops_test_option_lst)
        return ops_test_option_lst

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, format='[%(asctime)s][%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    result = ParserSoc.main()
    logging.info(result)
