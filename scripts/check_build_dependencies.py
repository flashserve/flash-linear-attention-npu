 #!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""检查构建依赖。"""

import argparse
import logging
import os
import sys
from typing import Iterator, List, NamedTuple, Optional


class Receiver(NamedTuple):
    """消息接收器。"""
    warn_msgs: List[str]
    err_msgs: List[str]


def parse_version_line(line: str) -> str:
    """解析版本行。"""
    version = line.strip().split('=', maxsplit=1)[1]
    return version.split('-')[0]


def read_pkg_version(recv: Receiver, ascend_install_path: str, name: str) -> Optional[str]:
    """读取包版本。"""
    filepath = os.path.join(ascend_install_path, 'share', 'info', name, 'version.info')
    if not os.path.isfile(filepath):
        recv.err_msgs.append(f'{filepath} does not exist in read_pkg_version!')
        return None

    with open(filepath, encoding='utf-8') as file:
        for line in file:
            if line.startswith('Version='):
                return parse_version_line(line)

    recv.err_msgs.append(f'The version field was not found in {filepath} in read_pkg_version!')
    return None


def check_build_dep_item(version: str, dep: str) -> bool:
    """检查构建依赖项。"""
    def split_version(version: str) -> List[int]:
        return [int(num) for num in version.split('.')]

    def check_ge() -> bool:
        for req, rel in zip(dep_parts, version_parts):
            if req > rel:
                return False
            if rel > req:
                return True
        return True

    def check_gt() -> bool:
        for req, rel in zip(dep_parts, version_parts):
            if req > rel:
                return False
            if rel > req:
                return True
        return False

    def check_le() -> bool:
        for req, rel in zip(dep_parts, version_parts):
            if rel > req:
                return False
            if req > rel:
                return True
        return True

    def check_lt() -> bool:
        for req, rel in zip(dep_parts, version_parts):
            if rel > req:
                return False
            if req > rel:
                return True
        return False

    def check_eq() -> bool:
        for req, rel in zip(dep_parts, version_parts):
            if req != rel:
                return False
        return True

    version_parts = split_version(version)

    if dep.startswith('>='):
        dep = dep[2:]
        check_func = check_ge
    elif dep.startswith('>'):
        dep = dep[1:]
        check_func = check_gt
    elif dep.startswith('<='):
        dep = dep[2:]
        check_func = check_le
    elif dep.startswith('<'):
        dep = dep[1:]
        check_func = check_lt
    else:
        check_func = check_eq

    dep_parts = split_version(dep)

    if check_func is check_eq:
        return check_func()

    if len(dep_parts) < len(version_parts):
        dep_parts.extend([0] * (len(version_parts) - len(dep_parts)))
    elif len(dep_parts) > len(version_parts):
        version_parts.extend([0] * (len(dep_parts) - len(version_parts)))

    return check_func()


def check_build_dep(version: str, dep_info: str) -> bool:
    """检查构建依赖。"""
    def check_range(dep: str, deps_iter: Iterator[str]) -> bool:
        result = check_build_dep_item(version, dep)
        for dep in deps_iter:
            if dep.startswith('<') or dep.startswith('<='):
                return result and check_build_dep_item(version, dep)
            result &= check_build_dep_item(version, dep)
        return result

    deps = [dep.strip() for dep in dep_info.split(',')]
    deps_iter = iter(deps)

    for dep in deps_iter:
        if dep.startswith('>=') or dep.startswith('>'):
            result = check_range(dep, deps_iter)
        else:
            result = check_build_dep_item(version, dep)
        if result:
            return True
    return False



def check_build_deps(recv: Receiver, ascend_install_path: str, deps: list):
    """检查构建依赖。"""
    deps_iter = iter(deps)
    for dep_pkg in deps_iter:
        dep_info = next(deps_iter)
        version = read_pkg_version(recv, ascend_install_path, dep_pkg)
        if not version:
            continue
        try:
            if not check_build_dep(version, dep_info):
                warn_msg = 'Check build dependency failed! ' \
                        f'Required {dep_pkg} version is {dep_info}, but {dep_pkg} version is {version}.'
                recv.warn_msgs.append(warn_msg)
        except ValueError:
            err_msg = f'Check build dependency error! version is {version}, dep_info is {dep_info}.'
            recv.err_msgs.append(err_msg)


def main():
    """主流程。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('ascend_install_path', help='Ascend install path.')
    parser.add_argument('deps', nargs='*', help='Dependency informations.')
    args = parser.parse_args()

    logging.basicConfig(format=f'{os.path.basename(__file__)}: %(levelname)s: %(message)s')
    if len(args.deps) % 2 != 0:
        logging.error('The deps argument must contain an even number of elements!')
        return False

    recv = Receiver([], [])
    check_build_deps(recv, args.ascend_install_path, args.deps)

    if recv.warn_msgs:
        for warn_msg in recv.warn_msgs:
            logging.warning(warn_msg)

    if recv.err_msgs:
        for err_msg in recv.err_msgs:
            logging.error(err_msg)
        return False

    return True


if __name__ == '__main__':
    if not main():
        sys.exit(1)