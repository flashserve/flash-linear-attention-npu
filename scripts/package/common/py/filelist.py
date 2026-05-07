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

"""filelist相关类。"""

import itertools
import os
from enum import IntEnum
from collections import Counter
from functools import partial
from itertools import chain, repeat
from operator import and_, attrgetter, contains, itemgetter, lt, methodcaller, ne, not_
from typing import Callable, Iterator, List, NamedTuple, Set, Tuple

from .utils.pkg_utils import (TOP_DIR, FilelistError, GenerateFilelistError,
                              conditional_apply, pairwise, swap_args, config_feature_to_string)
from .utils.funcbase import (any_, constant, dispatch, identity, invoke, pipe, side_effect, star_apply)
from .utils.comm_log import CommLog


class FileItem(NamedTuple):
    """文件条目"""
    module: str
    operation: str
    relative_path_in_pkg: str
    relative_install_path: str
    is_in_docker: str
    permission: str
    owner_group: str
    install_type: str
    softlink: List[str]
    feature: Set[str]
    is_common_path: str
    configurable: str
    hash_value: str
    block: str
    pkg_inner_softlink: List[str]
    chip: Set[str]
    is_dir: bool


def create_file_item(*args, **kwargs) -> FileItem:
    """创建文件条目。"""
    file_item = FileItem(*args, **kwargs)

    if not isinstance(file_item.feature, set):
        raise TypeError('The feature parameter should be a set.')
    if not isinstance(file_item.chip, set):
        raise TypeError('The chip parameter should be a set.')
    if not isinstance(file_item.softlink, list):
        raise TypeError('The softlink parameter should be a list.')
    if not isinstance(file_item.pkg_inner_softlink, list):
        raise TypeError('The pkg_inner_softlink parameter should be a list.')

    return file_item


# 文件列表
FileList = List[FileItem]


def soft_links_to_string(soft_links: List[str]) -> str:
    """软链接转换为字符串。"""
    if not soft_links:
        return 'NA'
    return ';'.join(soft_links)


def file_item_to_string(item: FileItem) -> str:
    """文件条目转换为字符串。"""
    return ','.join([
        item.module, item.operation, item.relative_path_in_pkg, item.relative_install_path,
        item.is_in_docker, item.permission, item.owner_group, item.install_type,
        soft_links_to_string(item.softlink), config_feature_to_string(item.feature),
        item.is_common_path, item.configurable, item.hash_value, item.block,
        soft_links_to_string(item.pkg_inner_softlink), config_feature_to_string(item.chip)
    ])


def get_filelist_header_string() -> str:
    """获取文件列表表头。"""
    return ','.join([
        'module', 'operation', 'relative_path_in_pkg', 'relative_install_path',
        'is_in_docker', 'permission', 'owner:group', 'install_type',
        'softlink', 'feature', 'is_common_path', 'configurable', 'hash',
        'block', 'pkg_inner_softlink', 'chip'
    ])


def get_soft_links_not_in_common_paths(filelist: FileList, target_env: str) -> Iterator[List[str]]:
    for file_item_t in filelist:
        if file_item_t.relative_install_path.startswith(target_env):
            for softlink in file_item_t.softlink:
                if not softlink.startswith(target_env):
                    yield softlink


def fill_is_common_path(filelist: FileList, target_env: str) -> Iterator[FileItem]:
    """填充文件条目中是否为公共目录字段。"""
    soft_links = set(get_soft_links_not_in_common_paths(filelist, target_env))
    for file_item in filelist:
        if file_item.relative_install_path.startswith(target_env):
            yield file_item._replace(is_common_path='Y')
        else:
            is_soft_links_prefix = map(
                methodcaller('startswith', f'{file_item.relative_install_path}/'), soft_links
            )
            if any(is_soft_links_prefix):
                yield file_item._replace(is_common_path='YY')
            else:
                yield file_item


def is_relative_install_path(path: str) -> bool:
    """是否为相对路径。"""
    if path.startswith('/'):
        return False
    return True


def is_specific_operations(file_item: FileItem, operations: List[str]) -> bool:
    """是否为特定的操作类型。"""
    if file_item.operation in operations:
        return True
    return False


def is_specific_install_type(file_item: FileItem, install_types: Set[str]) -> bool:
    """是否为特定的安装类型。"""
    item_install_types = set(file_item.install_type.split(';'))
    if 'all' in item_install_types:
        return True
    if item_install_types & install_types:
        return True
    return False


def get_install_path_dirs(install_path: str) -> Iterator[str]:
    """获取安装路径父目录。"""
    install_path = os.path.dirname(install_path)
    while install_path not in ('', '/'):
        yield install_path
        install_path = os.path.dirname(install_path)


def get_missing_dir_set(filelist: FileList) -> Set[str]:
    """获取缺失目录集合。

    文件列表可能出现某一级目录缺失情况。
    如配置了file_info:aaa/bbb/ccc.txt，但只配置了dir_info:aaa，
    那么缺失dir_info:aaa/bbb
    """
    parent_dirs: Set[str] = invoke(
        pipe(
            dispatch(
                pipe(
                    partial(
                        filter,
                        partial(is_specific_operations, operations={'copy', 'copy_entity'}),
                    ),
                    partial(map, attrgetter('relative_install_path')),
                    partial(filter, is_relative_install_path),
                    set,
                    partial(map, get_install_path_dirs),
                    chain.from_iterable,
                ),
                pipe(
                    partial(map, attrgetter('softlink')),
                    chain.from_iterable,
                    partial(
                        filter,
                        pipe(
                            dispatch(
                                bool,
                                is_relative_install_path,
                                partial(ne, 'NA'),
                            ),
                            all
                        )
                    ),
                    set,
                    partial(map, get_install_path_dirs),
                    chain.from_iterable,
                ),
                pipe(
                    partial(map, attrgetter('pkg_inner_softlink')),
                    chain.from_iterable,
                    partial(
                        filter,
                        pipe(
                            dispatch(
                                bool,
                                partial(ne, 'NA'),
                            ),
                            all
                        )
                    ),
                    set,
                    partial(map, get_install_path_dirs),
                    chain.from_iterable,
                ),
            ),
            chain.from_iterable,
            set,
        ),
        filelist
    )
    mkdir_installs: Set[str] = {
        file_item.relative_install_path
        for file_item in filter(
            partial(is_specific_operations, operations={'mkdir'}),
            filelist
        )
        if is_relative_install_path(file_item.relative_install_path)
    }

    mkdir_parent_dirs: Set[str] = set(
        itertools.chain.from_iterable(
            map(get_install_path_dirs, mkdir_installs)
        )
    )

    missing_dir_set = sorted((parent_dirs | mkdir_parent_dirs) - mkdir_installs)
    return set(missing_dir_set)


def print_missing_dir_set(missing_dir_set: Set[str], in_msg: str = None) -> Set[str]:
    """打印缺失目录集合。"""
    if in_msg:
        tail_msg = f' {in_msg}'
    else:
        tail_msg = ''
    for path in sorted(missing_dir_set):
        CommLog.cilog_error(f'missing dir info path "{path}"{tail_msg}')
    return missing_dir_set


def print_unsafe_paths(unsafe_paths: Tuple[str, ...]) -> Tuple[str, ...]:
    """打印非安全路径。"""
    for path in unsafe_paths:
        CommLog.cilog_error(f'unsafe path "{path}" in move scene.')
    return unsafe_paths


# 获取filelist中所有的特性集合
get_features_in_filelist = pipe(
    partial(map, attrgetter('feature')),
    chain.from_iterable,  # 展开集合序列为元素序列
    set,  # 去重
    partial(filter, partial(ne, 'comm')),  # 排除comm特性
    set,
)

# 获取filelist中所有的芯片集合
get_chips_in_filelist = pipe(
    partial(map, attrgetter('chip')),
    chain.from_iterable,  # 展开集合序列为元素序列
    set,  # 去重
)


def check_features_in_filelist(features: Set[str], filelist: FileList) -> Set[str]:
    """检查文件列表中特性配置目录规范。"""
    return invoke(
        pipe(
            # 过滤指定features的file_item
            partial(
                filter,
                pipe(attrgetter('feature'), partial(and_, features), bool)
            ),
            list,
            get_missing_dir_set,
            partial(print_missing_dir_set, in_msg=f'in features {features}'),
        ),
        filelist
    )


def check_chip_in_filelist(chip: str, filelist: FileList) -> Set[str]:
    """检查文件列表中芯片配置目录规范。"""
    return invoke(
        pipe(
            # 过滤指定chip的file_item
            partial(
                filter,
                any_(
                    pipe(
                        attrgetter('chip'), not_
                    ),  # 没有配置chip
                    pipe(
                        attrgetter('chip'), partial(swap_args(contains), chip), bool
                    ),  # 配置了指定chip
                ),
            ),
            list,
            get_missing_dir_set,
            partial(print_missing_dir_set, in_msg=f'in chip {chip}'),
        ),
        filelist
    )


check_filelist_features = any_(
    pipe(
        dispatch(
            pipe(
                get_features_in_filelist,
                # 对于每个feature，与comm组成一个set
                partial(map, lambda x: {x, 'comm'}),
                # 此时为feature集合序列
            ),
            repeat,  # 重复filelist
        ),
        tuple,
        star_apply(zip),
        # 此时为元组序列，元组的第1个元素是过滤的feature集合，第2个元素是filelist
        partial(itertools.starmap, check_features_in_filelist),
        # 此时为集合序列；合并为一个集合
        chain.from_iterable,
        set,
    ),
    pipe(
        dispatch(
            get_chips_in_filelist,
            repeat,  # 重复filelist
        ),
        tuple,
        star_apply(zip),
        # 此时为元组序列，元组的第1个元素是chip集合，第2个元素是filelist
        partial(itertools.starmap, check_chip_in_filelist),
        # 此时为集合序列；合并为一个集合
        chain.from_iterable,
        set,
    )
)


# 检查move是否安全，是否存在同一个源路径被mv多次
check_move_safe = pipe(
    partial(
        filter,
        partial(is_specific_operations, operations={'copy', 'copy_entity', 'move'}),
    ),
    partial(map, attrgetter('relative_path_in_pkg')),
    Counter,
    methodcaller('items'),
    partial(filter, pipe(itemgetter(1), partial(lt, 1))),
    partial(map, itemgetter(0)),
    tuple,
    print_unsafe_paths,
)


def check_filelist(filelist: FileList, check_features: bool, check_move: bool):
    """检查文件列表是否符合规范。"""
    if check_features:
        check_features_func = check_filelist_features
    else:
        check_features_func = constant(set())

    if check_move:
        check_move_func = check_move_safe
    else:
        check_move_func = constant(tuple())

    # 此处使用any_，短路部分报错
    check_func = any_(
        pipe(
            get_missing_dir_set,
            print_missing_dir_set,
        ),
        pipe(
            partial(filter, partial(is_specific_install_type, install_types={'run'})),
            list,
            get_missing_dir_set,
            partial(print_missing_dir_set, in_msg='in run install type'),
        ),
        check_features_func,
        check_move_func,
    )
    missing = check_func(filelist)

    if missing:
        raise FilelistError()


def get_common_path(args: List[str]) -> str:
    """公共路径前缀。"""
    try:
        return os.path.commonpath(args)
    except ValueError:
        return ''


class FileItemRelation(IntEnum):
    """文件条目之间的关系。"""
    NOT_NESTED = 0  # 不是嵌套文件
    NESTED = 1  # 嵌套文件
    SAME = 2  # 相同文件


def is_nested_file_item(item: FileItem, base_item: FileItem) -> FileItemRelation:
    """是否为嵌套的文件。"""
    if base_item is None:
        return FileItemRelation.NOT_NESTED

    if item == base_item:
        return FileItemRelation.SAME

    install_path = item.relative_install_path
    base_install_path = base_item.relative_install_path

    common_install_path = get_common_path([install_path, base_install_path])
    if common_install_path != base_install_path:
        return FileItemRelation.NOT_NESTED

    pkg_path = item.relative_path_in_pkg
    base_pkg_path = base_item.relative_path_in_pkg

    install_rel_path = os.path.relpath(install_path, base_install_path)
    pkg_rel_path = os.path.relpath(pkg_path, base_pkg_path)
    if install_rel_path != pkg_rel_path:
        # 确保打包与安装相对路径一致
        raise FilelistError(f'nested paths {item} and {base_item} are illegal.')
    return FileItemRelation.NESTED


def found_nested_file_item(item: FileItem, base_item: FileItem):
    """发现嵌套元素。"""
    raise FilelistError(f'found nested paths {item} and {base_item}!')


def convert_nested_path_in_filelist(filelist: FileList):
    """filelist中嵌套路径元素转为del。"""
    pre_item = None
    for item in filelist:
        ret = is_nested_file_item(item, pre_item)
        if ret == FileItemRelation.NESTED:
            yield item._replace(operation='del')
        elif any((
            ret == FileItemRelation.NOT_NESTED,
            (ret == FileItemRelation.SAME and not item.is_dir)
        )):
            yield item
            pre_item = item


# 检查文件列表中的嵌套路径。入参: filelist
check_nested_path_in_filelist = pipe(
    partial(filter, partial(is_specific_operations, operations={'copy', 'copy_entity'})),
    partial(sorted, key=attrgetter('relative_install_path')),
    pairwise,
    partial(
        map,
        conditional_apply(star_apply(is_nested_file_item), star_apply(found_nested_file_item))
    ),
    list,
)


# 变换文件列表中嵌套路径。入参: filelist
transform_nested_path_in_filelist = pipe(
    dispatch(
        partial(
            itertools.filterfalse, partial(is_specific_operations, operations={'copy'})
        ),
        pipe(
            partial(filter, partial(is_specific_operations, operations={'copy'})),
            partial(sorted, key=attrgetter('relative_install_path')),
            convert_nested_path_in_filelist
        ),
    ),
    chain.from_iterable,
    list,
    side_effect(check_nested_path_in_filelist),
)


def generate_filelist(filelist: FileList, filename: str, build_dir: str):
    """生成文件列表文件。"""
    content_list = list(
        itertools.chain(
            [get_filelist_header_string()],
            [file_item_to_string(item) for item in filelist]
        )
    )
    content = '\n'.join(content_list)
    filepath = os.path.join(build_dir, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
            # filelist.csv文件末尾补充一个换行符
            file.write('\n')
    except OSError as ex:
        raise GenerateFilelistError(filename) from ex


def get_transform_nested_path_func(parallel: bool) -> Callable[[FileList], FileList]:
    """获取转换嵌套路径函数。"""
    if parallel:
        return transform_nested_path_in_filelist
    return identity
