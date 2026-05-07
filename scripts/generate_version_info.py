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

"""生成version.info文件。"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator


def gen_version_info_content(version: str, deps: Iterable[str]) -> Iterator[str]:
    """生成version.info文件内容。"""
    yield f'Version={version}'
    yield 'version_dir=cann'
    deps_iter = iter(deps)
    for dep_pkg in deps_iter:
        dep_info = next(deps_iter)
        yield f'required_package_{dep_pkg}_version="{dep_info}"'
    if os.environ.get('tagInfo'):
        tag_info = os.environ.get('tagInfo')
        timestamp = '_'.join(tag_info.split('_')[-3:-1])
    else:
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S%f')[:-3]
    yield f'timestamp={timestamp}'
    yield ''  # for last \n


def main():
    """主流程。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('version', help='Version number.')
    parser.add_argument('deps', nargs='*', help='Dependency informations.')
    parser.add_argument('--output', required=True, help='Output file path.')
    args = parser.parse_args()

    logging.basicConfig(format=f'{os.path.basename(__file__)}: %(levelname)s: %(message)s')

    if len(args.deps) % 2 != 0:
        logging.error('The deps argument must contain an even number of elements!')
        return False

    content = '\n'.join(gen_version_info_content(args.version, args.deps))
    with open(args.output, 'w', encoding='utf-8') as file:
        file.write(content)

    return True


if __name__ == '__main__':
    if not main():
        sys.exit(1)