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
import urllib.request
import os
import subprocess


def down_files_native(url_list):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for url in url_list:
        # 判断是否为 Git 仓库 URL
        if url.endswith('.git'):
            # 提取仓库名（去掉 .git 后缀，取 URL 最后一部分）
            repo_name = url.rstrip('/').split('/')[-1]
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]  # 去掉 .git
            repo_path = os.path.join(current_dir, repo_name)

            if os.path.exists(repo_path):
                print(f"目录 {repo_path} 已存在，跳过克隆。")
                continue

            try:
                subprocess.run(['git', 'clone', url, repo_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"克隆失败: {e}")
            except FileNotFoundError:
                print("git 命令未找到，请安装 git。")
        else:
            # 普通文件下载
            file_name = url.split('/')[-1]
            if not file_name:
                file_name = "downloaded_file"
            file_path = os.path.join(current_dir, file_name)

            print(f"正在下载 {url} 到 {file_path}")
            urllib.request.urlretrieve(url, file_path)

if __name__ == "__main__":
    my_urls = [
        "https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz",
        "https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip",
        ("https://gitcode.com/cann-src-third-party/makeself/releases/download/"
         "release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz"),
        "https://gitcode.com/cann-src-third-party/pybind11/releases/download/v2.13.6/pybind11-2.13.6.tar.gz",
        "https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0-h0.trunk/eigen-5.0.0.tar.gz",
        "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz",
        ("https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/"
         "20230802.1/abseil-cpp-20230802.1.tar.gz"),
        "https://gitcode.com/cann/opbase.git"   # Git 仓库
    ]

    down_files_native(my_urls)