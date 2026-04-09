#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
import os
import sys
import shutil
import logging
import json

def to_camel_case(name: str) -> str:
    parts = name.split('_')
    return ''.join(p.capitalize() for p in parts if p)

def main():
    if len(sys.argv) != 4:
        logging.error("Usage: python clean_opapi_stub.py <op_name_list> <base_path> <chip_name>")
        sys.exit(1)

    op_names_raw = sys.argv[1]
    base_path = sys.argv[2]
    chip_name = sys.argv[3]

    op_names = [op.strip() for op in op_names_raw.split(",") if op.strip()]
    if not op_names:
        logging.info("No valid op names provided.")
        sys.exit(0)

    config_path = os.path.join(base_path, "opp", "built-in", "op_impl", "ai_core", "tbe", "kernel", "config", chip_name, "ops_legacy", "binary_info_config.json")
    if not os.path.exists(config_path):
        logging.error("binary_info_config.json not found at %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON file: %s", config_path)
            return

    ascend910b_dir = os.path.join(base_path, "opp", "built-in", "op_impl", "ai_core", "tbe", "kernel", chip_name, "ops_legacy")
    if not os.path.exists(ascend910b_dir):
        logging.warning("Base directory not found: %s", ascend910b_dir)
        sys.exit(0)

    modified = False
    for op in op_names:
        camel_op = to_camel_case(op)
        op_info = data.get(camel_op)
        if not op_info:
            continue
        binary_list = op_info.get("binaryList", [])
        if any("opapi_stub" in str(item) for item in binary_list):
            logging.info("Removing auto-generated stub for : %s", op)
            del data[camel_op]
            modified = True

    if modified:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info("binary_info_config.json stub cleanup completed.")

        op_dir = os.path.join(ascend910b_dir, op)
        stub_file = os.path.join(op_dir, "opapi_ut.stub")

        if os.path.isdir(op_dir) and os.path.isfile(stub_file):
            logging.info("Removing directory: %s", op_dir)
            shutil.rmtree(op_dir)
        else:
            logging.info("No stub found or directory missing for op: %s, skip", op)

    lib_paths = [
        os.path.join(base_path, "opp", "built-in", "op_impl", "ai_core", "tbe", "op_tiling", "lib", "linux", "x86_64"),
        os.path.join(base_path, "opp", "built-in", "op_impl", "ai_core", "tbe", "op_tiling", "lib", "linux", "aarch64"),
    ]

    for lib_dir in lib_paths:
        lib_file = os.path.join(lib_dir, "libopmaster_rt2.0.so")
        if os.path.islink(lib_file):
            symlink = os.readlink(lib_file)
            if symlink == "libopmaster_rt2.0.so":
                os.unlink(lib_file)
                logging.info("Deleted symlink:%s", symlink)

    logging.info("Cleanup finished.")

if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s][%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    main()
