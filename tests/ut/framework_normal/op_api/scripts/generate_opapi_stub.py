#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import sys
import json
import logging


def to_camel_case(name: str) -> str:
    parts = name.split('_')
    return ''.join(p.capitalize() for p in parts if p)

def main():
    if len(sys.argv) != 5:
        logging.error("Usage: python generate_opapi_stub.py <op_name_list> <base_path> <chip_name> <runtime_stub_path>")
        sys.exit(1)

    op_names_raw = sys.argv[1]
    base_path = sys.argv[2]
    chip_name = sys.argv[3]
    runtime_stub_path = sys.argv[4]

    op_names = [name.strip() for name in op_names_raw.split(',') if name.strip()]
    if not op_names:
        logging.error("no valid op names provided.")
        sys.exit(1)

    config_path = os.path.join(base_path, "opp", "built-in", "op_impl", "ai_core", "tbe", "kernel", "config", chip_name, "ops_legacy", "binary_info_config.json")
    if not os.path.exists(config_path):
        logging.error("binary_info_config.json not found at %s", config_path)
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Error reading JSON: {e}")
        sys.exit(1)

    kernel_base_dir = os.path.join(base_path, "opp", "built-in", "op_impl", "ai_core", "tbe", "kernel", chip_name, "ops_legacy")

    # 生成打桩.o .json
    for op in op_names:
        camel_op = to_camel_case(op)
        logging.info("Processing op: %s (CamelCase: %s)", op, camel_op)

        if camel_op not in data:
            logging.info("key [%s] not found in JSON, create.", camel_op)
            new_entry = {
                "dynamicRankSupport": True,
                "simplifiedKeyMode": 0,
                "binaryList": [
                    {
                        "coreType": 2,
                        "simplifiedKey": [
                            f"{camel_op}/d=0,p=0/1"
                        ],
                        "binPath": f"{chip_name}/{op}/{op}_opapi_stub.o",
                        "jsonPath": f"{chip_name}/{op}/{op}_opapi_stub.json"
                    }
                ]
            }
            data[camel_op] = new_entry
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
        op_info = data[camel_op]
        binary_list = op_info.get("binaryList", [])
        if not binary_list:
            logging.warning("[%s] has no binaryList, skipping.", camel_op)
            continue

        first_item = binary_list[0]
        bin_path = first_item.get("binPath")
        json_path = first_item.get("jsonPath")

        if not bin_path or not json_path:
            logging.warning("[%s] binaryList missing binPath/jsonPath, skipping.", camel_op)
            continue

        target_dir = os.path.join(kernel_base_dir, op)
        if os.path.exists(target_dir):
            continue

        os.makedirs(target_dir, exist_ok=True)
        bin_filename = os.path.basename(bin_path)
        json_filename = os.path.basename(json_path)

        open(os.path.join(target_dir, bin_filename), 'w').close()
        open(os.path.join(target_dir, json_filename), 'w').close()
        open(os.path.join(target_dir, "opapi_ut.stub"), 'w').close()

        logging.info("Created directory: %s", target_dir)
        logging.info("  - %s", bin_filename)
        logging.info("  - %s", json_filename)
        logging.info("  - opapi_ut.stub")
    
    # 生成打桩libopmaster_rt2.0.so
    lib_paths = [
        os.path.join(base_path, "opp", "built-in", "op_impl", "ai_core", "tbe", "op_tiling", "lib", "linux", "x86_64"),
        os.path.join(base_path, "opp", "built-in", "op_impl", "ai_core", "tbe", "op_tiling", "lib", "linux", "aarch64"),
    ]

    for lib_dir in lib_paths:
        os.makedirs(lib_dir, exist_ok=True)
        lib_file = os.path.join(lib_dir, "libopmaster_rt2.0.so")
        if os.path.lexists(lib_file):
            logging.info("%s already exists, skipping creation.", lib_file)
        else:
            symlink_path = lib_file
            if not os.path.lexists(symlink_path):
                os.symlink("libopmaster_rt2.0.so", symlink_path)
                logging.info("Created symlink:%s", symlink_path)

    # 生成runtime_stubs.cpp
    runtime_stub_path = os.path.join(runtime_stub_path)

    # 其余固定函数体内容（除了 rtGetSocVersion）
    other_stubs = r"""
    rtError_t rtGetDeviceInfo(uint32_t device_id, int32_t module_type, int32_t info_type, int64_t *val)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtCtxGetDevice(int32_t *device)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtKernelLaunchEx([[maybe_unused]] void *args, [[maybe_unused]] uint32_t argsSize,
                            [[maybe_unused]] uint32_t flags, [[maybe_unused]] rtStream_t stm)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **handle)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtDevBinaryUnRegister(void *handle)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey, uint32_t blockDim, rtArgsEx_t *argsInfo,
                                        rtSmDesc_t *smDesc, rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtKernelLaunchWithFlagV2(const void *stubFunc, uint32_t blockDim, rtArgsEx_t *argsInfo, rtSmDesc_t *smDesc,
                                    rtStream_t stm, uint32_t flags, const rtTaskCfgInfo_t *cfgInfo)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtLaunchKernelByFuncHandleV2(rtFuncHandle funcHandle, uint32_t blockDim, rtLaunchArgsHandle argsHandle,
                                        rtStream_t stm, const rtTaskCfgInfo_t *cfgInfo)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtGetTaskIdAndStreamID(uint32_t *taskId, uint32_t *streamId)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtGetDevice(int32_t *device)
    {
        *device = 0;
        return RT_ERROR_NONE;
    }

    rtError_t rtMemGetInfoEx(rtMemInfoType_t memInfoType, size_t *freeSize, size_t *totalSize)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char_t *stubName, const void *kernelInfoExt,
                                uint32_t funcMode)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtGetFunctionByName(const char_t *stubName, void **stubFunc)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtAiCoreMemorySizes(rtAiCoreMemorySize_t *aiCoreMemorySize)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtAicpuKernelLaunchExWithArgs(uint32_t kernelType, const char *opName, uint32_t blockDim,
                                            const rtAicpuArgsEx_t *argsInfo, rtSmDesc_t *smDesc, rtStream_t stream,
                                            uint32_t flags)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtStreamSynchronize(rtStream_t stream)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtStreamSynchronizeWithTimeout(rtStream_t stm, int32_t timeout)
    {
        return RT_ERROR_NONE;
    }
    int64_t gDeterministic = 0;
    rtFloatOverflowMode_t gOverflow = RT_OVERFLOW_MODE_SATURATION;

    rtError_t rtCtxSetSysParamOpt(const rtSysParamOpt configOpt, const int64_t configVal)
    {
        if (configOpt == SYS_OPT_DETERMINISTIC) {
            gDeterministic = configVal;
        }
        return RT_ERROR_NONE;
    }

    rtError_t rtCtxGetSysParamOpt(const rtSysParamOpt configOpt, int64_t *const configVal)
    {
        if (configOpt == SYS_OPT_DETERMINISTIC) {
            *configVal = gDeterministic;
        }
        return RT_ERROR_NONE;
    }

    rtError_t rtGetDeviceSatMode(rtFloatOverflowMode_t *floatOverflowMode)
    {
        *floatOverflowMode = gOverflow;
        return RT_ERROR_NONE;
    }

    rtError_t rtCtxGetOverflowAddr(void **overflowAddr)
    {
        *overflowAddr = reinterpret_cast<void *>(0x005);
        return RT_ERROR_NONE;
    }

    rtError_t rtSetDeviceSatMode(rtFloatOverflowMode_t floatOverflowMode)
    {
        gOverflow = floatOverflowMode;
        return RT_ERROR_NONE;
    }

    rtError_t rtMalloc(void **devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
    {
        *devPtr = new uint8_t[size];
        memset_s(*devPtr, size, 0, size);
        return RT_ERROR_NONE;
    }

    rtError_t rtFree(void *devptr)
    {
        delete[] (uint8_t *)devptr;
        return RT_ERROR_NONE;
    }

    rtError_t rtMemcpyAsync(void *dst, uint64_t destMax, const void *src, uint64_t count, rtMemcpyKind_t kind,
                            rtStream_t stream)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtCalcLaunchArgsSize(size_t argsSize, size_t hostInfoTotalSize, size_t hostInfoNum, size_t *launchArgsSize)
    {
        *launchArgsSize = argsSize + hostInfoTotalSize;
        return RT_ERROR_NONE;
    }

    rtError_t rtCreateLaunchArgs(size_t argsSize, size_t hostInfoTotalSize, size_t hostInfoNum, void *argsData,
                                rtLaunchArgsHandle *argsHandle)
    {
        static size_t hdlData = 0;
        static rtLaunchArgsHandle hdl = static_cast<void *>(&hdlData);
        *argsHandle = hdl;
        return RT_ERROR_NONE;
    }


    rtError_t rtDestroyLaunchArgs(rtLaunchArgsHandle argsHandle)
    {
        return RT_ERROR_NONE;
    }


    rtError_t rtAppendLaunchAddrInfo(rtLaunchArgsHandle argsHandle, void *addrInfo)
    {
        return RT_ERROR_NONE;
    }


    rtError_t rtAppendLaunchHostInfo(rtLaunchArgsHandle argsHandle, size_t hostInfoSize, void **hostInfo)
    {
        return RT_ERROR_NONE;
    }


    rtError_t rtBinaryLoad(const rtDevBinary_t *bin, rtBinHandle *binHandle)
    {
        return RT_ERROR_NONE;
    }


    rtError_t rtBinaryGetFunction(const rtBinHandle binHandle, const uint64_t tilingKey, rtFuncHandle *funcHandle)
    {
        return RT_ERROR_NONE;
    }


    rtError_t rtLaunchKernelByFuncHandle(rtFuncHandle funcHandle, uint32_t blockDim, rtLaunchArgsHandle argsHandle,
                                        rtStream_t stm)
    {
        return RT_ERROR_NONE;
    }


    rtError_t rtSetExceptionExtInfo(const rtArgsSizeInfo_t *const sizeInfo)
    {
        return RT_ERROR_NONE;
    }


    rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags)
    {
        return RT_ERROR_NONE;
    }


    rtError_t rtStreamDestroy(rtStream_t stream)
    {
        return RT_ERROR_NONE;
    }


    rtError_t rtEventCreateWithFlag(rtEvent_t *event, uint32_t flag)
    {
        return RT_ERROR_NONE;
    }


    rtError_t rtEventDestroy(rtEvent_t event)
    {
        return RT_ERROR_NONE;
    }


    rtError_t rtEventRecord(rtEvent_t event, rtStream_t stream)
    {
        return RT_ERROR_NONE;
    }


    rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event)
    {
        return RT_ERROR_NONE;
    }


    rtError_t rtEventReset(rtEvent_t event, rtStream_t stream)
    {
        return RT_ERROR_NONE;
    }

    static uint64_t floatDebugStatus = 0;

    rtError_t rtNpuGetFloatDebugStatus(void *outputAddrPtr, uint64_t outputSize, uint32_t checkMode, rtStream_t stm)
    {
        uint64_t *status = static_cast<uint64_t *>(outputAddrPtr);
        floatDebugStatus = 1;
        *status = floatDebugStatus;
        return RT_ERROR_NONE;
    }


    rtError_t rtNpuClearFloatDebugStatus(uint32_t checkMode, rtStream_t stm)
    {
        floatDebugStatus = 0;
        return RT_ERROR_NONE;
    }


    rtError_t rtCtxGetCurrent(rtContext_t *ctx)
    {
        int64_t x = 1;
        *ctx = (void *)x;
        return RT_ERROR_NONE;
    }

    rtError_t rtBinaryLoadWithoutTilingKey(const void *data, const uint64_t length, rtBinHandle *binHandle)
    {
        return RT_ERROR_NONE;
    }

    rtError_t rtBinaryGetFunctionByName(const rtBinHandle binHandle, const char *kernelName, rtFuncHandle *funcHandle)
    {
        return RT_ERROR_NONE;
    }
    """

    runtime_stub_content = f'''/**
    * This program is free software, you can redistribute it and/or modify.
    * Copyright (c) 2025 Huawei Technologies Co., Ltd.
    * This file is a part of the CANN Open Software.
    * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
    * Please refer to the License for details. You may not use this file except in compliance with the License.
    * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
    * See LICENSE in the root of the software repository for the full text of the License.
    */

    /*!
    * \\file runtime_stubs.cpp
    * \\brief
    */

    #include <securec.h>
    #include <cstdint>
    #include <map>
    #include <any>
    #include <string>
    #include <dlfcn.h>
    #include <runtime/rt.h>
    #include <runtime/base.h>

    #define CHIP_NAME "{chip_name}"

    extern "C" {{
    rtError_t rtGetSocVersion(char *version, const uint32_t maxLen)
    {{
        std::string ver = "";
        if (CHIP_NAME == std::string("ascend910_93")) {{
            ver = "Ascend910_9391";
        }} else if (CHIP_NAME == std::string("ascend910_95")) {{
            ver = "Ascend910_9599";
        }} else {{
            ver = "Ascend910B1";
        }}
        (void)strncpy_s(version, maxLen, ver.c_str(), ver.length());
        return RT_ERROR_NONE;
    }}
    {other_stubs}
    }} // extern "C"
    '''

    # 写入文件
    os.makedirs(os.path.dirname(runtime_stub_path), exist_ok=True)
    with open(runtime_stub_path, "w") as f:
        f.write(runtime_stub_content.replace("{other_stubs}", other_stubs))

    logging.info("Generated runtime_stubs.cpp at: %s", runtime_stub_path)


    logging.info("All done.")

if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s][%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    main()