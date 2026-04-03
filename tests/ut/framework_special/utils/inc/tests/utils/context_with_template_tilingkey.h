/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file context_with_template_tilingkey.h
 * \brief 提供 [使能 templatekey 功能下] CPU模式 Tiling / Kernel 阶段上下文功能, 辅助 Tiling / Kernel 运行.
 */

#pragma once

#include <utility>
#include <iostream>
#include <fstream>
#include <tikicpulib.h>
#include "context.h"
#include "tests/utils/log.h"
#include "tests/utils/platform.h"

namespace ops::adv::tests::utils {

template<typename... Args>
class ContextWithTemplateTilingKey : public Context {
public:
    /**
     * Kernel 运行回调函数
     *
     * \param func 算子 template kernel 入口函数
     * \param tilingKey TilingKey
     * \param blockDim Tiling 切分 BlockDim
     * \param inputs 算子输入
     * \param outputs 算子输出
     * \param workspace 运行所需的 workspace 空间
     * \param tilingData TilingData 结果
     */
    typedef bool (*KernelRunTemplateCbf)(std::function<void(Args...)> func,
                                        uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                                        std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData);
    
    /**
     * 属性设置
     * 设置算子 Kernel 处理回调函数(带模板参数)
     */
    [[maybe_unused]] [[nodiscard]] bool SetKernelRunTemplateCbf(KernelRunTemplateCbf cbf) {
        if (cbf == nullptr) {
            return false;
        }
        kernelRunTemplateCbf_ = cbf;
        return true;
    }

    /**
     * 属性设置
     * 设置算子 Kernel 总入口回调函数(带模板参数)
     */
    [[maybe_unused]] [[nodiscard]] bool SetKernelTemplateMainFunc(std::function<void(Args...)> func) {
        templateKernelFunc_ = func;
        return true;
    }

protected:
    bool RunKernelProcess(std::string &caseName) override {
        if (kernelRunTemplateCbf_ == nullptr) {
            LOG_ERR("[%s:%s] Can't get kernelRunCbf_", opName_.c_str(), caseName.c_str());
            return false;
        }
        LOG_DBG("[BGN] Run %s:%s Kernel async, TilingKey=%lu, BlockDim=%ld", opName_.c_str(), caseName.c_str(), tilingKey_,
                tilingBlockDim_);

    #ifdef TESTS_UT_OPS_TEST_CI_PR // 为便于定位, 仅在PR场景进行重定向
        /**
     * 重定向标准输入输出流
     *
     *  - Model 部分通过 stdout 输出流输出日志;
     *  - Ascend C 框架部分通过 std::cerr/std::clog/std::cout 输出流输出日志;
     *
     * 此处通过重定向上述输出流到文件, 以获取输出内容并做检查, 进而感知到 Kernel 执行的异常.
     */
        std::string mod = "Model";
        std::string fmk = "Framework";
        std::string casePath = std::string(platform_->GetExeAbsPath()) + "/" + opName_ + "_" + caseName;

        /* Model 部分, 重定向 stdout 到文件 */
        auto stdoutFileHdl = dup(1); // 保存原有输出流句柄
        std::string modelFilePath = casePath + "_kernel_" + mod + ".log";
        std::ofstream modelFileHdl(modelFilePath);
        freopen(modelFilePath.c_str(), "w", stdout);

        /* Ascend C 框架部分, 重定向 std::cerr/std::clog/std::cout 到同一个文件 */
        std::string fmkFilePath = casePath + "_kernel_" + fmk + ".log";
        std::ofstream fmkFileHdl(fmkFilePath);
        std::streambuf *stdErr = std::cerr.rdbuf(fmkFileHdl.rdbuf());
        std::streambuf *stdLog = std::clog.rdbuf(fmkFileHdl.rdbuf());
        std::streambuf *stdOut = std::cout.rdbuf(fmkFileHdl.rdbuf());
    #endif
        /* 调用回调函数, 触发具体算子 Kernel 执行 */
        ICPU_SET_TILING_KEY(tilingKey_);    
        bool ret = kernelRunTemplateCbf_(templateKernelFunc_, tilingKey_, tilingBlockDim_, inputs_, outputs_, workspacePtr_,
                                tilingData_.data());

    #ifdef TESTS_UT_OPS_TEST_CI_PR // 为便于定位, 仅在PR场景进行重定向
        /* 恢复重定向 */
        std::cout.rdbuf(stdOut);
        std::clog.rdbuf(stdLog);
        std::cerr.rdbuf(stdErr);
        fmkFileHdl.close();
        modelFileHdl.close();
        dup2(stdoutFileHdl, 1);

        /* 执行日志结果获取与结果校验 */
        ret = ret && this->CheckKernelResult(ret, caseName, modelFilePath, mod.c_str(), CheckModelKernelResultStr, false);
        ret = ret && this->CheckKernelResult(ret, caseName, fmkFilePath, fmk.c_str(), CheckFrameworkKernelResultStr, false);
    #endif
        return ret;
    }

private:
    void Destroy() {
        if (workspacePtr_ != nullptr) {
            this->FreeWorkspaceImpl(workspacePtr_);
            workspacePtr_ = nullptr;
        }
        workspaceSize_ = 0;
    }

protected:
    KernelRunTemplateCbf kernelRunTemplateCbf_ = nullptr;
    std::function<void(Args...)> templateKernelFunc_;
};
}