/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file op_info.cpp
 * \brief 测试用例算子信息.
 */

#include "tests/utils/op_info.h"
#include "tests/utils/platform.h"
#include "tests/utils/log.h"

using namespace ops::adv::tests::utils;

OpInfo::OpInfo() : OpInfo(ControlInfo(true, false))
{
}

OpInfo::OpInfo(const ControlInfo &ctr) : OpInfo(ctr, ExpectInfo())
{
}

OpInfo::OpInfo(const ControlInfo &ctr, const ExpectInfo &exp) : OpInfo("Undefined", ctr, exp)
{
}

OpInfo::OpInfo(const char *name, const ControlInfo &ctr, const ExpectInfo &exp)
    : mName(name), mCtr(ctr), mExp(exp), mCtx(nullptr)
{
}

bool OpInfo::SetContext(ContextIntf *ctxParam)
{
    this->mCtx = static_cast<ContextIntf *>(ctxParam);
    return true;
}

bool OpInfo::ProcessTiling(std::string &caseName)
{
    if (mCtr.mRunTiling) {
        if (!this->RunTiling(caseName)) {
            return false;
        }
        if (!this->ChkTiling(caseName)) {
            return false;
        }
    }
    return true;
}

bool OpInfo::ProcessKernel(std::string &caseName)
{
    if (mCtr.mRunKernel) {
        if (!this->RunKernel(caseName)) {
            return false;
        }
        if (!this->ChkKernel(caseName)) {
            return false;
        }
    }
    return true;
}

bool OpInfo::RunTiling(std::string &caseName)
{
    if (!mCtx->RunTiling(caseName, false)) {
        LOG_IF(mExp.mSuccess, LOG_ERR("Case[%s:%s] Run Tiling Failed.", caseName.c_str(), mName.c_str()));
        return false;
    }
    return true;
}

bool OpInfo::ChkTiling(std::string &caseName)
{
    if (mExp.mTilingKey != ExpectInfo::kInvalidTilingKey) {
        auto actTilingKey = mCtx->GetTilingKey();
        if (mExp.mTilingKey != actTilingKey) {
            LOG_IF(mExp.mSuccess, LOG_ERR("Case[%s:%s] Check Tiling result failed(TilingKey), Exp=%lu, Act=%lu",
                                          caseName.c_str(), mName.c_str(), mExp.mTilingKey, actTilingKey));
            return false;
        }
    }
    if (mExp.mTilingBlockDim != ExpectInfo::kInvalidTilingBlockDim) {
        int64_t expTilingBlockDim = mExp.mTilingBlockDim;
        if (expTilingBlockDim == ExpectInfo::kFullTilingBlockDim) {
            auto *platform = Platform::GetGlobalPlatform();
            expTilingBlockDim = platform != nullptr ? platform->GetBlockDim() : expTilingBlockDim;
        }
        auto actTilingBlockDim = mCtx->GetTilingBlockDim();
        if (expTilingBlockDim != actTilingBlockDim) {
            LOG_IF(mExp.mSuccess, LOG_ERR("Case[%s:%s] Check Tiling result failed(TilingBlockDim), Exp=%ld, Act=%ld",
                                          caseName.c_str(), mName.c_str(), expTilingBlockDim, actTilingBlockDim));
            return false;
        }
    }
    return true;
}

bool OpInfo::RunKernel(std::string &caseName)
{
    if (!mCtx->RunKernel(caseName)) {
        LOG_IF(mExp.mSuccess, LOG_ERR("Case[%s:%s] Run Kernel Failed.", caseName.c_str(), mName.c_str()));
        return false;
    }
    return true;
}

bool OpInfo::ChkKernel(std::string &caseName)
{
    LOG_DBG("Case[%s:%s] Run Kernel Finish.", caseName.c_str(), mName.c_str());
    return true;
}
