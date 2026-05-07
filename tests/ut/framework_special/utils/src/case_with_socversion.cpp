/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file case_with_socversion.cpp
 * \brief 测试用例.
 */

#include "tests/utils/case_with_socversion.h"
#include <utility>
#include "tests/utils/log.h"

using namespace ops::adv::tests::utils;

void *CaseWithSocversion::mCurrentCasePtr = nullptr;

CaseWithSocversion::CaseWithSocversion() : CaseWithSocversion("Undefined", true, "", kTilingTemplatePriority_Invalid)
{
}

CaseWithSocversion::CaseWithSocversion(const char *name, bool enable, const char *dbgInfo, int32_t tilingTemplatePriority)
    : mName(name), mEnable(enable), mDbgInfo(dbgInfo), mTilingTemplatePriority(tilingTemplatePriority)
{
}

bool CaseWithSocversion::Init(int32_t socVersionIn)
{
    bool rst = this->InitCurrentCasePtr();
    rst = rst && this->InitParam();
    rst = rst && this->InitOpInfo();
    this->socVersion = socVersionIn;
    LOG_IF(!rst, LOG_ERR("Case(%s, %s) Init failed", mName.c_str(), mDbgInfo.c_str()));
    return rst;
}

void *CaseWithSocversion::GetCurrentCase()
{
    return CaseWithSocversion::mCurrentCasePtr;
}

const char *CaseWithSocversion::GetRootPath()
{
    return mRootPath.c_str();
}

bool CaseWithSocversion::InitCurrentCasePtr()
{
    CaseWithSocversion::mCurrentCasePtr = this;
    return true;
}
