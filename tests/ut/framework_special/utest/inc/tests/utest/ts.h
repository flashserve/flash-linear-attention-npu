/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file ts.h
 * \brief UTest 基类定义, 用于辅助算子实现对应 TestSuite.
 */

#pragma once

#include <gtest/gtest.h>
#include "tests/utils/log.h"
#include "tests/utils/case.h"
#include "tests/utils/tensor.h"
#include "tests/utils/op_info.h"
#include "tests/utils/platform.h"

#ifdef TESTS_UT_OPS_TEST_CI_PR
/**
 * 用于控制个别用例在 PR 门禁场景不执行 Kernel
 */
constexpr bool RunKernelNotInPr = false;
#else
constexpr bool RunKernelNotInPr = true;
#endif

using Case = ops::adv::tests::utils::Case;
using Tensor = ops::adv::tests::utils::Tensor;
using OpInfo = ops::adv::tests::utils::OpInfo;
using ControlInfo = ops::adv::tests::utils::ControlInfo;
using ExpectInfo = ops::adv::tests::utils::ExpectInfo;
using Platform = ops::adv::tests::utils::Platform;

/**
 * 基础 TestSuite
 */
template <class C> class Ts : public testing::Test {
protected:
    C *case_ = nullptr;
    Platform *platform_ = nullptr;
    Platform::SocVersion socVersion_ = Platform::SocVersion::Ascend910B2;

protected:
    void SetUp() override
    {
        if (case_ == nullptr) {
            case_ = new C();
            ASSERT_NE(case_, nullptr);
        }
        platform_ = Platform::GetGlobalPlatform();
        ASSERT_NE(platform_, nullptr);
        platform_->SetSocVersion(socVersion_);
    }

    void TearDown() override
    {
        ASSERT_NE(case_, nullptr);
        delete case_;
        case_ = nullptr;
        platform_ = nullptr;
        socVersion_ = Platform::SocVersion::Ascend910B2;
        // ASSERT_TRUE(ops::adv::tests::utils::ChkLogErrCnt());
    }

    [[maybe_unused]] [[nodiscard]] int64_t GetCoreNum() const
    {
        return platform_->GetCoreNum();
    }
};

template <class C> class Ts_Ascend910B1 : public Ts<C> {
protected:
    void SetUp() override
    {
        Ts<C>::socVersion_ = Platform::SocVersion::Ascend910B1;
        Ts<C>::SetUp();
    }
};

template <class C> class Ts_Ascend910B2 : public Ts<C> {
protected:
    void SetUp() override
    {
        Ts<C>::socVersion_ = Platform::SocVersion::Ascend910B2;
        Ts<C>::SetUp();
    }
};

template <class C> class Ts_Ascend910B3 : public Ts<C> {
protected:
    void SetUp() override
    {
        Ts<C>::socVersion_ = Platform::SocVersion::Ascend910B3;
        Ts<C>::SetUp();
    }
};

template <class C> class Ts_Ascend310P3 : public Ts<C> {
protected:
    void SetUp() override
    {
        Ts<C>::socVersion_ = Platform::SocVersion::Ascend310P3;
        Ts<C>::SetUp();
    }
};

template <class C> class Ts_Ascend910_9591 : public Ts<C> {
protected:
    void SetUp() override
    {
        Ts<C>::socVersion_ = Platform::SocVersion::Ascend910_9591;
        Ts<C>::SetUp();
    }
};


/**
 * 支持 TEST_P 类型用例的 TestSuite
 */
template <class C> class Ts_WithParam : public Ts<C>, public ::testing::WithParamInterface<C> {
protected:
    void SetUp() override
    {
        auto &p = ::testing::WithParamInterface<C>::GetParam();
        Ts<C>::case_ = new C(p);
        ASSERT_NE(Ts<C>::case_, nullptr);
        Ts<C>::SetUp();
    }
};

template <class C> class Ts_WithParam_Ascend910B1 : public Ts_WithParam<C> {
protected:
    void SetUp() override
    {
        Ts_WithParam<C>::socVersion_ = Platform::SocVersion::Ascend910B1;
        Ts_WithParam<C>::SetUp();
    }
};

template <class C> class Ts_WithParam_Ascend910B2 : public Ts_WithParam<C> {
protected:
    void SetUp() override
    {
        Ts_WithParam<C>::socVersion_ = Platform::SocVersion::Ascend910B2;
        Ts_WithParam<C>::SetUp();
    }
};

template <class C> class Ts_WithParam_Ascend910B3 : public Ts_WithParam<C> {
protected:
    void SetUp() override
    {
        Ts_WithParam<C>::socVersion_ = Platform::SocVersion::Ascend910B3;
        Ts_WithParam<C>::SetUp();
    }
};


template <class C> class Ts_WithParam_Ascend310P3 : public Ts_WithParam<C> {
protected:
    void SetUp() override
    {
        Ts_WithParam<C>::socVersion_ = Platform::SocVersion::Ascend310P3;
        Ts_WithParam<C>::SetUp();
    }
};

template <class C> class Ts_WithParam_Ascend910_9591 : public Ts_WithParam<C> {
    protected:
        void SetUp() override
        {
            Ts_WithParam<C>::socVersion_ = Platform::SocVersion::Ascend910_9591;
            Ts_WithParam<C>::SetUp();
        }
    };