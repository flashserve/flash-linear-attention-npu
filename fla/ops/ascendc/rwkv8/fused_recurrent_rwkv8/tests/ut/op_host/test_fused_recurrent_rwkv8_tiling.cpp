/*!
 * \file test_fused_recurrent_rwkv8_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "../../../op_host/fused_recurrent_rwkv8_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;
using namespace optiling;
using namespace FusedRecurrentRwkv8;

class FusedRecurrentRwkv8TilingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedRecurrentRwkv8TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedRecurrentRwkv8TilingTest TearDown" << std::endl;
    }
};

TEST_F(FusedRecurrentRwkv8TilingTest, tiling_with_initial_state)
{
    optiling::FusedRecurrentRwkv8CompileInfo compileinfo = {40, 196608}; // aivNum, ubSize

    int b = 2;
    int t = 64;
    int h = 4;
    int n = 64;

    gert::StorageShape ioShape = {{b, h, t, n}, {b, h, t, n}};
    gert::StorageShape stateShape = {{b, h, n, n}, {b, h, n, n}};

    gert::TilingContextPara tilingContextPara("FusedRecurrentRwkv8",
        {
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},      // q
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},      // w
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},      // k
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},      // v
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},      // z
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},      // b
            {stateShape, ge::DT_FLOAT, ge::FORMAT_ND},   // initial_state
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // o
        },
        {
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.5)},
        },
        &compileinfo
    );

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0UL);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(FusedRecurrentRwkv8TilingData));

    auto *tilingData = reinterpret_cast<const FusedRecurrentRwkv8TilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->B, 2U);
    EXPECT_EQ(tilingData->T, 64U);
    EXPECT_EQ(tilingData->H, 4U);
    EXPECT_EQ(tilingData->K, 64U);
    EXPECT_EQ(tilingData->V, 64U);
    EXPECT_FLOAT_EQ(tilingData->scale, 0.5f);
    EXPECT_EQ(tilingData->hasInitialState, 1U);
}

TEST_F(FusedRecurrentRwkv8TilingTest, tiling_without_initial_state)
{
    optiling::FusedRecurrentRwkv8CompileInfo compileinfo = {40, 196608};

    int b = 1;
    int t = 33;
    int h = 1;
    int n = 64;

    gert::StorageShape ioShape = {{b, h, t, n}, {b, h, t, n}};

    gert::TilingContextPara tilingContextPara("FusedRecurrentRwkv8",
        {
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
        },
        &compileinfo
    );

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0UL);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(FusedRecurrentRwkv8TilingData));

    auto *tilingData = reinterpret_cast<const FusedRecurrentRwkv8TilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->T, 33U);
    EXPECT_EQ(tilingData->hasInitialState, 0U);
}

// bf16 io：tiling 正常通过（io dtype 只走编译期宏，tiling 侧仅校验合法性）
TEST_F(FusedRecurrentRwkv8TilingTest, tiling_bf16_io)
{
    optiling::FusedRecurrentRwkv8CompileInfo compileinfo = {40, 196608};

    int b = 2;
    int t = 64;
    int h = 4;
    int n = 64;

    gert::StorageShape ioShape = {{b, h, t, n}, {b, h, t, n}};
    gert::StorageShape stateShape = {{b, h, n, n}, {b, h, n, n}};

    gert::TilingContextPara tilingContextPara("FusedRecurrentRwkv8",
        {
            {ioShape, ge::DT_BF16, ge::FORMAT_ND},       // q
            {ioShape, ge::DT_BF16, ge::FORMAT_ND},       // w
            {ioShape, ge::DT_BF16, ge::FORMAT_ND},       // k
            {ioShape, ge::DT_BF16, ge::FORMAT_ND},       // v
            {ioShape, ge::DT_BF16, ge::FORMAT_ND},       // z
            {ioShape, ge::DT_BF16, ge::FORMAT_ND},       // b
            {stateShape, ge::DT_FLOAT, ge::FORMAT_ND},   // initial_state 恒 fp32
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},      // o 跟随 q
        },
        {
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.5)},
        },
        &compileinfo
    );

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0UL);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(FusedRecurrentRwkv8TilingData));

    auto *tilingData = reinterpret_cast<const FusedRecurrentRwkv8TilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->B, 2U);
    EXPECT_EQ(tilingData->T, 64U);
    EXPECT_EQ(tilingData->H, 4U);
    EXPECT_EQ(tilingData->K, 64U);
    EXPECT_EQ(tilingData->V, 64U);
    EXPECT_FLOAT_EQ(tilingData->scale, 0.5f);
    EXPECT_EQ(tilingData->hasInitialState, 1U);
}

// 训练预埋：output_chunk_state/output_sa/reverse 三个 bool attr 正确落 TilingData
TEST_F(FusedRecurrentRwkv8TilingTest, tiling_flag_attrs)
{
    optiling::FusedRecurrentRwkv8CompileInfo compileinfo = {40, 196608};

    int b = 2;
    int t = 64;
    int h = 4;
    int n = 64;

    gert::StorageShape ioShape = {{b, h, t, n}, {b, h, t, n}};
    gert::StorageShape stateShape = {{b, h, n, n}, {b, h, n, n}};

    gert::TilingContextPara tilingContextPara("FusedRecurrentRwkv8",
        {
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {stateShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // o
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // s
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // sa
        },
        {
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.5)},
            {"output_chunk_state", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_sa", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"reverse", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        },
        &compileinfo
    );

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0UL);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(FusedRecurrentRwkv8TilingData));

    auto *tilingData = reinterpret_cast<const FusedRecurrentRwkv8TilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->reverse, 1U);
    EXPECT_EQ(tilingData->outputChunkState, 1U);
    EXPECT_EQ(tilingData->outputSa, 1U);
    EXPECT_EQ(tilingData->chunkLen, 16U);   // 未带 chunk_len attr → 缺省 16
}

// 缺省（不带三个新 attr）→ 三 flag 全 0
TEST_F(FusedRecurrentRwkv8TilingTest, tiling_flag_attrs_default_off)
{
    optiling::FusedRecurrentRwkv8CompileInfo compileinfo = {40, 196608};

    int b = 1;
    int t = 16;
    int h = 2;
    int n = 64;

    gert::StorageShape ioShape = {{b, h, t, n}, {b, h, t, n}};

    gert::TilingContextPara tilingContextPara("FusedRecurrentRwkv8",
        {
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // o
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // s
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // sa
        },
        {
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
        },
        &compileinfo
    );

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    auto *tilingData = reinterpret_cast<const FusedRecurrentRwkv8TilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->reverse, 0U);
    EXPECT_EQ(tilingData->outputChunkState, 0U);
    EXPECT_EQ(tilingData->outputSa, 0U);
    EXPECT_EQ(tilingData->chunkLen, 16U);   // 缺省回落 16
}

// K≠V：q/w/k/z/b (B,H,T,K)、v (B,H,T,V)，K/V 分别落 TilingData
TEST_F(FusedRecurrentRwkv8TilingTest, tiling_k_ne_v)
{
    optiling::FusedRecurrentRwkv8CompileInfo compileinfo = {40, 196608};

    int b = 2;
    int t = 64;
    int h = 4;
    int kDim = 64;
    int vDim = 32;

    gert::StorageShape kShape = {{b, h, t, kDim}, {b, h, t, kDim}};
    gert::StorageShape vShape = {{b, h, t, vDim}, {b, h, t, vDim}};
    gert::StorageShape stateShape = {{b, h, kDim, vDim}, {b, h, kDim, vDim}};   // initial_state 接口朝向 (B,H,K,V)

    gert::TilingContextPara tilingContextPara("FusedRecurrentRwkv8",
        {
            {kShape, ge::DT_FLOAT, ge::FORMAT_ND},       // q
            {kShape, ge::DT_FLOAT, ge::FORMAT_ND},       // w
            {kShape, ge::DT_FLOAT, ge::FORMAT_ND},       // k
            {vShape, ge::DT_FLOAT, ge::FORMAT_ND},       // v (B,H,T,V)
            {kShape, ge::DT_FLOAT, ge::FORMAT_ND},       // z
            {kShape, ge::DT_FLOAT, ge::FORMAT_ND},       // b
            {stateShape, ge::DT_FLOAT, ge::FORMAT_ND},   // initial_state
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // o (B,H,T,V)
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // s
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // sa
        },
        {
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.5)},
        },
        &compileinfo
    );

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 0UL);
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(FusedRecurrentRwkv8TilingData));

    auto *tilingData = reinterpret_cast<const FusedRecurrentRwkv8TilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->K, 64U);
    EXPECT_EQ(tilingData->V, 32U);
    EXPECT_EQ(tilingData->B, 2U);
    EXPECT_EQ(tilingData->T, 64U);
}

// chunk_len attr：显式 8 → TilingData.chunkLen=8
TEST_F(FusedRecurrentRwkv8TilingTest, tiling_chunk_len_attr)
{
    optiling::FusedRecurrentRwkv8CompileInfo compileinfo = {40, 196608};

    int b = 2;
    int t = 64;
    int h = 4;
    int n = 64;

    gert::StorageShape ioShape = {{b, h, t, n}, {b, h, t, n}};

    gert::TilingContextPara tilingContextPara("FusedRecurrentRwkv8",
        {
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"output_chunk_state", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_sa", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"reverse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"chunk_len", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        },
        &compileinfo
    );

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    auto *tilingData = reinterpret_cast<const FusedRecurrentRwkv8TilingData *>(tilingInfo.tilingData.get());
    ASSERT_NE(tilingData, nullptr);
    EXPECT_EQ(tilingData->chunkLen, 8U);
    EXPECT_EQ(tilingData->outputChunkState, 1U);
}

// chunk_len attr 非法（< 1）→ tiling 失败
TEST_F(FusedRecurrentRwkv8TilingTest, tiling_chunk_len_invalid)
{
    optiling::FusedRecurrentRwkv8CompileInfo compileinfo = {40, 196608};

    int b = 2;
    int t = 64;
    int h = 4;
    int n = 64;

    gert::StorageShape ioShape = {{b, h, t, n}, {b, h, t, n}};

    gert::TilingContextPara tilingContextPara("FusedRecurrentRwkv8",
        {
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"output_chunk_state", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"output_sa", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"reverse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"chunk_len", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileinfo
    );

    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, tilingInfo));
}
