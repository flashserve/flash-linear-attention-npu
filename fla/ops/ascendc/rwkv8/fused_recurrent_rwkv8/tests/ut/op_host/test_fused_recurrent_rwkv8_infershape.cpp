/*!
 * \file test_fused_recurrent_rwkv8_infershape.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

class FusedRecurrentRwkv8Test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedRecurrentRwkv8Test Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedRecurrentRwkv8Test Proto TearDown" << std::endl;
    }
};

TEST_F(FusedRecurrentRwkv8Test, infershape_with_initial_state)
{
    int b = 2;
    int t = 64;
    int h = 4;
    int n = 64;

    gert::StorageShape ioShape = {{b, h, t, n}, {b, h, t, n}};
    gert::StorageShape stateShape = {{b, h, n, n}, {b, h, n, n}};

    gert::InfershapeContextPara infershapeContextPara("FusedRecurrentRwkv8",
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
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // s
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // sa
        },
        {
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
        }
    );

    // 训练预埋开关缺省 → s/sa 零尺寸占位
    std::vector<std::vector<int64_t>> expectOutputShape = {{b, h, t, n}, {0}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(FusedRecurrentRwkv8Test, infershape_without_initial_state)
{
    int b = 1;
    int t = 16;
    int h = 2;
    int n = 64;

    gert::StorageShape ioShape = {{b, h, t, n}, {b, h, t, n}};

    gert::InfershapeContextPara infershapeContextPara("FusedRecurrentRwkv8",
        {
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},      // q
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},      // w
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},      // k
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},      // v
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},      // z
            {ioShape, ge::DT_FLOAT, ge::FORMAT_ND},      // b
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // o
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // s
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // sa
        },
        {
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.125)},
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {{b, h, t, n}, {0}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// bf16 io：shape 推导与 fp32 一致
TEST_F(FusedRecurrentRwkv8Test, infershape_bf16_io)
{
    int b = 2;
    int t = 64;
    int h = 4;
    int n = 64;

    gert::StorageShape ioShape = {{b, h, t, n}, {b, h, t, n}};
    gert::StorageShape stateShape = {{b, h, n, n}, {b, h, n, n}};

    gert::InfershapeContextPara infershapeContextPara("FusedRecurrentRwkv8",
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
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // s 恒 fp32
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // sa 恒 fp32
        },
        {
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape = {{b, h, t, n}, {0}, {0}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 训练预埋：output_chunk_state/output_sa 打开 → s (B,H,T//16,N,N)、sa (B,H,T,N)
TEST_F(FusedRecurrentRwkv8Test, infershape_with_s_sa)
{
    int b = 2;
    int t = 64;
    int h = 4;
    int n = 64;

    gert::StorageShape ioShape = {{b, h, t, n}, {b, h, t, n}};
    gert::StorageShape stateShape = {{b, h, n, n}, {b, h, n, n}};

    gert::InfershapeContextPara infershapeContextPara("FusedRecurrentRwkv8",
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
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"output_chunk_state", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_sa", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"reverse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape =
        {{b, h, t, n}, {b, h, t / 16, n, n}, {b, h, t, n}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// bf16 io：dtype 传播 o=bf16，s/sa 恒 fp32
TEST_F(FusedRecurrentRwkv8Test, inferdtype_bf16_io)
{
    gert::InferDataTypeContextFaker contextFaker;
    contextFaker.SetOpType("FusedRecurrentRwkv8");
    contextFaker.NodeIoNum(7, 3);
    for (int i = 0; i < 6; i++) {
        contextFaker.NodeInputTd(i, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    contextFaker.NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);   // initial_state
    for (int i = 0; i < 3; i++) {
        contextFaker.NodeOutputTd(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto contextHolder = contextFaker.Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("FusedRecurrentRwkv8")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);

    auto *ctx = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(ctx, nullptr);
    ASSERT_EQ(inferDtypeFunc(ctx), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ctx->GetOutputDataType(0), ge::DT_BF16);   // o 跟随 q
    EXPECT_EQ(ctx->GetOutputDataType(1), ge::DT_FLOAT);  // s 恒 fp32
    EXPECT_EQ(ctx->GetOutputDataType(2), ge::DT_FLOAT);  // sa 恒 fp32
}

// K≠V：o/sa (B,H,T,V)，s (B,H,T//16,K,V)
TEST_F(FusedRecurrentRwkv8Test, infershape_k_ne_v)
{
    int b = 2;
    int t = 64;
    int h = 4;
    int kDim = 64;
    int vDim = 32;

    gert::StorageShape kShape = {{b, h, t, kDim}, {b, h, t, kDim}};
    gert::StorageShape vShape = {{b, h, t, vDim}, {b, h, t, vDim}};
    gert::StorageShape stateShape = {{b, h, kDim, vDim}, {b, h, kDim, vDim}};   // initial_state 接口朝向 (K,V)

    gert::InfershapeContextPara infershapeContextPara("FusedRecurrentRwkv8",
        {
            {kShape, ge::DT_FLOAT, ge::FORMAT_ND},       // q
            {kShape, ge::DT_FLOAT, ge::FORMAT_ND},       // w
            {kShape, ge::DT_FLOAT, ge::FORMAT_ND},       // k
            {vShape, ge::DT_FLOAT, ge::FORMAT_ND},       // v
            {kShape, ge::DT_FLOAT, ge::FORMAT_ND},       // z
            {kShape, ge::DT_FLOAT, ge::FORMAT_ND},       // b
            {stateShape, ge::DT_FLOAT, ge::FORMAT_ND},   // initial_state
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // o
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // s
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},     // sa
        },
        {
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"output_chunk_state", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_sa", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        }
    );

    std::vector<std::vector<int64_t>> expectOutputShape =
        {{b, h, t, vDim}, {b, h, t / 16, kDim, vDim}, {b, h, t, vDim}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 非默认 chunk_len：chunk_len=8 → s (B,H,T//8,N,N)；缺省 attr 时回落 16
TEST_F(FusedRecurrentRwkv8Test, infershape_chunk_len_attr)
{
    int b = 2;
    int t = 64;
    int h = 4;
    int n = 64;

    gert::StorageShape ioShape = {{b, h, t, n}, {b, h, t, n}};
    gert::StorageShape stateShape = {{b, h, n, n}, {b, h, n, n}};

    // chunk_len=8
    gert::InfershapeContextPara para8("FusedRecurrentRwkv8",
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
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"output_chunk_state", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
            {"output_sa", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"reverse", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"chunk_len", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
        }
    );
    std::vector<std::vector<int64_t>> expect8 = {{b, h, t, n}, {b, h, t / 8, n, n}, {0}};
    ExecuteTestCase(para8, ge::GRAPH_SUCCESS, expect8);

    // 缺省 chunk_len attr → 回落 16（与 infershape_with_s_sa 一致，这里显式再走一遍缺省路径）
    gert::InfershapeContextPara paraDefault("FusedRecurrentRwkv8",
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
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(1.0)},
            {"output_chunk_state", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        }
    );
    std::vector<std::vector<int64_t>> expectDefault = {{b, h, t, n}, {b, h, t / 16, n, n}, {0}};
    ExecuteTestCase(paraDefault, ge::GRAPH_SUCCESS, expectDefault);
}
