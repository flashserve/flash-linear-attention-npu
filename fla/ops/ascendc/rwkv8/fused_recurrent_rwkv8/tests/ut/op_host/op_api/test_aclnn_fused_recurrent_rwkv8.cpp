/*!
 * \file test_aclnn_fused_recurrent_rwkv8.cpp
 * \brief
 */

#include <vector>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_fused_recurrent_rwkv8.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class aclnnFusedRecurrentRwkv8_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "aclnnFusedRecurrentRwkv8_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "aclnnFusedRecurrentRwkv8_test TearDown" << endl;
    }
};

namespace {
TensorDesc IoDesc(int64_t b, int64_t h, int64_t t, int64_t n, aclDataType dtype = ACL_FLOAT)
{
    return TensorDesc({b, h, t, n}, dtype, ACL_FORMAT_ND).ValueRange(0, 1);   // BHTC
}

TensorDesc StateDesc(int64_t b, int64_t h, int64_t n)
{
    return TensorDesc({b, h, n, n}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
}
} // namespace

// 正常路径：带 initial_state
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_full)
{
    auto q = IoDesc(2, 4, 64, 64);
    auto init = StateDesc(2, 4, 64);
    auto out = IoDesc(2, 4, 64, 64);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, q, q, q, q, q, init, 1.0f, false, false, false, (int64_t)16),
                        OUTPUT(out, nullptr, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// initialState = nullptr（可选输入缺省，零初态）
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_no_initial_state)
{
    auto q = IoDesc(1, 2, 33, 64);
    auto out = IoDesc(1, 2, 33, 64);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, q, q, q, q, q, nullptr, 0.125f, false, false, false, (int64_t)16),
                        OUTPUT(out, nullptr, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常路径：bf16 io（state 张量恒 fp32）
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_bf16_io)
{
    auto q = IoDesc(2, 4, 64, 64, ACL_BF16);
    auto init = StateDesc(2, 4, 64);               // initial_state 恒 fp32
    auto out = IoDesc(2, 4, 64, 64, ACL_BF16);     // o 跟随 q

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, q, q, q, q, q, init, 1.0f, false, false, false, (int64_t)16),
                        OUTPUT(out, nullptr, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 异常路径：io dtype 不一致（w 用 fp16 其余 bf16）→ PARAM_INVALID
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_io_dtype_mismatch)
{
    auto q = IoDesc(2, 4, 64, 64, ACL_BF16);
    auto wBad = IoDesc(2, 4, 64, 64, ACL_FLOAT16);
    auto out = IoDesc(2, 4, 64, 64, ACL_BF16);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, wBad, q, q, q, q, nullptr, 1.0f, false, false, false, (int64_t)16),
                        OUTPUT(out, nullptr, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径：initial_state 用 bf16（state 恒 fp32）→ PARAM_INVALID
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_state_dtype_bad)
{
    auto q = IoDesc(2, 4, 64, 64, ACL_BF16);
    auto initBad = TensorDesc({2, 4, 64, 64}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto out = IoDesc(2, 4, 64, 64, ACL_BF16);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, q, q, q, q, q, initBad, 1.0f, false, false, false, (int64_t)16),
                        OUTPUT(out, nullptr, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径：dtype 不支持（int8）→ PARAM_INVALID
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_bad_dtype)
{
    auto qBad = TensorDesc({2, 4, 64, 64}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 1);
    auto q = IoDesc(2, 4, 64, 64);
    auto out = IoDesc(2, 4, 64, 64);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(qBad, q, q, q, q, q, nullptr, 1.0f, false, false, false, (int64_t)16),
                        OUTPUT(out, nullptr, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径：必选输入为 nullptr → PARAM_INVALID
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_null_required_input)
{
    auto q = IoDesc(2, 4, 64, 64);
    auto out = IoDesc(2, 4, 64, 64);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(nullptr, q, q, q, q, q, nullptr, 1.0f, false, false, false, (int64_t)16),
                        OUTPUT(out, nullptr, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径：out 为 nullptr → PARAM_INVALID
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_null_out)
{
    auto q = IoDesc(2, 4, 64, 64);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, q, q, q, q, q, nullptr, 1.0f, false, false, false, (int64_t)16),
                        OUTPUT(nullptr, nullptr, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径：w 与 q shape 不一致 → PARAM_INVALID
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_shape_mismatch)
{
    auto q = IoDesc(2, 4, 64, 64);
    auto wBad = IoDesc(2, 4, 32, 64);
    auto out = IoDesc(2, 4, 64, 64);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, wBad, q, q, q, q, nullptr, 1.0f, false, false, false, (int64_t)16),
                        OUTPUT(out, nullptr, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径：训练预埋全开（reverse + outputChunkState + outputSa）
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_s_sa_full)
{
    auto q = IoDesc(2, 4, 64, 64);
    auto init = StateDesc(2, 4, 64);
    auto out = IoDesc(2, 4, 64, 64);
    auto s = TensorDesc({2, 4, 4, 64, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);   // (B,H,T//16,N,N)
    auto sa = IoDesc(2, 4, 64, 64);                                                      // (B,H,T,N)

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, q, q, q, q, q, init, 1.0f, true, true, true, (int64_t)16),
                        OUTPUT(out, s, sa));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 异常路径：outputChunkState=true 但 sOut=nullptr → PARAM_INVALID
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_s_flag_on_null_out)
{
    auto q = IoDesc(2, 4, 64, 64);
    auto out = IoDesc(2, 4, 64, 64);
    auto sa = IoDesc(2, 4, 64, 64);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, q, q, q, q, q, nullptr, 1.0f, false, true, false, (int64_t)16),
                        OUTPUT(out, nullptr, sa));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径：K≠V（q/w/k/z/b (B,H,T,64)，v/o (B,H,T,32)，state (B,H,64,32)）
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_k_ne_v)
{
    auto q = IoDesc(2, 4, 64, 64);
    auto v = IoDesc(2, 4, 64, 32);
    auto init = TensorDesc({2, 4, 64, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);   // (B,H,K,V)
    auto out = IoDesc(2, 4, 64, 32);
    auto s = TensorDesc({2, 4, 4, 64, 32}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);    // (B,H,T//16,K,V)
    auto sa = IoDesc(2, 4, 64, 32);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, q, q, v, q, q, init, 1.0f, false, true, true, (int64_t)16),
                        OUTPUT(out, s, sa));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 异常路径：K≠V 时 out 用了 K 维（应等于 v 的 V 维）→ PARAM_INVALID
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_k_ne_v_out_bad)
{
    auto q = IoDesc(2, 4, 64, 64);
    auto v = IoDesc(2, 4, 64, 32);
    auto outBad = IoDesc(2, 4, 64, 64);   // 错：out 应随 v 的 V=32

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, q, q, v, q, q, nullptr, 1.0f, false, false, false, (int64_t)16),
                        OUTPUT(outBad, nullptr, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径：非默认 chunkLen=8 → s (B,H,T//8,N,N)
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_chunk_len_8)
{
    auto q = IoDesc(2, 4, 64, 64);
    auto out = IoDesc(2, 4, 64, 64);
    auto s = TensorDesc({2, 4, 8, 64, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);   // (B,H,T//8,N,N)
    auto sa = IoDesc(2, 4, 64, 64);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, q, q, q, q, q, nullptr, 1.0f, false, true, true, (int64_t)8),
                        OUTPUT(out, s, sa));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 异常路径：chunkLen=0 → PARAM_INVALID
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_chunk_len_zero)
{
    auto q = IoDesc(2, 4, 64, 64);
    auto out = IoDesc(2, 4, 64, 64);

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, q, q, q, q, q, nullptr, 1.0f, false, false, false, (int64_t)0),
                        OUTPUT(out, nullptr, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常路径：chunkLen=8 但 sOut 仍按 16 给 shape（T//16）→ PARAM_INVALID
TEST_F(aclnnFusedRecurrentRwkv8_test, ascend910B4_case_chunk_len_s_shape_bad)
{
    auto q = IoDesc(2, 4, 64, 64);
    auto out = IoDesc(2, 4, 64, 64);
    auto sBad = TensorDesc({2, 4, 4, 64, 64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);   // 错：应为 T//8=8

    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnFusedRecurrentRwkv8,
                        INPUT(q, q, q, q, q, q, nullptr, 1.0f, false, true, false, (int64_t)8),
                        OUTPUT(out, sBad, nullptr));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
