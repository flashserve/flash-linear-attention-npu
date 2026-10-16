/*!
 * \file aclnn_fused_recurrent_rwkv8.cpp
 * \brief
 */
#include "aclnn_fused_recurrent_rwkv8.h"
#include "fused_recurrent_rwkv8.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"

#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t IO_DIM_NUM = 4;
constexpr int64_t DEFAULT_CHUNK_LEN = 16;   // 对齐官方 wkv7_cuda.cu backward 的 chunk 重建粒度

op::Shape MakeShape(std::initializer_list<int64_t> dims)
{
    op::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    return shape;
}

struct FusedRecurrentRwkv8Params {
    // mandatory inputs
    const aclTensor *q {nullptr};
    const aclTensor *w {nullptr};
    const aclTensor *k {nullptr};
    const aclTensor *v {nullptr};
    const aclTensor *z {nullptr};
    const aclTensor *b {nullptr};
    // optional input
    const aclTensor *initialState {nullptr};
    // attrs
    float scale {1.0f};
    bool reverse {false};
    bool outputChunkState {false};
    bool outputSa {false};
    int64_t chunkLen {DEFAULT_CHUNK_LEN};
    // outputs
    const aclTensor *out {nullptr};
    const aclTensor *sOut {nullptr};
    const aclTensor *saOut {nullptr};
};

// io（q/w/k/v/z/b/out）支持 fp16/bf16/fp32；state 张量（initialState/s/sa）恒 fp32
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_BF16,
                                                                       op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> DTYPE_STATE_LIST = {op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const FusedRecurrentRwkv8Params &params)
{
    OP_CHECK_NULL(params.q, return false);
    OP_CHECK_NULL(params.w, return false);
    OP_CHECK_NULL(params.k, return false);
    OP_CHECK_NULL(params.v, return false);
    OP_CHECK_NULL(params.z, return false);
    OP_CHECK_NULL(params.b, return false);
    OP_CHECK_NULL(params.out, return false);
    // initialState 允许为空
    return true;
}

static inline bool CheckDtypeValid(const FusedRecurrentRwkv8Params &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.q, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.w, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.k, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.v, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.z, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.b, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.out, DTYPE_SUPPORT_LIST, return false);
    // state 张量恒 fp32
    if (params.initialState != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(params.initialState, DTYPE_STATE_LIST, return false);
    }
    // 训练预埋输出恒 fp32（仅在对应开关打开时要求非空）
    if (params.outputChunkState) {
        OP_CHECK_NULL(params.sOut, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params.sOut, DTYPE_STATE_LIST, return false);
    }
    if (params.outputSa) {
        OP_CHECK_NULL(params.saOut, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(params.saOut, DTYPE_STATE_LIST, return false);
    }
    // io 同 dtype：w/k/v/z/b/out 必须等于 q
    const auto ioDtype = params.q->GetDataType();
    const aclTensor *ioTensors[] = {params.w, params.k, params.v, params.z, params.b, params.out};
    for (const auto *t : ioTensors) {
        if (t->GetDataType() != ioDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "io dtypes must be identical (q/w/k/v/z/b/out).");
            return false;
        }
    }
    return true;
}

static inline size_t Rank(const aclTensor *tensor)
{
    return tensor->GetViewShape().GetDimNum();
}

static inline bool SameShape(const aclTensor *lhs, const aclTensor *rhs)
{
    if (lhs == nullptr || rhs == nullptr || Rank(lhs) != Rank(rhs)) {
        return false;
    }
    for (size_t i = 0; i < Rank(lhs); i++) {
        if (lhs->GetViewShape().GetDim(i) != rhs->GetViewShape().GetDim(i)) {
            return false;
        }
    }
    return true;
}

static inline bool HasShape(const aclTensor *tensor, std::initializer_list<int64_t> expected)
{
    if (tensor == nullptr || Rank(tensor) != expected.size()) {
        return false;
    }
    size_t i = 0;
    for (int64_t dim : expected) {
        if (tensor->GetViewShape().GetDim(i++) != dim) {
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckShapeValid(const FusedRecurrentRwkv8Params &params)
{
    CHECK_COND(Rank(params.q) == IO_DIM_NUM, ACLNN_ERR_PARAM_INVALID, "q should be rank-4 (B,H,T,K).");

    // K 侧与 q 全等；V 侧（v）仅前 3 维与 q 一致，末维 V 独立
    CHECK_COND(SameShape(params.w, params.q), ACLNN_ERR_PARAM_INVALID, "w shape must equal q shape.");
    CHECK_COND(SameShape(params.k, params.q), ACLNN_ERR_PARAM_INVALID, "k shape must equal q shape.");
    CHECK_COND(SameShape(params.z, params.q), ACLNN_ERR_PARAM_INVALID, "z shape must equal q shape.");
    CHECK_COND(SameShape(params.b, params.q), ACLNN_ERR_PARAM_INVALID, "b shape must equal q shape.");
    CHECK_COND(Rank(params.v) == IO_DIM_NUM, ACLNN_ERR_PARAM_INVALID, "v should be rank-4 (B,H,T,V).");
    for (size_t d = 0; d < 3; d++) {
        CHECK_COND(params.v->GetViewShape().GetDim(d) == params.q->GetViewShape().GetDim(d),
                   ACLNN_ERR_PARAM_INVALID, "v dims B/H/T must equal q.");
    }
    CHECK_COND(SameShape(params.out, params.v), ACLNN_ERR_PARAM_INVALID, "out shape must equal v shape (B,H,T,V).");

    const int64_t dimB = params.q->GetViewShape().GetDim(0);
    const int64_t dimH = params.q->GetViewShape().GetDim(1);
    const int64_t dimT = params.q->GetViewShape().GetDim(2);
    const int64_t dimK = params.q->GetViewShape().GetDim(3);
    const int64_t dimV = params.v->GetViewShape().GetDim(3);

    if (params.initialState != nullptr) {
        CHECK_COND(HasShape(params.initialState, {dimB, dimH, dimK, dimV}), ACLNN_ERR_PARAM_INVALID,
                   "initialState shape should be (B,H,K,V).");
    }
    if (params.outputChunkState) {
        CHECK_COND(HasShape(params.sOut, {dimB, dimH, dimT / params.chunkLen, dimK, dimV}), ACLNN_ERR_PARAM_INVALID,
                   "sOut shape should be (B,H,T//chunkLen,K,V).");
    }
    if (params.outputSa) {
        CHECK_COND(SameShape(params.saOut, params.v), ACLNN_ERR_PARAM_INVALID,
                   "saOut shape should be (B,H,T,V).");
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const FusedRecurrentRwkv8Params &params)
{
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(params.chunkLen >= 1, ACLNN_ERR_PARAM_INVALID, "chunkLen must be >= 1.");
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnFusedRecurrentRwkv8GetWorkspaceSize(const aclTensor *q, const aclTensor *w, const aclTensor *k,
                                                     const aclTensor *v, const aclTensor *z, const aclTensor *b,
                                                     const aclTensor *initialState, float scale, bool reverse,
                                                     bool outputChunkState, bool outputSa, int64_t chunkLen,
                                                     aclTensor *out, aclTensor *sOut, aclTensor *saOut,
                                                     uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnFusedRecurrentRwkv8, DFX_IN(q, w, k, v, z, b, initialState, scale, reverse,
                                                    outputChunkState, outputSa, chunkLen),
                   DFX_OUT(out, sOut, saOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    FusedRecurrentRwkv8Params params {q, w, k, v, z, b, initialState, scale, reverse, outputChunkState, outputSa,
                                      chunkLen, out, sOut, saOut};
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto q_ = l0op::Contiguous(q, executorPtr);
    auto w_ = l0op::Contiguous(w, executorPtr);
    auto k_ = l0op::Contiguous(k, executorPtr);
    auto v_ = l0op::Contiguous(v, executorPtr);
    auto z_ = l0op::Contiguous(z, executorPtr);
    auto b_ = l0op::Contiguous(b, executorPtr);
    if (initialState != nullptr) {
        initialState = l0op::Contiguous(initialState, executorPtr);
    }

    // s/sa 内部张量恒分配（开关关闭时零尺寸占位，kernel flag 门控跳过写出）
    const op::Shape &qShape = q->GetViewShape();
    const int64_t dimB = qShape.GetDim(0);
    const int64_t dimH = qShape.GetDim(1);
    const int64_t dimT = qShape.GetDim(2);
    const int64_t dimK = qShape.GetDim(3);
    const int64_t dimV = v->GetViewShape().GetDim(3);
    auto sCompute = executorPtr->AllocTensor(
        outputChunkState ? MakeShape({dimB, dimH, dimT / chunkLen, dimK, dimV}) : MakeShape({0}),
        DataType::DT_FLOAT, Format::FORMAT_ND);
    CHECK_RET(sCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto saCompute = executorPtr->AllocTensor(
        outputSa ? MakeShape({dimB, dimH, dimT, dimV}) : MakeShape({0}),
        DataType::DT_FLOAT, Format::FORMAT_ND);
    CHECK_RET(saCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0接口
    auto outRet = l0op::FusedRecurrentRwkv8(q_, w_, k_, v_, z_, b_, initialState, scale,
                                            outputChunkState, outputSa, reverse, chunkLen,
                                            const_cast<aclTensor *>(sCompute),
                                            const_cast<aclTensor *>(saCompute), executorPtr);
    CHECK_RET(outRet != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out_ = l0op::Contiguous(out, executorPtr);
    auto viewCopyResult = l0op::ViewCopy(outRet, out_, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (outputChunkState && sOut != nullptr) {
        auto sOut_ = l0op::Contiguous(sOut, executorPtr);
        auto sCopyResult = l0op::ViewCopy(sCompute, sOut_, executorPtr);
        CHECK_RET(sCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (outputSa && saOut != nullptr) {
        auto saOut_ = l0op::Contiguous(saOut, executorPtr);
        auto saCopyResult = l0op::ViewCopy(saCompute, saOut_, executorPtr);
        CHECK_RET(saCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 获取计算过程中需要使用的workspace大小。
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedRecurrentRwkv8(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedRecurrentRwkv8);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
