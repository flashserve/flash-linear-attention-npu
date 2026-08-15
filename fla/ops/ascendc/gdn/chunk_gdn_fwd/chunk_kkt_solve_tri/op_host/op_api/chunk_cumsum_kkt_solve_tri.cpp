#include "chunk_cumsum_kkt_solve_tri.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkCumsumKktSolveTri);

namespace {
const aclTensor *ToInt64Tensor(const aclIntArray *values, aclOpExecutor *executor)
{
    if (values == nullptr) {
        return nullptr;
    }
    const aclTensor *tensor = executor->ConvertToTensor(values, DataType::DT_INT64);
    if (tensor == nullptr) {
        return nullptr;
    }
    auto *mutableTensor = const_cast<aclTensor *>(tensor);
    mutableTensor->SetStorageFormat(Format::FORMAT_ND);
    mutableTensor->SetViewFormat(Format::FORMAT_ND);
    mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
    return tensor;
}
}  // namespace

const std::array<const aclTensor *, 2> ChunkCumsumKktSolveTri(
    const aclTensor *k,
    const aclTensor *g,
    const aclTensor *beta,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    const aclTensor *gCumsumOut,
    const aclTensor *aOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkCumsumKktSolveTri, k, g, beta, cuSeqlensOptional, chunkIndicesOptional,
           chunkSize, gCumsumOut, aOut);
    const aclTensor *cu = ToInt64Tensor(cuSeqlensOptional, executor);
    const aclTensor *indices = ToInt64Tensor(chunkIndicesOptional, executor);
    if ((cuSeqlensOptional != nullptr && cu == nullptr) ||
        (chunkIndicesOptional != nullptr && indices == nullptr)) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to convert variable-length metadata.");
        return {nullptr, nullptr};
    }
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkCumsumKktSolveTri,
        OP_INPUT(k, g, beta, cu, indices),
        OP_OUTPUT(gCumsumOut, aOut),
        OP_ATTR(chunkSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }
    return {gCumsumOut, aOut};
}
}  // namespace l0op
