#include "chunk_kkt_solve_tri_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint64_t DEFAULT_AIC_NUM = 20;
constexpr uint64_t DEFAULT_AIV_NUM = 40;
constexpr uint64_t DEFAULT_LIB_API_WORKSPACE = 32ULL * 1024ULL * 1024ULL;
constexpr uint64_t WORKSPACE_ALIGN = 512;
constexpr uint64_t FP32_BLOCK_ELEMS = 8;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : CeilDiv(value, alignment) * alignment;
}

constexpr uint64_t ChunkTemplateIndex(uint64_t chunkSize)
{
    return chunkSize == 128 ? 1 : 0;
}

static_assert(ChunkTemplateIndex(64) == 0 && ChunkTemplateIndex(128) == 1,
              "ChunkKktSolveTri tiling-key positions must match ASCENDC_TPL_SEL order");

bool MulOverflow(uint64_t lhs, uint64_t rhs, uint64_t *out)
{
    if (out == nullptr || (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
        return true;
    }
    *out = lhs * rhs;
    return false;
}

bool GetChunkCount(const gert::Shape &shape, uint64_t *count)
{
    if (count == nullptr) {
        return false;
    }
    if (shape.GetDimNum() == 1 && shape.GetDim(0) > 0 && shape.GetDim(0) % 2 == 0) {
        *count = static_cast<uint64_t>(shape.GetDim(0) / 2);
        return true;
    }
    if (shape.GetDimNum() == 2 && shape.GetDim(0) > 0 && shape.GetDim(1) == 2) {
        *count = static_cast<uint64_t>(shape.GetDim(0));
        return true;
    }
    return false;
}

matmul_tiling::DataType ToMatmulType(ge::DataType dtype)
{
    return dtype == ge::DT_BF16 ? matmul_tiling::DataType::DT_BF16 : matmul_tiling::DataType::DT_FLOAT16;
}

ge::graphStatus BuildCubeTiling(uint64_t bt, uint64_t k, ge::DataType dtype,
                                ChunkKktSolveTriTilingData &tiling)
{
    matmul_tiling::MatmulApiTiling mm;
    const auto inputType = ToMatmulType(dtype);
    if (mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputType, false) != 0 ||
        mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, inputType, true) != 0 ||
        mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT) != 0 ||
        mm.EnableBias(false) != 0) {
        return ge::GRAPH_FAILED;
    }
    const int32_t btI32 = static_cast<int32_t>(bt);
    const int32_t kI32 = static_cast<int32_t>(k);
    if (mm.SetShape(btI32, btI32, kI32) != 0 || mm.SetOrgShape(btI32, btI32, kI32) != 0 ||
        mm.SetFixSplit(btI32, btI32, -1) != 0 || mm.SetBufferSpace(-1, -1, -1, -1) != 0) {
        return ge::GRAPH_FAILED;
    }
    return mm.GetTiling(tiling.cubeTilingData) == -1 ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus ChunkKktSolveTriTilingFunc(gert::TilingContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetInputShape(2) == nullptr || context->GetInputDesc(0) == nullptr ||
        context->GetInputDesc(1) == nullptr || context->GetInputDesc(2) == nullptr ||
        context->GetRawTilingData() == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &kShape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape &gShape = context->GetInputShape(1)->GetStorageShape();
    const gert::Shape &betaShape = context->GetInputShape(2)->GetStorageShape();
    if (kShape.GetDimNum() != 4 || gShape.GetDimNum() != 3 || betaShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    const int64_t bI64 = kShape.GetDim(0);
    const int64_t hkI64 = kShape.GetDim(1);
    const int64_t tI64 = kShape.GetDim(2);
    const int64_t kI64 = kShape.GetDim(3);
    const int64_t hvI64 = gShape.GetDim(1);
    const int64_t *chunkSizeAttr = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (bI64 <= 0 || hkI64 <= 0 || hvI64 <= 0 || tI64 <= 0 || kI64 != 128 ||
        chunkSizeAttr == nullptr || (*chunkSizeAttr != 64 && *chunkSizeAttr != 128) ||
        gShape.GetDim(0) != bI64 || gShape.GetDim(2) != tI64 ||
        betaShape.GetDim(0) != bI64 || betaShape.GetDim(1) != hvI64 || betaShape.GetDim(2) != tI64 ||
        hvI64 != hkI64) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    if ((dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) ||
        context->GetInputDesc(1)->GetDataType() != ge::DT_FLOAT ||
        context->GetInputDesc(2)->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t b = static_cast<uint64_t>(bI64);
    const uint64_t hk = static_cast<uint64_t>(hkI64);
    const uint64_t hv = static_cast<uint64_t>(hvI64);
    const uint64_t t = static_cast<uint64_t>(tI64);
    const uint64_t k = static_cast<uint64_t>(kI64);
    const uint64_t bt = static_cast<uint64_t>(*chunkSizeAttr);
    uint64_t chunks = CeilDiv(t, bt);
    uint64_t isVarlen = 0;
    const auto *cuDesc = context->GetOptionalInputDesc(3);
    const auto *chunkDesc = context->GetOptionalInputDesc(4);
    const auto *cuShape = context->GetOptionalInputShape(3);
    const auto *chunkShape = context->GetOptionalInputShape(4);
    const bool hasCu = cuDesc != nullptr && cuShape != nullptr;
    const bool hasChunks = chunkDesc != nullptr && chunkShape != nullptr;
    if (hasCu != hasChunks) {
        return ge::GRAPH_FAILED;
    }
    if (hasCu) {
        if (b != 1 || cuDesc->GetDataType() != ge::DT_INT64 || chunkDesc->GetDataType() != ge::DT_INT64 ||
            cuShape->GetStorageShape().GetDimNum() != 1 || cuShape->GetStorageShape().GetDim(0) < 2 ||
            !GetChunkCount(chunkShape->GetStorageShape(), &chunks)) {
            return ge::GRAPH_FAILED;
        }
        isVarlen = 1;
    }

    uint64_t taskNum = 0;
    uint64_t scoreElems = 0;
    uint64_t scoreBytes = 0;
    uint64_t aElems = 0;
    uint64_t aBytes = 0;
    if (MulOverflow(b * hk, chunks, &taskNum) || MulOverflow(taskNum, bt * bt, &scoreElems) ||
        MulOverflow(scoreElems, sizeof(float), &scoreBytes) || MulOverflow(b * hk * t, bt, &aElems) ||
        MulOverflow(aElems, sizeof(uint16_t), &aBytes) || taskNum == 0) {
        return ge::GRAPH_FAILED;
    }
    scoreBytes = AlignUp(scoreBytes, WORKSPACE_ALIGN);
    aBytes = AlignUp(aBytes, WORKSPACE_ALIGN);

    uint64_t aicNum = DEFAULT_AIC_NUM;
    uint64_t aivNum = DEFAULT_AIV_NUM;
    uint64_t libWorkspace = DEFAULT_LIB_API_WORKSPACE;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC platform(platformInfo);
        aicNum = std::max<uint64_t>(1, platform.GetCoreNumAic());
        aivNum = std::max<uint64_t>(1, platform.GetCoreNumAiv());
        libWorkspace = platform.GetLibApiWorkSpaceSize();
    }
    const uint64_t usedAicNum = std::max<uint64_t>(1, std::min(taskNum, aicNum));
    const uint64_t pairedAivNum = std::min<uint64_t>(aivNum, usedAicNum * 2);
    const uint64_t usedAivNum = std::max<uint64_t>(1, pairedAivNum);

    ChunkKktSolveTriTilingData tiling;
    tiling.set_B(b);
    tiling.set_Hk(hk);
    tiling.set_Hv(hv);
    tiling.set_hvPerHk(hv / hk);
    tiling.set_T(t);
    tiling.set_K(k);
    tiling.set_BT(bt);
    tiling.set_NT(chunks);
    tiling.set_taskNum(taskNum);
    tiling.set_usedAicNum(usedAicNum);
    tiling.set_usedAivNum(usedAivNum);
    tiling.set_btAlign(AlignUp(bt, FP32_BLOCK_ELEMS));
    tiling.set_isVarlen(isVarlen);
    tiling.set_scoreWorkspaceBytes(scoreBytes);
    tiling.set_aWorkspaceBytes(aBytes);
    tiling.set_totalTiles(static_cast<int64_t>(taskNum));
    tiling.set_matrixSize(static_cast<int64_t>(bt));
    tiling.set_numHeads(static_cast<int64_t>(hk));
    tiling.set_seqLen(static_cast<int64_t>(t));
    tiling.set_batchSize(static_cast<int64_t>(b));
    tiling.set_isLower(1);
    tiling.set_hasCuSeqlens(static_cast<int64_t>(isVarlen));
    tiling.set_chunkSize(static_cast<int64_t>(bt));
    tiling.set_numChunks(isVarlen == 0 ? static_cast<int64_t>(chunks) : 0);
    const uint64_t remainder = t % bt;
    tiling.set_lastChunkValidSize(static_cast<int64_t>(remainder == 0 ? bt : remainder));
    tiling.set_totalChunks(static_cast<int64_t>(chunks));
    tiling.set_layoutMode(isVarlen == 0 ? 0 : 3);
    tiling.set_dtypeMode(dtype == ge::DT_BF16 ? 1 : 0);
    if (BuildCubeTiling(bt, k, dtype, tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Each MIX core group owns a contiguous tile range. Its two AIV sub-blocks
    // produce KKT and signal the paired AIC before the solve phase starts.
    // MIX_AIC_1_2 launches two paired AIV sub-blocks for every AIC block. Do not
    // expand this to the physical core count: idle groups would still enter the
    // stage barriers and index solve workspace that is allocated per active task.
    const uint32_t blockDim = static_cast<uint32_t>(usedAicNum);
    const uint64_t solveAicNum = usedAicNum;
    const uint64_t solveTilesPerCore = CeilDiv(taskNum, solveAicNum);
    tiling.set_tilesPerCore(static_cast<int64_t>(solveTilesPerCore));

    // Template list positions, rather than the declared BT values, occupy the
    // high tiling-key bits.
    const uint64_t chunkKey = ChunkTemplateIndex(bt);
    const uint64_t dtypeKey = dtype == ge::DT_BF16 ? 20 : 10;
    context->SetTilingKey(dtypeKey + (chunkKey << 8));
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1);
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t solveShared = 3 * bt * bt * sizeof(uint16_t);
    const uint64_t solvePerCore = 2 * bt * bt * sizeof(uint16_t);
    const uint64_t solveWorkspacePerCore = AlignUp(solveShared + solvePerCore, WORKSPACE_ALIGN);
    tiling.set_solveWorkspacePerCoreBytes(solveWorkspacePerCore);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    workspace[0] = libWorkspace + scoreBytes + aBytes + solveAicNum * solveWorkspacePerCore;
    return ge::GRAPH_SUCCESS;
}

struct ChunkKktSolveTriCompileInfo {};
ge::graphStatus ChunkKktSolveTriTilingParse(gert::TilingParseContext *context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkKktSolveTri)
    .Tiling(ChunkKktSolveTriTilingFunc)
    .TilingParse<ChunkKktSolveTriCompileInfo>(ChunkKktSolveTriTilingParse);

IMPL_OP_OPTILING(ChunkCumsumKktSolveTri)
    .Tiling(ChunkKktSolveTriTilingFunc)
    .TilingParse<ChunkKktSolveTriCompileInfo>(ChunkKktSolveTriTilingParse);
}  // namespace optiling
