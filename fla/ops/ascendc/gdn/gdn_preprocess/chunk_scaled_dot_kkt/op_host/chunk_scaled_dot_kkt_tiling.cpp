#include "chunk_scaled_dot_kkt_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint64_t kDefaultAicNum = 20;
constexpr uint64_t kDefaultAivNum = 40;
constexpr uint64_t kDefaultLibApiWorkspace = 32ULL * 1024ULL * 1024ULL;
constexpr uint64_t kFp32BlockElems = 8;
constexpr uint64_t kWorkspaceAlign = 512;
constexpr uint64_t kMaxInt32 = static_cast<uint64_t>(std::numeric_limits<int32_t>::max());

uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}

uint64_t AlignUp(uint64_t a, uint64_t align)
{
    return align == 0 ? a : CeilDiv(a, align) * align;
}

bool IsChunkSizeSupported(uint64_t chunkSize)
{
    return chunkSize == 16 || chunkSize == 32 || chunkSize == 64 || chunkSize == 128;
}

uint64_t ChunkKey(uint64_t chunkSize)
{
    if (chunkSize == 16) {
        return 1;
    }
    if (chunkSize == 32) {
        return 2;
    }
    if (chunkSize == 128) {
        return 3;
    }
    return 0;
}

matmul_tiling::DataType MatmulKDataType(ge::DataType dtype)
{
    return dtype == ge::DT_BF16 ? matmul_tiling::DataType::DT_BF16 : matmul_tiling::DataType::DT_FLOAT16;
}

uint64_t DtypeKey(ge::DataType dtype)
{
    return dtype == ge::DT_BF16 ? 20 : 10;
}

uint64_t TilingKey(ge::DataType dtype, uint64_t chunkSize)
{
    return DtypeKey(dtype) + (ChunkKey(chunkSize) << 8);
}

bool Shape3Equal(const gert::Shape &shape, int64_t b, int64_t h, int64_t t)
{
    return shape.GetDimNum() == 3 && shape.GetDim(0) == b && shape.GetDim(1) == h && shape.GetDim(2) == t;
}

bool MulOverflow(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == nullptr) {
        return true;
    }
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return true;
    }
    *out = a * b;
    return false;
}

ge::graphStatus BuildCubeTiling(uint64_t bt, uint64_t k, ge::DataType kDtype, ChunkScaledDotKktTilingData &tiling)
{
    matmul_tiling::MatmulApiTiling mm;
    const matmul_tiling::DataType matmulKType = MatmulKDataType(kDtype);
    if (mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmulKType, false) != 0) {
        return ge::GRAPH_FAILED;
    }
    if (mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmulKType, true) != 0) {
        return ge::GRAPH_FAILED;
    }
    if (mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT) != 0) {
        return ge::GRAPH_FAILED;
    }
    if (mm.EnableBias(false) != 0) {
        return ge::GRAPH_FAILED;
    }

    const int32_t btI32 = static_cast<int32_t>(bt);
    const int32_t kI32 = static_cast<int32_t>(k);
    if (mm.SetShape(btI32, btI32, kI32) != 0 || mm.SetOrgShape(btI32, btI32, kI32) != 0) {
        return ge::GRAPH_FAILED;
    }

    const int32_t base = static_cast<int32_t>(std::max<uint64_t>(16, std::min<uint64_t>(128, AlignUp(bt, 16))));
    if (mm.SetFixSplit(base, base, -1) != 0) {
        return ge::GRAPH_FAILED;
    }
    if (mm.SetBufferSpace(-1, -1, -1, -1) != 0) {
        return ge::GRAPH_FAILED;
    }
    return mm.GetTiling(tiling.cubeTilingData) == -1 ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetInputShape(2) == nullptr || context->GetInputDesc(0) == nullptr ||
        context->GetInputDesc(1) == nullptr || context->GetInputDesc(2) == nullptr ||
        context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &kShape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape &gShape = context->GetInputShape(1)->GetStorageShape();
    const gert::Shape &betaShape = context->GetInputShape(2)->GetStorageShape();
    if (kShape.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const int64_t bI64 = kShape.GetDim(0);
    const int64_t hI64 = kShape.GetDim(1);
    const int64_t tI64 = kShape.GetDim(2);
    const int64_t kI64 = kShape.GetDim(3);
    if (bI64 <= 0 || hI64 <= 0 || tI64 <= 0 || kI64 <= 0 || !Shape3Equal(gShape, bI64, hI64, tI64) ||
        !Shape3Equal(betaShape, bI64, hI64, tI64)) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType kDtype = context->GetInputDesc(0)->GetDataType();
    if ((kDtype != ge::DT_FLOAT16 && kDtype != ge::DT_BF16) || context->GetInputDesc(1)->GetDataType() != ge::DT_FLOAT ||
        context->GetInputDesc(2)->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    int64_t chunkSizeI64 = 64;
    if (context->GetAttrs() != nullptr && context->GetAttrs()->GetAttrPointer<int64_t>(0) != nullptr) {
        chunkSizeI64 = *context->GetAttrs()->GetAttrPointer<int64_t>(0);
    }
    if (chunkSizeI64 <= 0 || !IsChunkSizeSupported(static_cast<uint64_t>(chunkSizeI64))) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t b = static_cast<uint64_t>(bI64);
    const uint64_t h = static_cast<uint64_t>(hI64);
    const uint64_t t = static_cast<uint64_t>(tI64);
    const uint64_t k = static_cast<uint64_t>(kI64);
    const uint64_t bt = static_cast<uint64_t>(chunkSizeI64);
    if (bt > kMaxInt32 || k > kMaxInt32) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t nt = CeilDiv(t, bt);
    uint64_t bh = 0;
    uint64_t taskNum = 0;
    uint64_t scoreElems = 0;
    uint64_t scoreBytes = 0;
    if (MulOverflow(b, h, &bh) || MulOverflow(bh, nt, &taskNum) || MulOverflow(taskNum, bt * bt, &scoreElems) ||
        MulOverflow(scoreElems, sizeof(float), &scoreBytes) || taskNum == 0) {
        return ge::GRAPH_FAILED;
    }
    scoreBytes = AlignUp(scoreBytes, kWorkspaceAlign);

    uint64_t aicNum = kDefaultAicNum;
    uint64_t aivNum = kDefaultAivNum;
    uint64_t libApiWorkspace = kDefaultLibApiWorkspace;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC platform(platformInfo);
        aicNum = static_cast<uint64_t>(platform.GetCoreNumAic());
        aivNum = static_cast<uint64_t>(platform.GetCoreNumAiv());
        libApiWorkspace = static_cast<uint64_t>(platform.GetLibApiWorkSpaceSize());
    }
    if (aicNum == 0) {
        aicNum = kDefaultAicNum;
    }
    if (aivNum == 0) {
        aivNum = kDefaultAivNum;
    }

    const uint64_t usedAicNum = std::max<uint64_t>(1, std::min(taskNum, aicNum));
    const uint64_t pairedAivNum = std::min<uint64_t>(aivNum, usedAicNum * 2);
    const uint64_t usedAivNum = std::max<uint64_t>(1, std::min<uint64_t>(std::max<uint64_t>(taskNum, pairedAivNum),
                                                                        pairedAivNum));
    uint32_t blockDim = static_cast<uint32_t>(usedAicNum);
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC platform(platformInfo);
        blockDim = platform.CalcTschBlockDim(static_cast<uint32_t>(usedAivNum), static_cast<uint32_t>(aicNum),
                                             static_cast<uint32_t>(aivNum));
        blockDim = std::max<uint32_t>(blockDim, static_cast<uint32_t>(usedAicNum));
    }
    if (blockDim == 0) {
        return ge::GRAPH_FAILED;
    }

    ChunkScaledDotKktTilingData tiling;
    tiling.set_B(b);
    tiling.set_H(h);
    tiling.set_T(t);
    tiling.set_K(k);
    tiling.set_BT(bt);
    tiling.set_NT(nt);
    tiling.set_taskNum(taskNum);
    tiling.set_usedAicNum(usedAicNum);
    tiling.set_usedAivNum(usedAivNum);
    tiling.set_btAlign(AlignUp(bt, kFp32BlockElems));
    tiling.set_scoreWorkspaceBytes(scoreBytes);
    if (BuildCubeTiling(bt, k, kDtype, tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(TilingKey(kDtype, bt));
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1);

    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = libApiWorkspace + scoreBytes;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkScaledDotKkt)
    .Tiling(TilingFunc);
}  // namespace optiling
