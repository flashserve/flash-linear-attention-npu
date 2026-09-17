#ifndef GDN_PHASE6_SOLVE_LAYOUT_STAGING_H
#define GDN_PHASE6_SOLVE_LAYOUT_STAGING_H

#include "kernel_operator.h"

namespace NsPhase6SolveLayoutStaging {

template <typename T, typename TilingData>
__aicore__ inline void TransposeBhtTnd(GM_ADDR source, GM_ADDR destination,
                                       const TilingData *tiling, bool bhtToTnd)
{
    if ASCEND_IS_AIC {
        return;
    }

    const int64_t vectorIndex = static_cast<int64_t>(AscendC::GetBlockIdx());
    const int64_t vectorCount = static_cast<int64_t>(tiling->usedAivNum);
    if (vectorCount <= 0 || vectorIndex >= vectorCount) {
        return;
    }

    const int64_t batchCount = static_cast<int64_t>(tiling->B);
    const int64_t headCount = static_cast<int64_t>(tiling->numHeads);
    const int64_t tokenCount = static_cast<int64_t>(tiling->T);
    const int64_t chunkSize = static_cast<int64_t>(tiling->BT);
    const int64_t rowCount = batchCount * headCount * tokenCount;
    const int64_t elementCount = rowCount * chunkSize;

    AscendC::GlobalTensor<T> sourceGm;
    AscendC::GlobalTensor<T> destinationGm;
    sourceGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(source), elementCount);
    destinationGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(destination), elementCount);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowBuffer;
    const uint32_t rowBytes = static_cast<uint32_t>(chunkSize * sizeof(T));
    const uint32_t alignedRowBytes = (rowBytes + 31U) / 32U * 32U;
    pipe.InitBuffer(rowBuffer, alignedRowBytes);
    AscendC::LocalTensor<T> rowLocal = rowBuffer.Get<T>();

    AscendC::DataCopyExtParams rowParams{1, rowBytes, 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    const event_t mte2ToMte3 =
        static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    const event_t mte3ToMte2 =
        static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE3_MTE2));

    for (int64_t row = vectorIndex; row < rowCount; row += vectorCount) {
        const int64_t token = row % tokenCount;
        const int64_t head = (row / tokenCount) % headCount;
        const int64_t batch = row / (tokenCount * headCount);
        const int64_t bhtOffset = ((batch * headCount + head) * tokenCount + token) * chunkSize;
        const int64_t tndOffset = ((batch * tokenCount + token) * headCount + head) * chunkSize;
        const int64_t sourceOffset = bhtToTnd ? bhtOffset : tndOffset;
        const int64_t destinationOffset = bhtToTnd ? tndOffset : bhtOffset;

        AscendC::DataCopyPad(rowLocal, sourceGm[sourceOffset], rowParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(mte2ToMte3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mte2ToMte3);
        AscendC::DataCopyPad(destinationGm[destinationOffset], rowLocal, rowParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2);
    }

    pipe.ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(mte2ToMte3);
    pipe.ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(mte3ToMte2);
}

}  // namespace NsPhase6SolveLayoutStaging

#endif  // GDN_PHASE6_SOLVE_LAYOUT_STAGING_H
