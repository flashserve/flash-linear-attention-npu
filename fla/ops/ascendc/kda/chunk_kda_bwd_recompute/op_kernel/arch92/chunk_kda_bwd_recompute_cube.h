/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_RECOMPUTE_ARCH92_CUBE_H
#define CHUNK_KDA_BWD_RECOMPUTE_ARCH92_CUBE_H

#include "chunk_kda_bwd_recompute_common.h"

namespace KDA {
using namespace AscendC;

template <typename QkType>
class ChunkKdaBwdRecomputeCubeProcess {
public:
    __aicore__ inline ChunkKdaBwdRecomputeCubeProcess(
        GM_ADDR a, GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR w, GM_ADDR u, GM_ADDR workspace)
        : cu_(cuSeqlens), indices_(chunkIndices)
    {
        gmA_.SetGlobalBuffer(reinterpret_cast<__gm__ QkType *>(a));
        gmW_.SetGlobalBuffer(reinterpret_cast<__gm__ QkType *>(w));
        gmU_.SetGlobalBuffer(reinterpret_cast<__gm__ QkType *>(u));
        (void)workspace;
    }

    __aicore__ inline void Init(const ChunkKdaBwdRecomputeTilingData *tiling)
    {
        using namespace KdaBwdRecomputeArch92;
        td_ = tiling;
        l1_ = LocalTensor<uint8_t>(TPosition::A1, 0, kL1Bytes);
        l0A_ = LocalTensor<QkType>(TPosition::A2, 0, kL0ABytes / sizeof(QkType));
        l0B_ = LocalTensor<QkType>(TPosition::B2, 0, kL0BBytes / sizeof(QkType));
        l0C_ = LocalTensor<float>(TPosition::CO1, 0, kL0CBytes / sizeof(float));
        for (uint32_t s = 0; s < kSlots; ++s) {
            SetFlag<HardEvent::MTE1_MTE2>(s);
        }
        SetFlag<HardEvent::FIX_MTE1>(0);
    }

    __aicore__ inline void Process()
    {
        using namespace KdaBwdRecomputeArch92;
        const uint64_t total = td_->chunkNum * td_->Hv;
        const uint64_t stride = td_->usedAic;
        uint64_t id = GetBlockIdx();
        uint64_t ordinal = 0;
        if (stride != 0 && id < stride && id < total) {
            WorkInfo current = GetWork(*td_, cu_, indices_, id, ordinal);
            CopyA(current);
            while (id < total) {
                const bool hasNext = id + stride < total;
                WorkInfo next{};
                if (hasNext) {
                    next = GetWork(*td_, cu_, indices_, id + stride, ordinal + 1);
                    CopyA(next);
                }
                Compute(current);
                current = next;
                id += stride;
                ++ordinal;
            }
        }
        // Drain initialized free tokens even on an empty/extra block.
        for (uint32_t s = 0; s < kSlots; ++s) {
            WaitFlag<HardEvent::MTE1_MTE2>(s);
        }
        WaitFlag<HardEvent::FIX_MTE1>(0);
    }

private:
    __aicore__ inline void CopyA(const KdaBwdRecomputeArch92::WorkInfo &work)
    {
        using namespace KdaBwdRecomputeArch92;
        const uint32_t s = work.slot;
        auto aL1 = l1_[AOffset(s)].template ReinterpretCast<QkType>();
        WaitFlag<HardEvent::MTE1_MTE2>(s);

        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = work.rows;
        p.dValue = work.rows;
        p.srcDValue = kRows;
        p.srcNdMatrixStride = 0;
        p.dstNzC0Stride = kRows;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;
        DataCopy(aL1, gmA_[work.aOffset], p);
        SetFlag<HardEvent::MTE2_MTE1>(s);
    }

    // Same A6 LoadData2DParamsV2 convention as Prepare's WuMatmulToL0C:
    // A NZ [64,64] -> L0A zz; RHS zN [64,128] -> L0B zn, transpose=true.
    __aicore__ inline void LoadOperand(LocalTensor<QkType> dst, LocalTensor<QkType> src, bool rhs)
    {
        LoadData2DParamsV2 p;
        p.mStartPosition = 0;
        p.kStartPosition = 0;
        p.mStep = 4;
        p.kStep = rhs ? 8 : 4;
        p.srcStride = 4;
        p.dstStride = rhs ? 8 : 4;
        p.ifTranspose = rhs;
        p.sid = 0;
        LoadData(dst, src, p);
    }

    __aicore__ inline void MmadAndStore(GlobalTensor<QkType> dst, uint32_t validRows)
    {
        using namespace KdaBwdRecomputeArch92;
        SetFlag<HardEvent::MTE1_M>(0);
        WaitFlag<HardEvent::MTE1_M>(0);
        MmadParams p;
        p.m = kRows;
        p.n = kChannels;
        p.k = kRows;
        p.cmatrixInitVal = true;
        p.cmatrixSource = false;
        p.unitFlag = 0;
        Mmad(l0C_, l0A_, l0B_, p);
        SetFlag<HardEvent::M_FIX>(0);
        WaitFlag<HardEvent::M_FIX>(0);

        // Prepare uses this SDK parameter spelling on 9201 as well. Output is
        // a complete tile owned by AIV0; GM only receives the valid rows.
        FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fix;
        fix.nSize = kChannels;
        fix.mSize = validRows;
        fix.srcStride = kRows;
        fix.dstStride = kChannels;
        fix.quantPre = QuantMode_t::F322BF16;
        fix.dualDstCtl = 0;
        fix.subBlockId = 0;
        fix.isChannelSplit = false;
        fix.params.ndNum = 1;
        fix.params.srcNdStride = 0;
        fix.params.dstNdStride = 0;
        Fixpipe<QkType, float, CFG_ROW_MAJOR>(dst, l0C_, fix);
        // One physical A/B/C set. Its next user waits for this Fixpipe,
        // which also proves the preceding MMAD has finished reading A/B.
        SetFlag<HardEvent::FIX_MTE1>(0);
    }

    __aicore__ inline void Compute(const KdaBwdRecomputeArch92::WorkInfo &work)
    {
        using namespace KdaBwdRecomputeArch92;
        const uint32_t s = work.slot;
        WaitFlag<HardEvent::FIX_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(s);
        LoadOperand(l0A_, l1_[AOffset(s)].template ReinterpretCast<QkType>(), false);
        SetFlag<HardEvent::MTE1_MTE2>(s);
        WaitRhsReady<PIPE_MTE1>(s);
        LoadOperand(l0B_, l1_[VbOffset(s)].template ReinterpretCast<QkType>(), true);
        MmadAndStore(gmU_[work.valueOffset], work.rows);

        WaitFlag<HardEvent::FIX_MTE1>(0);
        LoadOperand(l0B_, l1_[KbgOffset(s)].template ReinterpretCast<QkType>(), true);
        // Both L1 RHS tiles have now been read; U and W have no data dependency.
        SetRhsFree<PIPE_MTE1>(s);
        MmadAndStore(gmW_[work.valueOffset], work.rows);
    }

    GM_ADDR cu_;
    GM_ADDR indices_;
    const ChunkKdaBwdRecomputeTilingData *td_ = nullptr;
    GlobalTensor<QkType> gmA_, gmW_, gmU_;
    LocalTensor<uint8_t> l1_;
    LocalTensor<QkType> l0A_, l0B_;
    LocalTensor<float> l0C_;
};

} // namespace KDA

#endif
