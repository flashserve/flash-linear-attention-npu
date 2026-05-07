/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul.h
 * \brief
 */
#ifndef ASCEND_OPS_GROUPED_MATMUL_H
#define ASCEND_OPS_GROUPED_MATMUL_H

// Data type definitions
#define ORIG_DTYPE_X DT_BFLOAT16
#define DTYPE_X bfloat16_t

#define ORIG_DTYPE_WEIGHT DT_BFLOAT16
#define DTYPE_WEIGHT bfloat16_t

#define ORIG_DTYPE_Y DT_BFLOAT16
#define DTYPE_Y bfloat16_t

#define DTYPE_BIAS int
#define DTYPE_SCALE uint64_t
#define DTYPE_ANTIQUANT_SCALE half

#include "gmm/grouped_matmul/op_kernel/grouped_matmul_utils.h"
#include "gmm/grouped_matmul/op_kernel/grouped_matmul_tiling_key.h"


#include "grouped_matmul_tiling.h"

namespace GROUPED_MATMUL {

constexpr uint32_t thresholdBlockNum = 8;   // 8 is obtained by tests, indicating the threshold of basic block numbers
                                            // in both directions when assigning data blocks to cube cores when using
                                            // diagnal strategy
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
constexpr uint32_t thresholdDimM = 1;       // not needs any special strategies
#else
constexpr uint32_t thresholdDimM = 5;       // 5 is obtained by tests, indicating the threshold for distinguishing
                                            // strategies for large/small shapes
#endif

/*@brief store variables for core split configuration
*/
struct MNConfig {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t mIdx = 0;
    uint32_t nIdx = 0;
    uint32_t vecNIdx = 0;  // for A8W4 MSD NEW
    uint32_t blockDimM = 0;
    uint32_t blockDimN = 0;
    uint32_t vecBlockDimN = 0;  // for A8W4 MSD NEW
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t vecSingleN = 0;  // for A8W4 MSD NEW
    uint32_t offsetM = 0;  // for A8W4 MSD
    uint64_t wBaseOffset = 0;
    uint64_t nAxisBaseOffset = 0;
    uint64_t mAxisBaseOffset = 0;
    uint64_t xBaseOffset = 0;
    uint64_t yBaseOffset = 0;
    uint64_t wOutOffset = 0;
    uint64_t workSpaceOffset = 0;
    int64_t scaleIndex = -1;
};

__aicore__ inline void MNBlockIdxCompute(MNConfig &mnConfig, const uint32_t curBlock,
    const uint32_t count, const uint32_t thresholdM_dimN) {
    if (mnConfig.blockDimM <= thresholdDimM || thresholdDimM == 1) {
        mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
        mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
    } else {
        uint32_t relativeBlock = curBlock - count;
        uint32_t curThresholdM = relativeBlock >= AlignDown(mnConfig.blockDimM * mnConfig.blockDimN, thresholdM_dimN) ?
            mnConfig.blockDimM % thresholdBlockNum : thresholdBlockNum;
        uint32_t curThresholdM_thresholdN = curThresholdM * thresholdBlockNum;
        uint32_t curThresholdN = relativeBlock % thresholdM_dimN >= AlignDown(curThresholdM * mnConfig.blockDimN,
            curThresholdM_thresholdN) ? mnConfig.blockDimN % thresholdBlockNum : thresholdBlockNum;

        uint32_t localRelativeBlock = relativeBlock % thresholdM_dimN % curThresholdM_thresholdN;
        mnConfig.mIdx = localRelativeBlock % curThresholdM + relativeBlock / thresholdM_dimN * thresholdBlockNum;
        mnConfig.nIdx = (localRelativeBlock + localRelativeBlock /
            LeastCommonMultiple(curThresholdM, curThresholdN)) % curThresholdN + relativeBlock %
            thresholdM_dimN / curThresholdM_thresholdN * thresholdBlockNum;
    }
}

/** @brief GroupMatmul operator Class
*/
template <typename ComputeType>
class GMMProcess {
 protected:
    using B = typename ComputeType::B;
    ComputeType& computeOp;   // inernal computation operator
    const GMMBaseParams* __restrict gmmBaseParams;
    const TCubeTiling* __restrict mmTilingData;

    uint32_t blockIdx;
    uint32_t coreIdx;
    uint32_t groupNum;
    int32_t preOffset = 0;
    GM_ADDR groupListPtr;
    GlobalTensor<int64_t> groupListGm;
    int32_t* mListGm;
    int32_t* kListGm;
    int32_t* nListGm;
    uint32_t baseM_ = 0;
    uint32_t baseN_ = 0;

 public:
    /** @brief constructor */
    __aicore__ inline GMMProcess(ComputeType& computeOp_) : computeOp(computeOp_) {}

    __aicore__ inline void Init(const GMMBaseParams* __restrict gmmBaseParamsIn,
                                const TCubeTiling* __restrict mmTilingDataIn, int32_t* gmmArrayAddrIn,
                                GM_ADDR groupList);

    __aicore__ inline void Process();

 protected:
    __aicore__ inline void SetMNConfig(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig);

    __aicore__ inline void SetMKN(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig);

    __aicore__ inline void UpdateMnConfig(MNConfig &mnConfig);
};

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Init(const GMMBaseParams* __restrict gmmBaseParamsIn,
    const TCubeTiling* __restrict mmTilingDataIn, int32_t* gmmArrayAddrIn, GM_ADDR groupList) {
    blockIdx = GetBlockIdx();
    coreIdx = blockIdx;
    int64_t coreRation = GetTaskRation();
    if (coreRation > 1) {
        coreIdx /= coreRation;
    }
    gmmBaseParams = gmmBaseParamsIn;
    mmTilingData = mmTilingDataIn;
    groupNum = gmmBaseParams->groupNum;
    groupListPtr = groupList;
    if (groupListPtr != nullptr) {
        groupListGm.SetGlobalBuffer((__gm__ int64_t*)groupList);
    }
    mListGm = gmmArrayAddrIn;
    kListGm = gmmArrayAddrIn + MKN_LIST_LEN;
    nListGm = gmmArrayAddrIn + MKN_LIST_LEN * 2;
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::SetMNConfig(const int32_t splitValue, const uint32_t groupIdx, MNConfig &mnConfig) {
    SetMKN(splitValue, groupIdx, mnConfig);
    if (mmTilingData != nullptr) {
        mnConfig.baseM = mmTilingData->baseM;
        mnConfig.baseN = mmTilingData->baseN;
    } else {
        mnConfig.baseM = baseM_;
        mnConfig.baseN = baseN_;
    }
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::SetMKN(const int32_t splitValue, const uint32_t groupIdx,
                                                       MNConfig &mnConfig) {
    uint32_t singleWeight = gmmBaseParams->singleWeight;
    uint32_t singleX = gmmBaseParams->singleX;
    uint32_t singleY = gmmBaseParams->singleY;
    bool isAllSingleTensor = singleWeight == 1 && singleX == 1 && singleY == 1;
    uint32_t valueIdx = isAllSingleTensor ? 0 : groupIdx;
    if (mmTilingData == nullptr) {
        mnConfig.m = splitValue;
        mnConfig.k = gmmBaseParams->k;
        mnConfig.n = gmmBaseParams->n;
        return;
    }

    if (gmmBaseParams->groupType == 0) {
        mnConfig.m = splitValue;
        mnConfig.k = kListGm[valueIdx];
        mnConfig.n = nListGm[valueIdx];
        return;
    }

    if (gmmBaseParams->groupType == 2) {
        mnConfig.m = mListGm[valueIdx];
        mnConfig.k = splitValue;
        mnConfig.n = nListGm[valueIdx];
        return;
    }

    mnConfig.m = mListGm[groupIdx];
    mnConfig.k = kListGm[groupIdx];
    mnConfig.n = nListGm[groupIdx];
    return;
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::UpdateMnConfig(MNConfig &mnConfig) {
    if constexpr (B::format == CubeFormat::NZ) {
        mnConfig.wBaseOffset += AlignUp<16>(mnConfig.k) * AlignUp<16>(mnConfig.n);  // 16: nz format last two dim size
    } else {
        mnConfig.wBaseOffset += mnConfig.k * mnConfig.n;
    }
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.scaleIndex++;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += mnConfig.m * mnConfig.k;
    mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
}

template <typename ComputeType>
__aicore__ inline void GMMProcess<ComputeType>::Process() {
    MNConfig mnConfig;
    if (gmmBaseParams->groupType != -1) {  // -1: no split
        if (unlikely(groupListPtr == nullptr)) {
            return;
        }
        preOffset = 0;
    }
    AscendC::WaitPreTaskEnd();
    for (uint32_t groupIdx = 0, count = 0; groupIdx < groupNum; ++groupIdx) {
        UpdateMnConfig(mnConfig);
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, preOffset, gmmBaseParams, groupListGm);
        SetMNConfig(splitValue, groupIdx, mnConfig);
        if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
            continue;
        }
        mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.singleM);
        mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.singleN);

        uint32_t curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
        uint32_t curBlock = coreIdx >= count ? coreIdx : coreIdx + gmmBaseParams->coreNum;
        uint32_t thresholdM_dimN = thresholdBlockNum * mnConfig.blockDimN;

        while (curBlock < curCount) {
            MNBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN);
            computeOp.MMCompute(groupIdx, mnConfig, coreIdx);
            computeOp.VectorCompute(mnConfig);
            curBlock += gmmBaseParams->coreNum;
        }
        count = curCount % gmmBaseParams->coreNum;
    }
    AscendC::SetNextTaskStart();
}

/** @brief intenal computation class
*/
template <class mmType, bool sync = false>
class GMMCompute {
 public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using B = typename mmType::BT;
    using CT = typename mmType::CT::T;
    using BiasT = typename mmType::BiasT::T;
    using WT = DTYPE_WEIGHT;
    constexpr static bool transposeX = mmType::AT::isTrans;
    constexpr static bool transposeW = mmType::BT::isTrans;

    /** @brief constructor */
    __aicore__ inline GMMCompute(typename mmType::MT& mm_) : mm(mm_) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
                                GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR groupList,
                                GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
                                const GMMBaseParams* __restrict gmmBaseParams,
                                const TCubeTiling* __restrict mmTilingData, TPipe* tPipe);

    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig, uint32_t coreIdx);
    __aicore__ inline void VectorCompute(MNConfig& mnConfig) {}

    __aicore__ inline GlobalTensor<BT> SetGlobalBufferW(uint32_t groupIdx, uint32_t tailN, MNConfig& mnConfig);

    __aicore__ inline uint64_t SetWOffset(uint32_t tailN, uint32_t k);

 protected:
    TPipe* pipe;
    typename mmType::MT& mm;  // matmul operator
    bool hasBias = false;
    GM_ADDR xTensorPtr;
    GM_ADDR weightTensorPtr;
    GM_ADDR biasTensorPtr;
    GM_ADDR yTensorPtr;
    GlobalTensor<AT> xGm;
    GlobalTensor<BT> weightGm;
    GlobalTensor<BiasT> biasGm;
    GlobalTensor<DTYPE_Y> yGm;

    uint32_t ubBaseN;
    uint32_t ubBaseK;
    uint32_t ubCalSize;
    uint32_t singleWeight;
    uint32_t singleX;
    uint32_t singleY;
    uint32_t coreNum;
    uint32_t subBlockIdx;
    bool mmWaitStatus;
    uint32_t activeType;
};

template <typename mmType, bool sync>
__aicore__ inline void GMMCompute<mmType, sync>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                                      GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                                      GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                                      GM_ADDR workspace, const GMMBaseParams* __restrict gmmBaseParams,
                                                      const TCubeTiling* __restrict mmTilingData,
                                                      TPipe* tPipe) {
    xTensorPtr = x;
    weightTensorPtr = weight;
    biasTensorPtr = bias;
    yTensorPtr = y;
    pipe = tPipe;
    ubBaseN = gmmBaseParams->ubBaseN;
    ubBaseK = gmmBaseParams->ubBaseK;
    ubCalSize = gmmBaseParams->ubCalSize;
    singleWeight = gmmBaseParams->singleWeight;
    singleX = gmmBaseParams->singleX;
    singleY = gmmBaseParams->singleY;
    coreNum = gmmBaseParams->coreNum;
    subBlockIdx = GetSubBlockIdx();
    if (mmTilingData != nullptr) {
        hasBias = mmTilingData->isBias != 0;
    }
    activeType = gmmBaseParams->activeType;
    mmWaitStatus = false;
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    TBuf<> ubBuf;
    pipe->InitBuffer(ubBuf, TOTAL_UB_SIZE / 2);
    LocalTensor<uint8_t> buf = ubBuf.template Get<uint8_t>();
    mm.SetLocalWorkspace(buf);
#endif
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (gmmBaseParams->isOutputDisableL2Cache != 0) {
        yGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
#endif
}

template <typename mmType, bool sync>
__aicore__ inline uint64_t GMMCompute<mmType, sync>::SetWOffset(uint32_t tailN, uint32_t k) {
    uint64_t wOffset = 0;
    if constexpr (mmType::BT::format == CubeFormat::NZ && transposeW) {
        wOffset = tailN * (UB_BLOCK_UNIT_SIZE / sizeof(BT));  // 32: quant is 32, float16 is 16
    } else if constexpr (mmType::BT::format == CubeFormat::NZ) {
        wOffset = tailN * AlignUp<16>(k);  // 16: nz format last two dim size
    } else if constexpr (transposeW) {
        wOffset = tailN * k;
    } else {
        wOffset = tailN;
    }
    return wOffset;
}

template <typename mmType, bool sync>
__aicore__ inline GlobalTensor<typename mmType::BT::T> GMMCompute<mmType, sync>::SetGlobalBufferW(
        uint32_t groupIdx, uint32_t tailN, MNConfig& mnConfig) {
    uint64_t wOffset = SetWOffset(tailN, mnConfig.k);

    GlobalTensor<BT> weightGmLocal;
    weightGmLocal.SetGlobalBuffer((__gm__ bfloat16_t *)weightTensorPtr + mnConfig.wBaseOffset + wOffset);
    return weightGmLocal;

}

template <typename mmType, bool sync>
__aicore__ inline void GMMCompute<mmType, sync>::MMCompute(uint32_t groupIdx, MNConfig& mnConfig, uint32_t coreIdx) {
    if (subBlockIdx != 0) {
        return;
    }
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM = mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM
                                                                 : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    if constexpr (transposeX) {
        xOffset = mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    xGm.SetGlobalBuffer((__gm__ bfloat16_t *) xTensorPtr);
    GlobalTensor<BT> weightGmLocal = SetGlobalBufferW(groupIdx, tailN, mnConfig);
    mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGm[xOffset], false);
    mm.SetTensorB(weightGmLocal, transposeW);

    yGm.SetGlobalBuffer((__gm__ bfloat16_t *)yTensorPtr + mnConfig.yBaseOffset);
    mm.template IterateAll<sync>(yGm[outOffset], 0);
}

}  // namespace GROUPED_MATMUL
#endif  // ASCEND_OPS_GROUPED_MATMUL_H
