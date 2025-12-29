/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_full_mesh.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_H
#define MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "moe_distribute_dispatch_v2_tiling.h"
#include "../common/inc/kernel/moe_distribute_base.h"
#if __has_include("../moe_distribute_dispatch/check_winsize.h")
#include "../moe_distribute_dispatch/check_winsize.h"
#else
#include "../../moe_distribute_dispatch/op_kernel/check_winsize.h"
#endif

namespace MoeDistributeDispatchV2FullMeshImpl {
constexpr uint8_t BUFFER_NUM = 2;        // 多buf
constexpr uint32_t STATE_OFFSET = 32U;  // 状态空间偏移地址
constexpr uint32_t UB_ALIGN = 32U;       // UB按32字节对齐
constexpr uint8_t COMM_NUM = 2;  // 通信域大小
constexpr uint8_t COMM_EP_IDX = 0;
constexpr uint64_t WIN_STATE_OFFSET = 500UL * 1024UL;
constexpr uint64_t STATE_WIN_OFFSET = 950UL * 1024UL;
constexpr uint64_t CUMSUM_FLAG_OFFSET = 1000UL * 1024UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t SPLIT_BLOCK_SIZE = 512UL;
constexpr uint32_t SYNC_OFFSET = 3U * 1024U; // 核间软同步偏移地址 
constexpr uint32_t EXPAND_IDX_INFO = 3U;  // expand_idx是按3元组保存信息，分别为rank_id token_id topk_id
constexpr int32_t  MAX_UB_SIZE = 190 * 1024;
constexpr uint32_t COMPARE_COUNT_PER_BLOCK = 256 / sizeof(int32_t);
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t AIV_STATE_SIZE = 64U;
constexpr uint32_t SIZE_ALIGN_256 = 256U;
constexpr AscendC::CumSumConfig cumSumConfig{true, true, false};
template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist, bool IsNeedAllgather
#define TemplateMC2TypeFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist, IsNeedAllgather

using namespace AscendC;
template <TemplateMC2TypeClass>
class MoeDistributeDispatchV2FullMesh {
public:
    __aicore__ inline MoeDistributeDispatchV2FullMesh() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR elasticInfo, 
                                GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, 
                                GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void activeMaskProc();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void SetDataStatus();
    __aicore__ inline void SetTilingDataAndCal(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void SendToSharedExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf);
    __aicore__ inline void SendToMoeExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> expertMaskBuf, TBuf<> outBuf);
    __aicore__ inline void CalExpertSendNum(TBuf<> outBuf, TBuf<> expertMaskBuf);
    __aicore__ inline void AllToAllDispatch();
    __aicore__ inline void CalCumSum();
    __aicore__ inline void WaitCumSumFlag();
    __aicore__ inline void CalAndSendCnt();
    __aicore__ inline void WaitTokenNumAndCopy();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum);
    __aicore__ inline uint32_t CheckDataArriveWithFlag(uint32_t srcExpDataIdx, int32_t beginIdx, int32_t copyCnt);
    __aicore__ inline void CopyInAndOut(LocalTensor<float> xOutFp32Tensor, LocalTensor<int32_t> xOutInt32Tensor,
                                        GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount);
    __aicore__ inline void WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum);
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& srcLocal, uint32_t count);
    __aicore__ inline void QuantProcess(LocalTensor<XType> xInTensor, uint32_t expertIndex);
    __aicore__ inline void UpdataTokenNumsOut(LocalTensor<int32_t> cumSumTensor);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void SplitExpertNumToCore(uint32_t &delCurExpertGroupNum, uint32_t &groupIdx);
    __aicore__ inline void FillTriple(LocalTensor<ExpandXOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void TokenToExpertInQuant(GlobalTensor<ExpandXOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue,
                                                uint32_t srcTokenIndex, uint32_t toExpertId, uint32_t toExpertIndex);
    __aicore__ inline void TokenToExpert(GlobalTensor<ExpandXOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue,
                                        uint32_t srcTokenIndex, uint32_t toExpertIndex);

    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        if (rankId == epRankId_) {
            return (GM_ADDR)(winContext_[COMM_EP_IDX]->localWindowsIn) + winDataSizeOffset_;
        }
        return (GM_ADDR)(((HcclRankRelationResV2*)(winContext_[COMM_EP_IDX]->remoteRes[rankId].nextDevicePtr))->windowsIn)
                         + winDataSizeOffset_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        if (rankId == epRankId_) {
            return (GM_ADDR)(winContext_[COMM_EP_IDX]->localWindowsExp) + dataState_ * WIN_STATE_OFFSET;
        }
        return (GM_ADDR)(((HcclRankRelationResV2*)(winContext_[COMM_EP_IDX]->remoteRes[rankId].nextDevicePtr))->windowsExp)
                         + dataState_ * WIN_STATE_OFFSET;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<ExpandXOutType> winTpGatherOutGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<int32_t> elasticInfoGMTensor_;

    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> workLocalTensor_;
    LocalTensor<int32_t> validExpertIdsTensor_;
    LocalTensor<int32_t> tokenNumToExpertTensor_;
    LocalTensor<float> cumSumTime1Tensor_;
    LocalTensor<float> cumSumTime2Tensor_;
    LocalTensor<float> tempTime1Tensor_;
    LocalTensor<float> tempTime2Tensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> statusSumOutTensor_;
    LocalTensor<int32_t> cleanStatusTensor_;
    LocalTensor<uint32_t> gatherTmpTensor_;
    LocalTensor<uint8_t> sharedTmpBufTensor_;
    LocalTensor<float> syncOnCoreTensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<float> statusCleanFp32Tensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<ExpandXOutType> outTensor_;
    LocalTensor<ExpandXOutType> tempTensor_;
    LocalTensor<float> floatLocalAbsTemp_;
    LocalTensor<float> floatLocalTemp_;
    LocalTensor<uint32_t> expertMapTensor_;
    LocalTensor<uint32_t> expertFinishNumTensor_;
    LocalTensor<uint32_t> expertLeftNumTensor_;
    LocalTensor<uint8_t> flagCompResultU8_;
    LocalTensor<uint64_t> flagCompResultLtU64_;
    LocalTensor<uint32_t> flagRecvGatherMask_;
    LocalTensor<ExpandXOutType> xTmpTensor_;

    LocalTensor<float> flagGatherOutTensor_;
    LocalTensor<float> flagRecvTensor_;

    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> expertIdsBuf_;
    GM_ADDR expandXOutGM_;
    GM_ADDR sendCountsOutGM_;
    GM_ADDR sendTpCountOutGM_;
    GM_ADDR statusSpaceGM_;
    GM_ADDR windowGM_;
    GM_ADDR recvCntWorkspaceGM_;
    
    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    int32_t epRankId_{0};
    uint32_t aivId_{0};           // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};     // 共享专家卡数
    uint32_t rankNumPerSharedExpert_{0};  // 部署单个共享专家所用的卡数
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertRankNum_{0};  // moe专家卡数，等于epWorldSize_ - sharedExpertRankNum_
    uint32_t moeExpertNumPerRank_{0};
    uint32_t totalExpertNum_{0};
    uint32_t dealRankPerCore_{0};
    uint32_t hOutSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t startId_;
    uint32_t endId_;
    uint32_t sendNum_;
    uint32_t statusCntAlign_;
    uint32_t lastCore_{0};
    uint32_t dataState_{0};
    uint32_t tBufRealSize_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t sendToMoeExpTokenCnt_{0};
    uint64_t flagPadOffset_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    bool hasElasticInfoFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    uint64_t totalWinSize_{0};
    uint32_t gatherCount_{0};
    uint32_t expertTokenNumsType_{1};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    int32_t zeroComputeExpertNum_{0};
    uint32_t axisHExpandXAlignSize_{0};
    uint32_t blockCntPerToken_{0};
    uint32_t axisHCommu_{0};
    uint32_t hCommuSize_{0};
    uint32_t maskSizePerExpert_{0};
    uint32_t expertIdsBufSize_{0};
    uint32_t cumSumUBMinValue_{0};
    uint32_t rscvStatusNum_{0};
    uint64_t cumSumUB_{0};
    uint32_t cumSumTimes_{0};
    uint32_t delLastExpertId_{0};
    uint32_t remainderExpertNum_{0};
    uint32_t aivUsedCumSum_{0};
    uint32_t aivUsedAllToAll_{0};
    __gm__ HcclOpResParam *winContext_[COMM_NUM]{nullptr, nullptr};

    DataCopyExtParams expandXCopyParams_;
    DataCopyExtParams hCopyParams_;
};


template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::SetTilingDataAndCal(
    const MoeDistributeDispatchV2TilingData *tilingData)
{   
    axisBS_ = tilingData->moeDistributeDispatchV2Info.bs;
    axisH_ = tilingData->moeDistributeDispatchV2Info.h;
    hasElasticInfoFlag_ = tilingData->moeDistributeDispatchV2Info.hasElasticInfo;
    epRankId_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    epWorldSize_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    moeExpertNum_ = tilingData->moeDistributeDispatchV2Info.moeExpertNum;
    axisMaxBS_ = tilingData->moeDistributeDispatchV2Info.globalBs / epWorldSize_;
    sharedExpertNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertNum;
    expertTokenNumsType_ = tilingData->moeDistributeDispatchV2Info.expertTokenNumsType;
    zeroComputeExpertNum_ = tilingData->moeDistributeDispatchV2Info.zeroComputeExpertNum;
    isTokenMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isTokenMask;
    isExpertMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isExpertMask;
    axisK_ = tilingData->moeDistributeDispatchV2Info.k;
    aivNum_ = tilingData->moeDistributeDispatchV2Info.aivNum;
    cumSumUBMinValue_ = tilingData->moeDistributeDispatchV2Info.cumSumUBMinValue;
    isShareExpertRankFlag_ = (epRankId_ < sharedExpertRankNum_);
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;
    lastCore_ = aivNum_ - 1;
    expertIdsCnt_ = axisBS_ * axisK_;
    hOutSize_ = axisH_ * sizeof(ExpandXOutType);
    axisHExpandXAlignSize_ = Ceil(axisH_ * sizeof(ExpandXOutType), UB_ALIGN) * UB_ALIGN;
    tokenQuantAlign_ = axisHExpandXAlignSize_ / sizeof(int32_t);
    hOutSizeAlign_ = axisHExpandXAlignSize_ + UB_ALIGN * BUFFER_NUM;
    blockCntPerToken_ = Ceil(hOutSizeAlign_, SPLIT_BLOCK_DATA_SIZE);
    hCommuSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;
    axisHCommu_ = hCommuSize_ / sizeof(ExpandXOutType);
    expertPerSizeOnWin_ = axisMaxBS_ * hCommuSize_;
    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    statusCntAlign_ = Ceil(totalExpertNum_, 8) * 8;   // 8 = UB_ALIGN / sizeof(int32_t)
    aivUsedCumSum_ = totalExpertNum_ / 32; // 单核处理32个专家cnt发送
    aivUsedCumSum_ = (aivUsedCumSum_ == 0) ? 1 : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= (aivNum_ / 2)) ? (aivNum_ / 2) : aivUsedCumSum_;
    aivUsedAllToAll_ = aivNum_ - aivUsedCumSum_;
    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = (aivUsedAllToAll_ * sharedExpertNum_) / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    moeUsedAivNum_ = aivUsedAllToAll_ - sharedUsedAivNum_;
    rscvStatusNum_ = isShareExpertRankFlag_ ? epWorldSize_ : (epWorldSize_ * moeExpertNumPerRank_);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::SetDataStatus()
{
    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGM = (GM_ADDR)(winContext_[COMM_EP_IDX]->localWindowsExp);
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t*)(statusDataSpaceGM + STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    dataState_ = selfDataStatusTensor(0);
    if (dataState_ == 0) {
        selfDataStatusTensor(0) = 1;
    } else {
        selfDataStatusTensor(0) = 0;
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfDataStatusTensor);
    PipeBarrier<PIPE_ALL>();
    uint64_t hSizeAlignCombine = Ceil(axisH_ * sizeof(XType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    winDataSizeOffset_ = dataState_ * (totalWinSize_ / BUFFER_NUM)
                         + axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine;
}


template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR elasticInfo, GM_ADDR expandXOut, 
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut,
    GM_ADDR workspaceGM, TPipe *pipe, const MoeDistributeDispatchV2TilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_[COMM_EP_IDX] = (__gm__ HcclOpResParam*)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    totalWinSize_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeEp);
    auto realWinSize = winContext_[COMM_EP_IDX]->winSize;
    CheckWindowSize(totalWinSize_, realWinSize, tpipe_, expandXOut);
    SetTilingDataAndCal(tilingData);
    SetDataStatus();

    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ bool*)xActiveMask);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float*)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t*)expertTokenNumsOut);
    expandIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)(expandIdxOut));
    elasticInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t*)(elasticInfo));
    scalesGMTensor_.SetGlobalBuffer((__gm__ float*)scales);
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut;
    sendTpCountOutGM_ = tpSendCountsOut;
    recvCntWorkspaceGM_ = workspaceGM;
    statusSpaceGM_ = GetWindStateAddrByRankId(epRankId_);
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float*)(statusSpaceGM_));
    windowGM_ = GetWindAddrByRankId(epRankId_);
    hCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(XType)), 0U, 0U, 0U};
    expandXCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(ExpandXOutType)), 0U, 0U, 0U};
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::ReduceMaxInplace(const LocalTensor<float>& srcLocal,
    uint32_t count)
{
    uint64_t repsFp32 = count >> 6;        // 6 is count / elemPerRefFp32
    uint64_t offsetsFp32 = repsFp32 << 6;  // 6 is repsFp32 * elemPerRefFp32
    uint64_t remsFp32 = count & 0x3f;      // 0x3f = 63, count % elemPerRefFp32
    const uint64_t elemPerRefFp32 = 64UL;  // 64 = 256 bit / sizeof(float)
    if (likely(repsFp32 > 1)) {
        Max(srcLocal, srcLocal[elemPerRefFp32], srcLocal, elemPerRefFp32, repsFp32 - 1, {1, 1, 1, 0, 8, 0}); // 8 is rep stride
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0}); // 8 is rep stride
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? elemPerRefFp32 : count;
    WholeReduceMax(srcLocal, srcLocal, mask, 1, 8, 1, 8);  // 8 is rep stride
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::FillTriple(
    LocalTensor<ExpandXOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k)
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    xOutTint32(tokenQuantAlign_ + 8) = epRankId_;        // 8 = UB_ALIGN / sizeof(int32_t) 0:epRankId index
    xOutTint32(tokenQuantAlign_ + 8 + 1) = tokenIndex;   // 8 = UB_ALIGN / sizeof(int32_t) 1:token index
    xOutTint32(tokenQuantAlign_ + 8 + 2) = k;            // 8 = UB_ALIGN / sizeof(int32_t) 2:topK value index
    SyncFunc<AscendC::HardEvent::S_MTE3>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::QuantProcess(LocalTensor<XType> xInTensor,
    uint32_t expertIndex)
{
    float dynamicScale = 0.0;
    Cast(floatLocalTemp_, xInTensor, RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    if constexpr (IsSmoothScaleExist) {
        if constexpr (DynamicQuant) {
            SyncFunc<AscendC::HardEvent::V_MTE2>(); // ub复用，循环同步
        }
        DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(axisH_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadExtParams{false, 0U, 0U, 0U};
        DataCopyPad(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], scalesCopyInParams, copyPadExtParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Mul(floatLocalTemp_, floatLocalTemp_, smoothScalesTensor_, axisH_);
        PipeBarrier<PIPE_V>();
    }

    if constexpr (DynamicQuant) {
        Abs(floatLocalAbsTemp_, floatLocalTemp_, axisH_);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(floatLocalAbsTemp_, axisH_);

        SyncFunc<AscendC::HardEvent::V_S>();
        dynamicScale = float(127.0) / floatLocalAbsTemp_.GetValue(0);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(floatLocalTemp_, floatLocalTemp_, dynamicScale, axisH_);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<half> halfLocalTemp = floatLocalTemp_.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = floatLocalTemp_.ReinterpretCast<int32_t>();
    Cast(int32LocalTemp, floatLocalTemp_, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();

    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);

    PipeBarrier<PIPE_V>();
    Cast(tempTensor_, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> floatTempTensor = tempTensor_.template ReinterpretCast<float>();
    floatTempTensor.SetValue(tokenQuantAlign_, float(1.0) / dynamicScale);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::TokenToExpertInQuant(GlobalTensor<ExpandXOutType> dstWinGMTensor,
    TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex, uint32_t fillExpertIdx, uint32_t quantExpertIdx)
{
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    LocalTensor<XType> xInTensor = inQueue.AllocTensor<XType>();
    DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * axisH_], hCopyParams_, copyPadExtParams);
    inQueue.EnQue(xInTensor);
    xInTensor = inQueue.DeQue<XType>();
    QuantProcess(xInTensor, quantExpertIdx);
    FillTriple(tempTensor_, srcTokenIndex, fillExpertIdx);
    inQueue.FreeTensor<XType>(xInTensor);
    SyncFunc<AscendC::HardEvent::S_V>();
    LocalTensor<int32_t> tempTensorInt32 = tempTensor_.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> outTensorInt32 = outTensor_.template ReinterpretCast<int32_t>();
    // 64 = 256 / sizeof(int32_t) 一次操作字节数; 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32[flagPadOffset_ / sizeof(int32_t)], tempTensorInt32, uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15}); 
    // 64：偏移前一次拷贝的256字节； 56 = （480 - 256） / sizeof(int32_t); 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32[flagPadOffset_ / sizeof(int32_t) + 64], tempTensorInt32[64], uint64_t(56), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(dstWinGMTensor, outTensor_[flagPadOffset_ / sizeof(ExpandXOutType)], axisHCommu_);
    flagPadOffset_ = hCommuSize_ - flagPadOffset_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::TokenToExpert(GlobalTensor<ExpandXOutType> dstWinGMTensor,
    TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex, uint32_t toExpertIndex)
{
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    LocalTensor<XType> xInTensor = inQueue.AllocTensor<XType>();
    DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * axisH_], expandXCopyParams_, copyPadExtParams);
    inQueue.EnQue(xInTensor);
    xInTensor = inQueue.DeQue<XType>();
    FillTriple(xInTensor, srcTokenIndex, toExpertIndex);
    SyncFunc<AscendC::HardEvent::S_V>();
    // 128 = 256 / sizeof(ExpandXOutType) 一次操作字节数; 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensor_[flagPadOffset_ / sizeof(ExpandXOutType)], xInTensor, uint64_t(128), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    // 128：偏移前一次拷贝的256字节； 112 = （480 - 256） / sizeof(ExpandXOutType); 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensor_[flagPadOffset_ / sizeof(ExpandXOutType) + 128], xInTensor[128], uint64_t(112), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(dstWinGMTensor, outTensor_[flagPadOffset_ / sizeof(ExpandXOutType)], axisHCommu_);
    flagPadOffset_ = hCommuSize_ - flagPadOffset_;
    inQueue.FreeTensor<XType>(xInTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
    uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;                // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;  // 余数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else if (aivId_ >= aivUsedAllToAll_) { // aiv中后面aivUsedCumSum_个核给cusum计算使用
        newAivId = aivId_ - aivUsedAllToAll_;
    } else {
        newAivId = aivId_ - moeUsedAivNum_; // aivUsedAllToAll_中后面的核发送给共享专家
    }
    startTokenId = sendTokenNum * newAivId;  // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {      // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::SendToSharedExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf)
{
    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * BUFFER_NUM / sizeof(float));
    PipeBarrier<PIPE_V>();
    // 分核
    uint32_t startTokenId, endTokenId, sendTokenNum;
    uint32_t curSendCnt = activeMaskBsCnt_ * sharedExpertNum_; // 参数 validBsCnt_、sharedExpertNum_、sharedUsedAivNum_
    SplitToCore(curSendCnt, sharedUsedAivNum_, startTokenId, endTokenId, sendTokenNum, false);
    if (startTokenId >= curSendCnt) {return;}
    // 发送
    GlobalTensor<ExpandXOutType> dstWinGMTensor;
    uint32_t idInSharedGroup = epRankId_ % rankNumPerSharedExpert_;  // 计算目的共享专家卡在其所在共享专家组的id
    for (uint32_t virtualTokenIndex = startTokenId; virtualTokenIndex < endTokenId; ++virtualTokenIndex) {
        uint32_t sendTokenIndex = virtualTokenIndex % activeMaskBsCnt_;
        uint32_t toSharedExpertIndex = virtualTokenIndex / activeMaskBsCnt_;
        int32_t toRankId = idInSharedGroup + toSharedExpertIndex * rankNumPerSharedExpert_;
        dstWinGMTensor.SetGlobalBuffer(
            (__gm__ ExpandXOutType*)(GetWindAddrByRankId(toRankId) + expertPerSizeOnWin_ * epRankId_ + sendTokenIndex * hCommuSize_));
        uint32_t srcTokenIndex = sendTokenIndex;
        if constexpr (DynamicQuant || StaticQuant) {
            uint32_t fillExpertIdx = axisK_ + toSharedExpertIndex;
            uint32_t quantExpertIdx = toSharedExpertIndex;
            TokenToExpertInQuant(dstWinGMTensor, inQueue, srcTokenIndex, fillExpertIdx, quantExpertIdx);
        } else {
            TokenToExpert(dstWinGMTensor, inQueue, srcTokenIndex, axisK_ + toSharedExpertIndex);
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::CalExpertSendNum(TBuf<> outBuf, TBuf<> expertMaskBuf)
{
    uint64_t maskCnt = 0;
    uint32_t mask = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    uint32_t compareCount = Ceil(mask * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(int32_t);
    LocalTensor<uint8_t> expertMaskTensorU8 = expertMaskBuf.Get<uint8_t>();
    LocalTensor<uint32_t> expertMaskTensorU32 = expertMaskBuf.Get<uint32_t>();
    LocalTensor<int32_t> gatherTempTensor = outBuf.Get<int32_t>();
    for (int32_t expertIndex = 0; expertIndex < sendNum_; expertIndex++) {
        int32_t dstExpertId = expertIndex + startId_;
        if ((expertIndex == sendNum_ - 1) && (remainderExpertNum_ != 0)) {
            dstExpertId = delLastExpertId_;
        }
        CompareScalar(expertMaskTensorU8[maskSizePerExpert_ * expertIndex], validExpertIdsTensor_, dstExpertId, CMPMODE::EQ, compareCount);
        PipeBarrier<PIPE_V>();
        GatherMask(gatherTempTensor, validExpertIdsTensor_, expertMaskTensorU32[maskSizePerExpert_ * expertIndex / sizeof(uint32_t)],
            true, mask, {1, 1, 0, 0}, maskCnt); // 是否可以简化计算
        SyncFunc<AscendC::HardEvent::V_S>();
        tokenNumToExpertTensor_.SetValue(expertIndex, static_cast<uint32_t>(maskCnt));
    }
    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * BUFFER_NUM / sizeof(float));
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::SplitExpertNumToCore(uint32_t &delCurExpertGroupNum, uint32_t &groupIdx)
{
    sendNum_ = moeExpertNum_ / moeUsedAivNum_;
    remainderExpertNum_ = moeExpertNum_ % moeUsedAivNum_;
    startId_ = sendNum_ * aivId_;
    if (remainderExpertNum_ != 0) {
        int32_t remainderGroupSize = remainderExpertNum_;
        delLastExpertId_ = aivId_ % remainderExpertNum_;
        delCurExpertGroupNum = moeUsedAivNum_ / remainderGroupSize;
        if (delLastExpertId_ < moeUsedAivNum_ % remainderGroupSize) {
            delCurExpertGroupNum++;
        }
        groupIdx = aivId_ / remainderGroupSize;
        delLastExpertId_ += moeUsedAivNum_ * sendNum_;
        sendNum_ += 1;
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::SendToMoeExpert(TQue<QuePosition::VECIN, 1> inQueue,
    TBuf<> expertMaskBuf, TBuf<> outBuf)
{
    // 分核
    uint32_t delCurExpertGroupNum, groupIdx;
    SplitExpertNumToCore(delCurExpertGroupNum, groupIdx);
    // 计算专家发送数据量 && 发送
    CalExpertSendNum(outBuf, expertMaskBuf);
    uint32_t maskN64Num = Ceil(expertIdsCnt_, 64); // 64：ScalarGetSFFValue按照64长度一次计算
    GlobalTensor<ExpandXOutType> dstWinGMTensor;
    LocalTensor<uint64_t> expertMaskTensorU64 = expertMaskBuf.Get<uint64_t>();
    for (int32_t index = 0; index < sendNum_; index++) {
        int32_t dstTokenPreCnt = 0;
        int32_t expertIndex = (index + epRankId_ % sendNum_) % sendNum_;
        int32_t maskExpertU64Cnt = maskSizePerExpert_ * expertIndex / sizeof(uint64_t);
        if (tokenNumToExpertTensor_(expertIndex) == 0) {
            continue;
        }
        int32_t dstExpertId = expertIndex + startId_;
        dstExpertId = ((expertIndex == sendNum_ - 1) && (remainderExpertNum_ != 0)) ? delLastExpertId_ : dstExpertId;
        for (int32_t maskIndex = 0; maskIndex < maskN64Num; maskIndex++) {
            uint64_t dstExpInfoMask = expertMaskTensorU64(maskIndex + maskExpertU64Cnt);
            int64_t curValidIdx = ScalarGetSFFValue<1>(dstExpInfoMask);
            while (curValidIdx >= 0) {
                int32_t calExpertIdsIdx = curValidIdx + 64 * maskIndex;  // 64：ScalarGetSFFValue按照64长度一次计算
                if (calExpertIdsIdx >= expertIdsCnt_) {
                    break;
                }
                int32_t topKIndex = calExpertIdsIdx % axisK_;
                int32_t srcTokenIndex = calExpertIdsIdx / axisK_;
                int32_t toRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_;
                GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindAddrByRankId(toRankId) +
                                                (expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_ + dstExpertId % moeExpertNumPerRank_)) +
                                                hCommuSize_ * dstTokenPreCnt); // 计算地址偏移
                dstWinGMTensor.SetGlobalBuffer((__gm__ ExpandXOutType*)rankGM);
                if (!((expertIndex == sendNum_ - 1) && (remainderExpertNum_ != 0) && (dstTokenPreCnt % delCurExpertGroupNum != groupIdx))) {
                    if constexpr (DynamicQuant || StaticQuant) {
                        uint32_t quantExpertIdx = dstExpertId + sharedExpertNum_;
                        TokenToExpertInQuant(dstWinGMTensor, inQueue, srcTokenIndex, topKIndex, quantExpertIdx);
                    } else {
                        TokenToExpert(dstWinGMTensor, inQueue, srcTokenIndex, topKIndex);
                    }
                }
                dstTokenPreCnt++;
                uint64_t cleanMask = ~(uint64_t(1) << curValidIdx);
                dstExpInfoMask = cleanMask & dstExpInfoMask; // 将当前64bit中处理的1置0
                curValidIdx = ScalarGetSFFValue<1>(dstExpInfoMask);
            }
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::AllToAllDispatch()
{
    // 使用的全局参数
    TQue<QuePosition::VECIN, 1> inQueue;
    TBuf<> tempBuf, outBuf, expertIdsBuf, expertMaskBuf;
    TBuf<> smoothScalesBuf, tokenNumToExpertBuf, receiveDataCastFloatBuf;
    tpipe_->InitBuffer(inQueue, BUFFER_NUM, Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(outBuf, hCommuSize_ * BUFFER_NUM);
    outTensor_ = outBuf.Get<ExpandXOutType>();
    uint32_t hFp32Size = Ceil(axisH_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint32_t expertIdsSize = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t maxSize = hFp32Size > expertIdsSize ? hFp32Size : expertIdsSize;
    if constexpr (DynamicQuant || StaticQuant) {
        tpipe_->InitBuffer(tempBuf, hOutSizeAlign_);
        tpipe_->InitBuffer(receiveDataCastFloatBuf, maxSize);
        tpipe_->InitBuffer(smoothScalesBuf, maxSize);
        tempTensor_ = tempBuf.Get<ExpandXOutType>();
        floatLocalTemp_ = receiveDataCastFloatBuf.Get<float>();
        smoothScalesTensor_ = smoothScalesBuf.Get<float>();
        if constexpr (DynamicQuant) {
            floatLocalAbsTemp_ = smoothScalesBuf.Get<float>();
        }
        dstExpBuf_ = receiveDataCastFloatBuf; // 内存复用
        subExpBuf_ = smoothScalesBuf;         // 内存复用
    }

    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256; // 支持compareScalar
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    if (isTokenMaskFlag_ || isExpertMaskFlag_ || zeroComputeExpertNum_ != 0) {
        if constexpr (!(DynamicQuant || StaticQuant)) { // 非量化单独申请
            tpipe_->InitBuffer(dstExpBuf_, expertIdsSize);
            tpipe_->InitBuffer(subExpBuf_, expertIdsSize); 
        }
        tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_); 
    }
    activeMaskProc();
    if (activeMaskBsCnt_ == 0) {
        return;
    }

    if ((aivId_ >= moeUsedAivNum_) && (sharedExpertRankNum_ != 0)) {
        SendToSharedExpert(inQueue, outBuf);
    } else {
        maskSizePerExpert_ = Ceil((expertIdsBufSize_ / sizeof(int32_t)) / 8, UB_ALIGN) * UB_ALIGN; // 8 is 1byte->8bit
        uint32_t expertMaskBufSize = maskSizePerExpert_ * Ceil(moeExpertNum_, moeUsedAivNum_);
        tpipe_->InitBuffer(expertMaskBuf, expertMaskBufSize);
        tpipe_->InitBuffer(tokenNumToExpertBuf, Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
        tokenNumToExpertTensor_ = tokenNumToExpertBuf.Get<int32_t>();
        LocalTensor<int32_t> expertMaskTensor = expertMaskBuf.Get<int32_t>();
        Duplicate<int32_t>(expertMaskTensor, 0, int32_t(expertMaskBufSize / sizeof(int32_t)));
        SyncFunc<AscendC::HardEvent::V_S>();
        SendToMoeExpert(inQueue, expertMaskBuf, outBuf);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt)
{
    LocalTensor<int32_t> dstExpIdTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> subExpIdTensor = subExpBuf_.Get<int32_t>();
    Duplicate<int32_t>(dstExpIdTensor, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor, validExpertIdsTensor_, dstExpIdTensor, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor, dstExpIdTensor, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor(0);
    if (calCnt >= curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    } else {
        curExpertCnt = 0;
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::CalAndSendCnt()
{
    uint32_t startExpertId, endExpertId, sendExpertNum;
    SplitToCore(totalExpertNum_, aivUsedCumSum_, startExpertId, endExpertId, sendExpertNum, false);
    if (startExpertId >= totalExpertNum_) {return;}
    uint64_t mask[2] = { 0x101010101010101, 0 }; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    Duplicate<int32_t>(statusTensor_, 0, statusCntAlign_ * 8);  // 8 = UB_ALIGN / sizeof(int32_t)
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusCntAlign_ / 8, 1, 8); // 0x3F800000为float的1 8为一次操作8个block
    PipeBarrier<PIPE_ALL>();

    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t curExpertId = startExpertId; curExpertId < endExpertId; ++curExpertId) {
        int32_t curExpertCnt = 0;
        int32_t cntPosIndex = (curExpertId - startExpertId) * 8 + 1;               // 一个block有8个int32的元素，第一个元素为flag位，第二个为发送token数
        if ((curExpertId < sharedExpertRankNum_) && (activeMaskBsCnt_ > 0)) {     // 当前处理专家id为共享专家
            if (curExpertId % rankNumPerSharedExpert_ == epRankId_ % rankNumPerSharedExpert_) {
                curExpertCnt = activeMaskBsCnt_;
            }
        } else if (sendToMoeExpTokenCnt_ > 0) { // 当前处理卡为moe专家卡
            int32_t curMoeExpertId = curExpertId - sharedExpertRankNum_;
            CalTokenSendExpertCnt(curMoeExpertId, expertIdsCnt_, curExpertCnt);
        }
        statusTensor_.SetValue(cntPosIndex, curExpertCnt);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        uint32_t dstRankId = curExpertId;
        uint32_t offset = STATE_OFFSET * epRankId_;
        if (curExpertId >= sharedExpertRankNum_) {
            dstRankId = ((curExpertId - sharedExpertRankNum_) / moeExpertNumPerRank_ + sharedExpertRankNum_);
            offset += ((curExpertId - sharedExpertRankNum_) % moeExpertNumPerRank_ * epWorldSize_ * STATE_OFFSET);
        }
        GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindStateAddrByRankId(dstRankId) + offset);
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t*)rankGM);
        DataCopy<int32_t>(rankGMTensor, statusTensor_[(curExpertId - startExpertId) * 8], 8UL);  // 8 = UB_ALIGN / sizeof(int32_t)
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::UpdataTokenNumsOut(LocalTensor<int32_t> cumSumTensor)
{
    // 一卡一专家场景, 直接获取SendCounts最后的累加结果
    int32_t tokenNum = cumSumTensor.GetValue(epWorldSize_ - 1);
    expertTokenNumsOutGMTensor_.SetValue(0, tokenNum);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expertTokenNumsOutGMTensor_);
    // 一卡多专家场景, 更新moe专家卡除0号专家外的expertTokenNums数据
    if ((moeExpertNumPerRank_ != 1) && (!isShareExpertRankFlag_)) {
        uint32_t tokenSums = 0;
        uint32_t firstMoeCnt = cumSumTensor.GetValue(epWorldSize_ - 1);
        tokenSums = firstMoeCnt + gatherCount_;
        expertTokenNumsOutGMTensor_.SetValue(0, tokenSums);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expertTokenNumsOutGMTensor_[0]);
        for (uint32_t localMoeIndex = 1; localMoeIndex < moeExpertNumPerRank_; ++localMoeIndex) {
            uint32_t preOffset = epWorldSize_ * (localMoeIndex - 1) + epWorldSize_ - 1;
            uint32_t curOffset = epWorldSize_ * localMoeIndex + epWorldSize_ - 1;
            uint32_t preMoeIndexCnt = cumSumTensor.GetValue(preOffset);
            uint32_t curMoeIndexCnt = cumSumTensor.GetValue(curOffset);
            tokenSums = ((expertTokenNumsType_ == 0) ? tokenSums : 0) + (curMoeIndexCnt - preMoeIndexCnt) + gatherCount_;
            expertTokenNumsOutGMTensor_.SetValue(localMoeIndex, tokenSums);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(expertTokenNumsOutGMTensor_[localMoeIndex]);
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::WaitTokenNumAndCopy()
{
    // 等cnt+状态位并清理状态位
    float sumOfFlag = static_cast<float>(-1.0);
    float compareTarget = static_cast<float>(1.0) * rscvStatusNum_;
    DataCopyParams copyParams{static_cast<uint16_t>(rscvStatusNum_), 1, 0, 0};
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_, copyParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor_, statusFp32Tensor_, tempTime1Tensor_, 1, rscvStatusNum_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor_.GetValue(0);
    }
    uint64_t duplicateMask[2] = { 0x101010101010101, 0 }; // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0
    PipeBarrier<PIPE_ALL>();
    Duplicate<int32_t>(cleanStatusTensor_, static_cast<int32_t>(0), duplicateMask, Ceil(rscvStatusNum_, 8), 1, 8); // 一次操作8个block
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_, cleanStatusTensor_.ReinterpretCast<float>(), copyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    // cumsum计算结果输出值
    uint64_t rsvdCnt = 0;
    GlobalTensor<int32_t> sendCountsGlobal, workspaceGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sendCountsOutGM_));
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(recvCntWorkspaceGM_));
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(rscvStatusNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    uint32_t recStatusNumInner = Ceil(rscvStatusNum_ / cumSumTimes_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float); // 对inner要求32对齐
    const CumSumInfo cumSumInfo{1, recStatusNumInner}; // 第一个参数outter=1代表仅有1行，第二个参数inner=recStatusNumInner,表示一共接收的状态的个数，例如6个moe专家*16张卡；
    gatherTmpTensor_.SetValue(0, 2);  // 2 = 0010 源操作数每个datablock取下标为1的元素
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(tempTime1Tensor_, statusFp32Tensor_, gatherTmpTensor_, true, 2, {1, (uint16_t)(rscvStatusNum_), 1, 0}, rsvdCnt); // 2 = 0010
    PipeBarrier<PIPE_V>();
    CumSum<float, cumSumConfig>(cumSumTime1Tensor_, cumSumTime1Tensor_, tempTime1Tensor_, sharedTmpBufTensor_, cumSumInfo); // 计算ep_recv_count 行累加得到的ep_recv_count
    if (cumSumTimes_ == 2) { // 2: 当前ub大小空间需要分两次计算
        SyncFunc<AscendC::HardEvent::V_S>();
        tempTime2Tensor_(0) = tempTime2Tensor_(0) + cumSumTime1Tensor_(recStatusNumInner - 1);
        SyncFunc<AscendC::HardEvent::S_V>();
        CumSum<float, cumSumConfig>(cumSumTime2Tensor_, cumSumTime2Tensor_, tempTime2Tensor_, sharedTmpBufTensor_, cumSumInfo);
    }
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    LocalTensor<int32_t> cumSumResTensor = cumSumTime1Tensor_.ReinterpretCast<int32_t>();
    DataCopyPad(sendCountsGlobal, cumSumResTensor, dataCopyOutParams);
    for (uint32_t index = 0; index < aivNum_; ++index) { // 通过aivNum_次循环的方式，复制aivNum_份
        DataCopyPad(workspaceGlobal[index * rscvStatusNum_], cumSumResTensor, dataCopyOutParams);
    }
    UpdataTokenNumsOut(cumSumResTensor); // 计算各专家的recv_count
    PipeBarrier<PIPE_V>();
    Duplicate<float>(syncOnCoreTensor_, (float)1, aivNum_ * AIV_STATE_SIZE / sizeof(float));  // 8 = UB_ALIGN / sizeof(int32_t)
    PipeBarrier<PIPE_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[CUMSUM_FLAG_OFFSET / sizeof(float)], syncOnCoreTensor_, aivNum_ * AIV_STATE_SIZE / sizeof(float));  // 软同步 
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::CalCumSum()
{
    // 进来的核统一做发送，各专家的token总数发送
    TBuf<> statusBuf;
    uint32_t expertIdsSize = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256; // 支持compareScalar
    tpipe_->InitBuffer(dstExpBuf_, expertIdsSize);           // BS * K * 4
    tpipe_->InitBuffer(subExpBuf_, expertIdsSize);           // BS * K * 4
    tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);      // BS * K * 4
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    tpipe_->InitBuffer(statusBuf, statusCntAlign_ * UB_ALIGN);
    workLocalTensor_ = gatherMaskTBuf_.Get<float>();
    statusTensor_ = statusBuf.Get<int32_t>();
    activeMaskProc();
    CalAndSendCnt();

    // 最后一个核接收发给当前卡上专家的token数
    if (aivId_ != lastCore_) {
        return;
    }
    tpipe_->Reset();                
    TBuf<> sharedTmpBuf, gatherMaskOutBuf, cumSumTensorBuf;
    uint64_t gatherMaskOutSize = Ceil(rscvStatusNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t waitStatusBufSize = (((rscvStatusNum_ * UB_ALIGN) > SIZE_ALIGN_256) ?
        (rscvStatusNum_ * UB_ALIGN) : SIZE_ALIGN_256);
    uint64_t statusBufferCnt = waitStatusBufSize / sizeof(float);
    cumSumUB_ = MAX_UB_SIZE - BUFFER_NUM * gatherMaskOutSize;
    tpipe_->InitBuffer(gatherMaskOutBuf, gatherMaskOutSize);
    tpipe_->InitBuffer(cumSumTensorBuf, gatherMaskOutSize);
    tpipe_->InitBuffer(sharedTmpBuf, cumSumUB_);
    tempTime1Tensor_ = gatherMaskOutBuf.Get<float>();
    cumSumTime1Tensor_ = cumSumTensorBuf.Get<float>();
    if (isShareExpertRankFlag_) {
        cumSumTimes_ = 1;
    } else {
        cumSumTimes_ = (cumSumUBMinValue_ > cumSumUB_) ? 2 : 1; // 2:表示当前cumsum需要分开两次计算
        uint32_t rscvStatusTime1Num = rscvStatusNum_ / cumSumTimes_;
        uint64_t recStatusNumSpace = Ceil(rscvStatusTime1Num * sizeof(float), UB_ALIGN) * UB_ALIGN;
        uint64_t reTimesSize = (gatherMaskOutSize - recStatusNumSpace) / sizeof(float);
        cumSumTime2Tensor_ = cumSumTensorBuf.GetWithOffset<float>(reTimesSize, recStatusNumSpace);
        tempTime2Tensor_ = gatherMaskOutBuf.GetWithOffset<float>(reTimesSize, recStatusNumSpace);
    }
    statusFp32Tensor_ = sharedTmpBuf.GetWithOffset<float>(statusBufferCnt, 0);
    cleanStatusTensor_ = sharedTmpBuf.GetWithOffset<int32_t>(statusBufferCnt, 0);
    statusSumOutTensor_ = sharedTmpBuf.GetWithOffset<float>(UB_ALIGN / sizeof(float), waitStatusBufSize);
    gatherTmpTensor_ = sharedTmpBuf.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), waitStatusBufSize + UB_ALIGN);
    sharedTmpBufTensor_ = sharedTmpBuf.Get<uint8_t>();
    syncOnCoreTensor_ = sharedTmpBuf.GetWithOffset<float>(SYNC_OFFSET / sizeof(float), 0);
    WaitTokenNumAndCopy();
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::WaitCumSumFlag()
{
    // Check cumsum is finished
    // 当前每个aiv搬16个float(1)
    float cumSumFlag = 0;
    uint32_t cumSumFlagOffset = (CUMSUM_FLAG_OFFSET + AIV_STATE_SIZE * aivId_ ) / sizeof(float);
    while (true) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[cumSumFlagOffset], 8); // 8 = UB_ALIGN / sizeof(int32_t)
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        cumSumFlag = statusFp32Tensor_.GetValue(0);
        if (cumSumFlag == float(1)) {
            break;
        }
    }
    // Clean flag for next round
    Duplicate<float>(statusCleanFp32Tensor_, (float)0, AIV_STATE_SIZE / sizeof(float));
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[cumSumFlagOffset], statusCleanFp32Tensor_, AIV_STATE_SIZE / sizeof(float));
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum)
{
    // 获取cumSum
    GlobalTensor<int32_t> workspaceGlobal;
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(recvCntWorkspaceGM_));
    DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(rscvStatusNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyPad(sendCntTensor_, workspaceGlobal[aivId_ * rscvStatusNum_], scalesCopyInParams, copyPadExtParams);
    PipeBarrier<PIPE_ALL>();

    Duplicate<uint32_t>(expertFinishNumTensor_, 0, expInfoSize / sizeof(uint32_t));
    for (uint32_t index = startId_; index < endId_; index++) { // 从sendCnt中挑选当前有发送过来的卡的token数量
        expertMapTensor_(validNum) = index;
        if (index == 0) {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index);
        } else {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index) - sendCntTensor_(index - 1);
        }
        if (expertLeftNumTensor_(validNum) != 0) {
            validNum += 1;
        } 
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::CheckDataArriveWithFlag(uint32_t srcExpDataIdx,
    int32_t beginIdx, int32_t copyCnt)
{
    uint64_t rsvdCnt = 0;
    uint32_t arriveFlagNum = 0;
    uint32_t flagNum = blockCntPerToken_ * uint32_t(copyCnt);
    uint32_t compareCount = Ceil(flagNum, COMPARE_COUNT_PER_BLOCK) * COMPARE_COUNT_PER_BLOCK;
    uint32_t compResultU64Num = Ceil(flagNum, 64); // 64：按照64bit位进行划分
    DataCopyExtParams expFlagCopyParams{static_cast<uint16_t>(flagNum), static_cast<uint32_t>(sizeof(float)),
        static_cast<uint32_t>(SPLIT_BLOCK_SIZE - sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> expFlagPadParams{false, 0U, 0U, 0U};
    GlobalTensor<float> dataFlagGlobal;
    GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + srcExpDataIdx * expertPerSizeOnWin_ + // 拿到第一个起始位置
        beginIdx * hCommuSize_ + SPLIT_BLOCK_DATA_SIZE;
    dataFlagGlobal.SetGlobalBuffer((__gm__ float *)(wAddr));
    DataCopyPad(flagRecvTensor_, dataFlagGlobal, expFlagCopyParams, expFlagPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    GatherMask(flagGatherOutTensor_, flagRecvTensor_, flagRecvGatherMask_, true, uint32_t(1),
         {1, (uint16_t)(flagNum), 1, 0}, rsvdCnt); 
    PipeBarrier<PIPE_V>();
    CompareScalar(flagCompResultU8_, flagGatherOutTensor_, float(1), AscendC::CMPMODE::EQ, compareCount);
    SyncFunc<AscendC::HardEvent::V_S>();

    for (uint32_t i = 0; i < compResultU64Num; i++) { 
        uint64_t flagCompMask = flagCompResultLtU64_(i);
        int64_t firstValidIdx = ScalarGetSFFValue<0>(flagCompMask); // 找到0则表示数据没到
        if (firstValidIdx == -1) { // 本次数据全到
            arriveFlagNum += 64U; // 64：ScalarGetSFFValue操作单位为64bit位
        } else {
            arriveFlagNum += uint32_t(firstValidIdx);
            break;
        }
    }
    if (arriveFlagNum > flagNum) {
        arriveFlagNum = flagNum;
    }
    return uint32_t(arriveFlagNum / blockCntPerToken_); // 返回token总数
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::CopyInAndOut(LocalTensor<float> xOutFp32Tensor, 
    LocalTensor<int32_t> xOutInt32Tensor, GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount)
{
    GlobalTensor<ExpandXOutType> dataFlagGlobal, expandXOutGlobal;
    dataFlagGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(wAddr));
    expandXOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOutGM_) + (dstPosition) * axisH_);
    DataCopyExtParams srcTokenCopyParams{static_cast<uint16_t>(blockCntPerToken_ * arriveCount), 
        static_cast<uint32_t>(SPLIT_BLOCK_DATA_SIZE), static_cast<uint32_t>(UB_ALIGN), 0, 0};
    DataCopyExtParams scalesCopyParams{uint16_t(arriveCount), static_cast<uint32_t>(sizeof(float)), 
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1), 0U, 0U};
    DataCopyExtParams tokenCopyParams{uint16_t(arriveCount), hOutSize_, 
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - hOutSize_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams expandIdxCopyParams{uint16_t(arriveCount), EXPAND_IDX_INFO * sizeof(int32_t),
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1), 0U, 0U};
    DataCopyPadExtParams<ExpandXOutType> srcTokenPadParams{false, 0U, 0U, 0U};

    DataCopyPad(xTmpTensor_, dataFlagGlobal[expertFinishNumTensor_(index) * hCommuSize_ / sizeof(ExpandXOutType)],
                srcTokenCopyParams, srcTokenPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    if constexpr (DynamicQuant || StaticQuant) {
        DataCopyPad(dynamicScalesOutGMTensor_[dstPosition], xOutFp32Tensor[tokenQuantAlign_], scalesCopyParams);
    }
    DataCopyPad(expandXOutGlobal, xTmpTensor_, tokenCopyParams);
    DataCopyPad(expandIdxGMTensor_[dstPosition * EXPAND_IDX_INFO], xOutInt32Tensor[tokenQuantAlign_ + (UB_ALIGN / sizeof(int32_t))],
                expandIdxCopyParams);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum)
{
    uint32_t index = 0;
    uint32_t finishNum = 0;
    uint32_t maxCopyTokenCnt = tBufRealSize_ / hCommuSize_;
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
    uint32_t srcExpRankId, dstPosition, arriveCount, copyCnt, srcDataBlockIdx;
    uint32_t flagMaxRecvNum = (blockCntPerToken_ * maxCopyTokenCnt * UB_ALIGN) / sizeof(uint32_t);
    uint32_t gatherOutSize = Ceil(blockCntPerToken_ * maxCopyTokenCnt * sizeof(uint32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    GlobalTensor<int32_t> cleanGlobal;
    flagGatherOutTensor_ = tBuf.GetWithOffset<float>(gatherOutSize / sizeof(float), 0); // buf复用
    flagRecvTensor_ = tBuf.GetWithOffset<float>(flagMaxRecvNum, gatherOutSize);  // buf复用
    LocalTensor<float> xOutFp32Tensor = xTmpTensor_.template ReinterpretCast<float>();
    LocalTensor<int32_t> xOutInt32Tensor = xTmpTensor_.template ReinterpretCast<int32_t>();
    while (true) {
        if (expertLeftNumTensor_(index) == 0) { // 当前核负责的不需要收集
            index = (index + 1) % validNum; // 轮询查询每个有效的index
            continue;
        }
        srcExpRankId = expertMapTensor_(index);
        copyCnt = expertLeftNumTensor_(index) > maxCopyTokenCnt ? maxCopyTokenCnt : expertLeftNumTensor_(index); // 按照ub大小一次搬入多个token
        srcDataBlockIdx = srcExpRankId % epWorldSize_ * localExpertNum + srcExpRankId / epWorldSize_; // 转换成数据区的排布偏移
        arriveCount = CheckDataArriveWithFlag(srcDataBlockIdx, expertFinishNumTensor_(index), copyCnt);
        if (arriveCount == copyCnt) {
            dstPosition = srcExpRankId != 0 ? sendCntTensor_(srcExpRankId - 1) : 0;
            dstPosition += expertFinishNumTensor_(index);
            GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + srcDataBlockIdx * expertPerSizeOnWin_;
            CopyInAndOut(xOutFp32Tensor, xOutInt32Tensor, wAddr, index, dstPosition, arriveCount);
            // finish更新并clean
            expertFinishNumTensor_(index) += arriveCount;
            expertLeftNumTensor_(index) -= arriveCount;
            if (expertLeftNumTensor_(index) == 0) {
                uint32_t cleanUpNum = expertFinishNumTensor_(index) * blockCntPerToken_;
                DataCopyExtParams cleanUoParams = {uint16_t(cleanUpNum), sizeof(int32_t), 0U, SPLIT_BLOCK_SIZE - sizeof(int32_t), 0U};
                LocalTensor<int32_t> cleanTensor = tBuf.GetWithOffset<int32_t>(UB_ALIGN / sizeof(int32_t), 0); // 在0偏移位置存放比较结果
                cleanGlobal.SetGlobalBuffer((__gm__ int32_t *)(wAddr));
                SyncFunc<AscendC::HardEvent::MTE3_V>();
                Duplicate<int32_t>(cleanTensor, 0, 8); // 8 = UB_ALIGN / 4
                SyncFunc<AscendC::HardEvent::V_MTE3>();
                DataCopyPad(cleanGlobal[SPLIT_BLOCK_DATA_SIZE / sizeof(int32_t)], cleanTensor, cleanUoParams);
                finishNum++;
            }
            PipeBarrier<PIPE_ALL>();
        } else {
            index = (index + 1) % validNum;
        }
        if (validNum == finishNum) {
            break;
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::LocalWindowCopy()
{
    // 分核负责源专家数量
    tpipe_->Reset();
    TBuf<> cumSumBuf, statusWaitBuf, statusCleanBuf;
    uint32_t rscvNumAlign = Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusWaitBuf, rscvNumAlign);
    tpipe_->InitBuffer(cumSumBuf, rscvNumAlign);
    tpipe_->InitBuffer(statusCleanBuf, UB_ALIGN * BUFFER_NUM);
    statusFp32Tensor_ = statusWaitBuf.Get<float>();
    statusCleanFp32Tensor_ = statusCleanBuf.Get<float>();
    sendCntTensor_ = cumSumBuf.Get<int32_t>();
    SplitToCore(rscvStatusNum_, aivNum_, startId_, endId_, sendNum_, true);
    // 软同步
    WaitCumSumFlag();
    if (sendNum_ == 0) {
        return;
    }
    // 连续化
    TBuf<> expertMapBuf, expertFinishBuf, expertLeftBuf, flagMaskBuf, tBuf;
    uint32_t validNum = 0;
    uint32_t expInfoSize = Ceil(sendNum_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tBufRealSize_ = MAX_UB_SIZE - ((rscvNumAlign + UB_ALIGN) * BUFFER_NUM) - (expInfoSize * 3) - BUFFER_NUM * UB_ALIGN; // 3为expInfoSize大小buffer申请个数
    tpipe_->InitBuffer(expertMapBuf, expInfoSize);
    tpipe_->InitBuffer(expertFinishBuf, expInfoSize);
    tpipe_->InitBuffer(expertLeftBuf, expInfoSize);
    tpipe_->InitBuffer(flagMaskBuf, BUFFER_NUM * UB_ALIGN);  // max CompareScalar
    tpipe_->InitBuffer(tBuf, tBufRealSize_); // 其余buffer空间统一申请
    expertMapTensor_ = expertMapBuf.Get<uint32_t>();
    expertFinishNumTensor_ = expertFinishBuf.Get<uint32_t>();
    expertLeftNumTensor_ = expertLeftBuf.Get<uint32_t>();
    SetValidExpertInfo(expInfoSize, validNum);
    if (validNum == 0) { // 本核负责的Expert对应rank收到数据
        return;
    }
    flagCompResultU8_ = flagMaskBuf.Get<uint8_t>();
    flagCompResultLtU64_ = flagMaskBuf.Get<uint64_t>();
    flagRecvGatherMask_ = statusCleanBuf.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    xTmpTensor_ = tBuf.Get<ExpandXOutType>(); 
    LocalTensor<uint32_t> flagCompResultLtU32 = flagMaskBuf.Get<uint32_t>();
    Duplicate<uint32_t>(flagCompResultLtU32, 0, BUFFER_NUM * UB_ALIGN / sizeof(uint32_t));
    Duplicate<uint32_t>(flagRecvGatherMask_, 0, UB_ALIGN / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::V_S>();
    flagRecvGatherMask_.SetValue(0, 1);
    SyncFunc<AscendC::HardEvent::S_V>();
    WaitAndFormatOutput(tBuf, validNum);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::TokenActiveMaskCal()
{
    // 搬运x_active_mask，当前仅用于计算有效token总数
    LocalTensor<half> maskTmpTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<bool> maskInputTensor;
    uint32_t axisBsAlignSize = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    maskInputTensor = dstExpBuf_.Get<bool>();
    maskTmpTensor = subExpBuf_.Get<half>();
    sumOutTensor = gatherMaskTBuf_.Get<half>();
    DataCopyExtParams maskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(maskTmpTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize, axisBS_};
    Sum(sumOutTensor, maskTmpTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::activeMaskProc()
{
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    validExpertIdsTensor_ = expertIdsBuf_.Get<int32_t>();

    if (isTokenMaskFlag_) {
        TokenActiveMaskCal();
    }

    if (activeMaskBsCnt_ == 0) { // DataCopyPad不能拷贝0个
        return;
    }
    Duplicate<int32_t>(validExpertIdsTensor_, -1, int32_t(expertIdsBufSize_ / sizeof(int32_t)));
    PipeBarrier<PIPE_V>();

    uint32_t expertIdsAlignCnt = Ceil(activeMaskBsCnt_ * axisK_, 8) * 8; // 8 = UB_ALIGN / sizeof(int32_t)
    uint32_t rightPadding = expertIdsAlignCnt - (activeMaskBsCnt_ * axisK_);
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{true, 0U, uint8_t(rightPadding), -1}; // rightPadding字节数不能超过 32，不能超过8个u32
    DataCopyExtParams expertIdsCntParams{1U, static_cast<uint32_t>(activeMaskBsCnt_ * axisK_ * sizeof(uint32_t)), 0U, 0U, 0U}; //第二个参数blockLen 范围[1, 2097151] 不能为0
    // 拷贝bs*k个专家id 到local  补齐到32 字节对齐  bs*k = 3*3 = 9 expertIdsAlignCnt= 16 补7个 填充-1
    DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}


template <TemplateMC2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2FullMesh<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV {          // 全aiv处理
        if (aivId_ < aivUsedAllToAll_) {
            AllToAllDispatch(); // 前面核all2all发送
        } else {
            CalCumSum();        // 后面核发送当前卡给每个专家的tokenCnt，输出epRecvCnt/exportTokenNums
        }
        LocalWindowCopy();      // 本卡上专家数据连续化，输出expandX/scales/expandIdx
    }
}

} // MoeDistributeDispatchV2FullMeshImpl
#endif // MOE_DISTRIBUTE_DISPATCH_V2_FULL_MESH_H