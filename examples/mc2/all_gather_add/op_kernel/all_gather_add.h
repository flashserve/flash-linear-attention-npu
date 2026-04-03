/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/* !
 * \file all_gather_add.h
 * \brief
 */
#ifndef ALL_GATHER_ADD_H
#define ALL_GATHER_ADD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "all_gather_add_tiling.h"

constexpr int32_t ALLGATHER_ADD_BUFFER_NUM = 1;

namespace AscendC {
class AllGatherAdd {
public:
    __aicore__ inline AllGatherAdd(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR gatherGM, 
                                GM_ADDR workspaceGM, GM_ADDR contextGM, AllGatherAddTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void HcclPrepare();
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute();
    __aicore__ inline void HcclFinalize();

private:

    AllGatherAddTilingData *tilingData_;

    TPipe *tPipe_;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;

    TQue<QuePosition::VECIN, ALLGATHER_ADD_BUFFER_NUM> inputQueueGather;
    TQue<QuePosition::VECIN, ALLGATHER_ADD_BUFFER_NUM> inputQueueB;
    TQue<QuePosition::VECOUT, ALLGATHER_ADD_BUFFER_NUM> outputQueueC;

    GlobalTensor<half> inputAGM;
    GlobalTensor<half> gatherOutGM;
    GlobalTensor<half> inputBGM;
    GlobalTensor<half> outputCGM;

    int64_t blockElemNum_ = 0;
    int64_t tileNum_ = 0;
    uint32_t addTileElemNum_ = 0;

    HcclHandle handleId_{ INVALID_HANDLE_ID };
};

__aicore__ inline void AllGatherAdd::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR gatherGM, 
                                          GM_ADDR workspaceGM, GM_ADDR contextGM, AllGatherAddTilingData *tilingData, TPipe *tPipe)
{
    tilingData_ = tilingData;
    tPipe_ = tPipe;
    blockElemNum_ = tilingData->blockElemNum;
    addTileElemNum_ = tilingData->addTileElemNum;
    tileNum_ = tilingData->tileNum;

    // 初始化hccl对象
    hccl_.InitV2(contextGM, tilingData);
    hccl_.SetCcTilingV2(offsetof(AllGatherAddTilingData, mc2CcTiling));

    // 传入全局数据的指针，并设置存储大小
    inputAGM.SetGlobalBuffer((__gm__ half*)aGM, tilingData->gatherTileElemNum); // 非多轮切分AllGather场景，每张卡参与Gather的数据大小为{240，256}
    gatherOutGM.SetGlobalBuffer((__gm__ half*)gatherGM + blockElemNum_ * AscendC::GetBlockIdx(), blockElemNum_);
    inputBGM.SetGlobalBuffer((__gm__ half*)bGM + blockElemNum_ * AscendC::GetBlockIdx(), blockElemNum_);
    outputCGM.SetGlobalBuffer((__gm__ half*)cGM + blockElemNum_ * AscendC::GetBlockIdx(), blockElemNum_);
    
    tPipe_->InitBuffer(inputQueueGather, ALLGATHER_ADD_BUFFER_NUM, addTileElemNum_ * sizeof(half));
    tPipe_->InitBuffer(inputQueueB, ALLGATHER_ADD_BUFFER_NUM, addTileElemNum_ * sizeof(half));
    tPipe_->InitBuffer(outputQueueC, ALLGATHER_ADD_BUFFER_NUM, addTileElemNum_ * sizeof(half));
}

__aicore__ inline void AllGatherAdd::HcclPrepare()
{
    // 下发通信任务
    handleId_ = hccl_.AllGather<true>((__gm__ uint8_t*)this->inputAGM.GetPhyAddr(), (__gm__ uint8_t*)this->gatherOutGM.GetPhyAddr(), tilingData_->gatherTileElemNum,
                                      HcclDataType::HCCL_DATA_TYPE_FP16, 0, tilingData_->commTurn);
}

__aicore__ inline void AllGatherAdd::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<half> gatherLocal = inputQueueGather.AllocTensor<half>();
    AscendC::LocalTensor<half> bLocal = inputQueueB.AllocTensor<half>();
    AscendC::DataCopy(gatherLocal, gatherOutGM[progress * addTileElemNum_], addTileElemNum_);
    AscendC::DataCopy(bLocal, inputBGM[progress * addTileElemNum_], addTileElemNum_);
    inputQueueGather.EnQue(gatherLocal);
    inputQueueB.EnQue(bLocal);
}

__aicore__ inline void AllGatherAdd::Compute()
{
    AscendC::LocalTensor<half> gatherLocal = inputQueueGather.DeQue<half>();
    AscendC::LocalTensor<half> bLocal = inputQueueB.DeQue<half>();
    AscendC::LocalTensor<half> cLocal = outputQueueC.AllocTensor<half>();
    AscendC::Add(cLocal, gatherLocal, bLocal, addTileElemNum_);
    outputQueueC.EnQue<half>(cLocal);
    inputQueueGather.FreeTensor(gatherLocal);
    inputQueueB.FreeTensor(bLocal);
}

__aicore__ inline void AllGatherAdd::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<half> cLocal = outputQueueC.DeQue<half>();
    AscendC::DataCopy(outputCGM[progress * addTileElemNum_], cLocal, addTileElemNum_);
    outputQueueC.FreeTensor(cLocal);
}

__aicore__ inline void AllGatherAdd::HcclFinalize()
{
    AscendC::SyncAll<true>();
    hccl_.Finalize();
}

__aicore__ inline void AllGatherAdd::Process()
{
    HcclPrepare();
    for (int i = 0; i < tilingData_->commTurn; i++) {
        hccl_.Wait(handleId_);
        for (int j = 0; j < tileNum_; j++) {
            CopyIn(j);
            Compute();
            CopyOut(j);
        }
    }
    HcclFinalize();
}
}
#endif