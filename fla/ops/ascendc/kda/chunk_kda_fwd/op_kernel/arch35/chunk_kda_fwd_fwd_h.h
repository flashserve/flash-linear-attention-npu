#ifndef CHUNK_KDA_FWD_ARCH35_FWD_H_H
#define CHUNK_KDA_FWD_ARCH35_FWD_H_H

#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "../chunk_kda_fwd_plan.h"

namespace KdaForward::arch35 {

using namespace AscendC;

// 目标 Catlass 只允许使用 0..7，FwdH 在合法范围 0..5 内为三条 raw
// mode2 通道分配互不冲突的 flag。同一通道的 ready/free 使用不同 ID：
// direct 由 AIC 发布、两个 AIV 消费，state/vNew 由两个 AIV 共同发布、
// AIC 消费。每条通道的两个真实物理 slot 通过入口 seed、复用前 wait
// 和阶段退出 drain 形成完整闭环，不让计数状态跨越 FwdH 阶段。
constexpr uint64_t KDA_FWD_H_DIRECT_FREE_FLAG = 0;
constexpr uint64_t KDA_FWD_H_DIRECT_READY_FLAG = 1;
constexpr uint32_t KDA_FWD_H_DIRECT_BUFFER_DEPTH = 2;
constexpr uint64_t KDA_FWD_H_STATE_FREE_FLAG = 2;
constexpr uint64_t KDA_FWD_H_STATE_READY_FLAG = 3;
constexpr uint64_t KDA_FWD_H_VNEW_FREE_FLAG = 4;
constexpr uint64_t KDA_FWD_H_VNEW_READY_FLAG = 5;
constexpr uint32_t KDA_FWD_H_L1_BUFFER_DEPTH = 2;

constexpr TEventID KDA_FWD_H_MTE_W_EVENT = 0;
constexpr TEventID KDA_FWD_H_MTE_Q_EVENT = 1;
constexpr TEventID KDA_FWD_H_MTE_B_EVENT = 2;
constexpr TEventID KDA_FWD_H_MTE_A_EVENT = 3;
constexpr TEventID KDA_FWD_H_M_EVENT = 4;
// 手工管理的本地事件只能使用 0..5。相同数字在不同 HardEvent 事件池中
// 彼此独立，因此这里可以复用 MTE_W 的数字 0，而不会与 MTE2_MTE1 冲突。
constexpr TEventID KDA_FWD_H_IO_REUSE_EVENT = 0;
static_assert(KDA_FWD_H_IO_REUSE_EVENT <= 5,
              "FwdH manually managed local event IDs must stay within 0..5");

// L0C 只服务 Cube/Fixpipe 的 product 流水。4-head 任务窗口连续向同一条
// 流水喂入 product，物理槽始终按 ping/pong 两份轮转；槽间距按最大的
// 128 x 128 state-update product 固定，64 x 128 product 只占其中一半。
constexpr uint32_t KDA_FWD_H_L0C_BUFFER_DEPTH = 2;
constexpr uint32_t KDA_FWD_H_L0C_SLOT_BYTES = 128 * 128 * sizeof(float);
constexpr TEventID KDA_FWD_H_L0C_PING_EVENT = 0;
constexpr TEventID KDA_FWD_H_L0C_PONG_EVENT = 1;

constexpr uint32_t KDA_FWD_H_CHUNK = 64;
constexpr uint32_t KDA_FWD_H_DIM = 128;
constexpr uint32_t KDA_FWD_H_SUB_CHUNK = KDA_FWD_H_CHUNK / 2;
constexpr uint32_t KDA_FWD_H_SUB_DIM = KDA_FWD_H_DIM / 2;
constexpr uint32_t KDA_FWD_H_STATE_SUB_ELEMS = KDA_FWD_H_SUB_DIM * KDA_FWD_H_DIM;
constexpr uint32_t KDA_FWD_H_TOKEN_SUB_ELEMS = KDA_FWD_H_SUB_CHUNK * KDA_FWD_H_DIM;

constexpr uint32_t KDA_FWD_H_L1_W_OFFSET = 0;
constexpr uint32_t KDA_FWD_H_L1_Q_OFFSET = 16 * 1024;
constexpr uint32_t KDA_FWD_H_L1_H_OFFSET = 32 * 1024;
constexpr uint32_t KDA_FWD_H_L1_KG_OFFSET = 64 * 1024;
constexpr uint32_t KDA_FWD_H_L1_AQK_OFFSET = 80 * 1024;
constexpr uint32_t KDA_FWD_H_L1_V_OFFSET = 96 * 1024;
constexpr uint32_t KDA_FWD_H_L1_W1_OFFSET = 112 * 1024;
constexpr uint32_t KDA_FWD_H_L1_Q1_OFFSET = 128 * 1024;
constexpr uint32_t KDA_FWD_H_L1_KG1_OFFSET = 144 * 1024;
constexpr uint32_t KDA_FWD_H_L1_AQK1_OFFSET = 160 * 1024;
constexpr uint32_t KDA_FWD_H_L1_W2_OFFSET = 176 * 1024;
constexpr uint32_t KDA_FWD_H_L1_Q2_OFFSET = 192 * 1024;
constexpr uint32_t KDA_FWD_H_L1_KG2_OFFSET = 208 * 1024;
constexpr uint32_t KDA_FWD_H_L1_AQK2_OFFSET = 224 * 1024;
constexpr uint32_t KDA_FWD_H_L1_W3_OFFSET = 240 * 1024;
constexpr uint32_t KDA_FWD_H_L1_Q3_OFFSET = 256 * 1024;
constexpr uint32_t KDA_FWD_H_L1_KG3_OFFSET = 272 * 1024;
constexpr uint32_t KDA_FWD_H_L1_AQK3_OFFSET = 288 * 1024;
// FwdH 已不再在 L1 暂存 Akk/U，复用该区域作为 state/vNew 的第二个
// 物理 slot，确保 ping-pong 的两个逻辑槽对应互不重叠的真实地址。
constexpr uint32_t KDA_FWD_H_L1_H1_OFFSET = 304 * 1024;
constexpr uint32_t KDA_FWD_H_L1_V1_OFFSET = 336 * 1024;
constexpr uint32_t KDA_FWD_H_L1_STAGING_DEPTH = 4;
static_assert(KDA_FWD_H_L1_STAGING_DEPTH == 4,
              "FwdH literal staging-event dispatch requires four slots");

constexpr uint32_t KDA_FWD_H_L0A_STATE_OFFSET = 0;
constexpr uint32_t KDA_FWD_H_L0A_VNEW_OFFSET = 16 * 1024;
constexpr uint32_t KDA_FWD_H_L0A_POST_OFFSET = 32 * 1024;
constexpr uint32_t KDA_FWD_H_L0B_STATE_OFFSET = 0;
constexpr uint32_t KDA_FWD_H_L0B_VNEW_OFFSET = 32 * 1024;
constexpr uint32_t KDA_FWD_H_L0B_POST_OFFSET = 32 * 1024;

constexpr uint32_t KDA_FWD_H_UB_STATE_OFFSET = 0;
constexpr uint32_t KDA_FWD_H_UB_STATE_TYPED_OFFSET = 32 * 1024;
constexpr uint32_t KDA_FWD_H_UB_DIRECT_OFFSET = 48 * 1024;
constexpr uint32_t KDA_FWD_H_UB_DIRECT_SLOT_BYTES = 32 * 1024;
constexpr uint32_t KDA_FWD_H_UB_VNEW_OFFSET = 112 * 1024;
constexpr uint32_t KDA_FWD_H_UB_IO_OFFSET = 128 * 1024;
constexpr uint32_t KDA_FWD_H_UB_GATE_OFFSET = 144 * 1024;
static_assert(
    KDA_FWD_H_UB_DIRECT_OFFSET +
        KDA_FWD_H_DIRECT_BUFFER_DEPTH * KDA_FWD_H_UB_DIRECT_SLOT_BYTES <=
        KDA_FWD_H_UB_VNEW_OFFSET,
    "direct UB ping-pong slots overlap vNew scratch");
static_assert(
    KDA_FWD_H_L1_H1_OFFSET + KDA_FWD_H_DIM * KDA_FWD_H_DIM * sizeof(uint16_t) <=
        KDA_FWD_H_L1_V1_OFFSET,
    "state L1 ping-pong slot overlaps vNew slot");

template <typename T, typename GK_T, typename TilingData>
class ChunkKdaFwdFwdH {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutRM = Catlass::layout::RowMajor;
    using LayoutCM = Catlass::layout::ColumnMajor;
    using TileCopyRM = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, T, LayoutRM, T, LayoutRM, float, LayoutRM>;
    using DirectTileCopyRM = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, T, LayoutRM, T, LayoutRM, float, LayoutRM, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileCopyCM = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, T, LayoutCM, T, LayoutRM, float, LayoutRM>;
    using DirectTileCopyCM = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, T, LayoutCM, T, LayoutRM, float, LayoutRM, void,
        Catlass::Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    static_assert(
        KDA_FWD_H_L0C_BUFFER_DEPTH * KDA_FWD_H_L0C_SLOT_BYTES <= ArchTag::L0C_SIZE,
        "FwdH L0C ping-pong slots exceed the target L0C capacity");

    __aicore__ inline void Init(
        GM_ADDR gk, GM_ADDR initialState, GM_ADDR finalState,
        GM_ADDR w, GM_ADDR u, GM_ADDR kg,
        GM_ADDR vNew, GM_ADDR h, GM_ADDR cuSeqlens, GM_ADDR compactPlan,
        const TilingData &tiling)
    {
        gk_.SetGlobalBuffer(reinterpret_cast<__gm__ GK_T *>(gk));
        if (initialState != nullptr) {
            initialState_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initialState));
        }
        finalState_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(finalState));
        w_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(w));
        u_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(u));
        kg_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kg));
        vNew_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(vNew));
        h_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(h));
        if (cuSeqlens != nullptr) {
            cuSeqlens_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cuSeqlens));
        }
        compactPlanAddr_ = compactPlan;
        batch_ = tiling.batch;
        seqNum_ = tiling.seqNum;
        heads_ = tiling.vHeadNum;
        seqlen_ = tiling.seqlen;
        totalChunks_ = tiling.totalChunks;
        isVarLen_ = tiling.isVarLen;
        hasInitialState_ = tiling.hasInitialState;
        storeFinalState_ = tiling.storeFinalState;
        fwdCoreNum_ = tiling.prepareUsedCoreNum;
        statePublishCount_[0] = 0;
        statePublishCount_[1] = 0;
        vnewPublishCount_[0] = 0;
        vnewPublishCount_[1] = 0;
        stateConsumeIndex_ = 0;
        vnewConsumeIndex_ = 0;
        directPublishIndex_ = 0;
        directConsumeIndex_ = 0;
        l0cProductIndex_ = 0;
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            ProcessAic();
        }
        if ASCEND_IS_AIV {
            ProcessAiv();
        }
    }

private:
    __aicore__ inline void WaitL1SlotFreeMte3(
        uint64_t freeFlag, uint32_t publishCount)
    {
        // 前两次发布分别写入两个物理 slot；从第三次开始，两个 AIV
        // 复用最旧 slot 前必须等到 AIC 回传一份聚合 free credit。
        if (publishCount >= KDA_FWD_H_L1_BUFFER_DEPTH) {
            CrossCoreWaitFlag(freeFlag);
        }
    }

    template <pipe_t PIPE>
    __aicore__ inline void SetL1SlotFlagAicToAiv(uint64_t flag)
    {
        CrossCoreSetFlag<0x2, PIPE>(flag);
    }

    template <pipe_t PIPE>
    __aicore__ inline void SetL1SlotFlagAivToAic(uint64_t flag)
    {
        CrossCoreSetFlag<0x2, PIPE>(flag);
    }

    __aicore__ inline void WaitL1SlotReadyMte1(uint64_t readyFlag)
    {
        CrossCoreWaitFlag(readyFlag);
    }

    __aicore__ inline void WaitDirectFreeAic()
    {
        CrossCoreWaitFlag(KDA_FWD_H_DIRECT_FREE_FLAG);
    }

    __aicore__ inline void SetDirectReadyAic()
    {
        CrossCoreSetFlag<0x2, PIPE_FIX>(KDA_FWD_H_DIRECT_READY_FLAG);
    }

    __aicore__ inline LocalTensor<float> AcquireDirectBufferAiv()
    {
        CrossCoreWaitFlag(KDA_FWD_H_DIRECT_READY_FLAG);
        const uint32_t slot = directConsumeIndex_ % KDA_FWD_H_DIRECT_BUFFER_DEPTH;
        ++directConsumeIndex_;
        return resource_.ubBuf.template GetBufferByByte<float>(
            KDA_FWD_H_UB_DIRECT_OFFSET + slot * KDA_FWD_H_UB_DIRECT_SLOT_BYTES);
    }

    __aicore__ inline void SetDirectFreeAiv()
    {
        CrossCoreSetFlag<0x2, PIPE_V>(KDA_FWD_H_DIRECT_FREE_FLAG);
    }

    __aicore__ inline void InitializeDirectFreeCreditsAiv()
    {
        // 两个 AIV 都执行此循环。每个物理 slot 对应两次同序 set，
        // mode2 将它们聚合成 AIC 可见的一份 free credit。
        for (uint32_t slot = 0; slot < KDA_FWD_H_DIRECT_BUFFER_DEPTH; ++slot) {
            SetDirectFreeAiv();
        }
    }

    __aicore__ inline void DrainDirectFreeCreditsAic()
    {
        // 每次消费都会归还一份 credit；阶段结束时排空两个初始 slot，
        // 不允许 mode2 计数状态跨越 FwdH 阶段边界。
        for (uint32_t slot = 0; slot < KDA_FWD_H_DIRECT_BUFFER_DEPTH; ++slot) {
            WaitDirectFreeAic();
        }
    }

    __aicore__ inline void DrainL1FreeCreditsAiv(uint32_t subBlockIdx)
    {
        // AIC 每消费一份 payload 就广播一个 mode2 free token。旧 credit
        // 在 slot 复用前已被消费，因此最后最多残留两份。两个 AIV 按
        // 相同次数排空，保证一 AIC/两 AIV 组退出时信号量深度为零。
        const uint32_t stateCredits = min(
            statePublishCount_[subBlockIdx], KDA_FWD_H_L1_BUFFER_DEPTH);
        for (uint32_t credit = 0; credit < stateCredits; ++credit) {
            CrossCoreWaitFlag(KDA_FWD_H_STATE_FREE_FLAG);
        }
        const uint32_t vnewCredits = min(
            vnewPublishCount_[subBlockIdx], KDA_FWD_H_L1_BUFFER_DEPTH);
        for (uint32_t credit = 0; credit < vnewCredits; ++credit) {
            CrossCoreWaitFlag(KDA_FWD_H_VNEW_FREE_FLAG);
        }
    }

    __aicore__ inline TEventID L0CEvent(uint32_t slot) const
    {
        return slot == 0 ? KDA_FWD_H_L0C_PING_EVENT : KDA_FWD_H_L0C_PONG_EVENT;
    }

    __aicore__ inline void InitL0CPipelineAic()
    {
        // 每个物理槽只在阶段入口投放一份free credit。后续所有head和chunk
        // 共用这两份credit，禁止helper反复seed同一个事件。
        for (uint32_t slot = 0; slot < KDA_FWD_H_L0C_BUFFER_DEPTH; ++slot) {
            SetFlag<HardEvent::FIX_M>(L0CEvent(slot));
        }
    }

    __aicore__ inline uint32_t AcquireL0CSlotAic()
    {
        const uint32_t slot = l0cProductIndex_ % KDA_FWD_H_L0C_BUFFER_DEPTH;
        WaitFlag<HardEvent::FIX_M>(L0CEvent(slot));
        return slot;
    }

    __aicore__ inline void DrainL0CPipelineAic()
    {
        // 阶段退出前消费两个槽的最终free credit，保证后续阶段不会继承
        // 本阶段的M/Fixpipe事件状态。
        for (uint32_t slot = 0; slot < KDA_FWD_H_L0C_BUFFER_DEPTH; ++slot) {
            WaitFlag<HardEvent::FIX_M>(L0CEvent(slot));
        }
    }

    __aicore__ inline void SelectSequence(uint64_t entity)
    {
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        if (!isVarLen_) {
            sequenceStart_ = 0;
            sequenceChunkStart_ = plan.SequenceChunkOffset(
                static_cast<uint32_t>(entity));
            sequenceChunks_ = plan.DenseFullChunksPerSequence();
            sequenceTailTokens_ = plan.DenseTailTokens();
            sequenceTotalChunks_ = sequenceChunks_ +
                static_cast<uint64_t>(sequenceTailTokens_ != 0);
            return;
        }
        sequenceChunkStart_ = plan.SequenceChunkOffset(
            static_cast<uint32_t>(entity));
        sequenceStart_ = static_cast<uint64_t>(cuSeqlens_.GetValue(entity));
        const uint64_t sequenceEnd =
            static_cast<uint64_t>(cuSeqlens_.GetValue(entity + 1));
        const uint64_t sequenceTokens = sequenceEnd - sequenceStart_;
        sequenceChunks_ = sequenceTokens / KDA_FWD_H_CHUNK;
        sequenceTailTokens_ = sequenceTokens % KDA_FWD_H_CHUNK;
        sequenceTotalChunks_ = sequenceChunks_ + (sequenceTailTokens_ != 0);
    }

    __aicore__ inline bool SelectHeadRange(uint64_t coreIdx)
    {
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        if (!plan.IsValid()) {
            return false;
        }
        fwdCoreNum_ = plan.FwdUsedCoreNum();
        if (fwdCoreNum_ == 0 || coreIdx >= fwdCoreNum_) {
            return false;
        }
        headBegin_ = coreIdx * heads_ / fwdCoreNum_;
        headEnd_ = (coreIdx + 1) * heads_ / fwdCoreNum_;
        return headBegin_ < headEnd_;
    }

    __aicore__ inline uint64_t SequenceChunks() const
    {
        return sequenceChunks_;
    }

    __aicore__ inline uint64_t SequenceChunkStart() const
    {
        return sequenceChunkStart_;
    }

    __aicore__ inline uint64_t SequenceTailTokens() const
    {
        return sequenceTailTokens_;
    }

    __aicore__ inline uint64_t MatrixOffset(
        uint64_t entity, uint64_t hv, uint64_t t) const
    {
        if (isVarLen_) {
            return (hv * seqlen_ + sequenceStart_ + t) * KDA_FWD_H_DIM;
        }
        return ((entity * heads_ + hv) * seqlen_ + t) * KDA_FWD_H_DIM;
    }

    __aicore__ inline uint64_t ChunkMatrixOffset(
        uint64_t entity, uint64_t hv, uint64_t chunk, uint64_t row = 0) const
    {
        return MatrixOffset(
            entity, hv, chunk * KDA_FWD_H_CHUNK + row);
    }

    __aicore__ inline uint64_t ScoreOffset(
        uint64_t entity, uint64_t hv, uint64_t t) const
    {
        if (isVarLen_) {
            return (hv * seqlen_ + sequenceStart_ + t) * KDA_FWD_H_CHUNK;
        }
        return ((entity * heads_ + hv) * seqlen_ + t) * KDA_FWD_H_CHUNK;
    }

    __aicore__ inline uint64_t ChunkScoreOffset(
        uint64_t entity, uint64_t hv, uint64_t chunk) const
    {
        return ScoreOffset(entity, hv, chunk * KDA_FWD_H_CHUNK);
    }

    __aicore__ inline uint64_t StateOffset(uint64_t entity, uint64_t hv) const
    {
        return (entity * heads_ + hv) * KDA_FWD_H_DIM * KDA_FWD_H_DIM;
    }

    __aicore__ inline uint64_t HOffset(
        uint64_t entity, uint64_t hv, uint64_t chunk) const
    {
        const uint64_t flatChunk = SequenceChunkStart() + chunk;
        const uint64_t chunkIndex = isVarLen_
            ? hv * totalChunks_ + flatChunk
            : (entity * heads_ + hv) * totalChunks_ + chunk;
        return chunkIndex *
               KDA_FWD_H_DIM * KDA_FWD_H_DIM;
    }

    __aicore__ inline uint64_t VNewOffset(
        uint64_t entity, uint64_t hv, uint64_t chunk, uint64_t row = 0) const
    {
        return ChunkMatrixOffset(entity, hv, chunk, row);
    }

    __aicore__ inline uint64_t OutputOffset(
        uint64_t entity, uint64_t hv, uint64_t t) const
    {
        const uint64_t token = isVarLen_ ? sequenceStart_ + t
                                         : entity * seqlen_ + t;
        return (token * heads_ + hv) * KDA_FWD_H_DIM;
    }

    __aicore__ inline uint32_t L1WOffset(uint32_t slot) const
    {
        if (slot == 0) {
            return KDA_FWD_H_L1_W_OFFSET;
        }
        if (slot == 1) {
            return KDA_FWD_H_L1_W1_OFFSET;
        }
        return slot == 2 ? KDA_FWD_H_L1_W2_OFFSET : KDA_FWD_H_L1_W3_OFFSET;
    }

    __aicore__ inline uint32_t L1QOffset(uint32_t slot) const
    {
        if (slot == 0) {
            return KDA_FWD_H_L1_Q_OFFSET;
        }
        if (slot == 1) {
            return KDA_FWD_H_L1_Q1_OFFSET;
        }
        return slot == 2 ? KDA_FWD_H_L1_Q2_OFFSET : KDA_FWD_H_L1_Q3_OFFSET;
    }

    __aicore__ inline uint32_t L1KgOffset(uint32_t slot) const
    {
        if (slot == 0) {
            return KDA_FWD_H_L1_KG_OFFSET;
        }
        if (slot == 1) {
            return KDA_FWD_H_L1_KG1_OFFSET;
        }
        return slot == 2 ? KDA_FWD_H_L1_KG2_OFFSET : KDA_FWD_H_L1_KG3_OFFSET;
    }

    __aicore__ inline uint32_t L1AqkOffset(uint32_t slot) const
    {
        if (slot == 0) {
            return KDA_FWD_H_L1_AQK_OFFSET;
        }
        if (slot == 1) {
            return KDA_FWD_H_L1_AQK1_OFFSET;
        }
        return slot == 2 ? KDA_FWD_H_L1_AQK2_OFFSET : KDA_FWD_H_L1_AQK3_OFFSET;
    }

    __aicore__ inline uint32_t L1StateOffset(uint32_t slot) const
    {
        return slot == 0 ? KDA_FWD_H_L1_H_OFFSET : KDA_FWD_H_L1_H1_OFFSET;
    }

    __aicore__ inline uint32_t L1VnewOffset(uint32_t slot) const
    {
        return slot == 0 ? KDA_FWD_H_L1_V_OFFSET : KDA_FWD_H_L1_V1_OFFSET;
    }

    template <typename DirectTileCopy, typename TensorL0C>
    __aicore__ inline void PublishDirectTile(
        TensorL0C tensorL0C, uint32_t m, uint32_t n, uint32_t l0cSlot)
    {
        auto layoutUb = tla::MakeLayout<float, LayoutRM>(m, n);
        const uint32_t directSlot =
            directPublishIndex_ % KDA_FWD_H_DIRECT_BUFFER_DEPTH;
        auto tensorUb = tla::MakeTensor(
            resource_.ubBuf.template GetBufferByByte<float>(
                KDA_FWD_H_UB_DIRECT_OFFSET +
                directSlot * KDA_FWD_H_UB_DIRECT_SLOT_BYTES),
            layoutUb, Catlass::Arch::PositionUB{});
        using CopyL0CToDst =
            typename DirectTileCopy::template CopyL0CToDst<decltype(tensorUb)>;
        CopyL0CToDst copyL0CToDst;
        const TEventID l0cEvent = L0CEvent(l0cSlot);

        WaitDirectFreeAic();
        SetFlag<HardEvent::M_FIX>(l0cEvent);
        WaitFlag<HardEvent::M_FIX>(l0cEvent);
        copyL0CToDst(tensorUb, tensorL0C);
        SetDirectReadyAic();
        ++directPublishIndex_;
        ++l0cProductIndex_;
        SetFlag<HardEvent::FIX_M>(l0cEvent);
    }

    template <typename DirectTileCopy>
    __aicore__ inline void PublishDirect(
        LocalTensor<float> l0C, uint32_t m, uint32_t n, uint32_t l0cSlot)
    {
        auto layoutL0C = tla::MakeLayoutL0C(m, n);
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        PublishDirectTile<DirectTileCopy>(tensorL0C, m, n, l0cSlot);
    }

    __aicore__ inline void ClearTailL1Tile(
        LocalTensor<T> tensor, uint32_t bytes)
    {
        LocalTensor<uint16_t> bits = tensor.template ReinterpretCast<uint16_t>();
        InitConstValueParams<uint16_t> clearParams(
            1, static_cast<uint16_t>(bytes / 32), 0, 0);
        InitConstValue(bits, clearParams);
    }

    __aicore__ inline void WaitL1StagingSlotFreeAic(uint32_t slot)
    {
        // EventID 必须在 API 调用点保持为字面量。成员 TEventID 数组的
        // 动态读取会让 A5 第四槽的归还事件无法唤醒 MTE2。
        switch (slot) {
            case 0:
                WaitFlag<HardEvent::MTE1_MTE2>(0);
                return;
            case 1:
                WaitFlag<HardEvent::MTE1_MTE2>(1);
                return;
            case 2:
                WaitFlag<HardEvent::MTE1_MTE2>(2);
                return;
            case 3:
                WaitFlag<HardEvent::MTE1_MTE2>(3);
                return;
            default:
                return;
        }
    }

    __aicore__ inline void SetL1StagingSlotFreeAic(uint32_t slot)
    {
        switch (slot) {
            case 0:
                SetFlag<HardEvent::MTE1_MTE2>(0);
                return;
            case 1:
                SetFlag<HardEvent::MTE1_MTE2>(1);
                return;
            case 2:
                SetFlag<HardEvent::MTE1_MTE2>(2);
                return;
            case 3:
                SetFlag<HardEvent::MTE1_MTE2>(3);
                return;
            default:
                return;
        }
    }

    template <bool IS_TAIL = false>
    __aicore__ inline void PrefetchIndependentProductsAic(
        uint64_t b, uint64_t hv, uint64_t chunk, uint32_t validRows = KDA_FWD_H_CHUNK)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 3);
        WaitL1StagingSlotFreeAic(slot);
        using LayoutTagL1ARm = typename TileCopyRM::LayoutTagL1A;
        using LayoutTagL1ACm = typename TileCopyCM::LayoutTagL1A;

        auto layoutToken = tla::MakeLayout<T, LayoutRM>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM);
        auto layoutKg = tla::MakeLayout<T, LayoutCM>(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK);
        auto tensorW = tla::MakeTensor(
            w_[ChunkMatrixOffset(b, hv, chunk)], layoutToken, Catlass::Arch::PositionGM{});
        auto tensorKg = tla::MakeTensor(
            kg_[ChunkMatrixOffset(b, hv, chunk)], layoutKg, Catlass::Arch::PositionGM{});
        const uint32_t copyRows = IS_TAIL ? validRows : KDA_FWD_H_CHUNK;
        auto blockW = GetTile(tensorW, tla::MakeCoord(0, 0),
                              tla::MakeShape(copyRows, KDA_FWD_H_DIM));
        auto blockKg = GetTile(tensorKg, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_DIM, copyRows));
        using CopyGmToL1ARmW = typename TileCopyRM::template CopyGmToL1A<decltype(blockW)>;
        using CopyGmToL1ACm = typename TileCopyCM::template CopyGmToL1A<decltype(blockKg)>;

        LocalTensor<T> l1W = resource_.l1Buf.template GetBufferByByte<T>(
            L1WOffset(slot));
        LocalTensor<T> l1Kg = resource_.l1Buf.template GetBufferByByte<T>(
            L1KgOffset(slot));
        auto tensorL1W = tla::MakeTensor(
            l1W, tla::MakeLayout<T, LayoutTagL1ARm>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM),
            Catlass::Arch::PositionL1{});
        auto tensorL1Kg = tla::MakeTensor(
            l1Kg, tla::MakeLayout<T, LayoutTagL1ACm>(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK),
            Catlass::Arch::PositionL1{});

        if constexpr (IS_TAIL) {
            ClearTailL1Tile(l1W, KDA_FWD_H_CHUNK * KDA_FWD_H_DIM * sizeof(T));
            ClearTailL1Tile(l1Kg, KDA_FWD_H_CHUNK * KDA_FWD_H_DIM * sizeof(T));
        }
        CopyGmToL1ARmW{}(tensorL1W, blockW);
        CopyGmToL1ACm{}(tensorL1Kg, blockKg);
        SetFlag<HardEvent::MTE2_MTE1>(aicMte2ToMte1Event_);
    }

    __aicore__ inline void ComputeStateProductsAic(
        uint64_t b, uint64_t hv, uint64_t chunk, bool inputsReady = false)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 3);
        using LayoutTagL1A = typename TileCopyRM::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyRM::LayoutTagL1B;
        using LayoutTagL0A = typename TileCopyRM::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyRM::LayoutTagL0B;
        using CopyL1ToL0A = typename TileCopyRM::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyRM::CopyL1ToL0B;
        using TileMmad = Catlass::Gemm::Tile::TileMmadTla<ArchTag, T, LayoutTagL1A>;

        LocalTensor<T> l1A0 = resource_.l1Buf.template GetBufferByByte<T>(
            L1WOffset(slot));
        const uint32_t stateSlot =
            stateConsumeIndex_ % KDA_FWD_H_L1_BUFFER_DEPTH;
        LocalTensor<T> l1B = resource_.l1Buf.template GetBufferByByte<T>(
            L1StateOffset(stateSlot));
        LocalTensor<T> l0A = resource_.l0ABuf.template GetBufferByByte<T>(
            KDA_FWD_H_L0A_STATE_OFFSET);
        LocalTensor<T> l0B = resource_.l0BBuf.template GetBufferByByte<T>(
            KDA_FWD_H_L0B_STATE_OFFSET);
        const uint32_t l0cSlot = AcquireL0CSlotAic();
        LocalTensor<float> l0C = resource_.l0CBuf.template GetBufferByByte<float>(
            l0cSlot * KDA_FWD_H_L0C_SLOT_BYTES);

        auto layoutL1A = tla::MakeLayout<T, LayoutTagL1A>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM);
        auto layoutL1B = tla::MakeLayout<T, LayoutTagL1B>(KDA_FWD_H_DIM, KDA_FWD_H_DIM);
        auto layoutL0A = tla::MakeLayout<T, LayoutTagL0A>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM);
        auto layoutL0B = tla::MakeLayout<T, LayoutTagL0B>(KDA_FWD_H_DIM, KDA_FWD_H_DIM);
        auto layoutL0C = tla::MakeLayoutL0C(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM);
        auto tensorL1A0 = tla::MakeTensor(l1A0, layoutL1A, Catlass::Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(l1B, layoutL1B, Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(l0A, layoutL0A, Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(l0B, layoutL0B, Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(l0C, layoutL0C, Catlass::Arch::PositionL0C{});
        auto tileL1B = GetTile(tensorL1B, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_DIM, KDA_FWD_H_DIM));
        auto tileL0A = GetTile(tensorL0A, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tileL1W = GetTile(tensorL1A0, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tileL0B = GetTile(tensorL0B, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_DIM, KDA_FWD_H_DIM));
        auto tileL0C = GetTile(tensorL0C, tla::MakeCoord(0, 0),
                               tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));

        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        TileMmad tileMmad;

        if (!inputsReady) {
            WaitFlag<HardEvent::MTE2_MTE1>(aicMte2ToMte1Event_);
        }
        WaitFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);
        copyL1ToL0A(tileL0A, tileL1W);
        copyL1ToL0B(tileL0B, tileL1B);
        SetFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        WaitFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        tileMmad(tileL0C, tileL0A, tileL0B, KDA_FWD_H_CHUNK,
                 KDA_FWD_H_DIM, KDA_FWD_H_DIM, true, 0);
        SetFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);
        PublishDirect<DirectTileCopyRM>(
            l0C, KDA_FWD_H_CHUNK, KDA_FWD_H_DIM, l0cSlot);
        SetL1SlotFlagAicToAiv<PIPE_FIX>(KDA_FWD_H_STATE_FREE_FLAG);
        ++stateConsumeIndex_;
    }

    __aicore__ inline void ComputeVnewProductsAic(
        uint64_t b, uint64_t hv, uint64_t chunk, bool prefetchNext)
    {
        const uint32_t slot = static_cast<uint32_t>(chunk & 3);
        using LayoutTagL1AK = typename TileCopyCM::LayoutTagL1A;
        using LayoutTagL1B = typename TileCopyRM::LayoutTagL1B;
        using LayoutTagL0AK = typename TileCopyCM::LayoutTagL0A;
        using LayoutTagL0B = typename TileCopyRM::LayoutTagL0B;
        using CopyL1ToL0AK = typename TileCopyCM::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopyRM::CopyL1ToL0B;
        using TileMmadK = Catlass::Gemm::Tile::TileMmadTla<ArchTag, T, LayoutTagL1AK>;

        LocalTensor<T> l1Kg = resource_.l1Buf.template GetBufferByByte<T>(
            L1KgOffset(slot));
        const uint32_t vnewSlot =
            vnewConsumeIndex_ % KDA_FWD_H_L1_BUFFER_DEPTH;
        LocalTensor<T> l1V = resource_.l1Buf.template GetBufferByByte<T>(
            L1VnewOffset(vnewSlot));
        LocalTensor<T> l0A = resource_.l0ABuf.template GetBufferByByte<T>(
            KDA_FWD_H_L0A_VNEW_OFFSET);
        LocalTensor<T> l0B = resource_.l0BBuf.template GetBufferByByte<T>(
            KDA_FWD_H_L0B_VNEW_OFFSET);
        const uint32_t l0cSlot = AcquireL0CSlotAic();
        LocalTensor<float> l0C = resource_.l0CBuf.template GetBufferByByte<float>(
            l0cSlot * KDA_FWD_H_L0C_SLOT_BYTES);

        auto layoutL1Kg = tla::MakeLayout<T, LayoutTagL1AK>(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK);
        auto layoutL1V = tla::MakeLayout<T, LayoutTagL1B>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM);
        auto layoutL0Kg = tla::MakeLayout<T, LayoutTagL0AK>(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK);
        auto layoutL0V = tla::MakeLayout<T, LayoutTagL0B>(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM);
        auto baseL1Kg = tla::MakeTensor(l1Kg, layoutL1Kg, Catlass::Arch::PositionL1{});
        auto baseL1V = tla::MakeTensor(l1V, layoutL1V, Catlass::Arch::PositionL1{});
        auto baseL0Kg = tla::MakeTensor(l0A, layoutL0Kg, Catlass::Arch::PositionL0A{});
        auto baseL0V = tla::MakeTensor(l0B, layoutL0V, Catlass::Arch::PositionL0B{});
        auto baseL0Update = tla::MakeTensor(
            l0C, tla::MakeLayoutL0C(KDA_FWD_H_DIM, KDA_FWD_H_DIM),
            Catlass::Arch::PositionL0C{});
        auto tensorL1Kg = GetTile(baseL1Kg, tla::MakeCoord(0, 0),
                                  tla::MakeShape(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK));
        auto tensorL1V = GetTile(baseL1V, tla::MakeCoord(0, 0),
                                 tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tensorL0Kg = GetTile(baseL0Kg, tla::MakeCoord(0, 0),
                                  tla::MakeShape(KDA_FWD_H_DIM, KDA_FWD_H_CHUNK));
        auto tensorL0V = GetTile(baseL0V, tla::MakeCoord(0, 0),
                                 tla::MakeShape(KDA_FWD_H_CHUNK, KDA_FWD_H_DIM));
        auto tensorL0Update = GetTile(baseL0Update, tla::MakeCoord(0, 0),
                                      tla::MakeShape(KDA_FWD_H_DIM, KDA_FWD_H_DIM));

        CopyL1ToL0AK copyL1ToL0AK;
        CopyL1ToL0B copyL1ToL0B;
        TileMmadK tileMmadK;

        WaitFlag<HardEvent::M_MTE1>(vnewL0FreeEvent_);
        copyL1ToL0AK(tensorL0Kg, tensorL1Kg);
        copyL1ToL0B(tensorL0V, tensorL1V);
        SetFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        WaitFlag<HardEvent::MTE1_M>(aicMte1ToMEvent_);
        tileMmadK(tensorL0Update, tensorL0Kg, tensorL0V,
                  KDA_FWD_H_DIM, KDA_FWD_H_DIM, KDA_FWD_H_CHUNK, true, 0);
        SetFlag<HardEvent::M_MTE1>(vnewL0FreeEvent_);
        PublishDirect<DirectTileCopyCM>(
            l0C, KDA_FWD_H_DIM, KDA_FWD_H_DIM, l0cSlot);
        SetL1SlotFlagAicToAiv<PIPE_FIX>(KDA_FWD_H_VNEW_FREE_FLAG);
        ++vnewConsumeIndex_;
        SetL1StagingSlotFreeAic(slot);
        if (prefetchNext) {
            PrefetchIndependentProductsAic(b, hv, chunk + 1);
        }
    }

    __aicore__ inline void ProcessAic()
    {
        const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx());
        if (!SelectHeadRange(coreIdx)) {
            return;
        }
        SetLoadDataPaddingValue<T>(static_cast<T>(0));
        SetFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);
        InitL0CPipelineAic();
        SetL1StagingSlotFreeAic(0);
        SetL1StagingSlotFreeAic(1);
        SetL1StagingSlotFreeAic(2);
        SetL1StagingSlotFreeAic(3);
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        for (uint32_t ordinal = 0; ordinal < plan.AlignedSequenceCount(); ++ordinal) {
            const uint64_t b = plan.AlignedSequenceId(ordinal);
            SelectSequence(b);
            ProcessSelectedSequenceAic<false>(b);
        }
        for (uint32_t ordinal = 0; ordinal < plan.TailedSequenceCount(); ++ordinal) {
            const uint64_t b = plan.TailedSequenceId(ordinal);
            SelectSequence(b);
            ProcessSelectedSequenceAic<true>(b);
        }
        DrainDirectFreeCreditsAic();
        DrainL0CPipelineAic();
        WaitFlag<HardEvent::M_MTE1>(stateL0FreeEvent_);
        WaitL1StagingSlotFreeAic(0);
        WaitL1StagingSlotFreeAic(1);
        WaitL1StagingSlotFreeAic(2);
        WaitL1StagingSlotFreeAic(3);
    }

    template <bool HAS_TAIL>
    __aicore__ inline void ProcessSelectedSequenceAic(uint64_t b)
    {
        const uint64_t sequenceChunks = SequenceChunks();
        const uint32_t tailTokens = static_cast<uint32_t>(SequenceTailTokens());
        for (uint64_t hv = headBegin_; hv < headEnd_; ++hv) {
            if (sequenceChunks != 0) {
                PrefetchIndependentProductsAic(b, hv, 0);
                for (uint64_t chunk = 0; chunk < sequenceChunks; ++chunk) {
                    WaitL1SlotReadyMte1(KDA_FWD_H_STATE_READY_FLAG);
                    ComputeStateProductsAic(b, hv, chunk);
                    WaitL1SlotReadyMte1(KDA_FWD_H_VNEW_READY_FLAG);
                    ComputeVnewProductsAic(
                        b, hv, chunk, chunk + 1 < sequenceChunks);
                }
            }
            if constexpr (HAS_TAIL) {
                const uint64_t tailChunk = sequenceChunks;
                WaitL1SlotReadyMte1(KDA_FWD_H_STATE_READY_FLAG);
                PrefetchIndependentProductsAic<true>(
                    b, hv, tailChunk, tailTokens);
                ComputeStateProductsAic(b, hv, tailChunk);
                WaitL1SlotReadyMte1(KDA_FWD_H_VNEW_READY_FLAG);
                ComputeVnewProductsAic(b, hv, tailChunk, false);
            }
        }
    }

    __aicore__ inline void InitializeStateAiv(
        uint64_t b, uint64_t hv, uint32_t rowBegin,
        LocalTensor<float> state)
    {
        if (hasInitialState_) {
            DataCopy(state, initialState_[StateOffset(b, hv) + rowBegin * KDA_FWD_H_DIM],
                     KDA_FWD_H_STATE_SUB_ELEMS);
            SetFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
        } else {
            Duplicate(state, 0.0f, KDA_FWD_H_STATE_SUB_ELEMS);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreCurrentStateAiv(
        uint64_t b, uint64_t hv, uint64_t chunk, uint32_t rowBegin,
        LocalTensor<float> state, LocalTensor<T> stateTyped)
    {
        Cast(stateTyped, state, RoundMode::CAST_RINT, KDA_FWD_H_STATE_SUB_ELEMS);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        DataCopy(h_[HOffset(b, hv, chunk) + rowBegin * KDA_FWD_H_DIM],
                 stateTyped, KDA_FWD_H_STATE_SUB_ELEMS);
        SetFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);

        constexpr uint32_t columnGroup = 64;
        constexpr uint32_t columnGroups = KDA_FWD_H_DIM / columnGroup;
        for (uint32_t group = 0; group < columnGroups; ++group) {
            Cast(stateTyped[group * KDA_FWD_H_SUB_DIM * columnGroup],
                 state[group * columnGroup], RoundMode::CAST_RINT,
                 columnGroup, KDA_FWD_H_SUB_DIM,
                 {static_cast<uint16_t>(KDA_FWD_H_SUB_DIM), 1, 1,
                  static_cast<uint8_t>(columnGroups * 8)});
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        const uint32_t subBlockIdx = rowBegin / KDA_FWD_H_SUB_DIM;
        const uint32_t stateSlot =
            statePublishCount_[subBlockIdx] % KDA_FWD_H_L1_BUFFER_DEPTH;
        LocalTensor<T> l1State = resource_.l1Buf.template GetBufferByByte<T>(
            L1StateOffset(stateSlot));
        WaitL1SlotFreeMte3(
            KDA_FWD_H_STATE_FREE_FLAG, statePublishCount_[subBlockIdx]);
        DataCopyParams stateCopyParams;
        stateCopyParams.blockCount = KDA_FWD_H_DIM / 16;
        stateCopyParams.blockLen = KDA_FWD_H_SUB_DIM;
        stateCopyParams.srcGap = 0;
        stateCopyParams.dstGap = KDA_FWD_H_DIM - KDA_FWD_H_SUB_DIM;
        DataCopy(l1State[rowBegin * 16], stateTyped, stateCopyParams);
        SetFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
        SetL1SlotFlagAivToAic<PIPE_MTE3>(KDA_FWD_H_STATE_READY_FLAG);
        ++statePublishCount_[subBlockIdx];
    }

    template <bool IS_TAIL = false>
    __aicore__ inline void ProcessChunkAiv(
        uint64_t b, uint64_t hv, uint32_t chunk,
        uint32_t subBlockIdx, LocalTensor<float> state,
        LocalTensor<T> stateTyped, LocalTensor<float> direct,
        LocalTensor<float> vnew, LocalTensor<T> ioTyped, LocalTensor<float> gate,
        uint32_t validRows = KDA_FWD_H_SUB_CHUNK)
    {
        const uint32_t tokenBegin = subBlockIdx * KDA_FWD_H_SUB_CHUNK;
        const uint32_t stateRowBegin = subBlockIdx * KDA_FWD_H_SUB_DIM;
        StoreCurrentStateAiv(b, hv, chunk, stateRowBegin, state, stateTyped);

        direct = AcquireDirectBufferAiv();
        WaitFlag<HardEvent::MTE3_MTE2>(aivMte3ToMte2Event_);
        const uint32_t validElems = validRows * KDA_FWD_H_DIM;
        if constexpr (IS_TAIL) {
            if (validElems != 0) {
                DataCopyExtParams copyParams{
                    1, static_cast<uint32_t>(validElems * sizeof(T)), 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
                DataCopyPad(
                    ioTyped, u_[ChunkMatrixOffset(b, hv, chunk, tokenBegin)],
                    copyParams, padParams);
                SetFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
                WaitFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
                Cast(vnew, ioTyped, RoundMode::CAST_NONE, validElems);
            }
            if (validElems < KDA_FWD_H_TOKEN_SUB_ELEMS) {
                Duplicate(
                    vnew[validElems], 0.0f,
                    KDA_FWD_H_TOKEN_SUB_ELEMS - validElems);
            }
        } else {
            DataCopy(ioTyped, u_[ChunkMatrixOffset(b, hv, chunk, tokenBegin)],
                     KDA_FWD_H_TOKEN_SUB_ELEMS);
            SetFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
            Cast(vnew, ioTyped, RoundMode::CAST_NONE, KDA_FWD_H_TOKEN_SUB_ELEMS);
        }
        PipeBarrier<PIPE_V>();
        Sub(vnew, vnew, direct, KDA_FWD_H_TOKEN_SUB_ELEMS);
        PipeBarrier<PIPE_V>();
        if constexpr (IS_TAIL) {
            if (validElems < KDA_FWD_H_TOKEN_SUB_ELEMS) {
                Duplicate(
                    vnew[validElems], 0.0f,
                    KDA_FWD_H_TOKEN_SUB_ELEMS - validElems);
                PipeBarrier<PIPE_V>();
            }
        }
        SetDirectFreeAiv();
        Cast(ioTyped, vnew, RoundMode::CAST_RINT, KDA_FWD_H_TOKEN_SUB_ELEMS);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        if constexpr (IS_TAIL) {
            if (validRows != 0) {
                DataCopyExtParams copyParams{
                    1,
                    static_cast<uint32_t>(validRows * KDA_FWD_H_DIM * sizeof(T)),
                    0, 0, 0};
                DataCopyPad(
                    vNew_[VNewOffset(b, hv, chunk, tokenBegin)],
                    ioTyped, copyParams);
            }
        } else {
            DataCopy(vNew_[VNewOffset(b, hv, chunk, tokenBegin)],
                     ioTyped, KDA_FWD_H_TOKEN_SUB_ELEMS);
        }
        SetFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);

        constexpr uint32_t columnGroup = 64;
        constexpr uint32_t columnGroups = KDA_FWD_H_DIM / columnGroup;
        for (uint32_t group = 0; group < columnGroups; ++group) {
            Cast(ioTyped[group * KDA_FWD_H_SUB_CHUNK * columnGroup],
                 vnew[group * columnGroup], RoundMode::CAST_RINT,
                 columnGroup, KDA_FWD_H_SUB_CHUNK,
                 {static_cast<uint16_t>(KDA_FWD_H_SUB_CHUNK), 1, 1,
                  static_cast<uint8_t>(columnGroups * 8)});
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        WaitFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
        const uint32_t vnewSlot =
            vnewPublishCount_[subBlockIdx] % KDA_FWD_H_L1_BUFFER_DEPTH;
        LocalTensor<T> l1Vnew = resource_.l1Buf.template GetBufferByByte<T>(
            L1VnewOffset(vnewSlot));
        WaitL1SlotFreeMte3(
            KDA_FWD_H_VNEW_FREE_FLAG, vnewPublishCount_[subBlockIdx]);
        DataCopyParams vnewL1CopyParams;
        vnewL1CopyParams.blockCount = KDA_FWD_H_DIM / 16;
        vnewL1CopyParams.blockLen = KDA_FWD_H_SUB_CHUNK;
        vnewL1CopyParams.srcGap = 0;
        vnewL1CopyParams.dstGap = KDA_FWD_H_CHUNK - KDA_FWD_H_SUB_CHUNK;
        DataCopy(l1Vnew[tokenBegin * 16], ioTyped, vnewL1CopyParams);
        SetFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
        SetL1SlotFlagAivToAic<PIPE_MTE3>(KDA_FWD_H_VNEW_READY_FLAG);
        ++vnewPublishCount_[subBlockIdx];

        direct = AcquireDirectBufferAiv();
        const uint32_t gateRow = IS_TAIL ? SequenceTailTokens() - 1
                                         : KDA_FWD_H_CHUNK - 1;
        DataCopy(gate, gk_[ChunkMatrixOffset(b, hv, chunk, gateRow) +
                           stateRowBegin], KDA_FWD_H_SUB_DIM);
        SetFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(aivMte2ToVEvent_);
        Muls(gate, gate, 0.6931471805599453f, KDA_FWD_H_SUB_DIM);
        PipeBarrier<PIPE_V>();
        Exp(gate, gate, KDA_FWD_H_SUB_DIM);
        PipeBarrier<PIPE_V>();
        AscendC::VF_CALL<Catlass::Epilogue::Block::detail::ApplyKGateUpdateRegbaseDualIssue<float>>(
            reinterpret_cast<__ubuf__ float *>(direct.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(state.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float *>(gate.GetPhyAddr()),
            static_cast<uint16_t>(KDA_FWD_H_SUB_DIM),
            static_cast<uint16_t>(KDA_FWD_H_DIM));
        PipeBarrier<PIPE_V>();
        Adds(state, direct, 0.0f, KDA_FWD_H_STATE_SUB_ELEMS);
        PipeBarrier<PIPE_V>();
        SetDirectFreeAiv();
        SetFlag<HardEvent::MTE3_MTE2>(aivMte3ToMte2Event_);
    }

    __aicore__ inline void ProcessAiv()
    {
        const uint32_t subBlockIdx = static_cast<uint32_t>(GetSubBlockIdx());
        const uint32_t subBlockNum = static_cast<uint32_t>(GetSubBlockNum());
        const uint64_t coreIdx = static_cast<uint64_t>(GetBlockIdx()) / subBlockNum;
        if (!SelectHeadRange(coreIdx)) {
            return;
        }
        LocalTensor<float> state =
            resource_.ubBuf.template GetBufferByByte<float>(KDA_FWD_H_UB_STATE_OFFSET);
        LocalTensor<T> stateTyped =
            resource_.ubBuf.template GetBufferByByte<T>(KDA_FWD_H_UB_STATE_TYPED_OFFSET);
        LocalTensor<float> direct =
            resource_.ubBuf.template GetBufferByByte<float>(KDA_FWD_H_UB_DIRECT_OFFSET);
        LocalTensor<float> vnew =
            resource_.ubBuf.template GetBufferByByte<float>(KDA_FWD_H_UB_VNEW_OFFSET);
        LocalTensor<T> ioTyped =
            resource_.ubBuf.template GetBufferByByte<T>(KDA_FWD_H_UB_IO_OFFSET);
        LocalTensor<float> gate =
            resource_.ubBuf.template GetBufferByByte<float>(KDA_FWD_H_UB_GATE_OFFSET);

        InitializeDirectFreeCreditsAiv();
        SetFlag<HardEvent::MTE3_MTE2>(aivMte3ToMte2Event_);
        KdaForward::CompactSequencePlanView plan(compactPlanAddr_);
        for (uint32_t ordinal = 0; ordinal < plan.AlignedSequenceCount(); ++ordinal) {
            const uint64_t b = plan.AlignedSequenceId(ordinal);
            SelectSequence(b);
            ProcessSelectedSequenceAiv<false>(
                b, subBlockIdx, state, stateTyped, direct, vnew, ioTyped, gate);
        }
        for (uint32_t ordinal = 0; ordinal < plan.TailedSequenceCount(); ++ordinal) {
            const uint64_t b = plan.TailedSequenceId(ordinal);
            SelectSequence(b);
            ProcessSelectedSequenceAiv<true>(
                b, subBlockIdx, state, stateTyped, direct, vnew, ioTyped, gate);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(aivMte3ToMte2Event_);
        DrainL1FreeCreditsAiv(subBlockIdx);
    }

    template <bool HAS_TAIL>
    __aicore__ inline void ProcessSelectedSequenceAiv(
        uint64_t b, uint32_t subBlockIdx, LocalTensor<float> state,
        LocalTensor<T> stateTyped, LocalTensor<float> direct,
        LocalTensor<float> vnew, LocalTensor<T> ioTyped,
        LocalTensor<float> gate)
    {
        const uint32_t sequenceChunks = static_cast<uint32_t>(SequenceChunks());
        const uint32_t tailTokens = static_cast<uint32_t>(SequenceTailTokens());
        for (uint64_t hv = headBegin_; hv < headEnd_; ++hv) {
            ProcessSequenceHeadAiv<HAS_TAIL>(
                b, hv, sequenceChunks, tailTokens, subBlockIdx,
                state, stateTyped, direct, vnew, ioTyped, gate);
        }
    }

    template <bool HAS_TAIL>
    __aicore__ inline void ProcessSequenceHeadAiv(
        uint64_t b, uint64_t hv, uint32_t sequenceChunks, uint32_t tailTokens,
        uint32_t subBlockIdx, LocalTensor<float> state,
        LocalTensor<T> stateTyped, LocalTensor<float> direct,
        LocalTensor<float> vnew, LocalTensor<T> ioTyped, LocalTensor<float> gate)
    {
        const uint32_t stateRowBegin = subBlockIdx * KDA_FWD_H_SUB_DIM;
        InitializeStateAiv(b, hv, stateRowBegin, state);
        if (sequenceChunks != 0 || HAS_TAIL) {
            for (uint32_t chunk = 0; chunk < sequenceChunks; ++chunk) {
                ProcessChunkAiv(
                    b, hv, chunk, subBlockIdx,
                    state, stateTyped, direct, vnew, ioTyped, gate);
            }
            if constexpr (HAS_TAIL) {
                const uint32_t tokenBegin = subBlockIdx * KDA_FWD_H_SUB_CHUNK;
                const uint32_t validRows = tailTokens > tokenBegin
                    ? min(tailTokens - tokenBegin, KDA_FWD_H_SUB_CHUNK)
                    : 0;
                ProcessChunkAiv<true>(
                    b, hv, sequenceChunks, subBlockIdx,
                    state, stateTyped, direct, vnew, ioTyped, gate,
                    validRows);
            }
        }
        if (storeFinalState_) {
            SetFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
            WaitFlag<HardEvent::V_MTE3>(aivVToMte3Event_);
            DataCopy(finalState_[StateOffset(b, hv) +
                     stateRowBegin * KDA_FWD_H_DIM],
                     state, KDA_FWD_H_STATE_SUB_ELEMS);
            SetFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
            WaitFlag<HardEvent::MTE3_V>(aivMte3ToVEvent_);
        }
    }

private:
    GlobalTensor<GK_T> gk_;
    GlobalTensor<float> initialState_;
    GlobalTensor<float> finalState_;
    GlobalTensor<T> w_;
    GlobalTensor<T> u_;
    GlobalTensor<T> kg_;
    GlobalTensor<T> vNew_;
    GlobalTensor<T> h_;
    GlobalTensor<int64_t> cuSeqlens_;
    GM_ADDR compactPlanAddr_ = nullptr;
    uint32_t statePublishCount_[2] = {0, 0};
    uint32_t vnewPublishCount_[2] = {0, 0};
    uint32_t stateConsumeIndex_ = 0;
    uint32_t vnewConsumeIndex_ = 0;
    uint32_t directPublishIndex_ = 0;
    uint32_t directConsumeIndex_ = 0;
    uint32_t l0cProductIndex_ = 0;
    TEventID aicMte2ToMte1Event_ = KDA_FWD_H_MTE_W_EVENT;
    TEventID stateL0FreeEvent_ = KDA_FWD_H_M_EVENT;
    TEventID vnewL0FreeEvent_ = KDA_FWD_H_M_EVENT;
    TEventID aicMte1ToMEvent_ = KDA_FWD_H_M_EVENT;
    TEventID aivMte2ToVEvent_ = KDA_FWD_H_MTE_W_EVENT;
    TEventID aivVToMte3Event_ = KDA_FWD_H_MTE_Q_EVENT;
    TEventID aivMte3ToVEvent_ = KDA_FWD_H_MTE_B_EVENT;
    TEventID aivMte3ToMte2Event_ = KDA_FWD_H_IO_REUSE_EVENT;
    Catlass::Arch::Resource<ArchTag> resource_;
    uint64_t batch_ = 0;
    uint64_t seqNum_ = 0;
    uint64_t heads_ = 0;
    uint64_t seqlen_ = 0;
    uint64_t totalChunks_ = 0;
    uint64_t sequenceStart_ = 0;
    uint64_t sequenceChunkStart_ = 0;
    uint64_t sequenceChunks_ = 0;
    uint64_t sequenceTotalChunks_ = 0;
    uint64_t sequenceTailTokens_ = 0;
    uint64_t fwdCoreNum_ = 1;
    uint64_t headBegin_ = 0;
    uint64_t headEnd_ = 0;
    bool hasInitialState_ = false;
    bool isVarLen_ = false;
    bool storeFinalState_ = false;
};

} // namespace KdaForward::arch35

#endif
