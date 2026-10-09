/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH35_CUBE_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH35_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "chunk_kda_bwd_finalize_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace KDA {

// Cube 流水：Process 只组织任务与阶段；各阶段负责选操作数、计算和发布结果。
// Stage10 类名沿用历史命名，实际矩阵阶段为 0、1、3、4、6、8。
class ChunkKdaBwdFinalizeCubeStage10 {
public:
    __aicore__ inline void Init(
        GM_ADDR v, GM_ADDR akk, GM_ADDR vNew, GM_ADDR h, GM_ADDR dh,
        GM_ADDR dvScan, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        GM_ADDR workspace, const ChunkKdaBwdFinalizeTilingData *tiling)
    {
        v_ = v;
        akk_ = StreamingInput(akk);
        vNew_ = StreamingInput(vNew);
        h_ = h;
        dh_ = dh;
        dvScan_ = StreamingInput(dvScan);
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        workspace_ = workspace;
        tiling_ = tiling;
    }

    // 这些输入每个 head/chunk 只从 GM 读取一次，之后保留在 L1。
    // 关闭其 L2 缓存，为 Cube/Vector 共用的状态和 gate 数据留出容量。
    __aicore__ inline GM_ADDR StreamingInput(GM_ADDR address)
    {
        AscendC::GlobalTensor<DT> tensor;
        tensor.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(address));
        tensor.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        return reinterpret_cast<GM_ADDR>(const_cast<__gm__ DT *>(tensor.GetPhyAddr()));
    }

    // 按核轮转处理双头窗口；headGeneration 在各阶段重放，以保持 AIV/slot 映射一致。
    __aicore__ inline void Process()
    {
        // 1. 配置矩阵布局、初始化本地事件，并允许 AIV 发布首个任务的 L1 操作数。
        AscendC::SetMMLayoutTransform(true);
        AscendC::SetHF32Mode(false);
        Catlass::Arch::Resource<ArchTag> resource;
        InitEvents();
        ReleaseTaskL1();
        const int64_t coreIdx = AscendC::GetBlockIdx();
        const int64_t coreNum = AscendC::GetBlockNum();
        uint64_t groupGeneration = 0;
        uint64_t headGeneration = 0;
        for (int64_t workTask = coreIdx; workTask < tiling_->workTaskNum;
             workTask += coreNum, ++groupGeneration) {
            const int64_t headWindow = workTask / tiling_->chunkTaskNum;
            const int64_t chunkTask = workTask - headWindow * tiling_->chunkTaskNum;
            const int64_t headBegin = headWindow * KDA_FINALIZE_HEADS_PER_WINDOW;
            const int64_t headEnd = FinalizeMin(
                headBegin + KDA_FINALIZE_HEADS_PER_WINDOW, tiling_->NV);
            FinalizeChunkInfo chunk;
            ResolveFinalizeChunk(chunkTask, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }

            // 2. Stage0 输入预取：先填充两份 112 KiB L1 stream。
            // 当前双头窗口各使用一份；保留的 head+2 补载分支在双头配置下不进入。
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
            const int64_t preloadEnd = FinalizeMin(headBegin + 2, headEnd);
            for (int64_t head = headBegin; head < preloadEnd; ++head) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                LoadStage0(resource, chunk, head, owner);
            }

            // 3. Stage0：逐 head 发射四项矩阵乘，结果按消费者分别写 GM、L1 或 UB。
            for (int64_t head = headBegin; head < headEnd; ++head, ++headGeneration) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(headGeneration & 1U);
                const uint32_t slot = static_cast<uint32_t>((headGeneration >> 1U) & 1U);
                ComputeStage0(resource, chunk, owner, aiv, slot,
                              coreIdx, groupGeneration);
                const int64_t nextHead = head + 2;
                if (nextHead < headEnd) {
                    LoadStage0(resource, chunk, nextHead,
                               static_cast<uint32_t>(nextHead - headBegin));
                }
            }

            headGeneration -= static_cast<uint64_t>(headEnd - headBegin);

            // 4. Stage1：逐 head 等待 dW/kE 高低位就绪，计算 dKgb 与 zW。
            for (int64_t head = headBegin; head < headEnd; ++head, ++headGeneration) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(headGeneration & 1U);
                const uint32_t slot = static_cast<uint32_t>((headGeneration >> 1U) & 1U);
                RunStage1(resource, chunk, owner, aiv, slot,
                          coreIdx, groupGeneration);
            }
            headGeneration -= static_cast<uint64_t>(headEnd - headBegin);

            // 5. Stage3：消费 AIV 发布的 Zb，复用驻留 Akk 计算 Tza。
            for (int64_t head = headBegin; head < headEnd; ++head, ++headGeneration) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(headGeneration & 1U);
                const uint32_t slot = static_cast<uint32_t>((headGeneration >> 1U) & 1U);
                RunStage3(resource, chunk, owner, aiv, slot);
            }
            headGeneration -= static_cast<uint64_t>(headEnd - headBegin);

            // 6. Stage4：消费 Akk/Tza 高低位，直接向目标 AIV 的 UB 写 FP32 dAkk。
            // 目标区由通知协议保护，AIV 的 StateAndBase 按独立依赖推进。
            for (int64_t head = headBegin; head < headEnd; ++head, ++headGeneration) {
                const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                const uint32_t aiv = static_cast<uint32_t>(headGeneration & 1U);
                const uint32_t slot = static_cast<uint32_t>((headGeneration >> 1U) & 1U);
                RunStage4(resource, chunk, owner, aiv, slot);
            }

            // 7. 每次处理 32 行：Stage6 计算 dq_local/left，Stage8 计算 right。
            // 每带使用独立 gate 平移中心，同一 owner 的 L1 操作数槽按带复用。
            for (uint32_t rowBegin = 0; rowBegin < chunk.validRows;
                 rowBegin += KDA_FINALIZE_INTRA_ROWS) {
                headGeneration -= static_cast<uint64_t>(headEnd - headBegin);
                for (int64_t head = headBegin; head < headEnd; ++head, ++headGeneration) {
                    const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                    const uint32_t aiv = static_cast<uint32_t>(headGeneration & 1U);
                    const uint32_t slot = static_cast<uint32_t>((headGeneration >> 1U) & 1U);
                    RunStage6(resource, chunk, owner, aiv, slot, rowBegin);
                }
                headGeneration -= static_cast<uint64_t>(headEnd - headBegin);
                for (int64_t head = headBegin; head < headEnd; ++head, ++headGeneration) {
                    const uint32_t owner = static_cast<uint32_t>(head - headBegin);
                    const uint32_t aiv = static_cast<uint32_t>(headGeneration & 1U);
                    const uint32_t slot = static_cast<uint32_t>((headGeneration >> 1U) & 1U);
                    RunStage8(resource, chunk, owner, aiv, slot, rowBegin);
                }
            }

            // 8. 本任务最后一次 L1 读取后，允许下一任务复用相同物理区。
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
            ReleaseTaskL1();
        }

        // 9. 消费末次本地 FREE，恢复布局设置后退出。
        DrainEvents();
        AscendC::SetMMLayoutTransform(false);
    }

private:
    // 向两个 AIV 归还任务级 L1 写权限；空闲 AIV 也必须消费该通知。
    __aicore__ inline void ReleaseTaskL1()
    {
        for (uint32_t aiv = 0; aiv < 2; ++aiv) {
            AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE1>(
                KDA_FINALIZE_TASK_L1_FREE + aiv * KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE);
        }
    }

    using ArchTag = Catlass::Arch::Ascend950;
    using DT = bfloat16_t;
    using Acc = float;
    using LayoutRM = Catlass::layout::RowMajor;
    using LayoutCM = Catlass::layout::ColumnMajor;
    using CopyTransB = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRM, DT, LayoutCM, Acc, LayoutRM>;
    using CopyTransA = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutCM, DT, LayoutRM, Acc, LayoutRM>;
    using CopyRegular = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRM, DT, LayoutRM, Acc, LayoutRM>;
    using CopyStage3 = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutCM, DT, LayoutCM, Acc, LayoutRM>;
    using CopyTransBToUb = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, DT, LayoutRM, DT, LayoutCM, Acc, LayoutRM>;
    using CopyTransAToUbBf16 = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, DT, LayoutCM, DT, LayoutRM, DT, LayoutRM>;
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, DT, typename CopyRegular::LayoutTagL1A>;

    // 矩阵输出路径：GM_FP32 写 workspace；L1_BF16 写高位；WITH_RAW 另送 FP32 到 UB；
    // AIV_UB_* 直接交给目标 AIV。路径同时决定对应 FREE/READY 的等待与发布。
    enum class ResultPath : uint32_t {
        GM_FP32,
        L1_BF16,
        L1_BF16_WITH_RAW,
        AIV_UB_FP32,
        AIV_UB_BF16,
    };

    // L1 按阶段复用：Akk/dW/kE/Zb/Tza 为驻留结果，两份 STREAM 为 Stage0 输入。
    // 以下保留物理偏移，不根据当前 head 数压缩；后续 LocalOperand 会复用其中部分区域。
    static constexpr uint32_t L1_AKK = 0;
    static constexpr uint32_t L1_DW = 32 * 1024;
    static constexpr uint32_t L1_KE = 96 * 1024;
    static constexpr uint32_t L1_ZB = 160 * 1024;
    static constexpr uint32_t L1_STREAM = 192 * 1024;
    static constexpr uint32_t STREAM_BYTES = 112 * 1024;
    static constexpr uint32_t STREAM_VNEW = 0;
    static constexpr uint32_t STREAM_DH = 16 * 1024;
    static constexpr uint32_t STREAM_DVSCAN = 48 * 1024;
    static constexpr uint32_t STREAM_H = 64 * 1024;
    static constexpr uint32_t STREAM_V = 96 * 1024;
    static constexpr uint32_t L1_TZA = 416 * 1024;
    static constexpr uint32_t STAGE0_READY_COUNT = 4;
    static constexpr uint32_t L0_BYTES = 32 * 1024;
    static constexpr uint32_t L0C_BYTES = 128 * 1024;
    static constexpr AscendC::FixpipeConfig FIX_NZ_L1 = {
        AscendC::CO2Layout::NZ, false};

    // GM leading dimension 与物理 tile 固定，Akk 行跨度仍为 64。
    // 尾块只缩小有效拷贝范围；先在同一 MTE2 pipe 清零 padding，再载入有效数据。
    template <typename TileCopy, typename GmLayout>
    __aicore__ inline void LoadGmToL1A(
        AscendC::LocalTensor<DT> dst, GM_ADDR src,
        uint32_t m, uint32_t k, uint32_t validM = 0, uint32_t validK = 0)
    {
        validM = validM == 0 ? m : validM;
        validK = validK == 0 ? k : validK;
        if (validM != m || validK != k) {
            AscendC::InitConstValue(dst, AscendC::InitConstValueParams<DT>(
                1, static_cast<uint16_t>(m * k * sizeof(DT) / 32), 0, static_cast<DT>(0)));
            // 清零和 ND→NZ 搬运写同一 L1 区；先完成清零，避免晚到的写覆盖有效列。
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        AscendC::GlobalTensor<DT> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(src));
        auto tensorGm = tla::MakeTensor(
            gm, tla::MakeLayout<DT, GmLayout>(m, k), Catlass::Arch::PositionGM{});
        auto block = tla::GetTile(tensorGm, tla::MakeCoord(0, 0), tla::MakeShape(validM, validK));
        auto tensorL1 = tla::MakeTensor(
            dst, tla::MakeLayout<DT, typename TileCopy::LayoutTagL1A>(m, k),
            Catlass::Arch::PositionL1{});
        typename TileCopy::template CopyGmToL1A<decltype(block)>{}(tensorL1, block);
    }

    template <typename TileCopy, typename GmLayout>
    __aicore__ inline void LoadGmToL1B(
        AscendC::LocalTensor<DT> dst, GM_ADDR src,
        uint32_t k, uint32_t n, uint32_t validK = 0, uint32_t validN = 0)
    {
        validK = validK == 0 ? k : validK;
        validN = validN == 0 ? n : validN;
        if (validK != k || validN != n) {
            AscendC::InitConstValue(dst, AscendC::InitConstValueParams<DT>(
                1, static_cast<uint16_t>(k * n * sizeof(DT) / 32), 0, static_cast<DT>(0)));
            // 与 A 操作数相同：先排空清零写入，再载入有效 B tile。
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        AscendC::GlobalTensor<DT> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(src));
        auto tensorGm = tla::MakeTensor(
            gm, tla::MakeLayout<DT, GmLayout>(k, n), Catlass::Arch::PositionGM{});
        auto block = tla::GetTile(tensorGm, tla::MakeCoord(0, 0), tla::MakeShape(validK, validN));
        auto tensorL1 = tla::MakeTensor(
            dst, tla::MakeLayout<DT, typename TileCopy::LayoutTagL1B>(k, n),
            Catlass::Arch::PositionL1{});
        typename TileCopy::template CopyGmToL1B<decltype(block)>{}(tensorL1, block);
    }

    // 基础 GEMM：BF16 操作数、FP32 累加；按 ResultPath 选择消费者。
    // SPLIT=0：仅高位；1/2：补左/右残差；3：补两侧交叉项，均不计算 low×low。
    // SIGNAL_L1_READY 发布本地 L1 通知；WAIT_AIV_FREE 控制 UB 写前等待；CONCAT_RIGHT 沿 K 拼接。
    template <typename TileCopy, ResultPath PATH, bool SIGNAL_L1_READY = false,
              bool WAIT_AIV_FREE = true, bool CONCAT_RIGHT = false, uint32_t SPLIT = 0>
    __aicore__ inline void RunGemm(
        Catlass::Arch::Resource<ArchTag> &resource,
        AscendC::LocalTensor<DT> l1A, AscendC::LocalTensor<DT> l1B,
        uint32_t m, uint32_t n, uint32_t k, uint32_t slot,
        GM_ADDR gmDst, AscendC::LocalTensor<DT> l1Dst,
        uint32_t aiv, uint32_t aivSlot, uint32_t aivUbOffset,
        uint64_t freeFlag, uint64_t readyFlag,
        AscendC::LocalTensor<DT> l1ASecond = {},
        AscendC::LocalTensor<DT> l1BSecond = {})
    {
        // 1. 绑定本次 L0 ping/pong 和操作数布局。
        auto l0A = resource.l0ABuf.template GetBufferByByte<DT>(slot * L0_BYTES);
        auto l0B = resource.l0BBuf.template GetBufferByByte<DT>(slot * L0_BYTES);
        auto l0C = resource.l0CBuf.template GetBufferByByte<Acc>(slot * L0C_BYTES);
        auto tensorL1A = tla::MakeTensor(
            l1A, tla::MakeLayout<DT, typename TileCopy::LayoutTagL1A>(m, k),
            Catlass::Arch::PositionL1{});
        auto tensorL1B = tla::MakeTensor(
            l1B, tla::MakeLayout<DT, typename TileCopy::LayoutTagL1B>(k, n),
            Catlass::Arch::PositionL1{});
        auto tensorL0A = tla::MakeTensor(
            l0A, tla::MakeLayout<DT, typename TileCopy::LayoutTagL0A>(m, k),
            Catlass::Arch::PositionL0A{});
        auto tensorL0B = tla::MakeTensor(
            l0B, tla::MakeLayout<DT, typename TileCopy::LayoutTagL0B>(k, n),
            Catlass::Arch::PositionL0B{});
        auto tensorL0C = tla::MakeTensor(
            l0C, tla::MakeLayoutL0C(m, n), Catlass::Arch::PositionL0C{});
        auto tileA = tla::GetTile(tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto tileB = tla::GetTile(tensorL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto tileC = tla::GetTile(tensorL0C, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        // 2. 等 L0A/B 上次读取完成，将当前 L1 操作数送入 L0。
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
        if constexpr (CONCAT_RIGHT) {
            // 沿归约维拼接两项：保持四份 L1 源不变，在 L0 组成一次 MMAD 的输入。
            auto aFirst = tla::MakeTensor(l1A,
                tla::MakeLayout<DT, typename TileCopy::LayoutTagL1A>(m, k / 2),
                Catlass::Arch::PositionL1{});
            auto aSecond = tla::MakeTensor(l1ASecond,
                tla::MakeLayout<DT, typename TileCopy::LayoutTagL1A>(m, k / 2),
                Catlass::Arch::PositionL1{});
            auto bFirst = tla::MakeTensor(l1B,
                tla::MakeLayout<DT, typename TileCopy::LayoutTagL1B>(k / 2, n),
                Catlass::Arch::PositionL1{});
            auto bSecond = tla::MakeTensor(l1BSecond,
                tla::MakeLayout<DT, typename TileCopy::LayoutTagL1B>(k / 2, n),
                Catlass::Arch::PositionL1{});
            auto a0 = tla::GetTile(tensorL0A, tla::MakeCoord(0, 0), tla::MakeShape(m, k / 2));
            auto a1 = tla::GetTile(tensorL0A, tla::MakeCoord(0, k / 2), tla::MakeShape(m, k / 2));
            auto b0 = tla::GetTile(tensorL0B, tla::MakeCoord(0, 0), tla::MakeShape(k / 2, n));
            auto b1 = tla::GetTile(tensorL0B, tla::MakeCoord(k / 2, 0), tla::MakeShape(k / 2, n));
            auto sa0 = tla::GetTile(aFirst, tla::MakeCoord(0, 0), tla::MakeShape(m, k / 2));
            auto sa1 = tla::GetTile(aSecond, tla::MakeCoord(0, 0), tla::MakeShape(m, k / 2));
            auto sb0 = tla::GetTile(bFirst, tla::MakeCoord(0, 0), tla::MakeShape(k / 2, n));
            auto sb1 = tla::GetTile(bSecond, tla::MakeCoord(0, 0), tla::MakeShape(k / 2, n));
            typename TileCopy::CopyL1ToL0A{}(a0, sa0);
            typename TileCopy::CopyL1ToL0A{}(a1, sa1);
            typename TileCopy::CopyL1ToL0B{}(b0, sb0);
            typename TileCopy::CopyL1ToL0B{}(b1, sb1);
        } else {
            typename TileCopy::CopyL1ToL0A{}(tensorL0A, tileA);
            typename TileCopy::CopyL1ToL0B{}(tensorL0B, tileB);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(slot);
        Catlass::Gemm::Tile::TileMmadTla<
            ArchTag, DT, typename TileCopy::LayoutTagL1A>{}(tileC, tensorL0A, tensorL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);

        // 3. 主项累加后按 SPLIT 追加残差项，沿用同一 L0C。
        if constexpr (SPLIT != 0) {
            for (uint32_t correction = 0; correction < (SPLIT == 3 ? 2U : 1U); ++correction) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
                const bool lowA = SPLIT == 1 || (SPLIT == 3 && correction == 1);
                auto residualA = tla::MakeTensor(lowA ? l1ASecond : l1A,
                    tla::MakeLayout<DT, typename TileCopy::LayoutTagL1A>(m, k), Catlass::Arch::PositionL1{});
                auto residualB = tla::MakeTensor(lowA ? l1B : l1BSecond,
                    tla::MakeLayout<DT, typename TileCopy::LayoutTagL1B>(k, n), Catlass::Arch::PositionL1{});
                auto extraA = tla::GetTile(residualA, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
                auto extraB = tla::GetTile(residualB, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
                typename TileCopy::CopyL1ToL0A{}(tensorL0A, extraA);
                typename TileCopy::CopyL1ToL0B{}(tensorL0B, extraB);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(slot);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slot);
                Catlass::Gemm::Tile::TileMmadTla<ArchTag, DT, typename TileCopy::LayoutTagL1A>{}(
                    tileC, tensorL0A, tensorL0B, false, 0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(slot);

        // 4. 等 MMAD 完成，再按目标路径写回；远端 UB 在 FREE 后才能覆盖。
        if constexpr (PATH == ResultPath::GM_FP32) {
            AscendC::GlobalTensor<float> dst;
            dst.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gmDst));
            auto params = AscendC::FixpipeParamsV220(n, m, (m + 15U) / 16U * 16U, n, false);
            params.quantPre = QuantMode_t::NoQuant;
            params.unitFlag = 0;
            AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(dst, l0C, params);
        } else if constexpr (PATH == ResultPath::L1_BF16 || PATH == ResultPath::L1_BF16_WITH_RAW) {
            AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> params;
            params.nSize = n;
            params.mSize = m;
            params.srcStride = (m + 15U) / 16U * 16U;
            params.dstStride = KDA_FINALIZE_CHUNK * 16U;
            params.quantPre = QuantMode_t::F322BF16;
            params.unitFlag = 0;
            AscendC::Fixpipe<DT, Acc, FIX_NZ_L1>(l1Dst, l0C, params);
            if constexpr (PATH == ResultPath::L1_BF16_WITH_RAW) {
                const uint64_t flagOffset = aiv * KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE;
                AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(freeFlag + flagOffset + aivSlot);
                auto dst = resource.ubBuf.template GetBufferByByte<float>(aivUbOffset);
                auto tensorUb = tla::MakeTensor(dst, tla::MakeLayout<float, LayoutRM>(m, n), Catlass::Arch::PositionUB{});
                auto tileUb = tla::GetTile(tensorUb, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
                typename CopyTransBToUb::template CopyL0CToDst<decltype(tileUb)>{}(
                    tileUb, tensorL0C, static_cast<uint8_t>(aiv), 0);
                AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(readyFlag + flagOffset + aivSlot);
            }
            if constexpr (SIGNAL_L1_READY) {
                AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(
                    static_cast<AscendC::TEventID>(readyFlag));
            }
        } else if constexpr (PATH == ResultPath::AIV_UB_FP32) {
            const uint64_t flagOffset =
                static_cast<uint64_t>(aiv) * KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE;
            if constexpr (WAIT_AIV_FREE) {
                AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(
                    freeFlag + flagOffset + aivSlot);
            }
            auto remoteUb = resource.ubBuf.template GetBufferByByte<float>(aivUbOffset);
            auto tensorUb = tla::MakeTensor(
                remoteUb, tla::MakeLayout<float, LayoutRM>(m, n),
                Catlass::Arch::PositionUB{});
            auto blockUb = tla::GetTile(
                tensorUb, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
            typename CopyTransBToUb::template CopyL0CToDst<decltype(blockUb)>{}(
                blockUb, tensorL0C, static_cast<uint8_t>(aiv), 0);
            AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(
                readyFlag + flagOffset + aivSlot);
        } else {
            const uint64_t flagOffset =
                static_cast<uint64_t>(aiv) * KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE;
            if constexpr (WAIT_AIV_FREE) {
                AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(
                    freeFlag + flagOffset + aivSlot);
            }
            auto remoteUb = resource.ubBuf.template GetBufferByByte<DT>(aivUbOffset);
            auto tensorUb = tla::MakeTensor(
                remoteUb, tla::MakeLayout<DT, LayoutRM>(m, n),
                Catlass::Arch::PositionUB{});
            auto blockUb = tla::GetTile(
                tensorUb, tla::MakeCoord(0, 0), tla::MakeShape(m, n));
            typename CopyTransAToUbBf16::template CopyL0CToDst<decltype(blockUb)>{}(
                blockUb, tensorL0C, m, static_cast<uint8_t>(aiv), 1, 0);
            AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(
                readyFlag + flagOffset + aivSlot);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(slot);
    }

    __aicore__ inline AscendC::TEventID Stage0Ready(
        uint32_t stream, uint32_t stage) const
    {
        return static_cast<AscendC::TEventID>(
            stream * STAGE0_READY_COUNT + stage);
    }

    // Stage0 输入：载入一头的矩阵，按首个消费者分组发布 READY。
    __aicore__ inline void LoadStage0(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        int64_t head, uint32_t owner)
    {
        constexpr uint32_t rows = KDA_FINALIZE_CHUNK;

        // 1. owner 选择输入 stream；等待上次 MTE1 读完后才能重用。
        const uint32_t stream = owner & 1U;
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(stream);
        const uint32_t streamBase = L1_STREAM + stream * STREAM_BYTES;
        auto akkL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        auto vNewL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_VNEW);
        auto dhL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_DH);
        auto dvScanL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_DVSCAN);
        auto hL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_H);
        auto vL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_V);

        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const int64_t akkToken = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_CHUNK);
        const int64_t hState = FinalizeHOffset(*tiling_, chunk, head);
        const int64_t state = FinalizeDhOffset(*tiling_, chunk, head);

        // 2. 载入 v_new/dh，随后依次准备 Akk/dv_scan、h、v。
        LoadGmToL1A<CopyTransB, LayoutRM>(
            vNewL1, vNew_ + token * sizeof(DT), rows, KDA_FINALIZE_DIM, chunk.validRows);
        LoadGmToL1B<CopyTransB, LayoutCM>(
            dhL1, dh_ + state * sizeof(DT), KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 0));

        LoadGmToL1A<CopyTransA, LayoutCM>(
            akkL1, akk_ + akkToken * sizeof(DT), rows, rows, chunk.validRows, chunk.validRows);
        LoadGmToL1A<CopyRegular, LayoutRM>(
            dvScanL1, dvScan_ + token * sizeof(DT), rows, KDA_FINALIZE_DIM, chunk.validRows);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 1));

        LoadGmToL1B<CopyTransB, LayoutCM>(
            hL1, h_ + hState * sizeof(DT), KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 2));
        LoadGmToL1B<CopyTransB, LayoutCM>(
            vL1, v_ + token * sizeof(DT), KDA_FINALIZE_DIM, rows, KDA_FINALIZE_DIM, chunk.validRows);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 3));
    }

    // Stage0 输入辅助路径：仅准备 stream 中的矩阵，Akk 由独立载入函数负责。
    __aicore__ inline void LoadStage0Stream(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        int64_t head, uint32_t owner)
    {
        constexpr uint32_t rows = KDA_FINALIZE_CHUNK;
        const uint32_t stream = owner & 1U;
        const uint32_t streamBase = L1_STREAM + stream * STREAM_BYTES;
        auto vNewL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_VNEW);
        auto dhL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_DH);
        auto dvScanL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_DVSCAN);
        auto hL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_H);
        auto vL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_V);
        const int64_t token = FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_DIM);
        const int64_t hState = FinalizeHOffset(*tiling_, chunk, head);
        const int64_t state = FinalizeDhOffset(*tiling_, chunk, head);

        LoadGmToL1A<CopyTransB, LayoutRM>(
            vNewL1, vNew_ + token * sizeof(DT), rows, KDA_FINALIZE_DIM, chunk.validRows);
        LoadGmToL1B<CopyTransB, LayoutCM>(
            dhL1, dh_ + state * sizeof(DT), KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 0));
        LoadGmToL1A<CopyRegular, LayoutRM>(
            dvScanL1, dvScan_ + token * sizeof(DT), rows, KDA_FINALIZE_DIM, chunk.validRows);
        LoadGmToL1B<CopyTransB, LayoutCM>(
            hL1, h_ + hState * sizeof(DT), KDA_FINALIZE_DIM, KDA_FINALIZE_DIM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 2));
        LoadGmToL1B<CopyTransB, LayoutCM>(
            vL1, v_ + token * sizeof(DT), KDA_FINALIZE_DIM, rows, KDA_FINALIZE_DIM, chunk.validRows);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 3));
    }

    // Stage0 Akk 辅助路径：保持 ColumnMajor L1 布局，并发布 Akk/dv_scan 联合 READY。
    __aicore__ inline void LoadStage0Akk(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        int64_t head, uint32_t owner)
    {
        constexpr uint32_t rows = KDA_FINALIZE_CHUNK;
        const uint32_t stream = owner & 1U;
        auto akkL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        const int64_t akkToken =
            FinalizeTokenOffset(*tiling_, chunk, head, KDA_FINALIZE_CHUNK);
        LoadGmToL1A<CopyTransA, LayoutCM>(
            akkL1, akk_ + akkToken * sizeof(DT), rows, rows, chunk.validRows, chunk.validRows);
        // dvScan 与 Akk 均已提交到同一 MTE2 队列；此 READY 同时保护二者。
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 1));
    }

    // Stage0 / Cube：四项 BF16 矩阵乘均 FP32 累加；按后续用途选择写回位置。
    __aicore__ inline void ComputeStage0(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        uint32_t owner, uint32_t aiv, uint32_t aivSlot,
        int64_t coreIdx, uint64_t groupGeneration)
    {
        // 1. 绑定本 head 的 L1 操作数与循环 workspace。
        constexpr uint32_t rows = KDA_FINALIZE_CHUNK;
        const uint32_t stream = owner & 1U;
        const uint32_t streamBase = L1_STREAM + stream * STREAM_BYTES;
        auto akkL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        auto dwL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_DW + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
        auto vNewL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_VNEW);
        auto dhL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_DH);
        auto dvScanL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_DVSCAN);
        auto hL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_H);
        auto vL1 = resource.l1Buf.template GetBufferByByte<DT>(streamBase + STREAM_V);

        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        GM_ADDR wsBase = workspace_ + ws;

        // 2. dk_state_raw = v_new @ dhᵀ，FP32 写 workspace。
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 0));
        RunGemm<CopyTransB, ResultPath::GM_FP32>(
            resource, vNewL1, dhL1, rows, KDA_FINALIZE_DIM, KDA_FINALIZE_DIM,
            NextSlot(), wsBase + KDA_FINALIZE_WS_DK_STATE_RAW, {}, 0, 0, 0, 0, 0);

        // 3. DVb = Akkᵀ @ dv_scan，FP32 写 workspace。
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 1));
        RunGemm<CopyTransA, ResultPath::GM_FP32>(
            resource, akkL1, dvScanL1, rows, KDA_FINALIZE_DIM, rows,
            NextSlot(), wsBase + KDA_FINALIZE_WS_DVB, {}, 0, 0, 0, 0, 0);

        // 4. dW = dv_scan @ hᵀ；BF16 高位写 L1，FP32 原值送 AIV 生成低位。
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 2));
        RunGemm<CopyTransB, ResultPath::L1_BF16_WITH_RAW>(
            resource, dvScanL1, hL1, rows, KDA_FINALIZE_DIM, KDA_FINALIZE_DIM,
            NextSlot(), nullptr, dwL1, aiv, aivSlot, 64 * 1024,
            KDA_FINALIZE_LOCAL_READY_BASE, KDA_FINALIZE_ZB_READY_BASE);

        // 5. zV = dv_scan @ vᵀ，FP32 直接写 AIV UB，供 BuildZ 使用。
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(Stage0Ready(stream, 3));
        RunGemm<CopyTransB, ResultPath::AIV_UB_FP32>(
            resource, dvScanL1, vL1, rows, rows, KDA_FINALIZE_DIM,
            NextSlot(), nullptr, {}, aiv, aivSlot,
            KDA_FINALIZE_UB_ZV + aivSlot * KDA_FINALIZE_MATRIX_FP32_BYTES,
            KDA_FINALIZE_ZV_FREE_BASE, KDA_FINALIZE_ZV_READY_BASE);

        // 6. 本头输入已读完，归还 stream 的搬入权限。
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(stream);
    }

    // Stage1 / Cube：消费 dW/kE 高低位，生成 dKgb_raw 与 zW。
    __aicore__ inline void RunStage1(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        uint32_t owner, uint32_t aiv, uint32_t aivSlot,
        int64_t coreIdx, uint64_t groupGeneration)
    {
        // 1. 绑定驻留 Akk/dW/kE 和本 head 的 workspace。
        constexpr uint32_t rows = KDA_FINALIZE_CHUNK;
        auto akkL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        auto dwL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_DW + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
        auto kEL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_KE + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
        const uint64_t flagOffset =
            static_cast<uint64_t>(aiv) * KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE;
        const uint64_t ws = FinalizeWorkspaceSlotBase(coreIdx, groupGeneration, owner);
        GM_ADDR wsBase = workspace_ + ws;

        // 2. 等 AIV 将 dW/kE 的高低位全部送入 L1。
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE1>(
            KDA_FINALIZE_KE_READY_BASE + flagOffset + aivSlot);
        auto dwLow = resource.l1Buf.template GetBufferByByte<DT>(
            64 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
        auto keLow = resource.l1Buf.template GetBufferByByte<DT>(
            128 * 1024 + owner * KDA_FINALIZE_VECTOR_BF16_BYTES);
        // Akk 保留 Stage0 的 ColumnMajor L1 解释，按该布局计算 Akkᵀ @ dW。
        // 3. dKgb_raw = Akkᵀ @ dW_hi + Akkᵀ @ dW_lo，FP32 写 workspace。
        RunGemm<CopyTransA, ResultPath::GM_FP32, false, true, false, 2>(
            resource, akkL1, dwL1, rows, KDA_FINALIZE_DIM, rows,
            NextSlot(), wsBase + KDA_FINALIZE_WS_DKGB_RAW, {}, 0, 0, 0, 0, 0, {}, dwLow);
        // 同一 READY 已保护 dW/kE 两项的高、低位平面。
        // 4. zW 累加 hi×hi、hi×lo、lo×hi，FP32 直接送 BuildZ 的 UB。
        RunGemm<CopyTransB, ResultPath::AIV_UB_FP32, false, true, false, 3>(
            resource, dwL1, kEL1, rows, rows, KDA_FINALIZE_DIM,
            NextSlot(), nullptr, {}, aiv, aivSlot,
            KDA_FINALIZE_UB_ZW + aivSlot * KDA_FINALIZE_MATRIX_FP32_BYTES,
            KDA_FINALIZE_ZW_FREE_BASE, KDA_FINALIZE_ZW_READY_BASE, dwLow, keLow);
    }

    // Stage3 / Cube：Tza = Zb @ Akkᵀ；高位写 L1，FP32 原值交 AIV 补残差。
    __aicore__ inline void RunStage3(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        uint32_t owner, uint32_t aiv, uint32_t aivSlot)
    {
        // 1. 等本 head 的 Zb 高低位写入 L1。
        constexpr uint32_t rows = KDA_FINALIZE_CHUNK;
        const uint64_t flagOffset =
            static_cast<uint64_t>(aiv) * KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE;
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE1>(
            KDA_FINALIZE_ZB_READY_BASE + flagOffset + aivSlot);
        auto zBL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_ZB + owner * 2 * KDA_FINALIZE_MATRIX_BF16_BYTES);
        auto akkL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        auto tzaL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_TZA + owner * 2 * KDA_FINALIZE_MATRIX_BF16_BYTES);
        auto zBLow = resource.l1Buf.template GetBufferByByte<DT>(
            L1_ZB + (owner * 2 + 1) * KDA_FINALIZE_MATRIX_BF16_BYTES);

        // 2. 累加 Zb 高位与低位两项，发布 Tza 原值。
        RunGemm<CopyTransB, ResultPath::L1_BF16_WITH_RAW, false, true, false, 1>(
            resource, zBL1, akkL1, rows, rows, rows,
            NextSlot(), nullptr,
            tzaL1, aiv, aivSlot, 0, KDA_FINALIZE_ZV_FREE_BASE,
            KDA_FINALIZE_ZW_READY_BASE, zBLow, {});

        // 3. 在 Tza Fixpipe 完成后归还 Zb 槽；此时 Cube 肯定已结束对 Zb 的读取。
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(
            KDA_FINALIZE_ZB_FREE_BASE + flagOffset + aivSlot);
    }

    // Stage4 / Cube：dAkk_raw = Akkᵀ @ Tza，FP32 交给 Stage5。
    __aicore__ inline void RunStage4(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        uint32_t owner, uint32_t aiv, uint32_t aivSlot)
    {
        // 1. 绑定驻留 Akk 和 Tza 高低位。
        constexpr uint32_t rows = KDA_FINALIZE_CHUNK;
        auto akkL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_AKK + owner * KDA_FINALIZE_MATRIX_BF16_BYTES);
        auto tzaL1 = resource.l1Buf.template GetBufferByByte<DT>(
            L1_TZA + owner * 2 * KDA_FINALIZE_MATRIX_BF16_BYTES);
        auto tzaLow = resource.l1Buf.template GetBufferByByte<DT>(
            L1_TZA + (owner * 2 + 1) * KDA_FINALIZE_MATRIX_BF16_BYTES);

        // 2. 等待 Tza 残差写 L1 完成，并取得 dAkk 目标 UB 的写权限。
        // 双头窗口每 AIV 一头，FP32 交接固定占 [0,16) KiB；写入前 StateAndBase 已释放该区。
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE1>(
            KDA_FINALIZE_KE_READY_BASE + aiv * KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE + aivSlot);

        // 3. 累加 Tza 的高位、低位贡献，写入 AIV 的 dAkk_raw 区。
        RunGemm<CopyTransA, ResultPath::AIV_UB_FP32, false, false, false, 2>(
            resource, akkL1, tzaL1, rows, rows, rows,
            NextSlot(), nullptr, {}, aiv, aivSlot,
            KDA_FINALIZE_UB_DAKK_RAW,
            KDA_FINALIZE_DAKK_FREE_BASE, KDA_FINALIZE_DAKK_READY_BASE, {}, tzaLow);
    }

    // 分带 GEMM：处理 32 行输出，支持沿 M 堆叠或沿 K 拼接两项。
    // TRANSPOSE_A 选择左矩阵布局；STACK_LEFT 堆叠 dq/left；CONCAT_RIGHT 拼接 right 两项。
    template <bool TRANSPOSE_A, bool CONCAT_RIGHT = false, bool WAIT_FREE = true,
              bool STACK_LEFT = false>
    __aicore__ inline void RunLocalGemm(
        Catlass::Arch::Resource<ArchTag> &resource,
        AscendC::LocalTensor<FinalizeLocalType> a, AscendC::LocalTensor<FinalizeLocalType> b,
        uint32_t aiv, uint32_t aivSlot, uint32_t ubOffset,
        uint64_t freeFlag, uint64_t readyFlag, uint32_t rowBegin,
        AscendC::LocalTensor<FinalizeLocalType> aSecond = {},
        AscendC::LocalTensor<FinalizeLocalType> bSecond = {})
    {
        // 1. 选择布局和 L0 槽；输出行数由是否堆叠决定。
        using ALayout = std::conditional_t<TRANSPOSE_A, LayoutCM, LayoutRM>;
        using Copy = Catlass::Gemm::Tile::PackedTileCopyTla<
            ArchTag, FinalizeLocalType, ALayout, FinalizeLocalType, LayoutRM, float, LayoutRM>;
        const uint32_t slot = NextSlot();
        auto l0a = resource.l0ABuf.template GetBufferByByte<FinalizeLocalType>(slot * L0_BYTES);
        auto l0b = resource.l0BBuf.template GetBufferByByte<FinalizeLocalType>(slot * L0_BYTES);
        auto l0c = resource.l0CBuf.template GetBufferByByte<float>(slot * L0C_BYTES);
        constexpr uint32_t reduction = CONCAT_RIGHT ? 128U : 64U;
        constexpr uint32_t outputRows = STACK_LEFT ? 2U * KDA_FINALIZE_INTRA_ROWS : KDA_FINALIZE_INTRA_ROWS;
        auto ta = tla::MakeTensor(l0a,
            tla::MakeLayout<FinalizeLocalType, typename Copy::LayoutTagL0A>(outputRows, reduction), Catlass::Arch::PositionL0A{});
        auto tb = tla::MakeTensor(l0b,
            tla::MakeLayout<FinalizeLocalType, typename Copy::LayoutTagL0B>(reduction, 128), Catlass::Arch::PositionL0B{});
        auto tc = tla::MakeTensor(l0c, tla::MakeLayoutL0C(outputRows, 128), Catlass::Arch::PositionL0C{});
        auto tileC = tla::GetTile(tc, tla::MakeCoord(0, 0), tla::MakeShape(outputRows, 128));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(slot);

        // 2. 按 high×low、low×high、high×high 的顺序累加三项，省略 low×low。
        // CONCAT_RIGHT 沿 K 拼接两项；BF16 K=128 恰好放入一份 32 KiB L0B。
        for (uint32_t term = 0; term < 3U; ++term) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
            for (uint32_t part = 0; part < ((CONCAT_RIGHT || STACK_LEFT) ? 2U : 1U); ++part) {
                auto sourceA = part == 0 ? a : aSecond;
                auto sourceB = (part == 0 || STACK_LEFT) ? b : bSecond;
                auto sa = tla::MakeTensor(sourceA[term == 1U ? KDA_FINALIZE_MATRIX_ELEMS : 0U],
                    tla::MakeLayout<FinalizeLocalType, typename Copy::LayoutTagL1A>(64, 64), Catlass::Arch::PositionL1{});
                auto sb = tla::MakeTensor(sourceB[term == 0U ? KDA_FINALIZE_VECTOR_ELEMS : 0U],
                    tla::MakeLayout<FinalizeLocalType, typename Copy::LayoutTagL1B>(64, 128), Catlass::Arch::PositionL1{});
                auto tileA = tla::GetTile(sa, tla::MakeCoord(rowBegin, 0), tla::MakeShape(KDA_FINALIZE_INTRA_ROWS, 64));
                auto tileB = tla::GetTile(sb, tla::MakeCoord(0, 0), tla::MakeShape(64, 128));
                auto dstA = tla::GetTile(ta,
                    tla::MakeCoord(STACK_LEFT ? part * KDA_FINALIZE_INTRA_ROWS : 0U,
                                   CONCAT_RIGHT ? part * 64U : 0U),
                    tla::MakeShape(KDA_FINALIZE_INTRA_ROWS, 64));
                auto dstB = tla::GetTile(tb, tla::MakeCoord(CONCAT_RIGHT ? part * 64U : 0U, 0), tla::MakeShape(64, 128));
                typename Copy::CopyL1ToL0A{}(dstA, tileA);
                if (!STACK_LEFT || part == 0) {
                    typename Copy::CopyL1ToL0B{}(dstB, tileB);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(slot);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(slot);
            Catlass::Gemm::Tile::TileMmadTla<ArchTag, FinalizeLocalType, typename Copy::LayoutTagL1A>{}(
                tileC, ta, tb, term == 0, 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(slot);

        // 3. 等目标 AIV 的结果区 FREE，再由 Fixpipe 写 FP32 并发布 READY。
        const uint64_t offset = aiv * KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE;
        if constexpr (WAIT_FREE) {
            AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(freeFlag + offset + aivSlot);
        }
        auto dst = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        auto tu = tla::MakeTensor(dst, tla::MakeLayout<float, LayoutRM>(outputRows, 128), Catlass::Arch::PositionUB{});
        auto tileU = tla::GetTile(tu, tla::MakeCoord(0, 0), tla::MakeShape(outputRows, 128));
        typename CopyTransBToUb::template CopyL0CToDst<decltype(tileU)>{}(
            tileU, tc, static_cast<uint8_t>(aiv), 0);
        AscendC::CrossCoreSetFlag<KDA_FINALIZE_CROSS_MODE, PIPE_FIX>(readyFlag + offset + aivSlot);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(slot);
    }

    // Stage6 / Cube：使用 Stage5 操作数，一次生成本带 dq_local 与 left。
    __aicore__ inline void RunStage6(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        uint32_t owner, uint32_t aiv, uint32_t aivSlot,
        uint32_t rowBegin)
    {
        // 1. 等本带五项 NZ 操作数发布完成。
        const uint64_t flagOffset =
            static_cast<uint64_t>(aiv) * KDA_FINALIZE_SUBBLOCK_FLAG_STRIDE;
        AscendC::CrossCoreWaitFlag<KDA_FINALIZE_CROSS_MODE, PIPE_MTE1>(
            KDA_FINALIZE_LOCAL_READY_BASE + flagOffset + aivSlot);

        // 2. 选择本 owner 的 dAqk、dAkk 与 kNeg 高低位。
        auto local = resource.l1Buf.template GetBufferByByte<FinalizeLocalType>(
            KDA_FINALIZE_LOCAL_BASE + owner * KDA_FINALIZE_LOCAL_BYTES);
        auto dAqkL1 = local[KDA_FINALIZE_LOCAL_DAQK / sizeof(FinalizeLocalType)];
        auto kNegL1 = local[KDA_FINALIZE_LOCAL_K_NEG / sizeof(FinalizeLocalType)];
        constexpr uint32_t rows = KDA_FINALIZE_CHUNK;

        // 3. dq_local 与 left 共用 kNeg，沿输出行拼成 [dq_local; left]。
        // 每带两个 32 行结果合成一份 64 行 FP32 UB 交接。
        RunLocalGemm<false, false, true, true>(
            resource, dAqkL1, kNegL1, aiv, aivSlot,
            KDA_FINALIZE_UB_DQ_LOCAL_RAW +
                aivSlot * KDA_FINALIZE_VECTOR_FP32_BYTES,
            KDA_FINALIZE_DQ_LOCAL_FREE_BASE,
            KDA_FINALIZE_DQ_LOCAL_READY_BASE, rowBegin,
            local[KDA_FINALIZE_LOCAL_DAKK / sizeof(FinalizeLocalType)]);
    }

    // Stage8 / Cube：right = dAqkᵀ @ qPos + dAkkᵀ @ bkPos。
    __aicore__ inline void RunStage8(
        Catlass::Arch::Resource<ArchTag> &resource, const FinalizeChunkInfo &chunk,
        uint32_t owner, uint32_t aiv, uint32_t slot, uint32_t rowBegin)
    {
        // 1. 复用本 owner 的分带 L1 操作数；Stage6 之后仍保持存活。
        auto local = resource.l1Buf.template GetBufferByByte<FinalizeLocalType>(
            KDA_FINALIZE_LOCAL_BASE + owner * KDA_FINALIZE_LOCAL_BYTES);
        auto daq = local[KDA_FINALIZE_LOCAL_DAQK / sizeof(FinalizeLocalType)];
        auto dak = local[KDA_FINALIZE_LOCAL_DAKK / sizeof(FinalizeLocalType)];
        auto kn = local[KDA_FINALIZE_LOCAL_K_NEG / sizeof(FinalizeLocalType)];
        auto qp = local[KDA_FINALIZE_LOCAL_Q_POS / sizeof(FinalizeLocalType)];
        auto bp = local[KDA_FINALIZE_LOCAL_BK_POS / sizeof(FinalizeLocalType)];
        constexpr uint32_t rows = KDA_FINALIZE_CHUNK;

        // 2. 两项沿 K 拼接，再累加三项高低位乘积；FP32 交给 Stage9。
        RunLocalGemm<true, true>(
            resource, daq, qp, aiv, slot, 64 * 1024 + slot * KDA_FINALIZE_VECTOR_FP32_BYTES,
            KDA_FINALIZE_KE_READY_BASE, KDA_FINALIZE_ZW_READY_BASE, rowBegin, dak, bp);
    }

    // 轮换 L0A/B/C bank；是否可复用由 RunGemm/RunLocalGemm 内的事件决定。
    __aicore__ inline uint32_t NextSlot()
    {
        const uint32_t current = l0Slot_;
        l0Slot_ ^= 1U;
        return current;
    }

    // 预置首轮 stream/L0 的 FREE；没有前序使用者需要等待。
    __aicore__ inline void InitEvents()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(slot);
        }
    }

    // 逐项消费末轮 FREE，与 InitEvents 及每次循环的发布保持配对。
    __aicore__ inline void DrainEvents()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * slot + 1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(slot);
        }
    }

    GM_ADDR v_ = nullptr;
    GM_ADDR akk_ = nullptr;
    GM_ADDR vNew_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR dh_ = nullptr;
    GM_ADDR dvScan_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    const ChunkKdaBwdFinalizeTilingData *tiling_ = nullptr;
    uint32_t l0Slot_ = 0;
};

} // namespace KDA

#endif // CHUNK_KDA_BWD_FINALIZE_ARCH35_CUBE_H
