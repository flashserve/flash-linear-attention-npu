#ifndef CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_CUBE_H
#define CHUNK_GATED_DELTA_RULE_BWD_FINALIZE_CUBE_H

#ifndef CATLASS_ARCH
#define CATLASS_ARCH 3510
#endif

#include "chunk_gated_delta_rule_bwd_finalize_common.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_utils/tile/copy_l0c_to_ub.hpp"
#include "kernel_operator.h"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace GDN {

template <typename DT>
class ChunkGatedDeltaRuleBwdFinalizeCube {
public:
    __aicore__ inline void Init(
        GM_ADDR q, GM_ADDR k, GM_ADDR vNew, GM_ADDR dO, GM_ADDR du, GM_ADDR h, GM_ADDR dh, GM_ADDR a,
        GM_ADDR kbg, GM_ADDR vb, GM_ADDR dA0, GM_ADDR ds0Out,
        GM_ADDR dkbOut, GM_ADDR dkbTOut, GM_ADDR dsIn,
        GM_ADDR doGIn, GM_ADDR vDecayIn, GM_ADDR workspace,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
        const ChunkGatedDeltaRuleBwdFinalizeTilingData *tiling)
    {
        q_ = q;
        k_ = k;
        vNew_ = vNew;
        do_ = dO;
        du_ = du;
        h_ = h;
        dh_ = dh;
        a_ = a;
        kbg_ = kbg;
        vb_ = vb;
        dA0_ = dA0;
        ds0Out_ = ds0Out;
        dkbOut_ = dkbOut;
        dkbTOut_ = dkbTOut;
        dsIn_ = dsIn;
        doGIn_ = doGIn;
        vDecayIn_ = vDecayIn;
        workspace_ = workspace;
        cuSeqlens_ = cuSeqlens;
        chunkIndices_ = chunkIndices;
        tiling_ = tiling;
        coreIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());
        coreNum_ = static_cast<int64_t>(AscendC::GetBlockNum());
        stateChunkNum_ = tiling_->isVariable != 0 ? tiling_->totalChunkNum : tiling_->chunkNumForT;
        if (coreNum_ <= 0) {
            coreNum_ = 1;
        }
    }

    __aicore__ inline void Process()
    {
        Catlass::Arch::Resource<ArchTag> resource;

        // L1 只由 AIC 写入。AIV 结果先落 GM，AIC 在对应 Stage 内再搬入
        // 下列物理槽，算子中不存在 UB->L1 通路。
        AscendC::LocalTensor<DT> hL1[L1_RESIDENT_HEAD_COUNT_4] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1_H_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1_H_OFFSET + STATE_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_H_OFFSET + 2 * STATE_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_H_OFFSET + 3 * STATE_BYTES)};
        AscendC::LocalTensor<DT> duL1[L1_RESIDENT_HEAD_COUNT_4] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1_DU_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1_DU_OFFSET + VECTOR_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_DU_OFFSET + 2 * VECTOR_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_DU_OFFSET + 3 * VECTOR_BYTES)};
        AscendC::LocalTensor<DT> vbL1[L1_RESIDENT_HEAD_COUNT_4] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1_VB_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1_VB_OFFSET + VECTOR_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_VB_OFFSET + 2 * VECTOR_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_VB_OFFSET + 3 * VECTOR_BYTES)};
        AscendC::LocalTensor<DT> kbgL1[L1_RESIDENT_HEAD_COUNT_4] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1_KBG_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1_KBG_OFFSET + VECTOR_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_KBG_OFFSET + 2 * VECTOR_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_KBG_OFFSET + 3 * VECTOR_BYTES)};
        AscendC::LocalTensor<DT> dw0L1[L1_RESIDENT_HEAD_COUNT_4] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1_DW0_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1_DW0_OFFSET + VECTOR_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_DW0_OFFSET + 2 * VECTOR_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_DW0_OFFSET + 3 * VECTOR_BYTES)};
        AscendC::LocalTensor<DT> aL1[L1_RESIDENT_HEAD_COUNT_4] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1_A_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1_A_OFFSET + MATRIX_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_A_OFFSET + 2 * MATRIX_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_A_OFFSET + 3 * MATRIX_BYTES)};
        AscendC::LocalTensor<DT> dA0L1[L1_RESIDENT_HEAD_COUNT_4] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE5_DA0_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE5_DA0_OFFSET + MATRIX_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE5_DA0_OFFSET + 2 * MATRIX_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE5_DA0_OFFSET + 3 * MATRIX_BYTES)};
        AscendC::LocalTensor<DT> kStage5L1[L1_RESIDENT_HEAD_COUNT_4] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE5_K_A_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE5_K_A_OFFSET + VECTOR_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE5_K_A_OFFSET + 2 * VECTOR_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE5_K_A_OFFSET + 3 * VECTOR_BYTES)};
        AscendC::LocalTensor<DT> doStage11L1[BANK_COUNT_2] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE11_DO_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE11_DO_OFFSET + VECTOR_BYTES)};
        AscendC::LocalTensor<DT> vNewStage11L1[L1_RESIDENT_HEAD_COUNT_4] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE11_VNEW_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE11_VNEW_OFFSET + VECTOR_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE11_VNEW_OFFSET + 2 * VECTOR_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE11_VNEW_OFFSET + 3 * VECTOR_BYTES)};
        AscendC::LocalTensor<DT> qStage13L1[BANK_COUNT_2] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE13_Q_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE13_Q_OFFSET + VECTOR_BYTES)};
        AscendC::LocalTensor<DT> dhStage13L1[BANK_COUNT_2] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE13_DH_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE13_DH_OFFSET + STATE_BYTES)};
        AscendC::LocalTensor<DT> dsStage13L1[L1_RESIDENT_HEAD_COUNT_4] = {
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE13_DS_OFFSET),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE13_DS_OFFSET + MATRIX_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE13_DS_OFFSET + 2 * MATRIX_BYTES),
            resource.l1Buf.template GetBufferByByte<DT>(L1_STAGE13_DS_OFFSET + 3 * MATRIX_BYTES)};

        // L0A/L0B 各自维护独立的 ping/pong slot 和事件，使用一次就取反一次。
        // L0B 按较大的 128x128 B 操作数预留 32 KiB，dA_u 的 128xM
        // 只使用其中有效部分，不拆走余量存放其他语义。
        AscendC::LocalTensor<DT> l0A[L0_BUFFER_COUNT_2] = {
            resource.l0ABuf.template GetBufferByByte<DT>(0),
            resource.l0ABuf.template GetBufferByByte<DT>(L0_A_BYTES)};
        AscendC::LocalTensor<DT> l0B[L0_BUFFER_COUNT_2] = {
            resource.l0BBuf.template GetBufferByByte<DT>(0),
            resource.l0BBuf.template GetBufferByByte<DT>(L0_B_BYTES)};
        AscendC::LocalTensor<ElementAccumulator> l0C[L0_BUFFER_COUNT_2] = {
            resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0),
            resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0_C_BYTES)};

        // dA_u 是 Cube-to-Vector resident。每个 AIV 最多承包两个 HEAD，
        // 因此当前物理 UB 中 [112,128) KiB 正好放两个 8 KiB slot；Fixpipe
        // 通过 subBlockId 定向写入 owner AIV，而不是双写到两个 AIV。
        AscendC::LocalTensor<DT> dAuCv[BANK_COUNT_2] = {
            resource.ubBuf.template GetBufferByByte<DT>(UB_DAU_OFFSET),
            resource.ubBuf.template GetBufferByByte<DT>(UB_DAU_OFFSET + MATRIX_BYTES)};
        AscendC::LocalTensor<DT> dvbCv[BANK_COUNT_2] = {
            resource.ubBuf.template GetBufferByByte<DT>(UB_DVB_OFFSET),
            resource.ubBuf.template GetBufferByByte<DT>(UB_DVB_OFFSET + VECTOR_BYTES)};
        AscendC::LocalTensor<DT> dAw0Cv[BANK_COUNT_2] = {
            resource.ubBuf.template GetBufferByByte<DT>(UB_DAW0_OFFSET),
            resource.ubBuf.template GetBufferByByte<DT>(UB_DAW0_OFFSET + VECTOR_BYTES)};
        AscendC::LocalTensor<DT> a2Cv[BANK_COUNT_2] = {
            resource.ubBuf.template GetBufferByByte<DT>(UB_A2_OFFSET),
            resource.ubBuf.template GetBufferByByte<DT>(UB_A2_OFFSET + MATRIX_BYTES)};
        AscendC::LocalTensor<DT> dA2Cv[BANK_COUNT_2] = {
            resource.ubBuf.template GetBufferByByte<DT>(UB_DA2_OFFSET),
            resource.ubBuf.template GetBufferByByte<DT>(UB_DA2_OFFSET + MATRIX_BYTES)};
        AscendC::LocalTensor<DT> dkbg0Cv[BANK_COUNT_2] = {
            resource.ubBuf.template GetBufferByByte<DT>(UB_DKBG0_OFFSET),
            resource.ubBuf.template GetBufferByByte<DT>(UB_DKBG0_OFFSET + VECTOR_BYTES)};
        AscendC::LocalTensor<DT> ds0Cv[BANK_COUNT_2] = {
            resource.ubBuf.template GetBufferByByte<DT>(UB_STAGE12_DS_OFFSET),
            resource.ubBuf.template GetBufferByByte<DT>(UB_STAGE12_DS_OFFSET + MATRIX_BYTES)};
        AscendC::LocalTensor<DT> dk0Cv[BANK_COUNT_2] = {
            resource.ubBuf.template GetBufferByByte<DT>(UB_STAGE13_DK0_OFFSET),
            resource.ubBuf.template GetBufferByByte<DT>(UB_STAGE13_DK0_OFFSET + VECTOR_BYTES)};
        AscendC::LocalTensor<DT> dqIntraCv[BANK_COUNT_2] = {
            resource.ubBuf.template GetBufferByByte<DT>(UB_STAGE13_DQ_INTRA_OFFSET),
            resource.ubBuf.template GetBufferByByte<DT>(UB_STAGE13_DQ_INTRA_OFFSET + VECTOR_BYTES)};
        AscendC::LocalTensor<DT> dkIntraCv[BANK_COUNT_2] = {
            resource.ubBuf.template GetBufferByByte<DT>(UB_STAGE13_DK_INTRA_OFFSET),
            resource.ubBuf.template GetBufferByByte<DT>(UB_STAGE13_DK_INTRA_OFFSET + VECTOR_BYTES)};

        uint32_t eventIndex = 0;
        int64_t taskIdx = 0;
        int64_t chunkTaskIdx = 0;
        int64_t headGroupIdx = 0;
        int64_t taskRound = 0;
        int64_t hvBase = 0;
        int64_t taskCount = 0;
        int64_t headOffset = 0;
        ChunkInfo chunk;

        // 4 份 du 与 4 份 h resident 各有独立 MTE2/MTE1 生命周期。
        // 初始不存在旧消费者，先发布 MTE2 可写；每次 HEAD 的末次 MTE1
        // 完成后重新发布，下一任务组才允许覆盖同一 resident。
        for (eventIndex = 0; eventIndex < L1_EVENT_COUNT_8; ++eventIndex) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(eventIndex);
        }
        for (eventIndex = 0; eventIndex < L0_BUFFER_COUNT_2; ++eventIndex) {
            // 严格按 dhu 的物理槽组织事件：ping 使用 A=0/B=1，
            // pong 使用 A=2/B=3，不把 A 和 B 各自排成连续 ID。
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * eventIndex);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2 * eventIndex + 1);
        }
        for (eventIndex = 0; eventIndex < L0_BUFFER_COUNT_2; ++eventIndex) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(eventIndex);
        }

        for (taskIdx = coreIdx_; taskIdx < tiling_->taskNum; taskIdx += coreNum_) {
            chunkTaskIdx = taskIdx / tiling_->headGroupNum;
            headGroupIdx = taskIdx % tiling_->headGroupNum;
            GetChunkInfo(chunkTaskIdx, cuSeqlens_, chunkIndices_, *tiling_, chunk);
            if (!chunk.valid) {
                continue;
            }
            hvBase = headGroupIdx * tiling_->taskGroupSize;
            taskCount = Min(tiling_->taskGroupSize, tiling_->HV - hvBase);
            taskRound = (taskIdx - coreIdx_) / coreNum_;
            workspaceGroupRound_ = taskRound;
                // 一个 AIC 顺序承包当前 chunk/task group 的全部 HEAD。du/h 的
                // MTE2 在等待 Stage 0 vb 前发射，可与两个 AIV 的 Stage 0 重叠。
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage1Head(
                        hL1, duL1, vbL1, dw0L1, l0A, l0B, l0C, dAuCv,
                        chunk, hvBase + headOffset, headOffset);
                }
                // Stage 1 已逐 HEAD消费 Vector->Cube 有序链，因此当前组每个
                // HEAD 的 kbg 都已写入 GM。Stage 3 写入 AIV 的 dAw0/dvb bank，
                // 与 Stage 2 使用的 dAu/k/beta/kb bank 不重叠，无需等待 Stage 0
                // 或 Stage 2 的整组完成信号。
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage3Head(
                        aL1, duL1, kbgL1, dw0L1, l0A, l0B, l0C, dvbCv, dAw0Cv,
                        chunk, hvBase + headOffset, headOffset);
                    // 当前 HEAD 的 dA_w0 和 dvb 分别占用两个 L0C slot。
                    // 两次跨核 Fixpipe 都完整落到 owner UB 后再处理下一 HEAD，
                    // 避免同一目标 AIV 上累计多个尚未闭环的远端 copyout。
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
                    // 当前 HEAD 的 dA_w0 和 dvb 均已完整落入 owner AIV 的 UB。
                    // 向两个 AIV 广播同一个逐 HEAD ready；owner 随后执行 Stage 4，
                    // non-owner 只消费信号，使固定 EventID 在任务组内逐次闭环。
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                }
                // Stage 5 只依赖已完成 Stage 4 并写入 GM 的 dA0。先连续完成
                // 当前任务组的全部 HEAD，使 Stage 5 内部的 MTE2/MTE1/Cube/
                // Fixpipe 双缓冲能够持续轮转，不在每个 HEAD 后切换到 Stage 7。
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage5Head(
                        aL1, dA0L1, kStage5L1,
                        l0A, l0B, l0C, a2Cv,
                        chunk, hvBase + headOffset, headOffset);
                }
                // Stage 5 已为当前任务组的每个 HEAD 保留独立 dA1 L1
                // resident。Stage 7 再按相同 HEAD 顺序连续消费；每个 HEAD
                // 进入 L0 后立即释放对应 L1 event，供下一任务组 Stage 5 复用。
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage7Head(
                        aL1, dA0L1, dw0L1, l0A, l0B, l0C, dA2Cv, dkbg0Cv,
                        chunk, hvBase + headOffset, headOffset);
                    // Stage 7 的 dA2/dkbg0 已进入 owner AIV 的独立 UB
                    // resident，逐 HEAD通知 Stage 8；Stage 6 使用另一条更早的
                    // Stage 5 ready，不在此处等待 Stage 7。
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                }
                // Stage 9 通过 Vector 侧 S6 入口 token 与 AIV 阶段对齐；
                // 该等待必须与每个 taskGroup 的批量信号一一匹配。
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage9Head(
                        dA0L1, vbL1, kStage5L1, l0A, l0B, l0C,
                        chunk, hvBase + headOffset, headOffset);
                }
                // Stage 11 与 Stage 10 无数据依赖。do/v_new 各从 GM 搬入一次，
                // 每个 HEAD 只完成 ds0 = do @ v_new.T。
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage11Head(
                        doStage11L1, vNewStage11L1, l0A, l0B, l0C, ds0Cv,
                        chunk, hvBase + headOffset, headOffset);
                    // ds0 Fixpipe 完成后再广播，owner AIV 才能搬入并执行 S12。
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                }
                // Stage 12 每个 HEAD 完成 ds 写 GM 后通过 V->C 有序链通知。
                // Stage 13 搬入 ds/q/dh，一次读取各输入并完成三条独立 GEMM。
                for (headOffset = 0; headOffset < taskCount; ++headOffset) {
                    ProcessStage13Head(
                        qStage13L1, dhStage13L1, dsStage13L1,
                        kStage5L1, hL1, vNewStage11L1, vbL1, l0A, l0B, l0C,
                        dk0Cv, dqIntraCv, dkIntraCv,
                        chunk, hvBase + headOffset, headOffset);
                    // 三份结果均已进入 owner AIV，通知后续 Stage 14 消费。
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
                }
                // Stage 13 的三份 UB resident 会被下一任务组的 Stage 13
                // Fixpipe 原址复用。等待两个 AIV 完成当前组的 MTE3 消费，
                // 防止下一轮 Fixpipe 覆盖仍在读取的源 UB。
                Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);
                // 下一任务组 Stage 1 必须逐 HEAD 等待下一组 Stage 0 的
                // Vector->Cube 有序链。两个 AIV 只有完成本组 Stage 4 后才会
                // 顺序进入下一组 Stage 0，因此该天然依赖已经保护跨组 UB 复用，
                // 这里不再增加任务组边界 wait。
        }

        // kernel 退出前消费所有初始/末轮事件，使每套 Set/Wait 都完整闭环。
        for (eventIndex = 0; eventIndex < L1_EVENT_COUNT_8; ++eventIndex) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(eventIndex);
        }
        for (eventIndex = 0; eventIndex < L0_BUFFER_COUNT_2; ++eventIndex) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * eventIndex);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2 * eventIndex + 1);
        }
        for (eventIndex = 0; eventIndex < L0_BUFFER_COUNT_2; ++eventIndex) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(eventIndex);
        }
    }

private:
    using ArchTag = Catlass::Arch::Ascend950;
    using LayoutRowMajor = Catlass::layout::RowMajor;
    using LayoutColumnMajor = Catlass::layout::ColumnMajor;
    using TileCopyDau = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRowMajor, DT, LayoutColumnMajor, DT, LayoutRowMajor>;
    using TileCopyStateVFirst = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRowMajor, DT, LayoutRowMajor, DT, LayoutRowMajor>;
    using TileCopyDw0 = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutRowMajor, DT, LayoutRowMajor, DT, LayoutRowMajor>;
    // dvb is A^T @ du: reinterpret the row-major GM A tile as column-major
    // for the Cube A operand, matching the standalone prepare_wy implementation.
    using TileCopyDvb = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutColumnMajor, DT, LayoutRowMajor, DT, LayoutRowMajor>;
    using TileCopyStage9DkbT = Catlass::Gemm::Tile::PackedTileCopyTla<
        ArchTag, DT, LayoutColumnMajor, DT, LayoutRowMajor, DT, LayoutRowMajor>;
    using TileCopyDauToUb = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, DT, LayoutRowMajor, DT, LayoutColumnMajor, DT, LayoutRowMajor>;
    using TileCopyDvbToUb = Common::Tile::PackedTileCopyTlaToUB<
        ArchTag, DT, LayoutRowMajor, DT, LayoutRowMajor, DT, LayoutRowMajor>;
    using ElementAccumulator = typename TileCopyDau::ElementAccumulator;
    using CopyDauL1ToL0A = typename TileCopyDau::CopyL1ToL0A;
    using CopyDauL1ToL0B = typename TileCopyDau::CopyL1ToL0B;
    using CopyStateVFirstL1ToL0B = typename TileCopyStateVFirst::CopyL1ToL0B;
    using CopyDw0L1ToL0A = typename TileCopyDw0::CopyL1ToL0A;
    using CopyDw0L1ToL0B = typename TileCopyDw0::CopyL1ToL0B;
    using CopyStage9DkbTL1ToL0A = typename TileCopyStage9DkbT::CopyL1ToL0A;
    using CopyStage9DkbTL1ToL0B = typename TileCopyStage9DkbT::CopyL1ToL0B;
    using TileMmad = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, DT, typename TileCopyDw0::LayoutTagL1A>;
    using TileMmadDvb = Catlass::Gemm::Tile::TileMmadTla<
        ArchTag, DT, typename TileCopyDvb::LayoutTagL1A>;

    static constexpr uint32_t L1_RESIDENT_HEAD_COUNT_4 = 4;
    static constexpr uint32_t L0_BUFFER_COUNT_2 = 2;
    static constexpr uint32_t L1_EVENT_COUNT_8 = 8;
    static constexpr uint32_t VECTOR_ELEMS = CHUNK_SIZE_64 * K_SIZE_128;
    static constexpr uint32_t MATRIX_ELEMS = CHUNK_SIZE_64 * CHUNK_SIZE_64;
    static constexpr uint32_t STATE_ELEMS = K_SIZE_128 * V_SIZE_128;
    static constexpr uint32_t VECTOR_BYTES = VECTOR_ELEMS * sizeof(DT);
    static constexpr uint32_t MATRIX_BYTES = MATRIX_ELEMS * sizeof(DT);
    static constexpr uint32_t STATE_BYTES = STATE_ELEMS * sizeof(DT);

    static constexpr uint32_t L1_H_OFFSET = 128 * 1024;
    static constexpr uint32_t L1_A_OFFSET = 0;
    static constexpr uint32_t L1_DU_OFFSET = 256 * 1024;
    static constexpr uint32_t L1_VB_OFFSET = 384 * 1024;
    static constexpr uint32_t L1_KBG_OFFSET = 320 * 1024;
    static constexpr uint32_t L1_DW0_OFFSET = 448 * 1024;
    static constexpr uint32_t L1_STAGE5_DA0_OFFSET = 96 * 1024;
    static constexpr uint32_t L1_STAGE5_K_A_OFFSET = 256 * 1024;
    static constexpr uint32_t L1_STAGE11_DO_OFFSET = 0;
    static constexpr uint32_t L1_STAGE11_VNEW_OFFSET = 320 * 1024;
    static constexpr uint32_t L1_STAGE13_Q_OFFSET = 0;
    static constexpr uint32_t L1_STAGE13_DH_OFFSET = 32 * 1024;
    static constexpr uint32_t L1_STAGE13_DS_OFFSET = 96 * 1024;
    static constexpr uint32_t UB_DAU_OFFSET = 112 * 1024;
    static constexpr uint32_t UB_DVB_OFFSET = 80 * 1024;
    static constexpr uint32_t UB_DAW0_OFFSET = 128 * 1024;
    static constexpr uint32_t UB_A2_OFFSET = 128 * 1024;
    static constexpr uint32_t UB_DA2_OFFSET = 192 * 1024;
    static constexpr uint32_t UB_DKBG0_OFFSET = 208 * 1024;
    static constexpr uint32_t UB_STAGE12_DS_OFFSET = 176 * 1024;
    static constexpr uint32_t UB_STAGE13_DK0_OFFSET = 32 * 1024;
    static constexpr uint32_t UB_STAGE13_DQ_INTRA_OFFSET = 80 * 1024;
    static constexpr uint32_t UB_STAGE13_DK_INTRA_OFFSET = 144 * 1024;

    // 64 KiB L0A 和 L0B 均直接二等分为两块 32 KiB ping/pong。
    // 当前有效 A 矩阵只占 16 KiB，也不能把 pong 提前到 16 KiB。
    static constexpr uint32_t L0_A_BYTES = 32 * 1024;
    static constexpr uint32_t L0_B_BYTES = STATE_BYTES;
    // Ascend950 的 256 KiB L0C 直接二等分为两块 128 KiB ping/pong。
    // dA_u/dw0 只使用当前 slot 的有效分形，不复用 slot 内剩余空间。
    static constexpr uint32_t L0_C_BYTES = 128 * 1024;
    static constexpr AscendC::FixpipeConfig FIXPIPE_NZ_L1_CONFIG = {
        AscendC::CO2Layout::NZ, false};

    static constexpr auto GM_DU_LAYOUT = tla::MakeLayout<DT, LayoutRowMajor>(
        tla::Int<CHUNK_SIZE_64>{}, tla::Int<V_SIZE_128>{});
    // h/dh are physically [K,V] row-major. Reinterpret the same bytes as a
    // [V,K] column-major view so Cube consumes h.T/dh.T without a copy.
    static constexpr auto GM_STATE_TRANSPOSE_LAYOUT = tla::MakeLayout<DT, LayoutColumnMajor>(
        tla::Int<V_SIZE_128>{}, tla::Int<K_SIZE_128>{});
    // state_v_first=true 时 GM 本身已是 [V,K]，直接按行主序搬入 L1B。
    static constexpr auto GM_STATE_V_FIRST_LAYOUT = tla::MakeLayout<DT, LayoutRowMajor>(
        tla::Int<V_SIZE_128>{}, tla::Int<K_SIZE_128>{});
    static constexpr auto GM_A_LAYOUT = tla::MakeLayout<DT, LayoutRowMajor>(
        tla::Int<CHUNK_SIZE_64>{}, tla::Int<CHUNK_SIZE_64>{});
    // GM 物理数据是 [M,128] 行主序。作为 GEMM B 操作数读取时，等价解释为
    // [128,M] 列主序，使 GM->L1B 直接生成转置乘所需的 L1 布局。
    static constexpr auto GM_VECTOR_TRANSPOSE_LAYOUT = tla::MakeLayout<DT, LayoutColumnMajor>(
        tla::Int<K_SIZE_128>{}, tla::Int<CHUNK_SIZE_64>{});
    static constexpr auto L1_DU_LAYOUT = tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1A>(
        tla::Int<CHUNK_SIZE_64>{}, tla::Int<V_SIZE_128>{});
    static constexpr auto L1_VB_LAYOUT = tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1B>(
        tla::Int<V_SIZE_128>{}, tla::Int<CHUNK_SIZE_64>{});
    static constexpr auto L1_STATE_TRANSPOSE_LAYOUT =
        tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1B>(
        tla::Int<V_SIZE_128>{}, tla::Int<K_SIZE_128>{});
    static constexpr auto L1_STATE_V_FIRST_LAYOUT =
        tla::MakeLayout<DT, typename TileCopyStateVFirst::LayoutTagL1B>(
        tla::Int<V_SIZE_128>{}, tla::Int<K_SIZE_128>{});
    static constexpr auto UB_DAU_LAYOUT = tla::MakeLayout<DT, LayoutRowMajor>(
        tla::Int<CHUNK_SIZE_64>{}, tla::Int<CHUNK_SIZE_64>{});
    static constexpr auto UB_DVB_LAYOUT = tla::MakeLayout<DT, LayoutRowMajor>(
        tla::Int<CHUNK_SIZE_64>{}, tla::Int<V_SIZE_128>{});

    static_assert(STATE_BYTES == 32 * 1024, "Stage 1 h slot must be 32 KiB.");
    static_assert(VECTOR_BYTES == 16 * 1024, "Stage 1 du/vb/dw0 slot must be 16 KiB.");
    static_assert(MATRIX_BYTES == 8 * 1024, "Stage 1 dA_u slot must be 8 KiB.");
    static_assert(L1_DW0_OFFSET + L1_RESIDENT_HEAD_COUNT_4 * VECTOR_BYTES == 512 * 1024,
                  "Stage 1 L1 resident layout must end at 512 KiB.");
    static_assert(UB_DAU_OFFSET + BANK_COUNT_2 * MATRIX_BYTES == 128 * 1024,
                  "Stage 1 dA_u resident must occupy UB[112,128) KiB.");
    static_assert(L1_A_OFFSET + L1_RESIDENT_HEAD_COUNT_4 * MATRIX_BYTES == 32 * 1024,
                  "Stage 3 A resident must occupy L1[0,32) KiB.");
    static_assert(UB_DVB_OFFSET + BANK_COUNT_2 * VECTOR_BYTES == UB_DAU_OFFSET,
                  "Stage 3 dvb slots must occupy UB[80,112) KiB.");
    static_assert(UB_DAW0_OFFSET + VECTOR_BYTES + MATRIX_BYTES == 152 * 1024,
                  "Stage 3 dA_w0 uses the first 8 KiB of UB banks at 128/144 KiB.");
    static_assert(L1_STAGE5_DA0_OFFSET + L1_RESIDENT_HEAD_COUNT_4 * MATRIX_BYTES == 128 * 1024,
                  "Stage 5 dA0 inputs must occupy L1[96,128) KiB.");
    static_assert(L1_STAGE5_K_A_OFFSET + L1_RESIDENT_HEAD_COUNT_4 * VECTOR_BYTES == 320 * 1024,
                  "Stage 5 k L1A resident must occupy L1[256,320) KiB.");
    static_assert(L1_STAGE11_DO_OFFSET + BANK_COUNT_2 * VECTOR_BYTES == 32 * 1024,
                  "Stage 11 do ping/pong must occupy L1[0,32) KiB.");
    static_assert(L1_STAGE11_VNEW_OFFSET + L1_RESIDENT_HEAD_COUNT_4 * VECTOR_BYTES == 384 * 1024,
                  "Stage 11 v_new resident must occupy L1[320,384) KiB.");
    static_assert(L1_STAGE13_Q_OFFSET + BANK_COUNT_2 * VECTOR_BYTES == L1_STAGE13_DH_OFFSET,
                  "Stage 13 q slots must occupy L1[0,32) KiB.");
    static_assert(L1_STAGE13_DH_OFFSET + BANK_COUNT_2 * STATE_BYTES == L1_STAGE13_DS_OFFSET,
                  "Stage 13 dh slots must occupy L1[32,96) KiB.");
    static_assert(L1_STAGE13_DS_OFFSET + L1_RESIDENT_HEAD_COUNT_4 * MATRIX_BYTES == 128 * 1024,
                  "Stage 13 ds slots must occupy L1[96,128) KiB.");
    static_assert(UB_A2_OFFSET + BANK_COUNT_2 * MATRIX_BYTES == 144 * 1024,
                  "Stage 5 a2 slots must occupy UB[128,144) KiB.");
    static_assert(UB_DA2_OFFSET + BANK_COUNT_2 * MATRIX_BYTES == UB_DKBG0_OFFSET,
                  "Stage 7 dA2 slots must occupy UB[192,208) KiB.");
    static_assert(UB_DKBG0_OFFSET + BANK_COUNT_2 * VECTOR_BYTES == 240 * 1024,
                  "Stage 7 dkbg0 slots must occupy UB[208,240) KiB.");
    static_assert(UB_STAGE13_DK0_OFFSET + BANK_COUNT_2 * VECTOR_BYTES == 64 * 1024,
                  "Stage 13 dk0 slots must occupy UB[32,64) KiB.");
    static_assert(UB_STAGE13_DK_INTRA_OFFSET + BANK_COUNT_2 * VECTOR_BYTES == 176 * 1024,
                  "Stage 13 dk_intra slots must occupy UB[144,176) KiB.");
    static_assert(UB_STAGE13_DQ_INTRA_OFFSET + BANK_COUNT_2 * VECTOR_BYTES == 112 * 1024,
                  "Stage 13 dq_hv slots must occupy UB[80,112) KiB.");
    static_assert(L0_C_BYTES * L0_BUFFER_COUNT_2 <= ArchTag::L0C_SIZE,
                  "Stage 1 L0C ping/pong exceeds the architecture limit.");

    __aicore__ inline void ProcessStage1Head(
        AscendC::LocalTensor<DT> (&hL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&duL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&vbL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&dw0L1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&l0A)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<DT> (&l0B)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<DT> (&dAuCv)[BANK_COUNT_2],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t mAlign = (chunkLen + 15U) / 16U * 16U;
        const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % AIV_COUNT_2);
        const uint32_t localHv = static_cast<uint32_t>(headOffset / AIV_COUNT_2);
        const uint32_t duEvent = static_cast<uint32_t>(headOffset);
        const uint32_t hEvent = L1_RESIDENT_HEAD_COUNT_4 + static_cast<uint32_t>(headOffset);
        const int64_t tokenBase =
            ((chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart) * tiling_->V;
        const int64_t stateBase =
            ((chunk.bIdx * tiling_->HV + hv) * stateChunkNum_ + chunk.stateChunkIdx) *
            tiling_->K * tiling_->V;

        AscendC::GlobalTensor<DT> gmDu;
        AscendC::GlobalTensor<DT> gmH;
        AscendC::GlobalTensor<DT> gmVb;
        gmDu.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(du_) + tokenBase);
        gmH.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(h_) + stateBase);
        gmVb.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT *>(vb_) + WorkspaceHeadOffset(headOffset));
        auto tensorGmDu = tla::MakeTensor(gmDu, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmVb = tla::MakeTensor(
            gmVb, GM_VECTOR_TRANSPOSE_LAYOUT, Catlass::Arch::PositionGM{});
        auto blockGmDu = tla::GetTile(
            tensorGmDu, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, V_SIZE_128));
        auto blockGmVb = tla::GetTile(
            tensorGmVb, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, chunkLen));
        auto tensorL1Du = tla::MakeTensor(
            duL1[headOffset], L1_DU_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1HTranspose = tla::MakeTensor(
            hL1[headOffset], L1_STATE_TRANSPOSE_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1HVFirst = tla::MakeTensor(
            hL1[headOffset], L1_STATE_V_FIRST_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1Vb = tla::MakeTensor(
            vbL1[headOffset], L1_VB_LAYOUT, Catlass::Arch::PositionL1{});

        // 每个 HEAD 只搬入自己的 du/h。等待对应 resident 可复用后完成
        // GM->L1，再由同一 HEAD 的 MTE1/Cube 消费，不提前发射下一 HEAD。
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(duEvent);
        typename TileCopyDau::template CopyGmToL1A<decltype(blockGmDu)>{}(tensorL1Du, blockGmDu);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(hEvent);
        if (tiling_->stateVFirst != 0) {
            auto tensorGmH = tla::MakeTensor(
                gmH, GM_STATE_V_FIRST_LAYOUT, Catlass::Arch::PositionGM{});
            auto blockGmH = tla::GetTile(
                tensorGmH, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            typename TileCopyStateVFirst::template CopyGmToL1B<decltype(blockGmH)>{}(
                tensorL1HVFirst, blockGmH);
        } else {
            auto tensorGmH = tla::MakeTensor(
                gmH, GM_STATE_TRANSPOSE_LAYOUT, Catlass::Arch::PositionGM{});
            auto blockGmH = tla::GetTile(
                tensorGmH, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            typename TileCopyDau::template CopyGmToL1B<decltype(blockGmH)>{}(
                tensorL1HTranspose, blockGmH);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(hEvent);

        // owner AIV 先用 MTE3 把本 HEAD 的 vb 写入 GM，另一个 AIV 发送
        // mode=0x2 参与信号。AIC 在 MTE2 pipe 聚合等待后执行 GM->L1；
        // duEvent 的 MTE2->MTE1 flag 同时覆盖此前的 du 和本次 vb 搬入。
        Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);
        typename TileCopyDau::template CopyGmToL1B<decltype(blockGmVb)>{}(tensorL1Vb, blockGmVb);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(duEvent);

        // GEMM 1: dA_u = du @ vb.T。du 首次进入 L0A 后仍保留在 L1，
        // vb 在本次 L1B->L0B 后完成末次消费，与 du 共用 duEvent 保护复用。
        const uint32_t dAuL0ASlot = l0ASlot_;
        const uint32_t dAuL0BSlot = l0BSlot_;
        const uint32_t dAuL0CSlot = l0CSlot_;
        const uint32_t dAuL0AEvent = 2 * dAuL0ASlot;
        const uint32_t dAuL0BEvent = 2 * dAuL0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        auto tensorDauL0A = tla::MakeTensor(
            l0A[dAuL0ASlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(mActual, V_SIZE_128),
            Catlass::Arch::PositionL0A{});
        auto tensorDauL0B = tla::MakeTensor(
            l0B[dAuL0BSlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(V_SIZE_128, chunkLen),
            Catlass::Arch::PositionL0B{});
        auto tileDauL1A = tla::GetTile(
            tensorL1Du, tla::MakeCoord(0, 0), tla::MakeShape(mActual, V_SIZE_128));
        auto tileDauL1B = tla::GetTile(
            tensorL1Vb, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, chunkLen));
        auto tensorDauL0C = tla::MakeTensor(
            l0C[dAuL0CSlot], tla::MakeLayoutL0C(mActual, chunkLen),
            Catlass::Arch::PositionL0C{});
        auto tileDauL0C = tla::GetTile(
            tensorDauL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(duEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dAuL0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dAuL0BEvent);
        CopyDauL1ToL0A{}(tensorDauL0A, tileDauL1A);
        CopyDauL1ToL0B{}(tensorDauL0B, tileDauL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dAuL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dAuL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dAuL0CSlot);
        tileMmad_(tileDauL0C, tensorDauL0A, tensorDauL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dAuL0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dAuL0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dAuL0CSlot);

        // dA_u 的 L0C 完成后立即定向 Fixpipe 到 owner AIV，不把
        // 本次 copyout 拖到下一条 MMAD 之后。这样每个 L0C slot 都按
        // M -> Fixpipe -> M 闭环。每发射一次 MMAD 就取反 L0C slot，
        // 避免 dA_u 的 [M,64] 与 dw0 的 [M,128] 异形结果复用同一物理槽。
        auto tensorDauCv = tla::MakeTensor(
            dAuCv[localHv], UB_DAU_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockDauCv = tla::GetTile(
            tensorDauCv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        // 当前任务组最多产生四份 dA_u，分别定向写入两个 AIV 上各两份
        // 固定 resident slot。Stage 1 内物理槽位不会复用，因此四次
        // Fixpipe 可以连续异步发射，不等待 AIV 反向 release。
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dAuL0CSlot);
        typename TileCopyDauToUb::template CopyL0CToDst<decltype(blockDauCv)>{}(
            blockDauCv, tensorDauL0C, static_cast<uint8_t>(ownerSubBlock), 0);
        // dA_u 的唯一 Fixpipe 已定向写入 owner AIV 的 UB。本信号逐 HEAD
        // 通知 owner 立即执行 Stage 2，通知 non-owner 完成本 HEAD 握手，
        // 并确认 AIC 已结束对 vb resident 的使用。两个 AIV 都消费回应后，
        // 固定 EventID 才能供下一 HEAD 复用，不再额外发送 Stage 1 组信号。
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVecFlag_);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dAuL0CSlot);

        // GEMM 2: dw0 = du @ h.T。L0A/L0B 每次 MMAD 后立即取反；
        // L0C 也在每次 MMAD 后取反。du/h 在末次 MTE1 后发布，
        // 下一任务组才能覆盖。
        const uint32_t dw0L0CSlot = l0CSlot_;
        l0CSlot_ ^= 1U;
        auto tensorDw0L0C = tla::MakeTensor(
            l0C[dw0L0CSlot], tla::MakeLayoutL0C(mActual, K_SIZE_128),
            Catlass::Arch::PositionL0C{});
        auto tileDw0L0C = tla::GetTile(
            tensorDw0L0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, K_SIZE_128));
        const uint32_t dw0L0ASlot = l0ASlot_;
        const uint32_t dw0L0BSlot = l0BSlot_;
        const uint32_t dw0L0AEvent = 2 * dw0L0ASlot;
        const uint32_t dw0L0BEvent = 2 * dw0L0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        auto tensorDw0L0A = tla::MakeTensor(
            l0A[dw0L0ASlot],
            tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL0A>(mActual, V_SIZE_128),
            Catlass::Arch::PositionL0A{});
        auto tensorDw0L0BTranspose = tla::MakeTensor(
            l0B[dw0L0BSlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(V_SIZE_128, K_SIZE_128),
            Catlass::Arch::PositionL0B{});
        auto tensorDw0L0BVFirst = tla::MakeTensor(
            l0B[dw0L0BSlot],
            tla::MakeLayout<DT, typename TileCopyStateVFirst::LayoutTagL0B>(V_SIZE_128, K_SIZE_128),
            Catlass::Arch::PositionL0B{});
        auto tileDw0L1A = tla::GetTile(
            tensorL1Du, tla::MakeCoord(0, 0), tla::MakeShape(mActual, V_SIZE_128));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(hEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dw0L0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dw0L0BEvent);
        CopyDw0L1ToL0A{}(tensorDw0L0A, tileDw0L1A);
        if (tiling_->stateVFirst != 0) {
            auto tileDw0L1B = tla::GetTile(
                tensorL1HVFirst, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            CopyStateVFirstL1ToL0B{}(tensorDw0L0BVFirst, tileDw0L1B);
        } else {
            auto tileDw0L1B = tla::GetTile(
                tensorL1HTranspose, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            CopyDauL1ToL0B{}(tensorDw0L0BTranspose, tileDw0L1B);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(duEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(hEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dw0L0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dw0L0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dw0L0CSlot);
        // [M,128] x [128,128] 的 K 维度一次完成，不拆成两个 K=64。
        if (tiling_->stateVFirst != 0) {
            tileMmad_(tileDw0L0C, tensorDw0L0A, tensorDw0L0BVFirst, true, 0);
        } else {
            tileMmad_(tileDw0L0C, tensorDw0L0A, tensorDw0L0BTranspose, true, 0);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dw0L0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dw0L0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dw0L0CSlot);

        // dw0 作为 Stage 3 的 A 操作数，以 NZ 形式保留在 L1[448,512) KiB。
        // 尾 chunk 只写有效 M 行；后续 Stage 3 同样只读取有效行，不清零 padding。
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dw0L0CSlot);
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> dw0FixpipeParams;
        dw0FixpipeParams.nSize = K_SIZE_128;
        dw0FixpipeParams.mSize = chunkLen;
        // NZ 模式下，srcStride 是 L0C 相邻 N 分形的起始地址间隔，
        // 单位为 L0C 的 C0_Size；它由 M 方向的 16 对齐长度决定，
        // 不能误用 N=128。dstStride 的单位是目的 BF16 元素；目的区域
        // 固定按 [64,128] L1A slot 解释，因此尾 chunk 也保持 64 * 16，
        // Stage 3 可直接读取同一物理布局，不需要二次搬运或重排。
        dw0FixpipeParams.srcStride = mAlign;
        dw0FixpipeParams.dstStride = CHUNK_SIZE_64 * 16U;
        dw0FixpipeParams.quantPre = QuantMode_t::F322BF16;
        // L0C 生命周期完全由 M_FIX/FIX_M 硬事件管理，不叠加 unit flag。
        dw0FixpipeParams.unitFlag = 0;
        AscendC::Fixpipe<DT, ElementAccumulator, FIXPIPE_NZ_L1_CONFIG>(
            dw0L1[headOffset], l0C[dw0L0CSlot], dw0FixpipeParams);
        // Stage 3 的 MTE1 将读取本 HEAD 的 dw0。该事件精确表达
        // Fixpipe 写 L1 -> MTE1 读 L1 的 RAW 依赖，不使用宽泛 barrier。
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(
            static_cast<uint32_t>(headOffset) + L1_RESIDENT_HEAD_COUNT_4);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dw0L0CSlot);
    }

    __aicore__ inline void ProcessStage3Head(
        AscendC::LocalTensor<DT> (&aL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&duL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&kbgL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&dw0L1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&l0A)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<DT> (&l0B)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<DT> (&dvbCv)[BANK_COUNT_2],
        AscendC::LocalTensor<DT> (&dAw0Cv)[BANK_COUNT_2],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t mAlign = (chunkLen + 15U) / 16U * 16U;
        const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % AIV_COUNT_2);
        const uint32_t localHv = static_cast<uint32_t>(headOffset / AIV_COUNT_2);
        const uint32_t aEvent = static_cast<uint32_t>(headOffset);
        const int64_t aBase =
            ((chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart) * CHUNK_SIZE_64;

        AscendC::GlobalTensor<DT> gmA;
        AscendC::GlobalTensor<DT> gmKbg;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(a_) + aBase);
        gmKbg.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT *>(kbg_) + WorkspaceHeadOffset(headOffset));
        auto tensorGmA = tla::MakeTensor(gmA, GM_A_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmKbg = tla::MakeTensor(
            gmKbg, GM_VECTOR_TRANSPOSE_LAYOUT, Catlass::Arch::PositionGM{});
        auto blockGmA = tla::GetTile(
            tensorGmA, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        auto blockGmKbg = tla::GetTile(
            tensorGmKbg, tla::MakeCoord(0, 0), tla::MakeShape(K_SIZE_128, chunkLen));
        auto tensorL1A = tla::MakeTensor(
            aL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});

        // Stage 1 已释放该事件槽。Stage 3 用一次 MTE2 生命周期同时搬入
        // A 和 Stage 0 已落 GM 的 kbg，不从 AIV UB 直写 L1。
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(aEvent);
        typename TileCopyDw0::template CopyGmToL1A<decltype(blockGmA)>{}(tensorL1A, blockGmA);
        auto tensorKbgL1 = tla::MakeTensor(
            kbgL1[headOffset], L1_VB_LAYOUT, Catlass::Arch::PositionL1{});
        typename TileCopyDau::template CopyGmToL1B<decltype(blockGmKbg)>{}(tensorKbgL1, blockGmKbg);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(aEvent);

        // GEMM 1: dA_w0 = dw0 @ kbg.T。
        const uint32_t dAwL0ASlot = l0ASlot_;
        const uint32_t dAwL0BSlot = l0BSlot_;
        const uint32_t dAwL0CSlot = l0CSlot_;
        const uint32_t dAwL0AEvent = 2 * dAwL0ASlot;
        const uint32_t dAwL0BEvent = 2 * dAwL0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        auto tensorDw0L1 = tla::MakeTensor(
            dw0L1[headOffset], L1_DU_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorDAwL0A = tla::MakeTensor(
            l0A[dAwL0ASlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(mActual, K_SIZE_128),
            Catlass::Arch::PositionL0A{});
        auto tensorDAwL0B = tla::MakeTensor(
            l0B[dAwL0BSlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(K_SIZE_128, chunkLen),
            Catlass::Arch::PositionL0B{});
        auto tileDw0L1 = tla::GetTile(
            tensorDw0L1, tla::MakeCoord(0, 0), tla::MakeShape(mActual, K_SIZE_128));
        auto tileKbgL1 = tla::GetTile(
            tensorKbgL1, tla::MakeCoord(0, 0), tla::MakeShape(K_SIZE_128, chunkLen));
        auto tensorDAwL0C = tla::MakeTensor(
            l0C[dAwL0CSlot], tla::MakeLayoutL0C(mActual, chunkLen),
            Catlass::Arch::PositionL0C{});
        auto tileDAwL0C = tla::GetTile(
            tensorDAwL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(
            static_cast<uint32_t>(headOffset) + L1_RESIDENT_HEAD_COUNT_4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(aEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dAwL0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dAwL0BEvent);
        CopyDauL1ToL0A{}(tensorDAwL0A, tileDw0L1);
        CopyDauL1ToL0B{}(tensorDAwL0B, tileKbgL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dAwL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dAwL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dAwL0CSlot);
        tileMmad_(tileDAwL0C, tensorDAwL0A, tensorDAwL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dAwL0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dAwL0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dAwL0CSlot);
        auto tensorDAwCv = tla::MakeTensor(
            dAw0Cv[localHv], UB_DAU_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockDAwCv = tla::GetTile(
            tensorDAwCv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dAwL0CSlot);
        typename TileCopyDauToUb::template CopyL0CToDst<decltype(blockDAwCv)>{}(
            blockDAwCv, tensorDAwL0C, static_cast<uint8_t>(ownerSubBlock), 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dAwL0CSlot);

        // GEMM 2: dvb = A @ du。A 和 du 都在 L1 resident 中，Stage 3
        // 不从 GM 重读 du；两条 GEMM 使用不同的 L0 ping/pong slot。
        const uint32_t dvbL0ASlot = l0ASlot_;
        const uint32_t dvbL0BSlot = l0BSlot_;
        const uint32_t dvbL0CSlot = l0CSlot_;
        const uint32_t dvbL0AEvent = 2 * dvbL0ASlot;
        const uint32_t dvbL0BEvent = 2 * dvbL0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        auto tensorDuL1B = tla::MakeTensor(
            duL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1B>(
                CHUNK_SIZE_64, V_SIZE_128), Catlass::Arch::PositionL1{});
        auto tensorL1AForDvb = tla::MakeTensor(
            aL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorDvbL0A = tla::MakeTensor(
            l0A[dvbL0ASlot],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL0A>(mActual, chunkLen),
            Catlass::Arch::PositionL0A{});
        auto tensorDvbL0B = tla::MakeTensor(
            l0B[dvbL0BSlot],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL0B>(chunkLen, V_SIZE_128),
            Catlass::Arch::PositionL0B{});
        auto tileAL1 = tla::GetTile(
            tensorL1AForDvb, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        auto tileDuL1B = tla::GetTile(
            tensorDuL1B, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, V_SIZE_128));
        auto tensorDvbL0C = tla::MakeTensor(
            l0C[dvbL0CSlot], tla::MakeLayoutL0C(mActual, V_SIZE_128),
            Catlass::Arch::PositionL0C{});
        auto tileDvbL0C = tla::GetTile(
            tensorDvbL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, V_SIZE_128));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dvbL0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dvbL0BEvent);
        typename TileCopyDvb::CopyL1ToL0A{}(tensorDvbL0A, tileAL1);
        typename TileCopyDvb::CopyL1ToL0B{}(tensorDvbL0B, tileDuL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(aEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dvbL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dvbL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dvbL0CSlot);
        TileMmadDvb{}(tileDvbL0C, tensorDvbL0A, tensorDvbL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dvbL0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dvbL0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dvbL0CSlot);
        auto tensorDvbCv = tla::MakeTensor(
            dvbCv[localHv], UB_DVB_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockDvbCv = tla::GetTile(
            tensorDvbCv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, V_SIZE_128));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dvbL0CSlot);
        typename TileCopyDvbToUb::template CopyL0CToDst<decltype(blockDvbCv)>{}(
            blockDvbCv, tensorDvbL0C, static_cast<uint8_t>(ownerSubBlock), 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dvbL0CSlot);
    }

    __aicore__ inline void ProcessStage5Head(
        AscendC::LocalTensor<DT> (&aL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&dA0L1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&kStage5L1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&l0A)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<DT> (&l0B)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<DT> (&a2Cv)[BANK_COUNT_2],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t mAlign = (chunkLen + 15U) / 16U * 16U;
        const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % AIV_COUNT_2);
        const uint32_t localHv = static_cast<uint32_t>(headOffset / AIV_COUNT_2);
        const uint32_t l1Event = static_cast<uint32_t>(headOffset);
        const int64_t hk = hv / tiling_->headRatio;
        const int64_t kBase =
            ((chunk.bIdx * tiling_->HK + hk) * tiling_->T + chunk.tokenStart) * tiling_->K;

        AscendC::GlobalTensor<DT> gmDA0;
        AscendC::GlobalTensor<DT> gmK;
        gmDA0.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT *>(dA0_) + WorkspaceHeadOffset(headOffset));
        gmK.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(k_) + kBase);
        auto tensorGmDA0 = tla::MakeTensor(gmDA0, GM_A_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmK = tla::MakeTensor(gmK, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto blockGmDA0 = tla::GetTile(
            tensorGmDA0, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        auto blockGmK = tla::GetTile(
            tensorGmK, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));

        // k 是原始输入，与 Stage 4 无依赖；先搬入独立的 L1 resident。
        // dA0 由 Stage 4 写入 GM，只在它的 MTE2 搬入前消费跨核 token。
        // 同一个事件覆盖本 HEAD 的两次 MTE2 写入，直到后面的两条
        // 矩阵乘都完成 L1 读取后才允许复用槽位。
        auto tensorL1DA0 = tla::MakeTensor(
            dA0L1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorL1K = tla::MakeTensor(
            kStage5L1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1A>(
                CHUNK_SIZE_64, K_SIZE_128), Catlass::Arch::PositionL1{});
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1Event);
        typename TileCopyDau::template CopyGmToL1A<decltype(blockGmK)>{}(
            tensorL1K, blockGmK);
        Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);
        typename TileCopyDw0::template CopyGmToL1A<decltype(blockGmDA0)>{}(
            tensorL1DA0, blockGmDA0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1Event);

        // GEMM 1: dA1 = dA0 @ A.T。A 沿用 Stage 3 的 row-major L1
        // resident，并在同一物理地址上按 column-major L1B 视图解释。
        const uint32_t dA1L0ASlot = l0ASlot_;
        const uint32_t dA1L0BSlot = l0BSlot_;
        const uint32_t dA1L0CSlot = l0CSlot_;
        const uint32_t dA1L0AEvent = 2 * dA1L0ASlot;
        const uint32_t dA1L0BEvent = 2 * dA1L0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        auto tensorL1AB = tla::MakeTensor(
            aL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1B>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorDA1L0A = tla::MakeTensor(
            l0A[dA1L0ASlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(mActual, chunkLen),
            Catlass::Arch::PositionL0A{});
        auto tensorDA1L0B = tla::MakeTensor(
            l0B[dA1L0BSlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(chunkLen, chunkLen),
            Catlass::Arch::PositionL0B{});
        auto tileDA0L1 = tla::GetTile(
            tensorL1DA0, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        auto tileAL1B = tla::GetTile(
            tensorL1AB, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        auto tensorDA1L0C = tla::MakeTensor(
            l0C[dA1L0CSlot], tla::MakeLayoutL0C(mActual, chunkLen),
            Catlass::Arch::PositionL0C{});
        auto tileDA1L0C = tla::GetTile(
            tensorDA1L0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1Event);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dA1L0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dA1L0BEvent);
        CopyDauL1ToL0A{}(tensorDA1L0A, tileDA0L1);
        CopyDauL1ToL0B{}(tensorDA1L0B, tileAL1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dA1L0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dA1L0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dA1L0CSlot);
        tileMmad_(tileDA1L0C, tensorDA1L0A, tensorDA1L0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dA1L0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dA1L0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dA1L0CSlot);
        // dA0 已被 MTE1 完整搬入 L0A，同一 HEAD 的 L1 slot 可以原址改为
        // dA1 resident。后续 Stage 7 直接按 L1B 布局读取，不经过 UB/GM。
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dA1L0CSlot);
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> dA1FixpipeParams;
        // NZ Fixpipe 的 M/N 搬出范围都按 16 对齐。Stage 7 只读取左上角
        // [chunkLen, chunkLen] 有效区域，因此 padding 不参与后续计算。
        dA1FixpipeParams.nSize = mAlign;
        dA1FixpipeParams.mSize = mAlign;
        dA1FixpipeParams.srcStride = mAlign;
        dA1FixpipeParams.dstStride = CHUNK_SIZE_64 * 16U;
        dA1FixpipeParams.quantPre = QuantMode_t::F322BF16;
        dA1FixpipeParams.unitFlag = 0;
        AscendC::Fixpipe<DT, ElementAccumulator, FIXPIPE_NZ_L1_CONFIG>(
            dA0L1[headOffset], l0C[dA1L0CSlot], dA1FixpipeParams);
        // Stage 7 的 MTE1 读取 dA1 前必须等待本次 Fixpipe 完成。
        AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(
            static_cast<uint32_t>(headOffset) + L1_RESIDENT_HEAD_COUNT_4);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dA1L0CSlot);

        // GEMM 2: a2 = k @ k.T。参考 prepare 算子，将同一个 k resident
        // 同时按 L1A 的 [M,128] 和 L1B 的 [128,M] 解释；L1 中没有 k.T
        // 副本，也没有任何位置移动操作。
        const uint32_t a2L0ASlot = l0ASlot_;
        const uint32_t a2L0BSlot = l0BSlot_;
        const uint32_t a2L0CSlot = l0CSlot_;
        const uint32_t a2L0AEvent = 2 * a2L0ASlot;
        const uint32_t a2L0BEvent = 2 * a2L0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        auto tensorL1KT = tla::MakeTensor(
            kStage5L1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1B>(
                K_SIZE_128, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorA2L0A = tla::MakeTensor(
            l0A[a2L0ASlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(mActual, K_SIZE_128),
            Catlass::Arch::PositionL0A{});
        auto tensorA2L0B = tla::MakeTensor(
            l0B[a2L0BSlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(K_SIZE_128, chunkLen),
            Catlass::Arch::PositionL0B{});
        auto tileKL1 = tla::GetTile(
            tensorL1K, tla::MakeCoord(0, 0), tla::MakeShape(mActual, K_SIZE_128));
        auto tileKTL1 = tla::GetTile(
            tensorL1KT, tla::MakeCoord(0, 0), tla::MakeShape(K_SIZE_128, chunkLen));
        auto tensorA2L0C = tla::MakeTensor(
            l0C[a2L0CSlot], tla::MakeLayoutL0C(mActual, chunkLen),
            Catlass::Arch::PositionL0C{});
        auto tileA2L0C = tla::GetTile(
            tensorA2L0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(a2L0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(a2L0BEvent);
        CopyDauL1ToL0A{}(tensorA2L0A, tileKL1);
        CopyDauL1ToL0B{}(tensorA2L0B, tileKTL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(a2L0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(a2L0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(a2L0CSlot);
        tileMmad_(tileA2L0C, tensorA2L0A, tensorA2L0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(a2L0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(a2L0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(a2L0CSlot);
        auto tensorA2Cv = tla::MakeTensor(
            a2Cv[localHv], UB_DAU_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockA2Cv = tla::GetTile(
            tensorA2Cv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(a2L0CSlot);
        typename TileCopyDauToUb::template CopyL0CToDst<decltype(blockA2Cv)>{}(
            blockA2Cv, tensorA2L0C, static_cast<uint8_t>(ownerSubBlock), 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(a2L0CSlot);
    }

    __aicore__ inline void ProcessStage7Head(
        AscendC::LocalTensor<DT> (&aL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&dA1L1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&dw0L1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&l0A)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<DT> (&l0B)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<DT> (&dA2Cv)[BANK_COUNT_2],
        AscendC::LocalTensor<DT> (&dkbg0Cv)[BANK_COUNT_2],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % AIV_COUNT_2);
        const uint32_t localHv = static_cast<uint32_t>(headOffset / AIV_COUNT_2);
        const uint32_t l1Event = static_cast<uint32_t>(headOffset);
        auto tensorL1DA1 = tla::MakeTensor(
            dA1L1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL1B>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});

        auto tensorL1A = tla::MakeTensor(
            aL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});

        // GEMM 1: dA2 = A.T @ dA1。A 的 row-major resident 按
        // column-major L1A 视图解释；dA1 直接读取 Stage 5 的 L1 resident。
        const uint32_t dA2L0ASlot = l0ASlot_;
        const uint32_t dA2L0BSlot = l0BSlot_;
        const uint32_t dA2L0CSlot = l0CSlot_;
        const uint32_t dA2L0AEvent = 2 * dA2L0ASlot;
        const uint32_t dA2L0BEvent = 2 * dA2L0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        auto tensorDA2L0A = tla::MakeTensor(
            l0A[dA2L0ASlot],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL0A>(mActual, chunkLen),
            Catlass::Arch::PositionL0A{});
        auto tensorDA2L0B = tla::MakeTensor(
            l0B[dA2L0BSlot],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL0B>(chunkLen, chunkLen),
            Catlass::Arch::PositionL0B{});
        auto tileAL1 = tla::GetTile(
            tensorL1A, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        auto tileDA1L1 = tla::GetTile(
            tensorL1DA1, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        auto tensorDA2L0C = tla::MakeTensor(
            l0C[dA2L0CSlot], tla::MakeLayoutL0C(mActual, chunkLen),
            Catlass::Arch::PositionL0C{});
        auto tileDA2L0C = tla::GetTile(
            tensorDA2L0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(
            static_cast<uint32_t>(headOffset) + L1_RESIDENT_HEAD_COUNT_4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dA2L0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dA2L0BEvent);
        typename TileCopyDvb::CopyL1ToL0A{}(tensorDA2L0A, tileAL1);
        typename TileCopyDvb::CopyL1ToL0B{}(tensorDA2L0B, tileDA1L1);
        // dA1 已进入 L0B，允许下一任务组的 Stage 5 MTE2 覆盖该 L1 slot。
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1Event);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dA2L0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dA2L0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dA2L0CSlot);
        TileMmadDvb{}(tileDA2L0C, tensorDA2L0A, tensorDA2L0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dA2L0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dA2L0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dA2L0CSlot);
        auto tensorDA2Cv = tla::MakeTensor(
            dA2Cv[localHv], UB_DAU_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockDA2Cv = tla::GetTile(
            tensorDA2Cv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dA2L0CSlot);
        typename TileCopyDauToUb::template CopyL0CToDst<decltype(blockDA2Cv)>{}(
            blockDA2Cv, tensorDA2L0C, static_cast<uint8_t>(ownerSubBlock), 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dA2L0CSlot);

        // GEMM 2: dkbg0 = A.T @ dw0。复用同一 A.T L1A 视图，dw0
        // 直接使用 Stage 1 Fixpipe 保存的 row-major L1B 布局。
        const uint32_t dkbgL0ASlot = l0ASlot_;
        const uint32_t dkbgL0BSlot = l0BSlot_;
        const uint32_t dkbgL0CSlot = l0CSlot_;
        const uint32_t dkbgL0AEvent = 2 * dkbgL0ASlot;
        const uint32_t dkbgL0BEvent = 2 * dkbgL0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        auto tensorL1Dw0B = tla::MakeTensor(
            dw0L1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL1B>(
                CHUNK_SIZE_64, K_SIZE_128), Catlass::Arch::PositionL1{});
        auto tensorDkbgL0A = tla::MakeTensor(
            l0A[dkbgL0ASlot],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL0A>(mActual, chunkLen),
            Catlass::Arch::PositionL0A{});
        auto tensorDkbgL0B = tla::MakeTensor(
            l0B[dkbgL0BSlot],
            tla::MakeLayout<DT, typename TileCopyDvb::LayoutTagL0B>(chunkLen, K_SIZE_128),
            Catlass::Arch::PositionL0B{});
        auto tileDw0L1B = tla::GetTile(
            tensorL1Dw0B, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));
        auto tensorDkbgL0C = tla::MakeTensor(
            l0C[dkbgL0CSlot], tla::MakeLayoutL0C(mActual, K_SIZE_128),
            Catlass::Arch::PositionL0C{});
        auto tileDkbgL0C = tla::GetTile(
            tensorDkbgL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, K_SIZE_128));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkbgL0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkbgL0BEvent);
        typename TileCopyDvb::CopyL1ToL0A{}(tensorDkbgL0A, tileAL1);
        typename TileCopyDvb::CopyL1ToL0B{}(tensorDkbgL0B, tileDw0L1B);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dkbgL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dkbgL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dkbgL0CSlot);
        TileMmadDvb{}(tileDkbgL0C, tensorDkbgL0A, tensorDkbgL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkbgL0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkbgL0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dkbgL0CSlot);
        auto tensorDkbgCv = tla::MakeTensor(
            dkbg0Cv[localHv], UB_DVB_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockDkbgCv = tla::GetTile(
            tensorDkbgCv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dkbgL0CSlot);
        typename TileCopyDvbToUb::template CopyL0CToDst<decltype(blockDkbgCv)>{}(
            blockDkbgCv, tensorDkbgL0C, static_cast<uint8_t>(ownerSubBlock), 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dkbgL0CSlot);
    }

    __aicore__ inline void ProcessStage9Head(
        AscendC::LocalTensor<DT> (&dAL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&kbL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&kL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&l0A)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<DT> (&l0B)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[L0_BUFFER_COUNT_2],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t l1Event = static_cast<uint32_t>(headOffset);
        const int64_t workspaceBase = WorkspaceHeadOffset(headOffset);

        AscendC::GlobalTensor<DT> gmDA;
        AscendC::GlobalTensor<DT> gmKb;
        AscendC::GlobalTensor<DT> gmDkb;
        AscendC::GlobalTensor<DT> gmDkbT;
        gmDA.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dA0_) + workspaceBase);
        gmKb.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(workspace_) + workspaceBase);
        gmDkb.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dkbOut_) + workspaceBase);
        gmDkbT.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dkbTOut_) + workspaceBase);
        auto tensorGmDA = tla::MakeTensor(gmDA, GM_A_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmKb = tla::MakeTensor(gmKb, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmDkb = tla::MakeTensor(gmDkb, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmDkbT = tla::MakeTensor(gmDkbT, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto blockGmDA = tla::GetTile(
            tensorGmDA, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        auto blockGmKb = tla::GetTile(
            tensorGmKb, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));
        auto blockGmDkb = tla::GetTile(
            tensorGmDkb, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));
        auto blockGmDkbT = tla::GetTile(
            tensorGmDkbT, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));

        auto tensorL1DA = tla::MakeTensor(
            dAL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorL1Kb = tla::MakeTensor(
            kbL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL1B>(
                CHUNK_SIZE_64, K_SIZE_128), Catlass::Arch::PositionL1{});
        // kb 由 Stage 2 早已写入 workspace，与当前 Stage 8 无依赖；
        // 先搬入它，再在读取 Stage 8 产生的 dA 前消费跨核 token。
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1Event);
        typename TileCopyStage9DkbT::template CopyGmToL1B<decltype(blockGmKb)>{}(
            tensorL1Kb, blockGmKb);
        Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);
        typename TileCopyDw0::template CopyGmToL1A<decltype(blockGmDA)>{}(
            tensorL1DA, blockGmDA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1Event);

        // GEMM 1: dkb = dA @ k。k 沿用 Stage 5 的 L1 resident，直接按
        // L1B 解释，不从 GM 重复搬运。
        const uint32_t dkbL0ASlot = l0ASlot_;
        const uint32_t dkbL0BSlot = l0BSlot_;
        const uint32_t dkbL0CSlot = l0CSlot_;
        const uint32_t dkbL0AEvent = 2 * dkbL0ASlot;
        const uint32_t dkbL0BEvent = 2 * dkbL0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        auto tensorL1K = tla::MakeTensor(
            kL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1B>(
                CHUNK_SIZE_64, K_SIZE_128), Catlass::Arch::PositionL1{});
        auto tensorDkbL0A = tla::MakeTensor(
            l0A[dkbL0ASlot],
            tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL0A>(mActual, chunkLen),
            Catlass::Arch::PositionL0A{});
        auto tensorDkbL0B = tla::MakeTensor(
            l0B[dkbL0BSlot],
            tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL0B>(chunkLen, K_SIZE_128),
            Catlass::Arch::PositionL0B{});
        auto tileDAL1 = tla::GetTile(
            tensorL1DA, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        auto tileKL1 = tla::GetTile(
            tensorL1K, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));
        auto tensorDkbL0C = tla::MakeTensor(
            l0C[dkbL0CSlot], tla::MakeLayoutL0C(mActual, K_SIZE_128),
            Catlass::Arch::PositionL0C{});
        auto tileDkbL0C = tla::GetTile(
            tensorDkbL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, K_SIZE_128));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1Event);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkbL0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkbL0BEvent);
        CopyDw0L1ToL0A{}(tensorDkbL0A, tileDAL1);
        CopyDw0L1ToL0B{}(tensorDkbL0B, tileKL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dkbL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dkbL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dkbL0CSlot);
        tileMmad_(tileDkbL0C, tensorDkbL0A, tensorDkbL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkbL0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkbL0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dkbL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dkbL0CSlot);
        typename TileCopyDw0::template CopyL0CToDst<decltype(blockGmDkb)>{}(
            blockGmDkb, tensorDkbL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dkbL0CSlot);

        // GEMM 2: dkb_t = dA.T @ kb。ColumnMajor A 标签直接表达 dA.T，
        // Fixpipe 输出已经是 [BT,128]，不计算 kb.T@dA，也不转置结果。
        const uint32_t dkbTL0ASlot = l0ASlot_;
        const uint32_t dkbTL0BSlot = l0BSlot_;
        const uint32_t dkbTL0CSlot = l0CSlot_;
        const uint32_t dkbTL0AEvent = 2 * dkbTL0ASlot;
        const uint32_t dkbTL0BEvent = 2 * dkbTL0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        auto tensorL1DAT = tla::MakeTensor(
            dAL1[headOffset],
            tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorDkbTL0A = tla::MakeTensor(
            l0A[dkbTL0ASlot],
            tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL0A>(mActual, chunkLen),
            Catlass::Arch::PositionL0A{});
        auto tensorDkbTL0B = tla::MakeTensor(
            l0B[dkbTL0BSlot],
            tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL0B>(chunkLen, K_SIZE_128),
            Catlass::Arch::PositionL0B{});
        auto tileDATL1 = tla::GetTile(
            tensorL1DAT, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        auto tileKbL1 = tla::GetTile(
            tensorL1Kb, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));
        auto tensorDkbTL0C = tla::MakeTensor(
            l0C[dkbTL0CSlot], tla::MakeLayoutL0C(mActual, K_SIZE_128),
            Catlass::Arch::PositionL0C{});
        auto tileDkbTL0C = tla::GetTile(
            tensorDkbTL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, K_SIZE_128));
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkbTL0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkbTL0BEvent);
        CopyStage9DkbTL1ToL0A{}(tensorDkbTL0A, tileDATL1);
        CopyStage9DkbTL1ToL0B{}(tensorDkbTL0B, tileKbL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1Event);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dkbTL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dkbTL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dkbTL0CSlot);
        tileMmad_(tileDkbTL0C, tensorDkbTL0A, tensorDkbTL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkbTL0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkbTL0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dkbTL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dkbTL0CSlot);
        typename TileCopyStage9DkbT::template CopyL0CToDst<decltype(blockGmDkbT)>{}(
            blockGmDkbT, tensorDkbTL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dkbTL0CSlot);
    }

    __aicore__ inline void ProcessStage11Head(
        AscendC::LocalTensor<DT> (&doL1)[BANK_COUNT_2],
        AscendC::LocalTensor<DT> (&vNewL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&l0A)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<DT> (&l0B)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<DT> (&ds0Cv)[BANK_COUNT_2],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t doSlot = static_cast<uint32_t>(headOffset) & 1U;
        const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % AIV_COUNT_2);
        const uint32_t localHv = static_cast<uint32_t>(headOffset / AIV_COUNT_2);
        const uint32_t doEvent = doSlot;
        const uint32_t vNewEvent = L1_RESIDENT_HEAD_COUNT_4 + static_cast<uint32_t>(headOffset);
        const int64_t vectorBase =
            ((chunk.bIdx * tiling_->HV + hv) * tiling_->T + chunk.tokenStart) * V_SIZE_128;
        AscendC::GlobalTensor<DT> gmDo;
        AscendC::GlobalTensor<DT> gmVNew;
        gmDo.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(do_) + vectorBase);
        gmVNew.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(vNew_) + vectorBase);
        auto tensorGmDo = tla::MakeTensor(gmDo, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmVNew = tla::MakeTensor(
            gmVNew, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto blockGmDo = tla::GetTile(
            tensorGmDo, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, V_SIZE_128));
        auto blockGmVNew = tla::GetTile(
            tensorGmVNew, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, V_SIZE_128));
        auto tensorDs0Cv = tla::MakeTensor(
            ds0Cv[localHv], UB_DAU_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockDs0Cv = tla::GetTile(
            tensorDs0Cv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        auto tensorL1Do = tla::MakeTensor(
            doL1[doSlot], L1_DU_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1VNewA = tla::MakeTensor(
            vNewL1[headOffset], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1A>(
                CHUNK_SIZE_64, V_SIZE_128), Catlass::Arch::PositionL1{});
        auto tensorL1VNewB = tla::MakeTensor(
            vNewL1[headOffset], L1_VB_LAYOUT, Catlass::Arch::PositionL1{});

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(doEvent);
        typename TileCopyDau::template CopyGmToL1A<decltype(blockGmDo)>{}(tensorL1Do, blockGmDo);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(vNewEvent);
        typename TileCopyDau::template CopyGmToL1A<decltype(blockGmVNew)>{}(
            tensorL1VNewA, blockGmVNew);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(doEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(vNewEvent);

        // GEMM 1: ds0 = do @ v_new.T。
        const uint32_t dsL0ASlot = l0ASlot_;
        const uint32_t dsL0BSlot = l0BSlot_;
        const uint32_t dsL0CSlot = l0CSlot_;
        const uint32_t dsL0AEvent = 2 * dsL0ASlot;
        const uint32_t dsL0BEvent = 2 * dsL0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        auto tensorDsL0A = tla::MakeTensor(
            l0A[dsL0ASlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(mActual, V_SIZE_128),
            Catlass::Arch::PositionL0A{});
        auto tensorDsL0B = tla::MakeTensor(
            l0B[dsL0BSlot],
            tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(V_SIZE_128, chunkLen),
            Catlass::Arch::PositionL0B{});
        auto tileDoL1 = tla::GetTile(
            tensorL1Do, tla::MakeCoord(0, 0), tla::MakeShape(mActual, V_SIZE_128));
        auto tileVNewL1 = tla::GetTile(
            tensorL1VNewB, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, chunkLen));
        auto tensorDsL0C = tla::MakeTensor(
            l0C[dsL0CSlot], tla::MakeLayoutL0C(mActual, chunkLen), Catlass::Arch::PositionL0C{});
        auto tileDsL0C = tla::GetTile(
            tensorDsL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(doEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(vNewEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dsL0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dsL0BEvent);
        CopyDauL1ToL0A{}(tensorDsL0A, tileDoL1);
        CopyDauL1ToL0B{}(tensorDsL0B, tileVNewL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(vNewEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dsL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dsL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dsL0CSlot);
        tileMmad_(tileDsL0C, tensorDsL0A, tensorDsL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dsL0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dsL0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dsL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dsL0CSlot);
        // S11 结果只供 S12 Vector 使用，Fixpipe 直接定向写入 owner AIV
        // 的固定 UB slot，避免经 GM workspace 往返。
        typename TileCopyDauToUb::template CopyL0CToDst<decltype(blockDs0Cv)>{}(
            blockDs0Cv, tensorDsL0C, static_cast<uint8_t>(ownerSubBlock), 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dsL0CSlot);

        // do 在本 Stage 的唯一 GEMM 中完成末次消费，允许下一 HEAD 复用。
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(doEvent);
    }

    __aicore__ inline void ProcessStage13Head(
        AscendC::LocalTensor<DT> (&qL1)[BANK_COUNT_2],
        AscendC::LocalTensor<DT> (&dhL1)[BANK_COUNT_2],
        AscendC::LocalTensor<DT> (&dsL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&kL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&hL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&doGL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&vDecayL1)[L1_RESIDENT_HEAD_COUNT_4],
        AscendC::LocalTensor<DT> (&l0A)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<DT> (&l0B)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<ElementAccumulator> (&l0C)[L0_BUFFER_COUNT_2],
        AscendC::LocalTensor<DT> (&dk0Cv)[BANK_COUNT_2],
        AscendC::LocalTensor<DT> (&dqIntraCv)[BANK_COUNT_2],
        AscendC::LocalTensor<DT> (&dkIntraCv)[BANK_COUNT_2],
        const ChunkInfo &chunk, int64_t hv, int64_t headOffset)
    {
        const uint32_t chunkLen = static_cast<uint32_t>(chunk.chunkLen);
        const uint32_t mActual = chunkLen == 1 ? 16 : chunkLen;
        const uint32_t ownerSubBlock = static_cast<uint32_t>(headOffset % AIV_COUNT_2);
        const uint32_t localHv = static_cast<uint32_t>(headOffset / AIV_COUNT_2);
        const int64_t hk = hv / tiling_->headRatio;
        const uint32_t qSlot = static_cast<uint32_t>(headOffset / tiling_->headRatio) & 1U;
        const uint32_t dhSlot = static_cast<uint32_t>(headOffset) & 1U;
        const uint32_t dsEvent = static_cast<uint32_t>(headOffset);
        const uint32_t qEvent = L1_RESIDENT_HEAD_COUNT_4 + qSlot;
        const uint32_t doGEvent = L1_RESIDENT_HEAD_COUNT_4 +
            (static_cast<uint32_t>(headOffset) & 1U);
        const uint32_t dhEvent = L1_RESIDENT_HEAD_COUNT_4 + BANK_COUNT_2 + dhSlot;
        const int64_t qBase =
            ((chunk.bIdx * tiling_->HK + hk) * tiling_->T + chunk.tokenStart) * K_SIZE_128;
        const int64_t workspaceBase = WorkspaceHeadOffset(headOffset);
        const int64_t stateBase =
            ((chunk.bIdx * tiling_->HV + hv) * stateChunkNum_ + chunk.stateChunkIdx) *
            K_SIZE_128 * V_SIZE_128;

        AscendC::GlobalTensor<DT> gmQ;
        AscendC::GlobalTensor<DT> gmDh;
        AscendC::GlobalTensor<DT> gmDs;
        AscendC::GlobalTensor<DT> gmDoG;
        AscendC::GlobalTensor<DT> gmVDecay;
        gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(q_) + qBase);
        gmDh.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dh_) + stateBase);
        gmDs.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(dsIn_) + workspaceBase);
        gmDoG.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(doGIn_) + workspaceBase);
        gmVDecay.SetGlobalBuffer(reinterpret_cast<__gm__ DT *>(vDecayIn_) + workspaceBase);
        auto tensorGmQ = tla::MakeTensor(gmQ, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmDs = tla::MakeTensor(gmDs, GM_A_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmDoG = tla::MakeTensor(gmDoG, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto tensorGmVDecay = tla::MakeTensor(gmVDecay, GM_DU_LAYOUT, Catlass::Arch::PositionGM{});
        auto blockGmQ = tla::GetTile(
            tensorGmQ, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));
        auto blockGmDs = tla::GetTile(
            tensorGmDs, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, chunkLen));
        auto blockGmDoG = tla::GetTile(
            tensorGmDoG, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, V_SIZE_128));
        auto blockGmVDecay = tla::GetTile(
            tensorGmVDecay, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, V_SIZE_128));
        auto tensorL1Q = tla::MakeTensor(
            qL1[qSlot], tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1B>(
                CHUNK_SIZE_64, K_SIZE_128), Catlass::Arch::PositionL1{});
        auto tensorL1DhTranspose = tla::MakeTensor(
            dhL1[dhSlot], L1_STATE_TRANSPOSE_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1DhVFirst = tla::MakeTensor(
            dhL1[dhSlot], L1_STATE_V_FIRST_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1Ds = tla::MakeTensor(
            dsL1[headOffset], tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorL1DsT = tla::MakeTensor(
            dsL1[headOffset], tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL1A>(
                CHUNK_SIZE_64, CHUNK_SIZE_64), Catlass::Arch::PositionL1{});
        auto tensorL1DoG = tla::MakeTensor(
            doGL1[headOffset], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1A>(
                CHUNK_SIZE_64, V_SIZE_128), Catlass::Arch::PositionL1{});
        auto tensorL1VDecay = tla::MakeTensor(
            vDecayL1[headOffset], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL1A>(
                CHUNK_SIZE_64, V_SIZE_128), Catlass::Arch::PositionL1{});

        // dh 是原始输入，与 Stage 12 无依赖，先搬入独立的双缓冲 L1。
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dhEvent);
        if (tiling_->stateVFirst != 0) {
            auto tensorGmDh = tla::MakeTensor(
                gmDh, GM_STATE_V_FIRST_LAYOUT, Catlass::Arch::PositionGM{});
            auto blockGmDh = tla::GetTile(
                tensorGmDh, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            typename TileCopyStateVFirst::template CopyGmToL1B<decltype(blockGmDh)>{}(
                tensorL1DhVFirst, blockGmDh);
        } else {
            auto tensorGmDh = tla::MakeTensor(
                gmDh, GM_STATE_TRANSPOSE_LAYOUT, Catlass::Arch::PositionGM{});
            auto blockGmDh = tla::GetTile(
                tensorGmDh, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            typename TileCopyDau::template CopyGmToL1B<decltype(blockGmDh)>{}(
                tensorL1DhTranspose, blockGmDh);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dhEvent);

        // ds/do_g 由 Stage 12 产生，只在读取它们前消费跨核 token。
        // ds 每 HEAD 搬一次；q 在同一 GVA 组的首个 HV 搬入并保持到组末。
        Catlass::Arch::CrossCoreWaitFlag(vecToCubeFlag_);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dsEvent);
        typename TileCopyDw0::template CopyGmToL1A<decltype(blockGmDs)>{}(tensorL1Ds, blockGmDs);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dsEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(doGEvent);
        typename TileCopyDau::template CopyGmToL1A<decltype(blockGmDoG)>{}(
            tensorL1DoG, blockGmDoG);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(doGEvent);

        auto tensorL1K = tla::MakeTensor(
            kL1[headOffset], tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL1B>(
                CHUNK_SIZE_64, K_SIZE_128), Catlass::Arch::PositionL1{});
        auto tileDsL1 = tla::GetTile(
            tensorL1Ds, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        auto tileKL1 = tla::GetTile(
            tensorL1K, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));
        auto tileDoGL1 = tla::GetTile(
            tensorL1DoG, tla::MakeCoord(0, 0), tla::MakeShape(mActual, V_SIZE_128));
        auto tensorL1HTranspose = tla::MakeTensor(
            hL1[headOffset], L1_STATE_TRANSPOSE_LAYOUT, Catlass::Arch::PositionL1{});
        auto tensorL1HVFirst = tla::MakeTensor(
            hL1[headOffset], L1_STATE_V_FIRST_LAYOUT, Catlass::Arch::PositionL1{});

        // GEMM 1: dq_hv = do_g @ h.T + ds @ k。两次 MMAD 使用同一块
        // L0C，第一次初始化、第二次累加，全部完成后只执行一次 Fixpipe。
        const uint32_t dqL0ASlot = l0ASlot_;
        const uint32_t dqL0BSlot = l0BSlot_;
        const uint32_t dqL0CSlot = l0CSlot_;
        const uint32_t dqL0AEvent = 2 * dqL0ASlot;
        const uint32_t dqL0BEvent = 2 * dqL0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        auto tensorDqL0A = tla::MakeTensor(
            l0A[dqL0ASlot], tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL0A>(
                mActual, chunkLen), Catlass::Arch::PositionL0A{});
        auto tensorDqL0B = tla::MakeTensor(
            l0B[dqL0BSlot], tla::MakeLayout<DT, typename TileCopyDw0::LayoutTagL0B>(
                chunkLen, K_SIZE_128), Catlass::Arch::PositionL0B{});
        auto tensorDqL0C = tla::MakeTensor(
            l0C[dqL0CSlot], tla::MakeLayoutL0C(mActual, K_SIZE_128), Catlass::Arch::PositionL0C{});
        auto tileDqL0C = tla::GetTile(
            tensorDqL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, K_SIZE_128));
        auto tensorDqCv = tla::MakeTensor(
            dqIntraCv[localHv], UB_DVB_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockDqCv = tla::GetTile(
            tensorDqCv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(doGEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dqL0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dqL0BEvent);
        auto tensorDoGL0A = tla::MakeTensor(
            l0A[dqL0ASlot], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(
                mActual, V_SIZE_128), Catlass::Arch::PositionL0A{});
        auto tensorHL0BTranspose = tla::MakeTensor(
            l0B[dqL0BSlot], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(
                V_SIZE_128, K_SIZE_128), Catlass::Arch::PositionL0B{});
        auto tensorHL0BVFirst = tla::MakeTensor(
            l0B[dqL0BSlot], tla::MakeLayout<DT, typename TileCopyStateVFirst::LayoutTagL0B>(
                V_SIZE_128, K_SIZE_128), Catlass::Arch::PositionL0B{});
        CopyDauL1ToL0A{}(tensorDoGL0A, tileDoGL1);
        if (tiling_->stateVFirst != 0) {
            auto tileHL1 = tla::GetTile(
                tensorL1HVFirst, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            CopyStateVFirstL1ToL0B{}(tensorHL0BVFirst, tileHL1);
        } else {
            auto tileHL1 = tla::GetTile(
                tensorL1HTranspose, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            CopyDauL1ToL0B{}(tensorHL0BTranspose, tileHL1);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(doGEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dqL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dqL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dqL0CSlot);
        if (tiling_->stateVFirst != 0) {
            tileMmad_(tileDqL0C, tensorDoGL0A, tensorHL0BVFirst, true, 0);
        } else {
            tileMmad_(tileDqL0C, tensorDoGL0A, tensorHL0BTranspose, true, 0);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dqL0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dqL0BEvent);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(dsEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dqL0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dqL0BEvent);
        CopyDw0L1ToL0A{}(tensorDqL0A, tileDsL1);
        CopyDw0L1ToL0B{}(tensorDqL0B, tileKL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dqL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dqL0CSlot);
        tileMmad_(tileDqL0C, tensorDqL0A, tensorDqL0B, false, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dqL0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dqL0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dqL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dqL0CSlot);
        typename TileCopyDvbToUb::template CopyL0CToDst<decltype(blockDqCv)>{}(
            blockDqCv, tensorDqL0C, static_cast<uint8_t>(ownerSubBlock), 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dqL0CSlot);

        // do_g resident 已进入 L0，复用同一事件搬入当前 HK 的 q。
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(qEvent);
        typename TileCopyDw0::template CopyGmToL1B<decltype(blockGmQ)>{}(tensorL1Q, blockGmQ);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(qEvent);

        // GEMM 2: dk_intra = ds.T @ q。同一份 ds L1 按列主序 A 视图读取。
        const uint32_t dkL0ASlot = l0ASlot_;
        const uint32_t dkL0BSlot = l0BSlot_;
        const uint32_t dkL0CSlot = l0CSlot_;
        const uint32_t dkL0AEvent = 2 * dkL0ASlot;
        const uint32_t dkL0BEvent = 2 * dkL0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        auto tensorDkL0A = tla::MakeTensor(
            l0A[dkL0ASlot], tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL0A>(
                mActual, chunkLen), Catlass::Arch::PositionL0A{});
        auto tensorDkL0B = tla::MakeTensor(
            l0B[dkL0BSlot], tla::MakeLayout<DT, typename TileCopyStage9DkbT::LayoutTagL0B>(
                chunkLen, K_SIZE_128), Catlass::Arch::PositionL0B{});
        auto tensorDkL0C = tla::MakeTensor(
            l0C[dkL0CSlot], tla::MakeLayoutL0C(mActual, K_SIZE_128), Catlass::Arch::PositionL0C{});
        auto tileDkL0C = tla::GetTile(
            tensorDkL0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, K_SIZE_128));
        auto tileQL1 = tla::GetTile(
            tensorL1Q, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));
        auto tileDsTL1 = tla::GetTile(
            tensorL1DsT, tla::MakeCoord(0, 0), tla::MakeShape(mActual, chunkLen));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(qEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkL0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dkL0BEvent);
        CopyStage9DkbTL1ToL0A{}(tensorDkL0A, tileDsTL1);
        CopyStage9DkbTL1ToL0B{}(tensorDkL0B, tileQL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dsEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(qEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dkL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dkL0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dkL0CSlot);
        tileMmad_(tileDkL0C, tensorDkL0A, tensorDkL0B, true, 0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkL0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dkL0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dkL0CSlot);
        auto tensorDkCv = tla::MakeTensor(
            dkIntraCv[localHv], UB_DVB_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockDkCv = tla::GetTile(
            tensorDkCv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));
        // dk_intra 暂留在独立 L0C。其目标 UB[144,176) 仍是
        // Stage 12 v_decay 的 MTE3 源，必须等 v_decay 搬入 L1 后再覆盖。

        // GEMM 3: dk_base = v_decay @ dh.T。ds 已完成末次 MTE1 消费，
        // 复用其事件将 Stage 12 写入 GM 的 v_decay 搬到固定 L1 resident。
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(dsEvent);
        typename TileCopyDau::template CopyGmToL1A<decltype(blockGmVDecay)>{}(
            tensorL1VDecay, blockGmVDecay);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(dsEvent);
        const uint32_t dk0L0ASlot = l0ASlot_;
        const uint32_t dk0L0BSlot = l0BSlot_;
        const uint32_t dk0L0CSlot = l0CSlot_;
        const uint32_t dk0L0AEvent = 2 * dk0L0ASlot;
        const uint32_t dk0L0BEvent = 2 * dk0L0BSlot + 1;
        l0ASlot_ ^= 1U;
        l0BSlot_ ^= 1U;
        l0CSlot_ ^= 1U;
        auto tileVDecayL1 = tla::GetTile(
            tensorL1VDecay, tla::MakeCoord(0, 0), tla::MakeShape(mActual, V_SIZE_128));
        auto tensorDk0L0A = tla::MakeTensor(
            l0A[dk0L0ASlot], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0A>(
                mActual, V_SIZE_128), Catlass::Arch::PositionL0A{});
        auto tensorDk0L0BTranspose = tla::MakeTensor(
            l0B[dk0L0BSlot], tla::MakeLayout<DT, typename TileCopyDau::LayoutTagL0B>(
                V_SIZE_128, K_SIZE_128), Catlass::Arch::PositionL0B{});
        auto tensorDk0L0BVFirst = tla::MakeTensor(
            l0B[dk0L0BSlot], tla::MakeLayout<DT, typename TileCopyStateVFirst::LayoutTagL0B>(
                V_SIZE_128, K_SIZE_128), Catlass::Arch::PositionL0B{});
        auto tensorDk0L0C = tla::MakeTensor(
            l0C[dk0L0CSlot], tla::MakeLayoutL0C(mActual, K_SIZE_128), Catlass::Arch::PositionL0C{});
        auto tileDk0L0C = tla::GetTile(
            tensorDk0L0C, tla::MakeCoord(0, 0), tla::MakeShape(mActual, K_SIZE_128));
        auto tensorDk0Cv = tla::MakeTensor(
            dk0Cv[localHv], UB_DVB_LAYOUT, Catlass::Arch::PositionUB{});
        auto blockDk0Cv = tla::GetTile(
            tensorDk0Cv, tla::MakeCoord(0, 0), tla::MakeShape(chunkLen, K_SIZE_128));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(dsEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(dhEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dk0L0AEvent);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(dk0L0BEvent);
        CopyDauL1ToL0A{}(tensorDk0L0A, tileVDecayL1);
        if (tiling_->stateVFirst != 0) {
            auto tileDhL1 = tla::GetTile(
                tensorL1DhVFirst, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            CopyStateVFirstL1ToL0B{}(tensorDk0L0BVFirst, tileDhL1);
        } else {
            auto tileDhL1 = tla::GetTile(
                tensorL1DhTranspose, tla::MakeCoord(0, 0), tla::MakeShape(V_SIZE_128, K_SIZE_128));
            CopyDauL1ToL0B{}(tensorDk0L0BTranspose, tileDhL1);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dsEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(dhEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(dk0L0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(dk0L0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(dk0L0CSlot);
        if (tiling_->stateVFirst != 0) {
            tileMmad_(tileDk0L0C, tensorDk0L0A, tensorDk0L0BVFirst, true, 0);
        } else {
            tileMmad_(tileDk0L0C, tensorDk0L0A, tensorDk0L0BTranspose, true, 0);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dk0L0AEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(dk0L0BEvent);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(dk0L0CSlot);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dk0L0CSlot);
        typename TileCopyDvbToUb::template CopyL0CToDst<decltype(blockDk0Cv)>{}(
            blockDk0Cv, tensorDk0L0C, static_cast<uint8_t>(ownerSubBlock), 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dk0L0CSlot);

        // v_decay 已完成 GM->L1，原 UB 源地址可以覆盖。此处执行
        // dk_intra 的唯一 Fixpipe，不增加第二份结果或 UB 位置移动。
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(dkL0CSlot);
        typename TileCopyDvbToUb::template CopyL0CToDst<decltype(blockDkCv)>{}(
            blockDkCv, tensorDkL0C, static_cast<uint8_t>(ownerSubBlock), 0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(dkL0CSlot);
    }

    __aicore__ inline int64_t WorkspaceHeadOffset(int64_t headOffset) const
    {
        return GetWorkspaceHeadOffset(coreIdx_, workspaceGroupRound_, headOffset);
    }

    GM_ADDR q_ = nullptr;
    GM_ADDR k_ = nullptr;
    GM_ADDR vNew_ = nullptr;
    GM_ADDR do_ = nullptr;
    GM_ADDR du_ = nullptr;
    GM_ADDR h_ = nullptr;
    GM_ADDR dh_ = nullptr;
    GM_ADDR a_ = nullptr;
    GM_ADDR kbg_ = nullptr;
    GM_ADDR vb_ = nullptr;
    GM_ADDR dA0_ = nullptr;
    GM_ADDR ds0Out_ = nullptr;
    GM_ADDR dkbOut_ = nullptr;
    GM_ADDR dkbTOut_ = nullptr;
    GM_ADDR dsIn_ = nullptr;
    GM_ADDR doGIn_ = nullptr;
    GM_ADDR vDecayIn_ = nullptr;
    GM_ADDR workspace_ = nullptr;
    GM_ADDR cuSeqlens_ = nullptr;
    GM_ADDR chunkIndices_ = nullptr;
    const ChunkGatedDeltaRuleBwdFinalizeTilingData *tiling_ = nullptr;
    int64_t coreIdx_ = 0;
    int64_t coreNum_ = 1;
    int64_t stateChunkNum_ = 0;
    int64_t workspaceGroupRound_ = 0;
    uint32_t l0ASlot_ = 0;
    uint32_t l0BSlot_ = 0;
    uint32_t l0CSlot_ = 0;
    // 与两个 AIV 共用两条普通事件链：V->C 只使用一个 id，C->V 只使用一个 id。
    // 相反方向的逐 HEAD 依赖形成天然背压，使同一方向的未消费计数不会超过硬件上限。
    Catlass::Arch::CrossCoreFlag vecToCubeFlag_{VEC_TO_CUBE_READY_FLAG};
    Catlass::Arch::CrossCoreFlag cubeToVecFlag_{CUBE_TO_VEC_READY_FLAG};
    TileMmad tileMmad_;
};

} // namespace GDN

#endif
