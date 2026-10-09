/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file pre_process_fwd_kernel_merged_vector.h
 * \brief arch22（A2/A3，910B / 910_93） 的 vector 实现（与另一 arch 同名类，由 _kernel.h 二选一）。
 */

/*!
 * \file pre_process_fwd_kernel_merged_vec.h
 * \brief pre_process_fwd_kernel_merged：AIV（vector）实现
 *
 * 本文件由 kernel 主文件按"机械搬运"拆出（代码与拆分前逐字符相同）；
 * Stage/布局/同步协议的完整说明见 pre_process_fwd_kernel_merged_common.h 顶部注释与 docs/design.md。
 */

#ifndef PREF_PROCESS_FWD_KERNEL_MERGED_VEC_H
#define PREF_PROCESS_FWD_KERNEL_MERGED_VEC_H

#include "../pre_process_fwd_kernel_merged_common.h"

namespace GDN {class PpFwdVector {
public:
    __aicore__ inline PpFwdVector(const PpFwdCtx &ctx) : ctx_(ctx) {}

    __aicore__ inline void Run()
    {
        const auto *t = ctx_.tiling;
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx_.k));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx_.w));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            (ctx_.v == nullptr) ? ctx_.u : ctx_.v));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx_.g));
        gkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx_.gk));
        cuGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ctx_.cu));
        hmGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ctx_.hm));

        // AIV 子核在 MIX 下拿到的是 AIV 编号：换算回 AIC 编号（两个子核做同一份工作，
        // 但**必须切分工作**，否则两个子核互相覆盖；flag 计数天然平衡）。
        const int64_t coreIdx = static_cast<int64_t>(GetBlockIdx()) / GetSubBlockNum();
        subIdx_ = static_cast<int32_t>(GetSubBlockIdx());
        subNum_ = static_cast<int32_t>(GetSubBlockNum());
        if (subNum_ <= 0) {
            subNum_ = 1;
        }
        // P5：列块切分（运行时可配，见 host tiling 的 colSplit）。cb_ =
        // 本工作项负责的列宽，
        // colBase_ = 该列块在整条链里的起始列（h 的 V 列 / m 的 K 列同一个 colBase_）。
        splitNum_ = (t->colSplit > 0) ? static_cast<int32_t>(t->colSplit) : 1;
        cb_ = static_cast<int32_t>(CV_V) / splitNum_;
        colBase_ = 0;
        __gm__ uint8_t *ws = reinterpret_cast<__gm__ uint8_t *>(ctx_.ws) + coreIdx * PPFM_CORE_WS_BYTES;

        hF32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_H_F32), CV_K * CV_V);
        mF32_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_M_F32), CV_K * CV_K);
        hBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_H_BF), CV_K * CV_V);
        mBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_M_BF), CV_K * CV_K);
        wBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_W_BF), CV_BT * CV_K);
        kBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_K_BF), CV_BT * CV_K);
        lBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_L_BF), CV_BT * CV_K);
        kBf1_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_K_BF_1), CV_BT * CV_K);  // 
        lBf1_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_L_BF_1), CV_BT * CV_K);  // 
        vBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_V_BF), CV_BT * CV_V);
        vTmpF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_VTMP_F32), CV_BT * CV_V);
        vNewBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_VNEW_BF), CV_BT * CV_V);
        dHF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_DH_F32), CV_K * CV_V);
        dHF1_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_DH_F32_1), CV_K * CV_V);
        t1F_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_T1_F32), CV_BT * CV_K);
        t1Bf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_T1_BF), CV_BT * CV_K);
        t2F_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_T2_F32), CV_K * CV_K);
        t2F1_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_T2_F32_1), CV_K * CV_K);
        gateF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_GATE), CV_BT + CV_K);
#if PPFM_RD_PROBE
        // 诊断暂存：WS_GATE 区共 4096B，gateF_ 只用前 192 个 float，后面 256 个 float 用来放
        // AIV 读回的 vTmp 行（仅诊断构建使用）。
        // 6 行 × 128 float：row0=vTmp 读回、row2=h 状态 prologue 回读、row3=h 状态读回、
        // row4=dH 读回。
        probeG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_GATE + 1024), 768);
#endif
#if PPFM_DIAG
        diagG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_DIAG), 16);
#endif

        // PPFM_VEC_UB_BYTES_TOTAL = PPFM_VEC_UB_BYTES + C10 的 factor 暂存（默认 0 ⇒ 值不变）
        pipe_.InitBuffer(ubBuf_, PPFM_VEC_UB_BYTES_TOTAL);
        // per-subcore 视图：两个 AIV 子核共享同一块 UB，凡"两个子核都会写"的 scratch
        // 都按 subIdx_ 偏移一份，段分区缓冲（kBlk/wBlk/vBlk/scr）保持不偏移。
        // PPFM_UB_SHARE=1 时不再偏移（每个 AIV 子核有独立 bank，见布局注释）。
        const int32_t sbf = PPFM_UB_SHARE ? 0 : subIdx_ * CV_K;
        const int32_t sf = PPFM_UB_SHARE ? 0 : subIdx_ * CV_K;
        const int32_t sdg = PPFM_UB_SHARE ? 0 : subIdx_ * CV_BT;
        const int32_t sdecay = PPFM_UB_SHARE ? 0 : subIdx_ * CV_K;
        const int32_t sexp = PPFM_UB_SHARE ? 0 : subIdx_ * CV_LANES;
        const int32_t sstate = PPFM_UB_SHARE ? 0 : subIdx_ * PPFM_RB * CV_V;
        const int32_t sext = PPFM_UB_SHARE ? 0 : subIdx_ * 2 * PPFM_SEG * CV_V;
        row0Bf_ = ubBuf_.Get<bfloat16_t>()[UB_ROW0_BF_ELEM + sbf];
        row1Bf_ = ubBuf_.Get<bfloat16_t>()[UB_ROW1_BF_ELEM + sbf];
        row2Bf_ = ubBuf_.Get<bfloat16_t>()[UB_ROW2_BF_ELEM + sbf];
        row0F_ = ubBuf_.Get<float>()[UB_ROW0_F_ELEM + sf];
        row1F_ = ubBuf_.Get<float>()[UB_ROW1_F_ELEM + sf];
        row2F_ = ubBuf_.Get<float>()[UB_ROW2_F_ELEM + sf];
        dgF_ = ubBuf_.Get<float>()[UB_DG_ELEM + sdg];
        decayF_ = ubBuf_.Get<float>()[UB_DECAY_ELEM + sdecay];
        expScratch_ = ubBuf_.Get<float>()[UB_EXP_ELEM + sexp];
        decayPrevF_ = ubBuf_.Get<float>()[UB_DECAY_PREV_ELEM + sdecay];
        gBlkF_ = ubBuf_.Get<float>()[UB_GBLK_ELEM + sdg];
        stateBlkF_ = ubBuf_.Get<float>()[UB_STATE_F_ELEM + sstate];
        extBlkF_ = ubBuf_.Get<float>()[UB_EXT_F_ELEM + sext];
#if PPFM_FAC8_ON
        // C10：factor 广播暂存（追加在 UB 末尾，见 _common.h 的 UB_FAC8）
        fac8_ = ubBuf_.Get<float>()[UB_FAC8_ELEM];
#endif
#if PPFM_H_UB
        // h 的常驻半步。共享基址（不偏移）：每个 AIV 子核在自己的 bank 里用同一偏移，
        // 各自存自己那 64 行 —— 正是 里 SPLIT_M 的同一套 bank 语义。
        hUb_ = ubBuf_.Get<float>()[UB_H_UB_ELEM];
#if PPFM_M_UB
        // m 的常驻半步（同上）
        mUb_ = ubBuf_.Get<float>()[UB_M_UB_ELEM];
#endif
#endif
        // fixpipe SPLIT_M 把两半写到**同一偏移**（各自 bank），这里用共享基址视图
        vTmpUb_ = ubBuf_.Get<float>()[UB_EXT_F_ELEM];
        stateBlkBf_ = ubBuf_.Get<bfloat16_t>()[UB_STATE_BF_ELEM + sstate];
        kBlkBf_ = ubBuf_.Get<bfloat16_t>()[UB_KBLK_BF_ELEM];
        wBlkBf_ = ubBuf_.Get<bfloat16_t>()[UB_WBLK_BF_ELEM];
        vBlkBf_ = ubBuf_.Get<bfloat16_t>()[UB_VBLK_BF_ELEM];
        scrF_ = ubBuf_.Get<float>()[UB_SCR_F_ELEM];
        scrBf_ = ubBuf_.Get<bfloat16_t>()[UB_SCR_BF_ELEM];
#if PPFM_DIAG
        dbgF_ = ubBuf_.Get<float>()[UB_DBG / 4 + subIdx_ * 16];
#endif

        // 任务空间 = 整宽链（前 hybridBase 条）+ 余数链的列片；hybridS<=1 时退化为原逻辑。
        const int64_t nChain = t->nSeq * t->Hv;
        const int64_t taskNum = (t->hybridS > 1)
            ? (t->hybridBase + (nChain - t->hybridBase) * t->hybridS)
            : (nChain * static_cast<int64_t>(splitNum_));
        for (int64_t task = coreIdx; task < taskNum; task += static_cast<int64_t>(t->usedAicNum)) {

            // P5/工作项 = (n, hv, 列块)。s 变化最快 ⇒
            // 同一条链的两个列块尽量落在不同核上。
            const PpFwdChunkInfo pos_ = GetChunkInfo(t, splitNum_, task);
            const int64_t hv = pos_.hv;
            const int64_t n = pos_.n;
            cb_ = pos_.cb;          // GetChunkInfo 已算好列窗（§4.4 ⑤）
            colBase_ = pos_.colBase;
            const int64_t bos = cuGm_.GetValue(n);
            const int64_t eos = cuGm_.GetValue(n + 1);
            ProcessChain(n, hv, bos, eos - bos);
        }
    }

private:
    // ---- 调试用：把一段 GM 内容用向量 Cast 转 fp32 后取前 2 个值 ----
    __aicore__ inline void ProbeBf16(const GlobalTensor<bfloat16_t> &src, int32_t idx)
    {
        DataCopy(row2Bf_, src, CV_K);
        PipeBarrier<PIPE_ALL>();
        Cast(row1F_, row2Bf_, RoundMode::CAST_NONE, CV_K);
        PipeBarrier<PIPE_ALL>();
        row0F_.SetValue(idx, row1F_.GetValue(0));
        row0F_.SetValue(idx + 1, row1F_.GetValue(1));
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProbeF32(const GlobalTensor<float> &src, int32_t idx, int32_t lane)
    {
        DataCopy(row1F_, src, CV_K);
        PipeBarrier<PIPE_ALL>();
        row0F_.SetValue(idx, row1F_.GetValue(lane));
        row0F_.SetValue(idx + 1, row1F_.GetValue(lane + 1));
        PipeBarrier<PIPE_ALL>();
    }

    // 把 src[0] 用 DataCopy 回读到 UB 后存进 row1F_ 指定 lane（供 epilogue 带出）
    __aicore__ inline void ProbeState(const GlobalTensor<float> &src, int32_t dstLane)
    {
        DataCopy(row0F_, src, CV_K);
        PipeBarrier<PIPE_ALL>();
        row1F_.SetValue(dstLane, row0F_.GetValue(0));
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline float Exp2Scalar(float x)
    {
        expScratch_.SetValue(0, x * 0.6931471805599453f);
        PipeBarrier<PIPE_ALL>();
        Exp(expScratch_, expScratch_, CV_LANES);
        PipeBarrier<PIPE_ALL>();
        return expScratch_.GetValue(0);
    }

    // glast 由调用方算好传入（原本这里又做了一次 GM 标量读）
    __aicore__ inline void SetDecay(int64_t hv, int64_t tGlobal, float glastIn)
    {
        if (ctx_.tiling->gateMode == PPFM_GATE_USE_G) {
            const float glast = PPFM_GLAST_UB ? glastIn
                : gGm_.GetValue(hv * ctx_.tiling->T + tGlobal);
            const float dc = Exp2Scalar(glast);
            // decayF_ 是全 128 项同值 ⇒ 一次 Duplicate 取代 128 次 SetValue
            Duplicate(decayF_, dc, CV_K);
        } else {
#if PPFM_KDA_DECAY_VEC
            // KDA（USE_GK）：decay[k] = 2^(gk_last[k])，整块向量化。

            // 原实现是 128 次 Exp2Scalar：每次都"标量写 UB → 全栅栏 → Exp → 全栅栏 →
            // 标量读"，

            // 实测是 AIV scalar 流水的最大单一来源。这里用 row0F_
            // 做暂存（本函数里它不承载数据），
            // 逐元素仍是 exp(x·ln2)，与逐点版本逐位等价（L1 位级门禁验证）。
            DataCopy(row0F_, gkGm_[(hv * ctx_.tiling->T + tGlobal) * CV_K], CV_K);
            AIV_SET_MTE2_V();
            AIV_WAIT_MTE2_V();
            Muls(row0F_, row0F_, 0.6931471805599453f, CV_K);
            PipeBarrier<PIPE_V>();
            Exp(decayF_, row0F_, CV_K);
#else
            for (int32_t k = 0; k < CV_K; ++k) {
                const float gk = gkGm_.GetValue((hv * ctx_.tiling->T + tGlobal) * CV_K + k);
                decayF_.SetValue(k, Exp2Scalar(gk));
            }
#endif
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessChain(int64_t n, int64_t hv, int64_t bos, int64_t len)
    {
        curN_ = n;
        const auto *t = ctx_.tiling;
        // ---- prologue：h = 0（910B/910_93）----
        // 实测（RD_PROBE 探针）：A2 上用"一次 Cast 出 bf16 行 + 循环内 128 次 MTE3 复用
        // 同一 UB 行"的写法，会把 bf16(h) 落到 GM 时写成 ~1e-3 量级的脏数据（同一循环里的
        // fp32 版本却是严格 0）。AIC 的 mm1 于是把非零的 bf16(h) 当输入，vTmp=W@h≠0，
        // 整条 h 链偏 1.5e-2；m 链因为用逐行写（见下）反而是对的。
        // 这里改成与 m 初值同款的逐行写法：每行重新 Cast，行间用 PIPE_ALL 隔离。
#if PPFM_H_UB
        // h 初值落在常驻 UB；GM 上只留 bf16(h)（逐行写法保持不变，见上面的 A2 实测）
        Duplicate(hUb_, 0.0f, H_UB_ROWS * cb_);
        PipeBarrier<PIPE_ALL>();
        for (int32_t r = subIdx_ * H_UB_ROWS; r < (subIdx_ + 1) * H_UB_ROWS; ++r) {
            Duplicate(row0F_, 0.0f, cb_);
            PipeBarrier<PIPE_ALL>();
            Cast(row0Bf_, row0F_, RoundMode::CAST_RINT, cb_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(hBf_[r * cb_], row0Bf_, cb_);
            PipeBarrier<PIPE_ALL>();
        }
#else
#if PPFM_A2_BLOCKWISE
        {
            const int32_t rowsPerSub_ = CV_K / PPFM_SUB;
            // extBlkF_ 的 fp32 容量由 UB_EXT_F_ELEMS 给出（A2 默认 = 2*PPFM_SEG*CV_V）
            // ⇒ 单次最多放 rowsPerCopy_ 行（cb_=128 时 32 行、cb_=64 时 64 行）。
            // 首版写死 64 行整块，cb_=128 时越界踩到相邻 UB（hy30/hy40 实测 m 半边出错），
            // 这里按 **UB 实际容量** 推导，避免以后再改布局时重复同一个错。
            const int32_t rowsPerCopy_ = UB_EXT_F_ELEMS / cb_;
            for (int32_t i0 = 0; i0 < rowsPerSub_; i0 += rowsPerCopy_) {
                const int32_t nr_ = (rowsPerSub_ - i0 < rowsPerCopy_)
                                        ? (rowsPerSub_ - i0) : rowsPerCopy_;
                Duplicate(extBlkF_, 0.0f, nr_ * cb_);
                PipeBarrier<PIPE_ALL>();
                Cast(stateBlkBf_, extBlkF_, RoundMode::CAST_RINT, nr_ * cb_);
                AIV_SET_V_MTE3();
                AIV_WAIT_V_MTE3();
                DataCopyParams hpF_{static_cast<uint16_t>(nr_),
                                   static_cast<uint16_t>((cb_ * 4) / 32), 0, 0};
                DataCopyParams hpB_{static_cast<uint16_t>(nr_),
                                   static_cast<uint16_t>((cb_ * 2) / 32), 0, 0};
                // [FIX-ROWPART] 与状态更新同构：本子核写/读自己那连续半区
                const int32_t r0_ = subIdx_ * rowsPerSub_ + i0;
                DataCopy(hF32_[r0_ * cb_], extBlkF_, hpF_);
                DataCopy(hBf_[r0_ * cb_], stateBlkBf_, hpB_);
                PipeBarrier<PIPE_ALL>();
            }
        }
#else
        for (int32_t r = subIdx_; r < CV_K; r += subNum_) {
            Duplicate(row0F_, 0.0f, cb_);
            PipeBarrier<PIPE_ALL>();
            Cast(row0Bf_, row0F_, RoundMode::CAST_RINT, cb_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(hF32_[r * cb_], row0F_, cb_);
            DataCopy(hBf_[r * cb_], row0Bf_, cb_);
            PipeBarrier<PIPE_ALL>();
        }
#endif
#endif
#if PPFM_RD_PROBE
        // 诊断：prologue 写完后立刻回读 h 状态第 0 行（期望全 0）
        if (subIdx_ == 0) {
            DataCopy(row1F_, hF32_, CV_V);           // GM -> UB
            PipeBarrier<PIPE_ALL>();
            DataCopy(probeG_[2 * CV_V], row1F_, CV_V);   // UB -> GM
            PipeBarrier<PIPE_ALL>();
            DataCopy(row2Bf_, hBf_, CV_V);                // bf16(h) 第 0 行（bf16->fp32 观察）
            PipeBarrier<PIPE_ALL>();
            Cast(row2F_, row2Bf_, RoundMode::CAST_NONE, CV_V);
            PipeBarrier<PIPE_ALL>();
            DataCopy(probeG_[1 * CV_V], row2F_, CV_V);
            PipeBarrier<PIPE_ALL>();
        }
#endif
        // m 初值 = I
        // 950：**逐行纯向量构造**（ArithProgression + |k-r| 造对角，省下 64 KiB UB）。
        // 注意： 不要用 row0F_.SetValue(r,1) 这类"标量写 UB + 向量写同一块 UB"的组合：
        //   实测标量写的落盘顺序不受 PipeBarrier<PIPE_V> 保护，会让个别行丢掉对角 1
        //   （表现为 m 只有 ~0.05% 元素错、max_abs≈1）。
        // 910B/910_93：同一套「ArithProgression + |k-r|」构造在 A2 上**实测退化**——
        //   m 变成"每行常数"（行 r 的值只随 r 变化、整行相同，对角与状态全错；
        //   用 w=0,g=0 探针可复现：m 应为 I，实到 m[r][:] 恒等于 [r%4<2]）。
        //   A2 上 ArithProgression 走 common 实现（标量写 8 拍 + 向量 Add 展开），
        //   与外层逐行向量组合相互干扰；且全仓仅本算子用到该原语（无先例）。
        //   这里改成最朴素、逐行可验证的构造：整行清零 + 单点写 1，
        //   标量写与搬运之间一律用 PIPE_ALL 全栅栏隔离（KDA 的逐点 SetValue 路径
        //   在 A2 上实测正确，说明标量写本身没问题）。
#if PPFM_A2_BLOCKWISE
        {
            const int32_t rowsPerSub_ = CV_K / PPFM_SUB;
            // 同 h 初值：按 extBlkF_ 的实际容量分块（cb_=128 ⇒ 32 行/块）
            const int32_t rowsPerCopy_ = UB_EXT_F_ELEMS / cb_;
            for (int32_t i0 = 0; i0 < rowsPerSub_; i0 += rowsPerCopy_) {
                const int32_t nr_ = (rowsPerSub_ - i0 < rowsPerCopy_)
                                        ? (rowsPerSub_ - i0) : rowsPerCopy_;
                Duplicate(extBlkF_, 0.0f, nr_ * cb_);
                PipeBarrier<PIPE_ALL>();
                for (int32_t n = 0; n < nr_; ++n) {
                    const int32_t r = subIdx_ * rowsPerSub_ + i0 + n;
                    const int32_t c = r - colBase_;
                    if (c >= 0 && c < cb_) {
                        extBlkF_.SetValue(n * cb_ + c, 1.0f);
                    }
                }
                PipeBarrier<PIPE_ALL>();
                Cast(stateBlkBf_, extBlkF_, RoundMode::CAST_RINT, nr_ * cb_);
                AIV_SET_V_MTE3();
                AIV_WAIT_V_MTE3();
                DataCopyParams mpF_{static_cast<uint16_t>(nr_),
                                   static_cast<uint16_t>((cb_ * 4) / 32), 0, 0};
                DataCopyParams mpB_{static_cast<uint16_t>(nr_),
                                   static_cast<uint16_t>((cb_ * 2) / 32), 0, 0};
                // [FIX-ROWPART] 与状态更新同构：本子核写/读自己那连续半区
                const int32_t r0_ = subIdx_ * rowsPerSub_ + i0;
                DataCopy(mF32_[r0_ * cb_], extBlkF_, mpF_);
                DataCopy(mBf_[r0_ * cb_], stateBlkBf_, mpB_);
#if PPFM_M_UB
                // m 常驻 UB：单位阵也要落到 mUb_（与 mpF_ 同套连续落点）
                DataCopy(mUb_[i0 * cb_], extBlkF_, mpF_);
#endif
                PipeBarrier<PIPE_ALL>();
            }
        }
#else
        for (int32_t r = subIdx_; r < CV_K; r += subNum_) {
            Duplicate(row0F_, 0.0f, cb_);
            PipeBarrier<PIPE_ALL>();
            if (r >= colBase_ && r < colBase_ + cb_) {
                row0F_.SetValue(r - colBase_, 1.0f);
            }
            PipeBarrier<PIPE_ALL>();
            Cast(row0Bf_, row0F_, RoundMode::CAST_RINT, cb_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(mF32_[r * cb_], row0F_, cb_);
            DataCopy(mBf_[r * cb_], row0Bf_, cb_);
            PipeBarrier<PIPE_ALL>();
        }
#endif
        // 临时诊断（当前不启用）：给 AIC 即将写的 C 缓冲预置哨兵。
        //   vTmpF_ = 7.0、t1F_ = 5.0 ⇒ 若 AIC 的 fixpipe 正常覆盖，chunk0 的结果不受影响；
        //   若结果里出现 7/5 量级的残留，说明 AIC→AIV 的 C 落点/可见性有问题。
#if PPFM_SENTINEL_PROBE
        Duplicate(scrF_, 7.0f, CV_BT * CV_V);
        PipeBarrier<PIPE_ALL>();
        DataCopy(vTmpF_, scrF_, static_cast<uint32_t>(CV_BT * CV_V));
        Duplicate(scrF_, 5.0f, CV_BT * CV_K);
        PipeBarrier<PIPE_ALL>();
        DataCopy(t1F_, scrF_, static_cast<uint32_t>(CV_BT * CV_K));
        PipeBarrier<PIPE_ALL>();
#endif
        const int64_t nt = (len + CV_BT - 1) / CV_BT;
        // 先独立做 chunk 0 的 staging；循环内把 staging(c+1) 提到
        // 「等 dH/T2(c)」之前 ⇒ staging 与 AIC 的 mm2/mm4(c) 重叠（原来 AIV 在这里纯等）
#if PPFM_DH_CV
        // /首 credit：dH 与 T2 的槽各先归还一次，否则 AIC 第一次 CV 写入会死等。

        // 每个 AIV 子核各置一次（硬件映射到 id / id+PPFM_SUBFLAG_STRIDE），AIC 侧按 subblock
        // 各等一次。
        CrossCoreSetFlag<0x4, PIPE_V>(static_cast<uint16_t>(kFlagDhFree));
#if PPFM_T2_CV
        CrossCoreSetFlag<0x4, PIPE_V>(static_cast<uint16_t>(kFlagT2Free));
#endif
#endif
        StageChunk(n, hv, bos, (len < CV_BT) ? len : CV_BT, 0);
        for (int64_t c = 0; c < nt; ++c) {
#if PPFM_DIAG
            curChunk_ = c;
#endif
            const int64_t t0 = bos + c * CV_BT;
            const int64_t left = len - c * CV_BT;
            const int64_t rows = (left < CV_BT) ? left : CV_BT;
            if (c > 0) {
                // dH/T2(c-1) 已在上一轮末尾等到；decayPrevF_ 此时是 decay(c-1)
                ApplyStateUpdates(true, c - 1);
            }
            // staging 产物 + 本 chunk 状态一次通知（原来分 kFlagInputs / kFlagState 两次）
            AivSetToAic(kFlagInputs);
            UpdateVNew(hv, t0, rows);
            if (c + 1 < nt) {
                // 保存本 chunk 的 decay（下一轮推迟状态更新要用），随后 staging 覆盖 decayF_
                Adds(decayPrevF_, decayF_, 0.0f, CV_K);
                PipeBarrier<PIPE_ALL>();
                const int64_t t0n = bos + (c + 1) * CV_BT;
                const int64_t leftn = len - (c + 1) * CV_BT;
                const int64_t rowsn = (leftn < CV_BT) ? leftn : CV_BT;
                StageChunk(n, hv, t0n, rowsn, static_cast<int32_t>((c + 1) & 1));
            }
            AivWaitFromAic(kFlagDH);
            // 读别的核（AIC）写的 GM 前必须让本核缓存行失效，否则会读到过期数据
            // （与 CANN matmul_client.h 中"读跨核 GM flag 前先 DCCI"的用法一致）
            // 注意： 原来只失效了**偶数奇偶**的 dHF_/t2F_，而下面按 chunk 奇偶读的是
            //   `dhBuf`/`t2Buf`（两份）⇒ 奇数 chunk 读的 `dHF1_`/`t2F1_` 从来没被失效过，
            //   命中旧行就会把过期 T2 减进 m（表现为 m 半边小范围错、h 正常）。
            //   这个缺口在 A2 上（PPFM_LEGACY_CACHEOPS=1）才暴露得出来；补齐两份。
#if PPFM_KEEP_CACHEOPS_R
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(dHF_);
#endif
#if PPFM_KEEP_CACHEOPS_R
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(dHF1_);
#endif
#if PPFM_KEEP_CACHEOPS_R
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(t2F_);
#endif
#if PPFM_KEEP_CACHEOPS_R
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(t2F1_);
#endif
#ifdef PPFM_DEBUG_HEADER
            // [临时调试] 在各更新点之后立刻回读状态（fp32 标量直读，最可靠）
            // 结果先放 UB（row1F_ 高位 lane），最后由 epilogue 的 header 块带出
            ProbeState(hF32_, 100);
            ProbeState(mF32_, 101);
            ProbeState(dHF_, 102);
            ProbeState(t2F_, 103);
            PipeBarrier<PIPE_ALL>();
#endif
        }
        // 最后一个 chunk 的状态更新（此时 decayF_ 就是它的 decay，dH/T2 也已落地）
        ApplyStateUpdates(false, nt - 1);
        // ---- epilogue：写 hm ----
        const int64_t hmBase = ((n * t->Hv + hv) * CV_K) * (CV_V + CV_K);
#if PPFM_H_UB
#if PPFM_EP_MERGE
        // h/m 各 1 次跨步 DataCopy 取代 64 次逐行搬 + 128 次 PIPE_ALL
        PipeBarrier<PIPE_V>();   // 状态更新的 V 运算先落地
        {
            DataCopyParams epParams{
                static_cast<uint16_t>(H_UB_ROWS),
                static_cast<uint16_t>((cb_ * 4) / 32),
                0,
                static_cast<uint16_t>(((CV_V + CV_K - cb_) * 4) / 32)};
            const int64_t epRow0 = hmBase +
                static_cast<int64_t>(subIdx_) * H_UB_ROWS * (CV_V + CV_K) + colBase_;
            DataCopy(hmGm_[epRow0], hUb_, epParams);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();
#if PPFM_M_UB
            DataCopy(hmGm_[epRow0 + CV_V], mUb_, epParams);
#else
            for (int32_t r = subIdx_ * H_UB_ROWS; r < (subIdx_ + 1) * H_UB_ROWS; ++r) {
                DataCopy(row0F_, mF32_[r * cb_], cb_);
                PipeBarrier<PIPE_ALL>();
                DataCopy(hmGm_[hmBase + r * (CV_V + CV_K) + CV_V + colBase_], row0F_, cb_);
                PipeBarrier<PIPE_ALL>();
            }
#endif
        }
        PipeBarrier<PIPE_ALL>();
#else
        // /h、m 都直接从常驻 UB 写 hm（同一套连续半区分配）
        for (int32_t r = subIdx_ * H_UB_ROWS; r < (subIdx_ + 1) * H_UB_ROWS; ++r) {
            DataCopy(hmGm_[hmBase + r * (CV_V + CV_K) + colBase_],
                     hUb_[(r - subIdx_ * H_UB_ROWS) * cb_], cb_);
            PipeBarrier<PIPE_ALL>();
#if PPFM_M_UB
            DataCopy(hmGm_[hmBase + r * (CV_V + CV_K) + CV_V + colBase_],
                     mUb_[(r - subIdx_ * H_UB_ROWS) * cb_], cb_);
#else
            DataCopy(row0F_, mF32_[r * cb_], cb_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(hmGm_[hmBase + r * (CV_V + CV_K) + CV_V + colBase_], row0F_, cb_);
#endif
            PipeBarrier<PIPE_ALL>();
        }
#endif
#else
#if PPFM_A2_BLOCKWISE
        {
            const int32_t rowsPerSub_ = CV_K / PPFM_SUB;
            // 同 prologue：按 extBlkF_ 的实际容量分块搬运，避免踩相邻 UB
            const int32_t rowsPerCopy_ = UB_EXT_F_ELEMS / cb_;
            for (int32_t i0 = 0; i0 < rowsPerSub_; i0 += rowsPerCopy_) {
                const int32_t nr_ = (rowsPerSub_ - i0 < rowsPerCopy_)
                                        ? (rowsPerSub_ - i0) : rowsPerCopy_;
                DataCopyParams rd_{static_cast<uint16_t>(nr_),
                                   static_cast<uint16_t>((cb_ * 4) / 32),
                                   0, 0};
                DataCopyParams wr_{static_cast<uint16_t>(nr_),
                                   static_cast<uint16_t>((cb_ * 4) / 32), 0,
                                   static_cast<uint16_t>(((CV_V + CV_K) - cb_) * 4 / 32)};
                // [FIX-ROWPART] 行归属必须与状态更新一致（连续半区）：否则收尾要读
                //               另一个子核刚写的行，而 AIV<->AIV 之间没有同步
                const int32_t r0_ = subIdx_ * rowsPerSub_ + i0;
                const int64_t epRow0_ = hmBase + static_cast<int64_t>(r0_) * (CV_V + CV_K);
                DataCopy(extBlkF_, hF32_[r0_ * cb_], rd_);
                PipeBarrier<PIPE_ALL>();
                DataCopy(hmGm_[epRow0_ + colBase_], extBlkF_, wr_);
                PipeBarrier<PIPE_ALL>();
                DataCopy(extBlkF_, mF32_[r0_ * cb_], rd_);
                PipeBarrier<PIPE_ALL>();
                DataCopy(hmGm_[epRow0_ + CV_V + colBase_], extBlkF_, wr_);
                PipeBarrier<PIPE_ALL>();
            }
        }
#else
        for (int32_t r = subIdx_; r < CV_K; r += subNum_) {
            DataCopy(row0F_, hF32_[r * cb_], cb_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(hmGm_[hmBase + r * (CV_V + CV_K) + colBase_], row0F_, cb_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(row0F_, mF32_[r * cb_], cb_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(hmGm_[hmBase + r * (CV_V + CV_K) + CV_V + colBase_], row0F_, cb_);
            PipeBarrier<PIPE_ALL>();
        }
#endif
#endif
#if PPFM_RD_PROBE
        // 诊断：把探针各行搬到 hm 的 m 半边第 0..4 行（验收时排除这些行）
        if (n == 0 && hv == 0 && subIdx_ == 0) {
            for (int32_t pr = 0; pr <= 5; ++pr) {
                DataCopy(row0F_, probeG_[pr * CV_V], CV_K);
                PipeBarrier<PIPE_ALL>();
                DataCopy(hmGm_[hmBase + pr * (CV_V + CV_K) + CV_V], row0F_, CV_K);
                PipeBarrier<PIPE_ALL>();
            }
        }
#endif
#if PPFM_DIAG
        // 诊断收尾（只由子核 0 写）：lane 0..3 = AIV 读到的 vTmpF_[0]（第 c 个 chunk），
        // lane 4 = AIV 读到的 T2 首元素；lane 8..11 = AIC 写出的 vTmpF_[0]，
        // lane 12 = AIC 写进 t2Buf 的首元素，lane 13 = AIC 读到的 bf16(m)[0]（chunk0 应为 1.0）。
        // 落在 hm 的 m 半边第 0 行（验收时排除该行）。
        if (subIdx_ == 0) {
            Duplicate(row0F_, 0.0f, CV_K);
            PipeBarrier<PIPE_V>();
            for (int32_t i = 0; i < 8; ++i) {
                row0F_.SetValue(i, dbgF_.GetValue(i));
            }
            PipeBarrier<PIPE_ALL>();
#if PPFM_LEGACY_CACHEOPS
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(diagG_);
#endif
            DataCopy(row1F_, diagG_, 8);          // 8 个 fp32 = 32B，满足对齐要求
            PipeBarrier<PIPE_ALL>();
            for (int32_t i = 0; i < 8; ++i) {
                row0F_.SetValue(8 + i, row1F_.GetValue(i));
            }
            PipeBarrier<PIPE_ALL>();
            DataCopy(hmGm_[hmBase + CV_V], row0F_, CV_K);
            PipeBarrier<PIPE_ALL>();
        }
#endif
#ifdef PPFM_DEBUG_HEADER
        // [临时调试] 收尾后覆盖 hm 第 0 行的 m 半边，写入 tiling/cu 关键值
        for (int32_t i = 0; i < CV_K; ++i) {
            row0F_.SetValue(i, 0.0f);
        }
        PipeBarrier<PIPE_ALL>();
        row0F_.SetValue(0, static_cast<float>(t->nSeq));
        row0F_.SetValue(1, static_cast<float>(t->Hv));
        row0F_.SetValue(2, static_cast<float>(t->T));
        row0F_.SetValue(3, static_cast<float>(t->K));
        row0F_.SetValue(4, static_cast<float>(t->V));
        row0F_.SetValue(5, static_cast<float>(t->chunkSize));
        row0F_.SetValue(6, static_cast<float>(t->gateMode));
        row0F_.SetValue(7, static_cast<float>(t->usedAicNum));
        row0F_.SetValue(8, static_cast<float>(t->taskNum));
        row0F_.SetValue(9, static_cast<float>(bos));
        row0F_.SetValue(10, static_cast<float>(len));
        row0F_.SetValue(11, static_cast<float>(nt));
        row0F_.SetValue(12, static_cast<float>(GetBlockIdx()));
        row0F_.SetValue(13, static_cast<float>(GetSubBlockNum()));
        // 各阶段中间量抽样：bf16 一律走"向量 Cast 转 fp32"（标量 bf16 转换不可靠）
        ProbeBf16(kGm_, 20);            // 输入 k（第 0 行前 2 个）
        ProbeBf16(wGm_, 22);            // 输入 w
        ProbeBf16(vGm_, 24);            // 输入 v
        row0F_.SetValue(26, gGm_.GetValue(0));
        row0F_.SetValue(27, gGm_.GetValue(1));
        ProbeBf16(kBf_, 28);            // staging k
        ProbeBf16(wBf_, 30);            // staging w
        ProbeBf16(lBf_, 32);            // staging left
        ProbeBf16(vBf_, 34);            // staging v
        ProbeF32(gateF_, 36, 0);        // dg[0], dg[1]
        ProbeF32(gateF_, 38, CV_BT);    // decay[0], decay[1]
        ProbeF32(vTmpF_, 40, 0);        // matmul1 输出
        ProbeF32(t1F_, 42, 0);          // matmul3 输出
        ProbeF32(dHF_, 44, 0);          // matmul2 输出
        ProbeF32(t2F_, 46, 0);          // matmul4 输出
        ProbeF32(hF32_, 48, 0);         // h 状态
        ProbeF32(mF32_, 50, 0);         // m 状态
        ProbeBf16(vNewBf_, 52);         // bf16(v_new) 前 2 个
        ProbeBf16(t1Bf_, 54);           // bf16(T1) 前 2 个
        ProbeBf16(mBf_, 56);            // bf16(m) 第 0 行前 2 个
        ProbeBf16(hBf_, 58);            // bf16(h) 第 0 行前 2 个
        // 更新点回读（见 chunk 循环里的插桩，值放在 row1F_ 的 100..103 lane）
        row0F_.SetValue(52, row1F_.GetValue(100));   // hF32_[0] @ UpdateH 之后
        row0F_.SetValue(53, row1F_.GetValue(101));   // mF32_[0] @ UpdateM 之后
        row0F_.SetValue(54, row1F_.GetValue(102));   // dHF_[0] @ DH 之后
        row0F_.SetValue(55, row1F_.GetValue(103));   // t2F_[0] @ T2 之后
        PipeBarrier<PIPE_ALL>();
        PipeBarrier<PIPE_ALL>();
        DataCopy(hmGm_[hmBase + CV_V], row0F_, CV_K);
        PipeBarrier<PIPE_ALL>();
#endif
    }

    // staging：W / k / left / v（bf16，尾块零填充）+ dg / decay
    __aicore__ inline void StageChunk(int64_t n, int64_t hv, int64_t t0, int64_t rows, int32_t slot)
    {
        // k/left 按 chunk 奇偶写不同 GM 槽，使 staging 可与上一 chunk 的 mm2/mm4 重叠
        GlobalTensor<bfloat16_t> &kOut = ((slot & 1) != 0) ? kBf1_ : kBf_;
        GlobalTensor<bfloat16_t> &lOut = ((slot & 1) != 0) ? lBf1_ : lBf_;
        const auto *t = ctx_.tiling;
        const int64_t hk = hv / (t->hvPerHk == 0 ? 1 : t->hvPerHk);
        const bool useG = (t->gateMode == PPFM_GATE_USE_G);
        float glast = 0.0f;
        if (useG) {
#if !PPFM_GLAST_UB
            glast = gGm_.GetValue(hv * t->T + (t0 + rows - 1));
#endif
            // 向量化：dg[t] = exp2(glast - g[t])（整块一次算完，替代逐 token 标量 Exp）
            // 注意： 尾块 rows 不是 8（32B）的整数倍时：DataCopy 的长度必须 32B 对齐，
            //    否则是 UB（越界读）；这里改用 DataCopyPad（blockLen 按字节给）。
            DataCopyExtParams gParams{1, static_cast<uint32_t>(rows * sizeof(float)), 0, 0, 0};
            DataCopyPad(gBlkF_, gGm_[hv * t->T + t0], gParams, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
#if PPFM_GLAST_UB
            // 同一个值已经在 UB 里（rows-1 就是本 chunk 最后一个 token 的 g）
            glast = gBlkF_.GetValue(rows - 1);
#endif
            Muls(gBlkF_, gBlkF_, -1.0f, static_cast<int32_t>(rows));
            PipeBarrier<PIPE_V>();
            Adds(gBlkF_, gBlkF_, glast, static_cast<int32_t>(rows));
            PipeBarrier<PIPE_V>();
            Muls(gBlkF_, gBlkF_, 0.6931471805599453f, static_cast<int32_t>(rows));
            PipeBarrier<PIPE_V>();
            // 注意： 先整块清零再覆盖前 rows 个：Duplicate 的目的地址必须 32B 对齐，
            //    rows=36 时 dgF_[rows] 落在 144B（非 32B 整数倍）→ VEC 访问 UB 非对齐
            //    （error code 340）直接 aicore exception。
            Duplicate(dgF_, 0.0f, static_cast<int32_t>(CV_BT));
            PipeBarrier<PIPE_V>();
            Exp(dgF_, gBlkF_, static_cast<int32_t>(rows));
            PipeBarrier<PIPE_ALL>();
        } else {
            // dgF_ 现在是 per-subcore 的私有 scratch，两个子核都要**各自填满**
            Duplicate(dgF_, 1.0f, static_cast<int32_t>(CV_BT));
        }
        PipeBarrier<PIPE_ALL>();
        SetDecay(hv, t0 + rows - 1, glast);

        // ---- staging：**按子核整半区（32 行）分配**，段内自己完成"清零/搬运/left
        // 计算/落盘"----
        // 原来切成 2×16 行是历史遗留 —— 子核 i 拿的是连续半区（`seg = i*SEG_PER_SUB + k`
        // ⇒ 行 [i*32,(i+1)*32)），之后缓冲也正好按 32 行分配 ⇒ **一段装齐**。
        // 依据：仿真流水显示 AIV 的 MTE3 占 53%、其中 UB→GM 的 `MOV_SRC_TO_DST_ALIGN` 平均
        // ~520 cycle/条（纯 latency）⇒ 段数减半最直接（对应参考文档 "row tile 太小"）。
        constexpr int32_t SEG = PPFM_SEGROWS;                 // 32
        // 段按**连续半区**分配给子核（子核 i 处理段 [i*2,(i+1)*2)），
        // 与 AIC fixpipe SPLIT_M 的落点（前一半行→低半区）对齐
        constexpr int32_t SEG_PER_SUB = (CV_BT / SEG) / PPFM_SUB;
        // 满 chunk 时 AIC 会直接读输入里的 w/k ⇒ 这里不必再 staging 它们
        const bool directInputs = (PPFM_AIC_DIRECT_INPUTS != 0) && (rows == CV_BT);
        // w 只在满 chunk 时交给 AIC 直读（省掉 AIV 的 w 读 + wBf_ 写）
        const bool directW = (PPFM_AIC_DIRECT_W != 0) && (rows == CV_BT);
        for (int32_t seg = subIdx_ * SEG_PER_SUB; seg < (subIdx_ + 1) * SEG_PER_SUB; ++seg) {
            const int32_t off = seg * SEG;
            // **UB 侧一律用子核本地段偏移**（`lo`），只有 GM 侧才用全局行号 `off`。
            // 依据：已证每个 AIV 子核有独立 UB bank ⇒ 段分区缓冲（kBlk/wBlk/vBlk/scr）
            // 每个子核只需放自己那 `SEG_PER_SUB*SEG = 32` 行，尺寸直接砍半（省 48 KiB）。
            // 这正是最初 里被误判为"做不到"的那条（当时按共享 UB 推导）。
            const int32_t lo = off - subIdx_ * (CV_BT / PPFM_SUB);
            const int32_t valid = (rows > off) ? ((rows - off < SEG) ? (rows - off) : SEG) : 0;
            // P1a：段间复用同一组 UB（kBlk/wBlk/vBlk/scr）。事件语义是"该流水此前所有操作
            // 都完成"，所以在这里成对 set/wait 即可覆盖"上一段的 MTE3 是否读完"，
            // 不需要额外的信用记账。
            AIV_SET_MTE3_MTE2();
            AIV_WAIT_MTE3_MTE2();
            AIV_SET_MTE3_V();
            AIV_WAIT_MTE3_V();
            // 只有尾块需要零填充（整段时下面的 DataCopy 会写满整段）
            if (valid < SEG) {
                Duplicate(kBlkBf_[lo * CV_K], static_cast<bfloat16_t>(0), SEG * CV_K);
                Duplicate(wBlkBf_[lo * CV_K], static_cast<bfloat16_t>(0), SEG * CV_K);
                Duplicate(vBlkBf_[lo * cb_], static_cast<bfloat16_t>(0), SEG * cb_);
                PipeBarrier<PIPE_V>();
                AIV_SET_V_MTE2();
                AIV_WAIT_V_MTE2();
            }
            if (valid > 0) {
                DataCopy(kBlkBf_[lo * CV_K], kGm_[(hk * t->T + t0 + off) * CV_K],
                         static_cast<uint32_t>(valid * CV_K));
                if (!directW) {
                    DataCopy(wBlkBf_[lo * CV_K], wGm_[(hv * t->T + t0 + off) * CV_K],
                             static_cast<uint32_t>(valid * CV_K));
                }
                // P5：v 只搬本工作项需要的列窗 [colBase_, colBase_+cb_)（列间隔用 srcStride
                // 跳过）
                DataCopyExtParams vParams{
                    static_cast<uint16_t>(valid),
                    static_cast<uint32_t>(cb_ * static_cast<int32_t>(sizeof(bfloat16_t))),
                    static_cast<uint32_t>((CV_V - cb_) * static_cast<int32_t>(sizeof(bfloat16_t))), 0, 0};
                DataCopyPad(vBlkBf_[lo * cb_], vGm_[(hv * t->T + t0 + off) * CV_V + colBase_],
                            vParams, {false, 0, 0, 0});
            }
            AIV_SET_MTE2_MTE3();
            AIV_WAIT_MTE2_MTE3();
            if (!directInputs) {
                DataCopy(kOut[off * CV_K], kBlkBf_[lo * CV_K], SEG * CV_K);
            }
            if (!directW) {
                DataCopy(wBf_[off * CV_K], wBlkBf_[lo * CV_K], SEG * CV_K);
            }
            // 注意：v 不再落到 GM（v_new 直接从 UB 的 vBlkBf_ 读），省一份 16 KiB/chunk 的 MTE3
            AIV_SET_MTE2_V();
            AIV_WAIT_MTE2_V();
            // left：USE_G 为 bf16(k·dg)，USE_GK 为 k 本身
            if (useG) {
                Cast(scrF_[lo * CV_K], kBlkBf_[lo * CV_K], RoundMode::CAST_NONE, SEG * CV_K);
                PipeBarrier<PIPE_V>();
#if PPFM_FAC8_ON
                // C10：Brcb 广播 factor（[SEG] → [SEG,8]）+ 反复式 Mul，取代 32 次标量读 + 32 次 Muls。
                // 行宽 CV_K=128 > 64 ⇒ 拆两段（向量一次最多 64 个 fp32）；乘数仍是 dgF_ 里的同一个 fp32 ⇒ 位级不变。
                Brcb(fac8_, dgF_[off], SEG / 8, {1, 8});
                PipeBarrier<PIPE_V>();
                for (int32_t half = 0; half < CV_K / 64; ++half) {
                    Mul(scrF_[lo * CV_K + half * 64], scrF_[lo * CV_K + half * 64], fac8_,
                        64, SEG, {1, 1, 0, CV_K / 8, CV_K / 8, 1});
                }
#else
#if PPFM_ROW_PREFETCH
                float facBuf_[PPFM_SEGROWS];
                for (int32_t i = 0; i < SEG; ++i) {
                    facBuf_[i] = dgF_.GetValue(off + i);   // 先把 32 个标量读完（隐藏 UB 标量读延迟）
                }
#pragma unroll
                for (int32_t i = 0; i < SEG; ++i) {
                    Muls(scrF_[(lo + i) * CV_K], scrF_[(lo + i) * CV_K], facBuf_[i], CV_K);
                }
#else
                for (int32_t i = 0; i < SEG; ++i) {
                    Muls(scrF_[(lo + i) * CV_K], scrF_[(lo + i) * CV_K], dgF_.GetValue(off + i), CV_K);
                }
#endif
#endif  // PPFM_FAC8_ON
                PipeBarrier<PIPE_V>();
                Cast(scrBf_[lo * CV_K], scrF_[lo * CV_K], RoundMode::CAST_RINT, SEG * CV_K);
                AIV_SET_V_MTE3();
                AIV_WAIT_V_MTE3();   // V -> MTE3
                DataCopy(lOut[off * CV_K], scrBf_[lo * CV_K], SEG * CV_K);
            } else {
#if PPFM_KDA_LEFT_ALIAS
                // KDA（USE_GK）下 left ≡ k ⇒ 这里不再重复写一份 lBf_，
                // AIC 侧 mm4 直接读 k 的槽（同 chunk 奇偶、同形状）。
                // directInputs 打开时 AIV 不写 kOut，保持原样。
                if (directInputs) {
                    DataCopy(lOut[off * CV_K], kBlkBf_[lo * CV_K], SEG * CV_K);
                }
#else
                DataCopy(lOut[off * CV_K], kBlkBf_[lo * CV_K], SEG * CV_K);
#endif
            }
            // 段末：本段两次 MTE3（kOut/wBf_ 与 lOut）读完后，下一段才能覆盖对应 UB
        }
        PipeBarrier<PIPE_ALL>();
        // dg / decay 落 GM 只是调试用途（只有 PPFM_DIAG 下的 ProbeF32 会读），
        //        当前不启用，每 chunk 省 2 次 DataCopy + 2 次全栅栏
#if PPFM_DIAG
        DataCopy(gateF_[WS_GATE_DG / 4], dgF_, CV_BT);
        PipeBarrier<PIPE_ALL>();
        DataCopy(gateF_[WS_GATE_DECAY / 4], decayF_, CV_K);
        PipeBarrier<PIPE_ALL>();
#endif
        // 不再在这里发通知——与状态更新合并成一次（见 ProcessChain）
    }

    __aicore__ inline void UpdateVNew(int64_t hv, int64_t t0, int64_t rows)
    {
        const bool useG = (ctx_.tiling->gateMode == PPFM_GATE_USE_G);
        AivWaitFromAic(kFlagHalf1);
#if PPFM_VTMP_UB_DIAG
        if (curChunk_ == 1) {
            PipeBarrier<PIPE_ALL>();
            const float g0 = vTmpF_.GetValue(0);
            const float g32 = vTmpF_.GetValue(32 * CV_V);
            const float uOwn0 = extBlkF_.GetValue(0);
            const float uOwn32 = extBlkF_.GetValue(32 * CV_V);
            const float uSh0 = vTmpUb_.GetValue(0);
            const float uSh32 = vTmpUb_.GetValue(32 * CV_V);
            PipeBarrier<PIPE_ALL>();
            row0F_.SetValue(0, g0);
            row0F_.SetValue(1, g32);
            row0F_.SetValue(2, uOwn0);
            row0F_.SetValue(3, uOwn32);
            row0F_.SetValue(4, uSh0);
            row0F_.SetValue(5, uSh32);
            PipeBarrier<PIPE_ALL>();
            DataCopy(hmGm_[((curN_ * ctx_.tiling->Hv + hv) * CV_K) * (CV_V + CV_K) + CV_V],
                     row0F_, 8);
            PipeBarrier<PIPE_ALL>();
        }
#endif
        // vTmp 走 GM 时保留过渡探读（A2 走 UB 时才省掉）
#if PPFM_LEGACY_PROBE_READS
#if !PPFM_VTMP_UB
        DataCopy(row2F_, vTmpF_, 8);
        PipeBarrier<PIPE_ALL>();
#endif
        DataCopy(row2F_, t1F_, 8);
        PipeBarrier<PIPE_ALL>();
#endif  // PPFM_LEGACY_PROBE_READS
#if PPFM_DIAG
        // 诊断：记下"本子核读到的 vTmpF_[0]"（chunk 0 时它必须恰好是 0）
        if (curChunk_ < PPFM_DIAG_CHUNKS) {
            PipeBarrier<PIPE_ALL>();
            dbgF_.SetValue(static_cast<int32_t>(curChunk_), vTmpF_.GetValue(0));
            PipeBarrier<PIPE_ALL>();
        }
#endif
#if !PPFM_VTMP_UB
#if PPFM_KEEP_CACHEOPS_R
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(vTmpF_);
#endif
#endif
#if PPFM_KEEP_CACHEOPS_R
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(t1F_);
#endif
        // v_new = (v - vTmp) · dg → bf16（逐行；整块版本会引入 ~0.4% 的 GDN 偏差，待查）

        // v_new = (v - vTmp)·dg → bf16：同样**按子核整半区（32 行）**，段内做完
        // Cast/Sub/缩放/Cast/落盘
        // 同 StageLeft —— 2×16 合并成 1×32（少一半 UB→GM 小搬运与段间同步对）
        constexpr int32_t SEG = PPFM_SEGROWS;                 // 32
        // 段按**连续半区**分配给子核（子核 i 处理段 [i*2,(i+1)*2)），
        // 与 AIC fixpipe SPLIT_M 的落点（前一半行→低半区）对齐
        constexpr int32_t SEG_PER_SUB = (CV_BT / SEG) / PPFM_SUB;
        for (int32_t seg = subIdx_ * SEG_PER_SUB; seg < (subIdx_ + 1) * SEG_PER_SUB; ++seg) {
            const int32_t off = seg * SEG;
            const int32_t lo = (seg - subIdx_ * SEG_PER_SUB) * SEG;
#if !PPFM_VTMP_UB
            // A5 主线：vTmp 仍从 GM 回读
            DataCopy(extBlkF_[lo * cb_], vTmpF_[off * cb_], SEG * cb_);
            AIV_SET_MTE2_V();
            AIV_WAIT_MTE2_V();
#if PPFM_RD_PROBE
        // 诊断：把本子核读到的 vTmp 第 0 行原样存到 GM 暂存（首个 chunk，子核 0）
            if (probeCnt_ == 0 && subIdx_ == 0) {
                DataCopy(probeG_, extBlkF_, cb_);
                PipeBarrier<PIPE_ALL>();
                probeCnt_ = 1;
            }
#endif
#endif
            // UB 侧用子核本地段偏移（GM 侧仍是全局 off）
            Cast(scrF_[lo * cb_], vBlkBf_[lo * cb_], RoundMode::CAST_NONE, SEG * cb_);
            PipeBarrier<PIPE_V>();
#if PPFM_VTMP_UB
            // 从共享基址视图读本子核那半（lo ∈ {0, SEG}）
            Sub(scrF_[lo * cb_], scrF_[lo * cb_], vTmpUb_[lo * cb_], SEG * cb_);
#else
            Sub(scrF_[lo * cb_], scrF_[lo * cb_], extBlkF_[lo * cb_], SEG * cb_);
#endif
            PipeBarrier<PIPE_V>();
            if (useG) {
#if PPFM_FAC8_ON
                // C10：同 left 路径 —— Brcb 广播 factor + 反复式 Mul（行宽 cb_，>64 时拆段）
                Brcb(fac8_, dgF_[off], SEG / 8, {1, 8});
                PipeBarrier<PIPE_V>();
                // ⚠ 列宽 cb_ 可能 < 64（hybridS=4 的分片任务 cb_=32）⇒ 必须按 ≤64 一段扫，
                //   不能写成 for (half = 0; half < cb_/64; ...)：cb_=32 时 cb_/64=0，整段缩放会被跳过
                //   （首版就是这个错，L1 在 gdn-hy64 上抓到 max|diff|=4.14e-02）。
                for (int32_t c0 = 0; c0 < cb_; c0 += 64) {
                    const int32_t n = (cb_ - c0 < 64) ? (cb_ - c0) : 64;
                    Mul(scrF_[lo * cb_ + c0], scrF_[lo * cb_ + c0], fac8_,
                        static_cast<uint16_t>(n), SEG,
                        {1, 1, 0, static_cast<uint8_t>(cb_ / 8),
                         static_cast<uint8_t>(cb_ / 8), 1});
                }
#else
#if PPFM_ROW_PREFETCH
                float facBuf_[PPFM_SEGROWS];
                for (int32_t i = 0; i < SEG; ++i) {
                    facBuf_[i] = dgF_.GetValue(off + i);
                }
#pragma unroll
                for (int32_t i = 0; i < SEG; ++i) {
                    Muls(scrF_[(lo + i) * cb_], scrF_[(lo + i) * cb_], facBuf_[i], cb_);
                }
#else
                for (int32_t i = 0; i < SEG; ++i) {
                    Muls(scrF_[(lo + i) * cb_], scrF_[(lo + i) * cb_], dgF_.GetValue(off + i), cb_);
                }
#endif
#endif  // PPFM_FAC8_ON
                PipeBarrier<PIPE_V>();
            }
            Cast(scrBf_[lo * cb_], scrF_[lo * cb_], RoundMode::CAST_RINT, SEG * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();   // V -> MTE3
            // 注意： UB 源必须用子核本地段偏移（`lo`），只有 GM 目标用全局 `off`
            DataCopy(vNewBf_[off * cb_], scrBf_[lo * cb_], SEG * cb_);
        }
        // 各子核只写自己那两段（off 互不相交），循环内无需段间信用；
        // 出口保留一次全栅栏，供后面的 staging 复用 scrF_/scrBf_（跨函数边界）。
        PipeBarrier<PIPE_ALL>();
        // bf16(T1)：同样按段分配
        // 段按**连续半区**分配给子核（子核 i 处理段 [i*2,(i+1)*2)），
        // 与 AIC fixpipe SPLIT_M 的落点（前一半行→低半区）对齐
        // SEG_PER_SUB 已在 v_new 循环前声明（同一函数内不能重复定义）
#if !PPFM_T1_FIXPIPE_BF16
        for (int32_t seg = subIdx_ * SEG_PER_SUB; seg < (subIdx_ + 1) * SEG_PER_SUB; ++seg) {
            const int32_t off = seg * SEG;
            DataCopy(scrF_[lo * cb_], t1F_[off * cb_], SEG * cb_);
            AIV_SET_MTE2_V();
            AIV_WAIT_MTE2_V();
            Cast(scrBf_[lo * cb_], scrF_[lo * cb_], RoundMode::CAST_RINT, SEG * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();   // V -> MTE3
            DataCopy(t1Bf_[off * cb_], scrBf_[lo * cb_], SEG * cb_);
        }
        PipeBarrier<PIPE_ALL>();
#endif
#if PPFM_RD_PROBE
        // 诊断：vNewBf_ 第 0 行（bf16→fp32）读回，确认 AIV 写出的 B 内容
        if (probeCnt_ == 1 && subIdx_ == 0) {
            DataCopy(row2Bf_, vNewBf_, CV_V);
            PipeBarrier<PIPE_ALL>();
            Cast(row2F_, row2Bf_, RoundMode::CAST_NONE, CV_V);
            PipeBarrier<PIPE_ALL>();
            DataCopy(probeG_[5 * CV_V], row2F_, CV_V);
            PipeBarrier<PIPE_ALL>();
        }
#endif
        AivSetToAic(kFlagVNew);
    }

    // h = decay⊙h + dH ; m = decay⊙m - T2
    // usePrevDecay=true 时用上一 chunk 的 decay（配合"状态更新推迟一个 chunk"）
    __aicore__ inline void ApplyStateUpdates(bool usePrevDecay, int64_t dataChunk)
    {
        const bool useG = (ctx_.tiling->gateMode == PPFM_GATE_USE_G);
        // dH / T2 双缓冲：按数据所属 chunk 的奇偶选 buffer（不用拷贝，直接分支）
        const bool evenChunk = ((dataChunk & 1) == 0);
        GlobalTensor<float> &dhBuf = evenChunk ? dHF_ : dHF1_;
        GlobalTensor<float> &t2Buf = evenChunk ? t2F_ : t2F1_;
#if PPFM_DH_CV
        // dH 在本子核的 UB bank 里（单槽），不再从 GM 回读
        LocalTensor<float> dHUb = ubBuf_.Get<float>()[UB_DH_CV_ELEM];
#endif
        // 同 UpdateVNew 的过渡探读：dH / T2 也是 AIC 刚写、本核刚读的 GM
#if PPFM_LEGACY_PROBE_READS
        DataCopy(row2F_, dhBuf, 8);
        PipeBarrier<PIPE_ALL>();
        DataCopy(row2F_, t2Buf, 8);
        PipeBarrier<PIPE_ALL>();
#endif  // PPFM_LEGACY_PROBE_READS
        // 每 RB 行一次搬运：块内逐行 Muls（廉价、无需栅栏），块级 Add/Sub/Cast
        // h/m 常驻 UB 时用 PPFM_SBRB（一块 64 行）；A2/回退路径仍是 PPFM_RB=32
        constexpr int32_t RB = PPFM_SBRB;
#if PPFM_H_UB
        // /h 常驻 UB 时本子核只持有自己那 64 行 ⇒ h 相位必须按**连续半区**分配
        // （子核 i 负责行 [i*CV_K/2, (i+1)*CV_K/2)）。状态更新是逐行 elementwise，
        // 换分法数值等价（GVA/KDA 的 decay 索引仍是全局行号）。
        const int32_t rbBeg = subIdx_ * H_UB_ROWS;
        const int32_t rbEnd = rbBeg + H_UB_ROWS;
        const int32_t rbStep = RB;
#elif PPFM_DH_CV
        // SPLIT_M 是「连续半区」⇒ 子核 i 负责行 [i*CV_K/2, (i+1)*CV_K/2)。
        const int32_t rbBeg = subIdx_ * (CV_K / PPFM_SUB);
        const int32_t rbEnd = rbBeg + (CV_K / PPFM_SUB);
        const int32_t rbStep = RB;
#else
        const int32_t rbBeg = subIdx_ * RB;
        const int32_t rbEnd = CV_K;
        const int32_t rbStep = subNum_ * RB;
#endif
        for (int32_t rb = rbBeg; rb < rbEnd; rb += rbStep) {
#if PPFM_H_UB
            // h 常驻 UB（本子核那 64 行）⇒ 就地更新，无 MTE2 载入、无 fp32 落盘。
            // 只有 bf16(h) 仍要写 GM —— 那是 AIC mm1 的输入。
            const int32_t lo = rb - rbBeg;
#if !PPFM_DH_CV
            // （A2/A3）：dH 仍从 GM 回读，但**提前发**（与下面 h 的 Muls 重叠）

            // 注意： 必须显式补 WAR 序：上一块（或上一相位）对 extBlkF_ 的 **V
            // 读**要先完成，
            //   否则这一个 MTE2 会覆盖它、dH 只写进去一部分 → h 的对应行整行错。
            //   （950 不踩这个坑是因为它的 dH 走 UB 槽，h 相位根本不碰 extBlkF_。）
            AIV_WAR_BEFORE_STATE_MTE2();   // C9：把上面注释要求的 V 读序真正补上
            AIV_SET_MTE3_MTE2();
            AIV_WAIT_MTE3_MTE2();
            DataCopy(extBlkF_, dhBuf[rb * cb_], RB * cb_);
            AIV_SET_MTE2_V();
#endif
            if (useG) {
                const float dc = usePrevDecay ? decayPrevF_.GetValue(0) : decayF_.GetValue(0);
                Muls(hUb_[lo * cb_], hUb_[lo * cb_], dc, RB * cb_);
            } else {
#if PPFM_KDA_ROW_PREFETCH
                // K1-a：先批量标量预取 factor，再整批 Muls（纯发射顺序，位级不变）
                for (int32_t kb = 0; kb < RB; kb += PPFM_KDA_ROW_BATCH) {
                    float decBuf_[PPFM_KDA_ROW_BATCH];
                    for (int32_t kj = 0; kj < PPFM_KDA_ROW_BATCH; ++kj) {
                        decBuf_[kj] = usePrevDecay ? decayPrevF_.GetValue(rb + kb + kj)
                                                   : decayF_.GetValue(rb + kb + kj);
                    }
#pragma unroll
                    for (int32_t kj = 0; kj < PPFM_KDA_ROW_BATCH; ++kj) {
                        Muls(hUb_[(lo + (kb + kj)) * cb_], hUb_[(lo + (kb + kj)) * cb_],
                             decBuf_[kj], cb_);
                    }
                }
#else
                for (int32_t r = rb; r < rb + RB; ++r) {
                    const float dc = usePrevDecay ? decayPrevF_.GetValue(r) : decayF_.GetValue(r);
                    Muls(hUb_[(lo + (r - rb)) * cb_], hUb_[(lo + (r - rb)) * cb_], dc, cb_);
                }
#endif
            }
            PipeBarrier<PIPE_V>();
#if PPFM_DH_CV
            Add(hUb_[lo * cb_], hUb_[lo * cb_], dHUb[lo * cb_], RB * cb_);
#else
            AIV_WAIT_MTE2_V();
            Add(hUb_[lo * cb_], hUb_[lo * cb_], extBlkF_, RB * cb_);
#endif
            AIV_SET_MTE3_V();
            AIV_WAIT_MTE3_V();   // 上一次 MTE3 读完 stateBlkBf_ 才能覆盖
            Cast(stateBlkBf_, hUb_[lo * cb_], RoundMode::CAST_RINT, RB * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();   // V -> MTE3
            DataCopy(hBf_[rb * cb_], stateBlkBf_, RB * cb_);
#else
            // 第一次搬运后的全栅栏冗余——紧随其后第二次搬运之后还有一次，
            //        足以保证两次 MTE2 都在 Muls/Add 之前完成
            // P1a：上一轮的 MTE3（写 hF32_/hBf_）读的是同一组 UB，先等它读完再覆盖
            AIV_WAR_BEFORE_STATE_MTE2();   // C9：补上一轮对 stateBlkF_/extBlkF_ 的 V 读序
            AIV_SET_MTE3_MTE2();
            AIV_WAIT_MTE3_MTE2();
            DataCopy(stateBlkF_, hF32_[rb * cb_], RB * cb_);
#if !PPFM_DH_CV
            DataCopy(extBlkF_, dhBuf[rb * cb_], RB * cb_);
#endif
            AIV_SET_MTE2_V();
            AIV_WAIT_MTE2_V();
#if PPFM_RD_PROBE
            // 诊断：首个 RB 块里，把"读到的 h 状态"和"读到的 dH"各留一行
            if (rb == subIdx_ * RB && subIdx_ == 0) {
                DataCopy(probeG_[3 * CV_V], stateBlkF_, CV_V);   // h 状态读回
                DataCopy(probeG_[4 * CV_V], extBlkF_, CV_V);     // dH 读回
                PipeBarrier<PIPE_ALL>();
            }
#endif
            if (useG) {
                // GDN：每 chunk 一个标量 decay ⇒ 整块一次 Muls（原来 16 次逐行 Muls +
                // 16 次 GetValue；h/m 合计每 chunk 每子核 128 次，是 AIV SCALAR 的主要来源）
                const float dc = usePrevDecay ? decayPrevF_.GetValue(0) : decayF_.GetValue(0);
                Muls(stateBlkF_, stateBlkF_, dc, RB * cb_);
            } else {
#if PPFM_KDA_ROW_PREFETCH
                // K1-a：先批量标量预取 factor，再整批 Muls（纯发射顺序，位级不变）
                for (int32_t kb = 0; kb < RB; kb += PPFM_KDA_ROW_BATCH) {
                    float decBuf_[PPFM_KDA_ROW_BATCH];
                    for (int32_t kj = 0; kj < PPFM_KDA_ROW_BATCH; ++kj) {
                        decBuf_[kj] = usePrevDecay ? decayPrevF_.GetValue(rb + kb + kj)
                                                   : decayF_.GetValue(rb + kb + kj);
                    }
#pragma unroll
                    for (int32_t kj = 0; kj < PPFM_KDA_ROW_BATCH; ++kj) {
                        Muls(stateBlkF_[(kb + kj) * cb_], stateBlkF_[(kb + kj) * cb_],
                             decBuf_[kj], cb_);
                    }
                }
#else
#if PPFM_FAC8_ON
                // C10b：FAC8 打开时本分支取代 K1-a（_common.h 已把 PPFM_KDA_ROW_PREFETCH 置 0）。
                // 乘数仍是 decayF_/decayPrevF_ 里同一个 fp32 ⇒ 位级不变。
                {
                    LocalTensor<float> decSrc = usePrevDecay ? decayPrevF_ : decayF_;
                    Brcb(fac8_, decSrc[rb], RB / 8, {1, 8});
                    PipeBarrier<PIPE_V>();
                    // 同 C10：按 ≤64 一段扫（cb_ 可为 32 ⇒ 不能假设 cb_/64 >= 1）
                    for (int32_t c0 = 0; c0 < cb_; c0 += 64) {
                        const int32_t n = (cb_ - c0 < 64) ? (cb_ - c0) : 64;
                        Mul(stateBlkF_[c0], stateBlkF_[c0], fac8_,
                            static_cast<uint16_t>(n), RB,
                            {1, 1, 0, static_cast<uint8_t>(cb_ / 8),
                             static_cast<uint8_t>(cb_ / 8), 1});
                    }
                }
#else
                for (int32_t r = rb; r < rb + RB; ++r) {
                    const float dc = usePrevDecay ? decayPrevF_.GetValue(r) : decayF_.GetValue(r);
                    Muls(stateBlkF_[(r - rb) * cb_], stateBlkF_[(r - rb) * cb_], dc, cb_);
                }
#endif
#endif
            }
            PipeBarrier<PIPE_V>();
#if PPFM_DH_CV
            Add(stateBlkF_, stateBlkF_, dHUb[(rb - rbBeg) * cb_], RB * cb_);
#else
            Add(stateBlkF_, stateBlkF_, extBlkF_, RB * cb_);
#endif
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();   // V -> MTE3
            DataCopy(hF32_[rb * cb_], stateBlkF_, RB * cb_);
            AIV_SET_MTE3_V();
            AIV_WAIT_MTE3_V();   // 上一次 MTE3 读完 stateBlkF_/stateBlkBf_ 才能覆盖
            Cast(stateBlkBf_, stateBlkF_, RoundMode::CAST_RINT, RB * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();   // V -> MTE3
            DataCopy(hBf_[rb * cb_], stateBlkBf_, RB * cb_);
#endif  // PPFM_H_UB
        }
#if PPFM_DH_CV
        // 归还本代 dH 的槽（PIPE_V：保证上面所有读该槽的 V 运算都已发射完）
        CrossCoreSetFlag<0x4, PIPE_V>(static_cast<uint16_t>(kFlagDhFree));
#endif
        PipeBarrier<PIPE_ALL>();
#if PPFM_T2_CV
        // T2 也在本子核的 UB bank 里（单槽，同样 SPLIT_M 连续半区）⇒ 本相位必须
        // 用与 h 相位一致的**连续半区**行分配（逐行 elementwise，数值等价）。
        LocalTensor<float> t2Ub = ubBuf_.Get<float>()[UB_T2_CV_ELEM];
        const int32_t mbBeg = subIdx_ * (CV_K / PPFM_SUB);
        const int32_t mbEnd = mbBeg + (CV_K / PPFM_SUB);
        for (int32_t rb = mbBeg; rb < mbEnd; rb += RB) {
            const int32_t lo = rb - mbBeg;
#if PPFM_M_UB
            // m 常驻 UB ⇒ 就地更新（无 MTE2 载入、无 fp32 落盘），只留 bf16(m) 给 AIC 的 mm3
            if (useG) {
                const float dc = usePrevDecay ? decayPrevF_.GetValue(0) : decayF_.GetValue(0);
                Muls(mUb_[lo * cb_], mUb_[lo * cb_], dc, RB * cb_);
            } else {
#if PPFM_KDA_ROW_PREFETCH
                // K1-a：先批量标量预取 factor，再整批 Muls（纯发射顺序，位级不变）
                for (int32_t kb = 0; kb < RB; kb += PPFM_KDA_ROW_BATCH) {
                    float decBuf_[PPFM_KDA_ROW_BATCH];
                    for (int32_t kj = 0; kj < PPFM_KDA_ROW_BATCH; ++kj) {
                        decBuf_[kj] = usePrevDecay ? decayPrevF_.GetValue(rb + kb + kj)
                                                   : decayF_.GetValue(rb + kb + kj);
                    }
#pragma unroll
                    for (int32_t kj = 0; kj < PPFM_KDA_ROW_BATCH; ++kj) {
                        Muls(mUb_[(lo + (kb + kj)) * cb_], mUb_[(lo + (kb + kj)) * cb_],
                             decBuf_[kj], cb_);
                    }
                }
#else
                for (int32_t r = rb; r < rb + RB; ++r) {
                    const float dc = usePrevDecay ? decayPrevF_.GetValue(r) : decayF_.GetValue(r);
                    Muls(mUb_[(lo + (r - rb)) * cb_], mUb_[(lo + (r - rb)) * cb_], dc, cb_);
                }
#endif
            }
            PipeBarrier<PIPE_V>();
            Sub(mUb_[lo * cb_], mUb_[lo * cb_], t2Ub[lo * cb_], RB * cb_);
            AIV_SET_MTE3_V();
            AIV_WAIT_MTE3_V();
            Cast(stateBlkBf_, mUb_[lo * cb_], RoundMode::CAST_RINT, RB * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();
            DataCopy(mBf_[rb * cb_], stateBlkBf_, RB * cb_);
#else
            AIV_SET_MTE3_MTE2();
            AIV_WAIT_MTE3_MTE2();
            DataCopy(stateBlkF_, mF32_[rb * cb_], RB * cb_);
            AIV_SET_MTE2_V();
            AIV_WAIT_MTE2_V();
            if (useG) {
                const float dc = usePrevDecay ? decayPrevF_.GetValue(0) : decayF_.GetValue(0);
                Muls(stateBlkF_, stateBlkF_, dc, RB * cb_);
            } else {
#if PPFM_KDA_ROW_PREFETCH
                // K1-a：先批量标量预取 factor，再整批 Muls（纯发射顺序，位级不变）
                for (int32_t kb = 0; kb < RB; kb += PPFM_KDA_ROW_BATCH) {
                    float decBuf_[PPFM_KDA_ROW_BATCH];
                    for (int32_t kj = 0; kj < PPFM_KDA_ROW_BATCH; ++kj) {
                        decBuf_[kj] = usePrevDecay ? decayPrevF_.GetValue(rb + kb + kj)
                                                   : decayF_.GetValue(rb + kb + kj);
                    }
#pragma unroll
                    for (int32_t kj = 0; kj < PPFM_KDA_ROW_BATCH; ++kj) {
                        Muls(stateBlkF_[(kb + kj) * cb_], stateBlkF_[(kb + kj) * cb_],
                             decBuf_[kj], cb_);
                    }
                }
#else
                for (int32_t r = rb; r < rb + RB; ++r) {
                    const float dc = usePrevDecay ? decayPrevF_.GetValue(r) : decayF_.GetValue(r);
                    Muls(stateBlkF_[(r - rb) * cb_], stateBlkF_[(r - rb) * cb_], dc, cb_);
                }
#endif
            }
            PipeBarrier<PIPE_V>();
            Sub(stateBlkF_, stateBlkF_, t2Ub[lo * cb_], RB * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();
            DataCopy(mF32_[rb * cb_], stateBlkF_, RB * cb_);
            AIV_SET_MTE3_V();
            AIV_WAIT_MTE3_V();
            Cast(stateBlkBf_, stateBlkF_, RoundMode::CAST_RINT, RB * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();
            DataCopy(mBf_[rb * cb_], stateBlkBf_, RB * cb_);
#endif  // PPFM_M_UB
        }
        CrossCoreSetFlag<0x4, PIPE_V>(static_cast<uint16_t>(kFlagT2Free));
#else
        for (int32_t rb = subIdx_ * RB; rb < CV_K; rb += subNum_ * RB) {
            AIV_WAR_BEFORE_STATE_MTE2();   // C9：m 相位的同一窗口
            AIV_SET_MTE3_MTE2();
            AIV_WAIT_MTE3_MTE2();
#if PPFM_M_UB
            // m 常驻 UB：不载入 fp32 状态（省每 chunk 64 KiB 读）
#else
            DataCopy(mState, mF32_[rb * cb_], RB * cb_);
#endif
            DataCopy(extBlkF_, t2Buf[rb * cb_], RB * cb_);
            AIV_SET_MTE2_V();
            AIV_WAIT_MTE2_V();
#if PPFM_M_UB
            // 本子核那一段（lo = rb - subIdx_*RB，与 epilogue 的连续半区一致）
            LocalTensor<float> mState = mUb_[(rb - subIdx_ * RB) * cb_];
#else
            LocalTensor<float> mState = stateBlkF_;
#endif
#if PPFM_DIAG
            // MDIAG：记录 AIV **实际读到的** T2 行首元素（行 rb 与 rb+32），
            // 与 AIC 写进 t2Buf 的那一份对比（槽位 4+2*subIdx_ / 5+2*subIdx_）。
            if (dataChunk == 0) {
                dbgF_.SetValue(4 + 2 * subIdx_, extBlkF_.GetValue(0));
                dbgF_.SetValue(5 + 2 * subIdx_, extBlkF_.GetValue(32 * cb_));
                PipeBarrier<PIPE_ALL>();
            }
#endif
            if (useG) {
                const float dc = usePrevDecay ? decayPrevF_.GetValue(0) : decayF_.GetValue(0);
                Muls(mState, mState, dc, RB * cb_);
            } else {
#if PPFM_KDA_ROW_PREFETCH
                // K1-a：先批量标量预取 factor，再整批 Muls（纯发射顺序，位级不变）
                for (int32_t kb = 0; kb < RB; kb += PPFM_KDA_ROW_BATCH) {
                    float decBuf_[PPFM_KDA_ROW_BATCH];
                    for (int32_t kj = 0; kj < PPFM_KDA_ROW_BATCH; ++kj) {
                        decBuf_[kj] = usePrevDecay ? decayPrevF_.GetValue(rb + kb + kj)
                                                   : decayF_.GetValue(rb + kb + kj);
                    }
#pragma unroll
                    for (int32_t kj = 0; kj < PPFM_KDA_ROW_BATCH; ++kj) {
                        Muls(mState[(kb + kj) * cb_], mState[(kb + kj) * cb_],
                             decBuf_[kj], cb_);
                    }
                }
#else
#if PPFM_FAC8_ON
                // C10b：m 相位的同一站点（非 M_UB / 非 T2_CV 路径，A2 活跃）
                {
                    LocalTensor<float> decSrc = usePrevDecay ? decayPrevF_ : decayF_;
                    Brcb(fac8_, decSrc[rb], RB / 8, {1, 8});
                    PipeBarrier<PIPE_V>();
                    // 同 C10：按 ≤64 一段扫（cb_ 可为 32 ⇒ 不能假设 cb_/64 >= 1）
                    for (int32_t c0 = 0; c0 < cb_; c0 += 64) {
                        const int32_t n = (cb_ - c0 < 64) ? (cb_ - c0) : 64;
                        Mul(mState[c0], mState[c0], fac8_,
                            static_cast<uint16_t>(n), RB,
                            {1, 1, 0, static_cast<uint8_t>(cb_ / 8),
                             static_cast<uint8_t>(cb_ / 8), 1});
                    }
                }
#else
                for (int32_t r = rb; r < rb + RB; ++r) {
                    const float dc = usePrevDecay ? decayPrevF_.GetValue(r) : decayF_.GetValue(r);
                    Muls(mState[(r - rb) * cb_], mState[(r - rb) * cb_], dc, cb_);
                }
#endif
#endif
            }
            PipeBarrier<PIPE_V>();
            Sub(mState, mState, extBlkF_, RB * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();   // V -> MTE3
#if !PPFM_M_UB
            DataCopy(mF32_[rb * cb_], mState, RB * cb_);
#endif
            AIV_SET_MTE3_V();
            AIV_WAIT_MTE3_V();
            Cast(stateBlkBf_, mState, RoundMode::CAST_RINT, RB * cb_);
            AIV_SET_V_MTE3();
            AIV_WAIT_V_MTE3();   // V -> MTE3
            DataCopy(mBf_[rb * cb_], stateBlkBf_, RB * cb_);
        }
#endif  // PPFM_T2_CV
        PipeBarrier<PIPE_ALL>();
    }

    const PpFwdCtx &ctx_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<bfloat16_t> row0Bf_;
    LocalTensor<bfloat16_t> row1Bf_;
    LocalTensor<bfloat16_t> row2Bf_;
    LocalTensor<float> row0F_;
    LocalTensor<float> row1F_;
    LocalTensor<float> row2F_;
    LocalTensor<float> dgF_;
    LocalTensor<float> decayF_;
    LocalTensor<float> expScratch_;
    LocalTensor<float> decayPrevF_;
    LocalTensor<float> gBlkF_;
    LocalTensor<float> stateBlkF_;
    LocalTensor<float> extBlkF_;
#if PPFM_H_UB
    LocalTensor<float> hUb_;          // 常驻 UB 的 h 半步（本子核的 64 行 × cb_）
#endif
#if PPFM_M_UB
    LocalTensor<float> mUb_;          // 常驻 UB 的 m 半步（同上）
#endif
    LocalTensor<float> vTmpUb_;   // 共享基址的 vTmp 落点视图
    LocalTensor<bfloat16_t> stateBlkBf_;
#if PPFM_FAC8_ON
    LocalTensor<float> fac8_;          // C10：factor 广播暂存 [FAC8_ROWS, 8] fp32
#endif
    LocalTensor<bfloat16_t> kBlkBf_;
    LocalTensor<bfloat16_t> wBlkBf_;
    LocalTensor<bfloat16_t> vBlkBf_;
    LocalTensor<float> scrF_;
    LocalTensor<bfloat16_t> scrBf_;
    int32_t subIdx_ = 0;
    int32_t subNum_ = 1;
    int32_t splitNum_ = 1;   // P5 列块数（1 或 2），由 tiling.colSplit 决定
    int32_t cb_ = CV_V;      // P5 本工作项的列宽（128 或 64）
    int32_t colBase_ = 0;    // P5 本工作项列块在整条链里的起始列
    int64_t curN_ = 0;   // UB 诊断探针要定位本链 hm 地址
    GlobalTensor<bfloat16_t> kGm_;
    GlobalTensor<bfloat16_t> wGm_;
    GlobalTensor<bfloat16_t> vGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<float> gkGm_;
    GlobalTensor<int64_t> cuGm_;
    GlobalTensor<float> hmGm_;
    GlobalTensor<float> hF32_;
    GlobalTensor<float> mF32_;
    GlobalTensor<bfloat16_t> hBf_;
    GlobalTensor<bfloat16_t> mBf_;
    GlobalTensor<bfloat16_t> wBf_;
    GlobalTensor<bfloat16_t> kBf_;
    GlobalTensor<bfloat16_t> lBf_;
    GlobalTensor<bfloat16_t> kBf1_;  // 
    GlobalTensor<bfloat16_t> lBf1_;  // 
    GlobalTensor<bfloat16_t> vBf_;
    GlobalTensor<float> vTmpF_;
    GlobalTensor<bfloat16_t> vNewBf_;
    GlobalTensor<float> dHF_;
    GlobalTensor<float> dHF1_;
    GlobalTensor<float> t1F_;
    GlobalTensor<bfloat16_t> t1Bf_;
    GlobalTensor<float> t2F_;
    GlobalTensor<float> t2F1_;
    GlobalTensor<float> gateF_;
#if PPFM_RD_PROBE
    GlobalTensor<float> probeG_;     // 诊断：AIV 读回的 vTmp 行暂存
    int32_t probeCnt_ = 0;
#endif
#if PPFM_DIAG
    GlobalTensor<float> diagG_;      // 每核诊断区（读 AIC 写的 vTmp 指纹）
    LocalTensor<float> dbgF_;        // 每子核 16 个诊断槽
    int64_t curChunk_ = 0;
#endif
};

// =====================================================================================
// AIC：4 个 matmul
// =====================================================================================
} // namespace GDN

#endif  // PREF_PROCESS_FWD_KERNEL_MERGED_VEC_H
