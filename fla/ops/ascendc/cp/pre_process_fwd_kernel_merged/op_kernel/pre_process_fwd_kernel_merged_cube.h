/*!
 * \file pre_process_fwd_kernel_merged_cube.h
 * \brief pre_process_fwd_kernel_merged：AIC（cube）实现
 *
 * 本文件由 kernel 主文件按"机械搬运"拆出（代码与拆分前逐字符相同）；
 * Stage/布局/同步协议的完整说明见 pre_process_fwd_kernel_merged_common.h 顶部注释与 docs/design.md。
 */

#ifndef PREF_PROCESS_FWD_KERNEL_MERGED_CUBE_H
#define PREF_PROCESS_FWD_KERNEL_MERGED_CUBE_H

#include "pre_process_fwd_kernel_merged_common.h"

namespace GDN {class PpFwdCube {
public:
    __aicore__ inline PpFwdCube(const PpFwdCtx &ctx) : ctx_(ctx) {}

    __aicore__ inline void Run()
    {
        const auto *t = ctx_.tiling;
        const int64_t coreIdx = static_cast<int64_t>(GetBlockIdx());
        // P5：列块切分（与 AIV 侧同一套解码；cb_/colBase_ 见 PpFwdVector::Run）
        splitNum_ = (t->colSplit > 0) ? static_cast<int32_t>(t->colSplit) : 1;
        cb_ = static_cast<int32_t>(CV_V) / splitNum_;
        colBase_ = 0;
        __gm__ uint8_t *ws = reinterpret_cast<__gm__ uint8_t *>(ctx_.ws) + coreIdx * PPFM_CORE_WS_BYTES;

        wBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_W_BF), CV_BT * CV_K);
        kBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_K_BF), CV_BT * CV_K);
        lBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_L_BF), CV_BT * CV_K);
        kBf1_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_K_BF_1), CV_BT * CV_K);  // 
        lBf1_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_L_BF_1), CV_BT * CV_K);  // 
        hBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_H_BF), CV_K * CV_V);
#if PPFM_DIAG
        diagG_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_DIAG), 16);
#endif
        mBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_M_BF), CV_K * CV_K);
        vNewBf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_VNEW_BF), CV_BT * CV_V);
        t1Bf_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ws + WS_T1_BF), CV_BT * CV_K);
        vTmpF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_VTMP_F32), CV_BT * CV_V);
        dHF_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_DH_F32), CV_K * CV_V);
        dHF1_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_DH_F32_1), CV_K * CV_V);
        t1F_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_T1_F32), CV_BT * CV_K);
        t2F_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_T2_F32), CV_K * CV_K);
        t2F1_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + WS_T2_F32_1), CV_K * CV_K);

        cuGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(ctx_.cu));
        // PPFM_AIC_DIRECT_INPUTS：满 chunk 时直接读输入张量里的 w/k（省掉 AIV 的 staging）
        wIn_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx_.w));
        kIn_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ctx_.k));
        // 与 AIV 侧同一套任务解码。
        const int64_t nChain = t->nSeq * t->Hv;
        const int64_t taskNum = (t->hybridS > 1)
            ? (t->hybridBase + (nChain - t->hybridBase) * t->hybridS)
            : (nChain * static_cast<int64_t>(splitNum_));
        for (int64_t task = coreIdx; task < taskNum; task += static_cast<int64_t>(t->usedAicNum)) {
            const PpFwdChunkInfo pos_ = GetChunkInfo(t, splitNum_, task);
            const int64_t hv = pos_.hv;
            const int64_t n = pos_.n;
            cb_ = pos_.cb;          // GetChunkInfo 已算好列窗（§4.4 ⑤）
            colBase_ = pos_.colBase;
            const int64_t bos = cuGm_.GetValue(n);
            const int64_t eos = cuGm_.GetValue(n + 1);
            const int64_t len = eos - bos;
            const int64_t nt = (len + CV_BT - 1) / CV_BT;
            for (int64_t c = 0; c < nt; ++c) {
                const int64_t leftLen = len - c * CV_BT;
                const int64_t rows = (leftLen < CV_BT) ? leftLen : CV_BT;
                ProcessChunk(c, hv, bos + c * CV_BT, rows);
            }
#if PPFM_DH_CV
            // /尾 drain：把本链 dH 槽与 T2 槽的归还信用收干净（每链 AIV prime 1 次、
            // 每 chunk set 1 次、这里 wait 1 次 ⇒ 逐链平衡）。对手算子同构做法见 arch35 的
            // DrainPipeFlags。
            CrossCoreWaitFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagDhFree));
            CrossCoreWaitFlag<0x4, PIPE_FIX>(
                static_cast<uint16_t>(kFlagDhFree + PPFM_SUBFLAG_STRIDE));
#if PPFM_T2_CV
            CrossCoreWaitFlag<0x4, PIPE_FIX>(static_cast<uint16_t>(kFlagT2Free));
            CrossCoreWaitFlag<0x4, PIPE_FIX>(
                static_cast<uint16_t>(kFlagT2Free + PPFM_SUBFLAG_STRIDE));
#endif
#endif
        }
    }

private:
    __aicore__ inline void ProcessChunk(int64_t c, int64_t hv, int64_t t0, int64_t rows)
    {
        // dH / T2 双缓冲：本 chunk 写到自己那一份（不用拷贝，直接分支）
        const bool evenChunk = ((c & 1) == 0);
        GlobalTensor<float> &dhBuf = evenChunk ? dHF_ : dHF1_;
        GlobalTensor<float> &t2Buf = evenChunk ? t2F_ : t2F1_;
        // 满 chunk 时 w/k 直接用输入张量（AIV 侧不再 staging）；尾块仍用零填充后的 staging
        const auto *tt = ctx_.tiling;
        const int64_t hk = hv / (tt->hvPerHk == 0 ? 1 : tt->hvPerHk);
        const bool directInputs = (PPFM_AIC_DIRECT_INPUTS != 0) && (rows == CV_BT);
        // w 由 AIC 直读输入（AIV 不再 staging w、也不写 wBf_）
        const bool directW = (PPFM_AIC_DIRECT_W != 0) && (rows == CV_BT);
        GlobalTensor<bfloat16_t> wTile =
            directW ? wIn_[(hv * tt->T + t0) * CV_K] : wBf_;
        // ① vTmp[BT,V] = W_c[BT,K] @ bf16(h)[K,V]
        // inputs 与 state 已合并为同一次通知
        AicWaitFromAiv(kFlagInputs);
        // 注意： 读别的核（AIV）刚写过的 GM 之前必须让本核的 cache 失效：h/m 每 chunk 都被
        //    AIV 重写，若 AIC 命中自己缓存的旧行，mm1/mm3 就会拿到过期的 h/m
        //    （实测表现为概率性的 h 半边大面积错、m 只是略偏，且随调度时好时坏）。
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(wBf_);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(hBf_);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(mBf_);
#endif
#if PPFM_TILE_MMAD
        RunTiledNT(wTile, hBf_, vTmpF_, CV_BT, static_cast<uint32_t>(cb_), CV_K,
                    /*toUb=*/(PPFM_VTMP_UB != 0));   // 按宏选择 A2 UB 落点
#else
        RunMmadNT(wTile, hBf_, vTmpF_, CV_BT, static_cast<uint32_t>(cb_), CV_K);
#endif

#if PPFM_T1_FIXPIPE_BF16
        // ---- ：**mm1 一做完就通知 AIV** ----
        // 原实现是"mm1 + mm3 都做完才 set kFlagHalf1"。但 AIV 的 `v_new` 只需要 mm1 的 C（vTmp）；
        // T1（mm3 的 C）只有 AIC 自己用（mm4 的 B）⇒ 通知提前到 mm1 之后，
        // AIC 的 mm3/mm4 就落进 AIV 的 `v_new` 窗口里（正是"错相"要的效果，
        // 但**不动 m 链相位、flag 数量完全不变**）。
        // 注意： 仅当 `PPFM_T1_FIXPIPE_BF16=1`（T1 直接以 bf16 落 GM、AIV 不消费）时成立。
#if !PPFM_VTMP_UB
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(vTmpF_);
#endif
#endif
#if PPFM_LEGACY_CACHEOPS
        DataSyncBarrier<MemDsbT::DDR>();
#endif
        AicSetToAiv(kFlagHalf1);
#endif

        // ③ T1[BT,K] = W_c[BT,K] @ bf16(m)[K,K]
#if PPFM_TILE_MMAD
#if PPFM_T1_FIXPIPE_BF16
        // fixpipe 直接把 T1 量化成 bf16 落 t1Bf_（mm4 的 B 操作数），AIV 侧不再往返
        // A（=W）复用 mm1 已经搬进 L1 的那一份，省一次 16 KiB 的 GM→L1
        RunTiledNT(wTile, mBf_, t1Bf_, CV_BT, static_cast<uint32_t>(cb_), CV_K,
                   /*toUb=*/false, /*keepA=*/true);
#else
        RunTiledNT(wTile, mBf_, t1F_, CV_BT, static_cast<uint32_t>(cb_), CV_K,
                   /*toUb=*/false, /*keepA=*/true);
#endif
#else
        RunMmadNT(wTile, mBf_, t1F_, CV_BT, static_cast<uint32_t>(cb_), CV_K);
#endif

        // 注意： 写侧也要 clean（写回），只靠读者 DCCI 不够：FIX
        // 写回可能还停在写缓冲里，
        //   此时 AIV 即便 DCCI 也会读到旧值。实测（PPFM_DIAG 指纹）：AIV 读到 vTmp 全 0，
        //   而 AIC 实际写了 -0.013/+0.0092 → v_new 退化成 v，h 半边随机整头崩（m
        // 不受影响）。
#if !PPFM_T1_FIXPIPE_BF16
#if !PPFM_VTMP_UB
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(vTmpF_);
#endif
#endif
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(t1F_);
#endif
#if !PPFM_T1_FIXPIPE_BF16
        // 注意： 正式原语：DDR 数据同步屏障——保证 C 的写回对其他核可见后再抬 flag。
        //   诊断版实验表明竞态是"flag 已到、写回仍在途"的时序窗口（加探针即掩盖）。
#if PPFM_LEGACY_CACHEOPS
        DataSyncBarrier<MemDsbT::DDR>();
#endif
        AicSetToAiv(kFlagHalf1);
#endif

        // ---- 把 **m 链的 mm4（T2）提前到「等 v_new」之前** ----

        // 依据（的 profile）：每 chunk 里 AIC 与 AIV 几乎完全串行 —— 两边的忙时都 ≈
        // 算子时长

        // （3069 / 3099 vs 3103 µs），而各自 pipe 只 52% / 92% 忙 ⇒
        // 卡在"乒乓"的关键路径上，不在吞吐。
        // mm4 = leftᵀ @ bf16(T1) 只依赖 left(c)（staging 已由 kFlagInputs 保证）与
        // T1(c)（刚算完），

        // **与 h 链的 v_new 无关** ⇒ 提前后可与 AIV 的 v_new 相位重叠，AIC 关键路径上少一个
        // MMAD；
        // m 链（m→T1→T2→m）也不再排在 h 链后面。

        // 安全性：T2 槽的归还信用由 AIV 在**本迭代开头的 m 相位**里置起（早于
        // kFlagInputs(c)），
        // 所以这里（kFlagInputs 之后）写槽一定在 AIV 消费完上一代之后。
#if PPFM_KDA_LEFT_ALIAS
        // KDA（USE_GK）下 left ≡ k：AIV 已不再单独写 lBf_（见 _vec.h 的 staging），
        // 这里直接复用 k 的槽。AIC_DIRECT_INPUTS 打开时不别名（那时仍走 lBf_ 旧路径）。
        const bool aliasLeft_ = (PPFM_AIC_DIRECT_INPUTS == 0) &&
                                (ctx_.tiling->gateMode != PPFM_GATE_USE_G);
        GlobalTensor<bfloat16_t> &lIn = aliasLeft_
            ? (((c & 1) != 0) ? kBf1_ : kBf_)
            : (((c & 1) != 0) ? lBf1_ : lBf_);
#else
        GlobalTensor<bfloat16_t> &lIn = ((c & 1) != 0) ? lBf1_ : lBf_;
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(lIn);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(t1Bf_);
#endif
#if PPFM_T2_CV
        // T2 直落 UB 槽（单槽）
        RunTiledTAUb(lIn, t1Bf_, CV_K, static_cast<uint32_t>(cb_), CV_BT,
                     UB_T2_CV, static_cast<uint16_t>(kFlagT2Free));
#else
        RunMmadTA(lIn, t1Bf_, t2Buf, CV_K, static_cast<uint32_t>(cb_), CV_BT);
#endif

        // ② dH[K,V] = k_c^T @ bf16(v_new)[BT,V]
        AicWaitFromAiv(kFlagVNew);
        // 按 chunk 奇偶取 k/left 槽（与 AIV staging 写入槽一致）
        GlobalTensor<bfloat16_t> &kIn = ((c & 1) != 0) ? kBf1_ : kBf_;

        // 满 chunk：k 也直接来自输入（AIV 不再写 kBf_）
        GlobalTensor<bfloat16_t> kTile = directInputs ? kIn_[(hk * tt->T + t0) * CV_K] : kIn;
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(kIn);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(vNewBf_);
#endif
#if PPFM_DH_CV
        // dH 的 C 直落 UB 槽（单槽），不再写 GM dHF_
        RunTiledTAUb(kTile, vNewBf_, CV_K, static_cast<uint32_t>(cb_), CV_BT,
                     UB_DH_CV, static_cast<uint16_t>(kFlagDhFree));
#else
        RunMmadTA(kTile, vNewBf_, dhBuf, CV_K, static_cast<uint32_t>(cb_), CV_BT);
#endif

        // ④ T2 已在上面（kFlagHalf1 之后）提前算完
#if !PPFM_DH_CV
#if PPFM_LEGACY_CACHEOPS
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(dhBuf);
#endif
#endif
#if PPFM_LEGACY_CACHEOPS && !PPFM_T2_CV
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                 DcciDst::CACHELINE_OUT>(t2Buf);
#endif
#if PPFM_LEGACY_CACHEOPS
        DataSyncBarrier<MemDsbT::DDR>();
#endif

        // dH 与 T2 都由 AIV 在**下一个 chunk 开头**使用，合并为一次跨核通知（省一次 flag
        // 往返）
        AicSetToAiv(kFlagDH);
#if PPFM_DIAG
        if (c < PPFM_DIAG_CHUNKS) {
#if PPFM_LEGACY_CACHEOPS
            DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(vTmpF_);
#endif
            PipeBarrier<PIPE_ALL>();
            diagG_.SetValue(static_cast<int32_t>(c), vTmpF_.GetValue(0));
            PipeBarrier<PIPE_ALL>();
        }
#endif
    }

    // A 行主：C[m,n] = A[m,k] @ B[k,n]（A/B 都是 bf16、行主；C 是 fp32 行主）
#if PPFM_TILE_MMAD
    // ---- 手写 tile 级（增量 A1）：GM→L1→L0A/L0B→MMAD→C 回写（落点仍是 GM）----

    // 目的：先用与 BlockMmad 相同的落点验证 tile/MMAD 数值一致；A5 的 L0C→UB 在 A2
    // 增量里接。
    // toUb=true 时 C 落 UB_EXT_F 区的共享槽（fixpipe SPLIT_M），否则仍落 gmC
    // CT = C 的元素类型：float（默认）或 bfloat16_t（fixpipe 直接按输入 dtype 量化，省掉
    // AIV 侧的"读回 fp32 → Cast → 写 bf16"整条回路，见 PPFM_T1_FIXPIPE_BF16）。
    template <class CT>
    __aicore__ inline void RunTiledNT(GlobalTensor<bfloat16_t> &gmA, GlobalTensor<bfloat16_t> &gmB,
                                      GlobalTensor<CT> &gmC, uint32_t m, uint32_t n, uint32_t k,
                                      bool toUb = false, bool keepA = false)
    {
        Catlass::Arch::Resource<MmArchTag> res;
        auto l1A = res.l1Buf.template GetBufferByByte<bfloat16_t>(TILED_L1_A_OFF);
        auto l1B = res.l1Buf.template GetBufferByByte<bfloat16_t>(TILED_L1_B_OFF);
        auto l0A = res.l0ABuf.template GetBufferByByte<bfloat16_t>(0);
        auto l0B = res.l0BBuf.template GetBufferByByte<bfloat16_t>(0);
        auto l0C = res.l0CBuf.template GetBufferByByte<float>(0);

        auto tA = tla::MakeTensor(gmA[0], tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k),
                                  Catlass::Arch::PositionGM{});
        auto tB = tla::MakeTensor(gmB[0], tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(k, n),
                                  Catlass::Arch::PositionGM{});
        auto tC = tla::MakeTensor(gmC[0], tla::MakeLayout<CT, Catlass::layout::RowMajor>(m, n),
                                  Catlass::Arch::PositionGM{});
        auto bA = GetTile(tA, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto bB = GetTile(tB, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto bC = GetTile(tC, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        auto tL1A = tla::MakeTensor(
            l1A, tla::MakeLayout<bfloat16_t, typename MmTileCopyNT::LayoutTagL1A>(TILED_L1_CAP_M, TILED_L1_CAP_K),
            Catlass::Arch::PositionL1{});
        auto tL1B = tla::MakeTensor(
            l1B, tla::MakeLayout<bfloat16_t, typename MmTileCopyNT::LayoutTagL1B>(TILED_L1_CAP_K, TILED_L1_CAP_N),
            Catlass::Arch::PositionL1{});
        typename MmTileCopyNT::template CopyGmToL1A<decltype(bA)> copyG2LA;
        typename MmTileCopyNT::template CopyGmToL1B<decltype(bB)> copyG2LB;
        // mm1 与 mm3 的 A 操作数都是同一个 `W`（同 shape、同 L1 槽）⇒ 第二次不必重搬。
        if (!keepA) {
            copyG2LA(tL1A, bA);
        }
        copyG2LB(tL1B, bB);
        // 跨流水必须用事件对（chunk_fwd_h_cube.h 的写法），PIPE_ALL 不保证 MTE1/M/FIX 次序
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        auto tL0A = tla::MakeTensor(
            l0A, tla::MakeLayout<bfloat16_t, typename MmTileCopyNT::LayoutTagL0A>(m, k),
            Catlass::Arch::PositionL0A{});
        auto tL0B = tla::MakeTensor(
            l0B, tla::MakeLayout<bfloat16_t, typename MmTileCopyNT::LayoutTagL0B>(k, n),
            Catlass::Arch::PositionL0B{});
        typename MmTileCopyNT::CopyL1ToL0A copyL2L0A;
        typename MmTileCopyNT::CopyL1ToL0B copyL2L0B;
        copyL2L0A(tL0A, GetTile(tL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k)));
        copyL2L0B(tL0B, GetTile(tL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n)));
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);

        auto tL0C = tla::MakeTensor(l0C, tla::MakeLayoutL0C(m, n), Catlass::Arch::PositionL0C{});
        MmTileMmadNT mmad;
        mmad(tL0C, tL0A, tL0B, m, n, k);
        SetFlag<HardEvent::M_FIX>(EVENT_ID2);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID2);

        if constexpr (std::is_same_v<CT, float>) {
            // fp32 C：950 可直接 L0C→UB（SPLIT_M），A2/A3 只能落 GM
            bool handled = false;
#if PPFM_ARCH_IS_950
            if (toUb) {
                // 写进 UB_EXT_F 区（64x128 fp32 = 32KB，正好是该区尺寸）。

                // SPLIT_M
                // 语义：整块的「前一半行」落在该地址的低半区、「后一半行」落高半区，
                // 与「段按连续半区分配 subcore」对齐 ⇒ 两个子核各读自己那半。
                AscendC::LocalTensor<float> vTmpUb(AscendC::TPosition::VECCALC, UB_EXT_F, CV_BT * CV_V);
                auto layoutUb = tla::MakeLayout<float, Catlass::layout::RowMajor>(m, n);
                auto tensorUb = tla::MakeTensor(vTmpUb, layoutUb, Catlass::Arch::PositionUB{});
                typename TiledCopyNTSplitUb::template CopyL0CToDst<decltype(tensorUb)> copyUb;
                copyUb(tensorUb, tL0C);
#if PPFM_VTMP_UB_DIAG
                typename MmTileCopyNT::template CopyL0CToDst<decltype(bC)> copyCRef;
                copyCRef(bC, tL0C, static_cast<uint8_t>(0));   // 诊断参照：同值再落一份 GM
#endif
                handled = true;
            }
#endif
            if (!handled) {
                MmCopyL0CToGm<decltype(bC)> copyC;
                // 注意： 必须走 3 参重载 (dst, src, unitFlag)：4 参会误选 (l0Batch, dstNdStride)
                //    批处理变体，l0Batch=0 ⇒ fixpipe 一个块都不搬，C 恒为初值 0。
                copyC(bC, tL0C, static_cast<uint8_t>(0));
            }
        } else {
            // bf16 C（PPFM_T1_FIXPIPE_BF16）：fixpipe 直接把 fp32 的 C 量化成 bf16 落 GM，
            // AIV 侧不再需要"读回 fp32 → Cast → 写 bf16"。
            MmCopyL0CToGm<decltype(bC)> copyC;
            copyC(bC, tL0C, static_cast<uint8_t>(0));
        }
        SetFlag<HardEvent::FIX_M>(EVENT_ID3);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID3);
        PipeBarrier<PIPE_ALL>();
    }

    // A 列主（A 在 GM 上是 [k,m] 列主，逻辑 [m,k]）
    __aicore__ inline void RunTiledTA(GlobalTensor<bfloat16_t> &gmA, GlobalTensor<bfloat16_t> &gmB,
                                      GlobalTensor<float> &gmC, uint32_t m, uint32_t n, uint32_t k)
    {
        Catlass::Arch::Resource<MmArchTag> res;
        auto l1A = res.l1Buf.template GetBufferByByte<bfloat16_t>(TILED_L1_A_OFF);
        auto l1B = res.l1Buf.template GetBufferByByte<bfloat16_t>(TILED_L1_B_OFF);
        auto l0A = res.l0ABuf.template GetBufferByByte<bfloat16_t>(0);
        auto l0B = res.l0BBuf.template GetBufferByByte<bfloat16_t>(0);
        auto l0C = res.l0CBuf.template GetBufferByByte<float>(0);

        auto tA = tla::MakeTensor(gmA[0], tla::MakeLayout<bfloat16_t, Catlass::layout::ColumnMajor>(m, k),
                                  Catlass::Arch::PositionGM{});
        auto tB = tla::MakeTensor(gmB[0], tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(k, n),
                                  Catlass::Arch::PositionGM{});
        auto tC = tla::MakeTensor(gmC[0], tla::MakeLayout<float, Catlass::layout::RowMajor>(m, n),
                                  Catlass::Arch::PositionGM{});
        auto bA = GetTile(tA, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto bB = GetTile(tB, tla::MakeCoord(0, 0), tla::MakeShape(k, n));
        auto bC = GetTile(tC, tla::MakeCoord(0, 0), tla::MakeShape(m, n));

        auto tL1A = tla::MakeTensor(
            l1A, tla::MakeLayout<bfloat16_t, typename MmTileCopyTA::LayoutTagL1A>(TILED_L1_CAP_M, TILED_L1_CAP_K),
            Catlass::Arch::PositionL1{});
        auto tL1B = tla::MakeTensor(
            l1B, tla::MakeLayout<bfloat16_t, typename MmTileCopyTA::LayoutTagL1B>(TILED_L1_CAP_K, TILED_L1_CAP_N),
            Catlass::Arch::PositionL1{});
        typename MmTileCopyTA::template CopyGmToL1A<decltype(bA)> copyG2LA;
        typename MmTileCopyTA::template CopyGmToL1B<decltype(bB)> copyG2LB;
        copyG2LA(tL1A, bA);
        copyG2LB(tL1B, bB);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        auto tL0A = tla::MakeTensor(
            l0A, tla::MakeLayout<bfloat16_t, typename MmTileCopyTA::LayoutTagL0A>(m, k),
            Catlass::Arch::PositionL0A{});
        auto tL0B = tla::MakeTensor(
            l0B, tla::MakeLayout<bfloat16_t, typename MmTileCopyTA::LayoutTagL0B>(k, n),
            Catlass::Arch::PositionL0B{});
        typename MmTileCopyTA::CopyL1ToL0A copyL2L0A;
        typename MmTileCopyTA::CopyL1ToL0B copyL2L0B;
        copyL2L0A(tL0A, GetTile(tL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k)));
        copyL2L0B(tL0B, GetTile(tL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n)));
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);

        auto tL0C = tla::MakeTensor(l0C, tla::MakeLayoutL0C(m, n), Catlass::Arch::PositionL0C{});
        MmTileMmadTA mmad;
        mmad(tL0C, tL0A, tL0B, m, n, k);
        SetFlag<HardEvent::M_FIX>(EVENT_ID2);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID2);

        MMTACopyL0CToGm<decltype(bC)> copyC;
        // 注意： 必须走 3 参重载 (dst, src, unitFlag)：4 参会误选 (l0Batch, dstNdStride)
        //    批处理变体，l0Batch=0 ⇒ fixpipe 一个块都不搬，C 恒为初值 0。
        copyC(bC, tL0C, static_cast<uint8_t>(0));
        SetFlag<HardEvent::FIX_M>(EVENT_ID3);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID3);
        PipeBarrier<PIPE_ALL>();
    }

#if PPFM_DH_CV
    // ---- /C 走 L0C->UB 的 TA 版本（mm2 = dH、mm4 = T2）----
    // L1/L0 装载与 MMAD 与 RunTiledTA 完全相同，只把出口从 GM 换成 UB 槽；
    // SPLIT_M 把 C 的 M 分成两半、分别写进两个 AIV 子核**各自 bank 的同一偏移**。
    __aicore__ inline void RunTiledTAUb(GlobalTensor<bfloat16_t> &gmA, GlobalTensor<bfloat16_t> &gmB,
                                        uint32_t m, uint32_t n, uint32_t k, int32_t ubOff,
                                        uint16_t freeFlag)
    {
        Catlass::Arch::Resource<MmArchTag> res;
        auto l1A = res.l1Buf.template GetBufferByByte<bfloat16_t>(TILED_L1_A_OFF);
        auto l1B = res.l1Buf.template GetBufferByByte<bfloat16_t>(TILED_L1_B_OFF);
        auto l0A = res.l0ABuf.template GetBufferByByte<bfloat16_t>(0);
        auto l0B = res.l0BBuf.template GetBufferByByte<bfloat16_t>(0);
        auto l0C = res.l0CBuf.template GetBufferByByte<float>(0);

        auto tA = tla::MakeTensor(gmA[0], tla::MakeLayout<bfloat16_t, Catlass::layout::ColumnMajor>(m, k),
                                  Catlass::Arch::PositionGM{});
        auto tB = tla::MakeTensor(gmB[0], tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(k, n),
                                  Catlass::Arch::PositionGM{});
        auto bA = GetTile(tA, tla::MakeCoord(0, 0), tla::MakeShape(m, k));
        auto bB = GetTile(tB, tla::MakeCoord(0, 0), tla::MakeShape(k, n));

        auto tL1A = tla::MakeTensor(
            l1A, tla::MakeLayout<bfloat16_t, typename MmTileCopyTA::LayoutTagL1A>(TILED_L1_CAP_M, TILED_L1_CAP_K),
            Catlass::Arch::PositionL1{});
        auto tL1B = tla::MakeTensor(
            l1B, tla::MakeLayout<bfloat16_t, typename MmTileCopyTA::LayoutTagL1B>(TILED_L1_CAP_K, TILED_L1_CAP_N),
            Catlass::Arch::PositionL1{});
        typename MmTileCopyTA::template CopyGmToL1A<decltype(bA)> copyG2LA;
        typename MmTileCopyTA::template CopyGmToL1B<decltype(bB)> copyG2LB;
        copyG2LA(tL1A, bA);
        copyG2LB(tL1B, bB);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        auto tL0A = tla::MakeTensor(
            l0A, tla::MakeLayout<bfloat16_t, typename MmTileCopyTA::LayoutTagL0A>(m, k),
            Catlass::Arch::PositionL0A{});
        auto tL0B = tla::MakeTensor(
            l0B, tla::MakeLayout<bfloat16_t, typename MmTileCopyTA::LayoutTagL0B>(k, n),
            Catlass::Arch::PositionL0B{});
        typename MmTileCopyTA::CopyL1ToL0A copyL2L0A;
        typename MmTileCopyTA::CopyL1ToL0B copyL2L0B;
        copyL2L0A(tL0A, GetTile(tL1A, tla::MakeCoord(0, 0), tla::MakeShape(m, k)));
        copyL2L0B(tL0B, GetTile(tL1B, tla::MakeCoord(0, 0), tla::MakeShape(k, n)));
        SetFlag<HardEvent::MTE1_M>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID1);

        auto tL0C = tla::MakeTensor(l0C, tla::MakeLayoutL0C(m, n), Catlass::Arch::PositionL0C{});
        MmTileMmadTA mmad;
        mmad(tL0C, tL0A, tL0B, m, n, k);
        SetFlag<HardEvent::M_FIX>(EVENT_ID2);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID2);

        // 槽归还信用：AIV 消费完上一代才置起（首 credit 由 AIV 侧 prime，尾 drain 在链末）
        CrossCoreWaitFlag<0x4, PIPE_FIX>(freeFlag);
        CrossCoreWaitFlag<0x4, PIPE_FIX>(
            static_cast<uint16_t>(freeFlag + PPFM_SUBFLAG_STRIDE));

        AscendC::LocalTensor<float> dHUb(AscendC::TPosition::VECCALC, ubOff, m * n);
        auto layoutUb = tla::MakeLayout<float, Catlass::layout::RowMajor>(m, n);
        auto tensorUb = tla::MakeTensor(dHUb, layoutUb, Catlass::Arch::PositionUB{});
        typename TiledCopyTASplitUb::template CopyL0CToDst<decltype(tensorUb)> copyUb;
        copyUb(tensorUb, tL0C);

        SetFlag<HardEvent::FIX_M>(EVENT_ID3);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID3);
        PipeBarrier<PIPE_ALL>();
    }
#endif  // PPFM_DH_CV
#endif  // PPFM_TILE_MMAD

    __aicore__ inline void RunMmadNT(GlobalTensor<bfloat16_t> &gmA, GlobalTensor<bfloat16_t> &gmB,
                                     GlobalTensor<float> &gmC, uint32_t m, uint32_t n, uint32_t k)
    {
        Catlass::Arch::Resource<MmArchTag> resource;
        MmBlockNT mm(resource);
        mm.preSetFlags();
        auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(m, k);
        auto layoutB = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(k, n);
        auto layoutC = tla::MakeLayout<float, Catlass::layout::RowMajor>(m, n);
        auto tA = tla::MakeTensor(gmA[0], layoutA, Catlass::Arch::PositionGM{});
        auto tB = tla::MakeTensor(gmB[0], layoutB, Catlass::Arch::PositionGM{});
        auto tC = tla::MakeTensor(gmC[0], layoutC, Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{m, n, k};
        auto bA = GetTile(tA, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto bB = GetTile(tB, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto bC = GetTile(tC, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        mm(bA, bB, bC, shape);
        mm.finalWaitFlags();
    }

    // A 列主（= 逻辑 [m,k] 在内存里按 [k,m] 存，正好对应"转置 A"）：C[m,n] = A^T @ B
    __aicore__ inline void RunMmadTA(GlobalTensor<bfloat16_t> &gmA, GlobalTensor<bfloat16_t> &gmB,
                                     GlobalTensor<float> &gmC, uint32_t m, uint32_t n, uint32_t k)
    {
        Catlass::Arch::Resource<MmArchTag> resource;
        MmBlockTA mm(resource);
        mm.preSetFlags();
        auto layoutA = tla::MakeLayout<bfloat16_t, Catlass::layout::ColumnMajor>(m, k);
        auto layoutB = tla::MakeLayout<bfloat16_t, Catlass::layout::RowMajor>(k, n);
        auto layoutC = tla::MakeLayout<float, Catlass::layout::RowMajor>(m, n);
        auto tA = tla::MakeTensor(gmA[0], layoutA, Catlass::Arch::PositionGM{});
        auto tB = tla::MakeTensor(gmB[0], layoutB, Catlass::Arch::PositionGM{});
        auto tC = tla::MakeTensor(gmC[0], layoutC, Catlass::Arch::PositionGM{});
        Catlass::GemmCoord shape{m, n, k};
        auto bA = GetTile(tA, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.k()));
        auto bB = GetTile(tB, tla::MakeCoord(0, 0), tla::MakeShape(shape.k(), shape.n()));
        auto bC = GetTile(tC, tla::MakeCoord(0, 0), tla::MakeShape(shape.m(), shape.n()));
        mm(bA, bB, bC, shape);
        mm.finalWaitFlags();
    }

    const PpFwdCtx &ctx_;
    GlobalTensor<bfloat16_t> wBf_;
    GlobalTensor<bfloat16_t> kBf_;
    GlobalTensor<bfloat16_t> lBf_;
    int32_t splitNum_ = 1;   // P5 列块数（1 或 2），与 AIV 侧同源（tiling.colSplit）
    int32_t cb_ = CV_V;      // P5 本工作项的列宽（128 或 64）
    int32_t colBase_ = 0;    // P5 本工作项列块起始列（AIC 侧只用于诊断/一致性）
    GlobalTensor<bfloat16_t> wIn_;   // PPFM_AIC_DIRECT_INPUTS：输入 w 视图
    GlobalTensor<bfloat16_t> kIn_;   // PPFM_AIC_DIRECT_INPUTS：输入 k 视图
    GlobalTensor<bfloat16_t> kBf1_;  // 
    GlobalTensor<bfloat16_t> lBf1_;  // 
    GlobalTensor<bfloat16_t> hBf_;
    GlobalTensor<bfloat16_t> mBf_;
    GlobalTensor<bfloat16_t> vNewBf_;
    GlobalTensor<bfloat16_t> t1Bf_;
    GlobalTensor<float> vTmpF_;
    GlobalTensor<float> dHF_;
    GlobalTensor<float> dHF1_;
    GlobalTensor<float> t1F_;
    GlobalTensor<float> t2F_;
    GlobalTensor<float> t2F1_;
    GlobalTensor<int64_t> cuGm_;
#if PPFM_DIAG
    GlobalTensor<float> diagG_;      // 每核 4 KiB 诊断区（AIV epilogue 会搬到 hm）
#endif
};

} // namespace GDN

#endif  // PREF_PROCESS_FWD_KERNEL_MERGED_CUBE_H
