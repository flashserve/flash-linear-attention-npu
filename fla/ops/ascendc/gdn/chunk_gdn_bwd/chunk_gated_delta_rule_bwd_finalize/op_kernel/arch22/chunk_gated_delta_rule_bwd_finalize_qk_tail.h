#ifndef GDN_BWD_FINALIZE_A2_QK_TAIL_H
#define GDN_BWD_FINALIZE_A2_QK_TAIL_H

#include "kernel_operator.h"

namespace GDN {

// 私有参数，不进入公开 Tiling ABI。leaf 不拥有 pipe、分配、事件或任务循环。
struct A2FinalizeQkTailArgs {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR qRstd;
    GM_ADDR kRstd;
    GM_ADDR dq1;
    GM_ADDR dk1;
    GM_ADDR dq;
    GM_ADDR dk;
    bool useQkL2norm;
};

template <typename QT>
struct A2FinalizeQkTailViews {
    AscendC::LocalTensor<QT> qInput, dqInput, dkInput, dkWyInput, kInput;
    AscendC::LocalTensor<float> gradQ, gradK, x, product, sum, reduceWork, broadcast;
    AscendC::LocalTensor<float> qRstdInput{}, kRstdInput{};
};

template <typename QT>
struct A2FinalizeQkTailGm {
    AscendC::GlobalTensor<QT> qGm, kGm, dq1Gm, dk1Gm, dk2Gm, dqGm, dkGm;
    AscendC::GlobalTensor<float> qRstdGm, kRstdGm;

    __aicore__ inline void Init(const A2FinalizeQkTailArgs &args, GM_ADDR dkWy)
    {
        qGm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(args.q));
        kGm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(args.k));
        dq1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(args.dq1));
        dk1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(args.dk1));
        dk2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(dkWy));
        dqGm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(args.dq));
        dkGm.SetGlobalBuffer(reinterpret_cast<__gm__ QT *>(args.dk));
        if (args.useQkL2norm) {
            qRstdGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.qRstd));
            kRstdGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(args.kRstd));
        }
    }
};

template <typename DT>
__aicore__ inline void A2FinalizeCopyIn(
    const AscendC::LocalTensor<DT> &dst, const AscendC::GlobalTensor<DT> &src,
    int64_t offset, uint32_t count)
{
    AscendC::DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(DT)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<DT> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src[offset], copy, pad);
}

template <typename QT, bool PREFETCH_RSTD = false>
__aicore__ inline void A2FinalizeNormTile(
    const AscendC::LocalTensor<float> &grad,
    const AscendC::LocalTensor<QT> &input,
    const AscendC::LocalTensor<float> &x,
    const AscendC::LocalTensor<float> &product,
    const AscendC::LocalTensor<float> &sum,
    const AscendC::LocalTensor<float> &reduceWork,
    const AscendC::LocalTensor<float> &broadcast,
    const AscendC::GlobalTensor<float> &rstd, int64_t offset, uint32_t rows,
    const AscendC::LocalTensor<float> &rstdInput)
{
    constexpr uint32_t K = 128;
    AscendC::Cast(x, input, AscendC::RoundMode::CAST_NONE, rows * K);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(product, x, grad, rows * K);
    AscendC::PipeBarrier<PIPE_V>();
    // 保留每行 64+64 partial 的布局；两批分离写之间保守保留 V barrier。
    AscendC::WholeReduceSum(reduceWork, product, 64, rows, 8, 1, 16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum(reduceWork[1], product[64], 64, rows, 8, 1, 16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum(sum, reduceWork, 2, rows, 1, 1, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Brcb(broadcast, sum, (rows + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(x, x, broadcast, 64, rows, {1, 1, 0, 16, 16, 1});
    AscendC::Mul(x[64], x[64], broadcast, 64, rows, {1, 1, 0, 16, 16, 1});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(grad, grad, x, rows * K);
    if constexpr (PREFETCH_RSTD) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(broadcast, rstdInput, (rows + 7) / 8, {1, 8});
    } else {
        AscendC::PipeBarrier<PIPE_ALL>();
        A2FinalizeCopyIn(sum, rstd, offset, rows);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Brcb(broadcast, sum, (rows + 7) / 8, {1, 8});
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(grad, grad, broadcast, 64, rows, {1, 1, 0, 16, 16, 1});
    AscendC::Mul(grad[64], grad[64], broadcast, 64, rows, {1, 1, 0, 16, 16, 1});
    AscendC::PipeBarrier<PIPE_V>();
}

// chunk 私有 opt-in。Q/K 为相邻的两组独立行，每行保持原归约与乘减顺序。
template <typename QT>
__aicore__ inline void A2FinalizeNormQkTile(
    const A2FinalizeQkTailViews<QT> &views, uint32_t rows)
{
    constexpr uint32_t K = 128;
    const uint32_t count = rows * K;
    const uint32_t normRows = 2 * rows;
    const auto &grad = views.gradQ;
    const auto &x = views.x;
    const auto &product = views.product;
    const auto &sum = views.sum;
    const auto &reduceWork = views.reduceWork;
    const auto &broadcast = views.broadcast;
    AscendC::Cast(x, views.qInput, AscendC::RoundMode::CAST_NONE, count);
    AscendC::Cast(x[count], views.kInput, AscendC::RoundMode::CAST_NONE, count);
    // calcC 内的 dk/dkWy/k 已全部读完，之后才能整块借给 product。
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(product, x, grad, 2 * count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum(reduceWork, product, 64, normRows, 8, 1, 16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum(reduceWork[1], product[64], 64, normRows, 8, 1, 16);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum(sum, reduceWork, 2, normRows, 1, 1, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Brcb(broadcast, sum, (normRows + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(x, x, broadcast, 64, normRows, {1, 1, 0, 16, 16, 1});
    AscendC::Mul(x[64], x[64], broadcast, 64, normRows, {1, 1, 0, 16, 16, 1});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(grad, grad, x, 2 * count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Brcb(broadcast, views.qRstdInput, (rows + 7) / 8, {1, 8});
    // 短尾时 K 的目的区会覆盖 Q 的 padded 行，先完成上一批写入。
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Brcb(broadcast[rows * 8], views.kRstdInput, (rows + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(grad, grad, broadcast, 64, normRows, {1, 1, 0, 16, 16, 1});
    AscendC::Mul(grad[64], grad[64], broadcast, 64, normRows, {1, 1, 0, 16, 16, 1});
    AscendC::PipeBarrier<PIPE_V>();
}

// 原 32 行 Tail 的搬运/加法/舍入/norm 顺序；DKwy offset 独立于全序列地址。
template <typename QT, bool USE_QK_L2NORM, bool PREFETCH_RSTD = false, bool FUSE_QK_NORM = false>
__aicore__ inline void A2FinalizeQkTailRows(
    const A2FinalizeQkTailGm<QT> &gm, const A2FinalizeQkTailViews<QT> &views,
    int64_t scalarOffset, int64_t dkWyOffset, uint32_t rows)
{
    static_assert(!FUSE_QK_NORM || (USE_QK_L2NORM && PREFETCH_RSTD),
                  "fused QK norm requires norm and prefetched rstd views");
    constexpr uint32_t K = 128;
    const uint32_t count = rows * K;
    const int64_t offset = scalarOffset * K;
    const auto &qInput = views.qInput;
    const auto &dqInput = views.dqInput;
    const auto &dkInput = views.dkInput;
    const auto &dkWyInput = views.dkWyInput;
    const auto &kInput = views.kInput;
    const auto &gradQ = views.gradQ;
    const auto &gradK = views.gradK;
    const auto &x = views.x;
    const auto &product = views.product;
    const auto &sum = views.sum;
    const auto &reduceWork = views.reduceWork;
    const auto &broadcast = views.broadcast;
    const auto &qGm = gm.qGm;
    const auto &kGm = gm.kGm;
    const auto &dq1Gm = gm.dq1Gm;
    const auto &dk1Gm = gm.dk1Gm;
    const auto &dk2Gm = gm.dk2Gm;
    const auto &dqGm = gm.dqGm;
    const auto &dkGm = gm.dkGm;
    const auto &qRstdGm = gm.qRstdGm;
    const auto &kRstdGm = gm.kRstdGm;
    AscendC::DataCopy(dqInput, dq1Gm[offset], count);
    AscendC::DataCopy(dkInput, dk1Gm[offset], count);
    AscendC::DataCopy(dkWyInput, dk2Gm[dkWyOffset], count);
    if constexpr (USE_QK_L2NORM) {
        AscendC::DataCopy(qInput, qGm[offset], count);
        AscendC::DataCopy(kInput, kGm[offset], count);
        if constexpr (PREFETCH_RSTD) {
            A2FinalizeCopyIn(views.qRstdInput, qRstdGm, scalarOffset, rows);
            A2FinalizeCopyIn(views.kRstdInput, kRstdGm, scalarOffset, rows);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::Cast(gradK, dkInput, AscendC::RoundMode::CAST_NONE, count);
    AscendC::Cast(x, dkWyInput, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(gradK, gradK, x, count);
    AscendC::PipeBarrier<PIPE_V>();
    // 原组合链的 add 输出为 BF16；norm 输入保留该 cast 边界。
    AscendC::Cast(dkInput, gradK, AscendC::RoundMode::CAST_RINT, count);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (USE_QK_L2NORM) {
        AscendC::Cast(gradQ, dqInput, AscendC::RoundMode::CAST_NONE, count);
        if constexpr (FUSE_QK_NORM) {
            AscendC::Cast(gradQ[count], dkInput, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            A2FinalizeNormQkTile(views, rows);
            AscendC::Cast(dqInput, gradQ, AscendC::RoundMode::CAST_RINT, count);
            AscendC::Cast(dkInput, gradQ[count], AscendC::RoundMode::CAST_RINT, count);
        } else {
            AscendC::Cast(gradK, dkInput, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            A2FinalizeNormTile<QT, PREFETCH_RSTD>(gradQ, qInput, x, product, sum, reduceWork, broadcast,
                                               qRstdGm, scalarOffset, rows, views.qRstdInput);
            A2FinalizeNormTile<QT, PREFETCH_RSTD>(gradK, kInput, x, product, sum, reduceWork, broadcast,
                                               kRstdGm, scalarOffset, rows, views.kRstdInput);
            AscendC::Cast(dqInput, gradQ, AscendC::RoundMode::CAST_RINT, count);
            AscendC::Cast(dkInput, gradK, AscendC::RoundMode::CAST_RINT, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    // 等待最终 Cast 写完 UB，再允许 MTE3 读取同一输出缓冲。
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopy(dqGm[offset], dqInput, count);
    AscendC::DataCopy(dkGm[offset], dkInput, count);
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace GDN

#endif
