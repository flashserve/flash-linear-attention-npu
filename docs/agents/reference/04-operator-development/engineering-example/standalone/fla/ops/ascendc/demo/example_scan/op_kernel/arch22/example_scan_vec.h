/**
 * 示例文件：.../op_kernel/arch22/example_scan_vec.h
 *
 * 注意事项：
 *   1. UB 生命周期按 TPipe/TQue 语义管理：AllocTensor -> 写 -> EnQue -> DeQue -> FreeTensor；
 *      FreeTensor 之后不得再读写，也不得把 tensor 留给后续异步 pipe 使用。
 *   2. MTE2→VEC、VEC→MTE3 必须各有事件同步；同一 UB 上的二次读写若同属 V pipe，
 *      才用 PipeBarrier<PIPE_V>()。
 *   3. 手工切分 LocalTensor（TBuf/切片/别名）时要在注释里说明区间与同步关系，
 *      sanitizer 对未建模的手工流水会给保守报告。
 *   4. 尾块/varlen 无效区按有效长度裁剪或写中性值，不能按整 tile 读写。
 *   5. 与 AIC 的交接只用 host 定义好的任务顺序与 workspace 协议，不依赖 core 启动顺序。
 */

#ifndef EXAMPLE_SCAN_VEC_ARCH22_H
#define EXAMPLE_SCAN_VEC_ARCH22_H

template <typename XT, typename GT, typename Policy>
class ExampleScanVector {
public:
    __aicore__ inline void Init(const ExampleScanArgs &args, AscendC::TPipe *pipe)
    {
        args_ = &args;
        pipe_ = pipe;
        // pipe_->InitBuffer(...)：UB 份数与 arch22 tiling 常量一致。
    }

    __aicore__ inline void Process()
    {
        // S0：算归一化系数；S2：写回 y，Save 档额外写 state/x_norm。
        // if constexpr (Policy::kOutput == ExampleScanOutputMode::Save) { ... }
    }

private:
    const ExampleScanArgs *args_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
};

#endif // EXAMPLE_SCAN_VEC_ARCH22_H
