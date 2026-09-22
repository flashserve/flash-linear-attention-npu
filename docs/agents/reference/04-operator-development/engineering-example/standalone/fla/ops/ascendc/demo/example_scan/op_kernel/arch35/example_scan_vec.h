/**
 * 示例文件：.../op_kernel/arch35/example_scan_vec.h
 *
 * 注意事项：
 *   1. A5 的 UB 更大，允许更大的 VF/向量粒度，但 UB 生命周期规则与 arch22 相同
 *      （TQue 语义、跨 pipe 事件、手工 buffer 显式建模）。
 *   2. A5 上如果有专用的 RegBase/VF 通路，必须与 host 侧 arch35 tiling 常量配套，
 *      并用 profiling 证明重叠确实发生（声明 ping/pong 就要有两份物理存储与重叠证据）。
 *   3. 首个 Stage 的初始化、尾块处理、空任务路径要与 arch22 保持同样的计数配平，
 *      不允许"某平台跳过 ready/free"。
 *   4. 任何平台相关宏只在本目录内使用，不要污染根目录共用头。
 */

#ifndef EXAMPLE_SCAN_VEC_ARCH35_H
#define EXAMPLE_SCAN_VEC_ARCH35_H

template <typename XT, typename GT, typename Policy>
class ExampleScanVector {
public:
    __aicore__ inline void Init(const ExampleScanArgs &args, AscendC::TPipe *pipe)
    {
        args_ = &args;
        pipe_ = pipe;
        // pipe_->InitBuffer(...)：份数取 arch35 常量。
    }

    __aicore__ inline void Process()
    {
        // A5 的 S0/S2 实现；Save 档额外搬出由 if constexpr 消除。
    }

private:
    const ExampleScanArgs *args_ = nullptr;
    AscendC::TPipe *pipe_ = nullptr;
};

#endif // EXAMPLE_SCAN_VEC_ARCH35_H
