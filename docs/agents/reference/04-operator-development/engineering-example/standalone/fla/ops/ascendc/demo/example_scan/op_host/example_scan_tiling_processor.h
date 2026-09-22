/**
 * 示例文件：fla/ops/ascendc/demo/example_scan/op_host/example_scan_tiling_processor.h
 *
 * 注意事项：
 *   1. 本文件只做"任务切分 + offset + workspace 区域"计算，不做张量校验（校验在 *_tiling.cpp）。
 *   2. 实现直接写在本头文件里（header-only）：host UT 只 include 本文件就能调用
 *      ExampleScanTilingProcessor，不需要造完整 TilingContext，也不需要额外链接目标。
 *      如果单独拆出 .cpp，UT 必须再链接该目标，且容易与算子编译目标重复定义。
 *   3. 每个 workspace 区域都要给出 size 与用途，并在注释里写清使用方与生命周期；
 *      复用同一区域时必须写明复用条件。
 *   4. 平台常量（tile、每核 head 数）从 arch22|arch35 的 tiling_impl.h 引入，不在这里写死。
 *   5. 余数分配规则必须写出来（示例：余数依次分给前面的 core），否则分核不可复现。
 *   6. 所有乘法都要做上界检查：长序列 + 大 D 时 uint64 乘法溢出会让 workspace 偏小，
 *      在设备侧表现为越界写而不是报错。返回 false 表示无法生成合法调度，上层必须报错退出。
 */

#ifndef EXAMPLE_SCAN_TILING_PROCESSOR_H
#define EXAMPLE_SCAN_TILING_PROCESSOR_H

#include <cstdint>
#include <limits>

#include "example_scan_output_mask.h"

namespace optiling {
namespace example_scan_detail {

inline bool MulChecked(uint64_t lhs, uint64_t rhs, uint64_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

} // namespace example_scan_detail

struct ExampleScanScheduleContext {
    uint64_t batch = 0;
    uint64_t headNum = 0;
    uint64_t chunksPerSequence = 0;
    uint64_t dim = 0;
    uint64_t chunkSize = 0;
    uint64_t coreNum = 0;
    uint64_t libApiWorkspaceBytes = 0;
    uint32_t outputMode = EXAMPLE_SCAN_OUTPUT_MODE_NONE;
    uint32_t dtypeBytes = 2;   // BF16/FP16 都是 2 字节
};

struct ExampleScanSchedule {
    uint32_t usedCoreNum = 0;
    uint32_t headsPerCore = 0;
    uint64_t stateSlotBytes = 0;    // save 档才分配；none 档为 0
    uint64_t normScratchBytes = 0;
    uint64_t workspaceBytes = 0;    // 含平台 LibApi 预留
};

class ExampleScanTilingProcessor {
public:
    explicit ExampleScanTilingProcessor(const ExampleScanScheduleContext &context)
        : context_(context) {}

    bool Process(ExampleScanSchedule &schedule) const
    {
        using example_scan_detail::MulChecked;
        if (context_.headNum == 0 || context_.chunksPerSequence == 0 || context_.coreNum == 0) {
            return false;
        }

        // 按 head 分核：H <= coreNum 时只用 H 个核；否则每核 head 数向上取整，
        // 余数依次分给前面的 core（同一序列不拆到多个核，保证状态递推顺序）。
        schedule.usedCoreNum = static_cast<uint32_t>(
            context_.headNum < context_.coreNum ? context_.headNum : context_.coreNum);
        schedule.headsPerCore = static_cast<uint32_t>(
            (context_.headNum + schedule.usedCoreNum - 1) / schedule.usedCoreNum);

        uint64_t stateElems = 0;
        if (context_.outputMode == EXAMPLE_SCAN_OUTPUT_MODE_SAVE) {
            // save 档：每核一段 [headsPerCore, chunksPerSequence, D] 的状态。
            if (!MulChecked(schedule.usedCoreNum, schedule.headsPerCore, stateElems) ||
                !MulChecked(stateElems, context_.chunksPerSequence, stateElems) ||
                !MulChecked(stateElems, context_.dim, stateElems) ||
                !MulChecked(stateElems, context_.dtypeBytes, schedule.stateSlotBytes)) {
                return false;
            }
        }

        uint64_t scratchElems = 0;
        if (!MulChecked(schedule.usedCoreNum, context_.chunksPerSequence, scratchElems) ||
            !MulChecked(scratchElems, context_.dim, scratchElems) ||
            !MulChecked(scratchElems, sizeof(float), schedule.normScratchBytes) ||
            context_.libApiWorkspaceBytes >
                std::numeric_limits<uint64_t>::max() - schedule.stateSlotBytes) {
            return false;
        }
        schedule.workspaceBytes = context_.libApiWorkspaceBytes + schedule.stateSlotBytes;
        if (schedule.workspaceBytes > std::numeric_limits<uint64_t>::max() - schedule.normScratchBytes) {
            return false;
        }
        schedule.workspaceBytes += schedule.normScratchBytes;
        return true;
    }

private:
    ExampleScanScheduleContext context_;
};

} // namespace optiling

#endif // EXAMPLE_SCAN_TILING_PROCESSOR_H
