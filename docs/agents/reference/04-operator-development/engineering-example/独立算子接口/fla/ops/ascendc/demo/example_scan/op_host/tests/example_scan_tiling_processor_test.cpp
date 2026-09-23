/**
 * 示例文件：.../op_host/tests/example_scan_tiling_processor_test.cpp
 *
 * 注意事项：
 *   1. 单测直接调用 tiling 计算入口与档位常量，不构造 TilingContext；这样任何平台都能跑。
 *   2. 必须覆盖：每个可达档位的 workspace 计算、分核余数分配、溢出保护、档位与区域数量一致。
 *   3. 输出掩码的互斥与全覆盖由 static_assert 静态保护，单测再测"档位 -> 掩码"的映射，
 *      两者缺一：只靠 static_assert 无法发现档位映射写错。
 *   4. 断言要带上下文（哪个档位、哪些输入），失败时能直接定位。
 *   5. 这里不测精度；精度、性能、内存检查在 tests/atk/<算子>/。
 */

#include "example_scan_tiling_processor.h"

#include <cstdio>
#include <cstdint>
#include <limits>

namespace {

using optiling::ExampleScanOutputMode;
using optiling::ExampleScanSchedule;
using optiling::ExampleScanScheduleContext;
using optiling::ExampleScanTilingProcessor;

int g_failures = 0;

void Expect(bool condition, const char *message)
{
    if (!condition) {
        ++g_failures;
        std::printf("[FAIL] %s\n", message);
    }
}

void TestNoneModeAllocatesNoStateSlot()
{
    ExampleScanScheduleContext context;
    context.batch = 1;
    context.headNum = 8;
    context.chunksPerSequence = 4;
    context.dim = 128;
    context.coreNum = 20;
    context.libApiWorkspaceBytes = 4096;
    context.outputMode = ExampleScanOutputMode::EXAMPLE_SCAN_OUTPUT_MODE_NONE;

    ExampleScanSchedule schedule;
    Expect(ExampleScanTilingProcessor(context).Process(schedule), "none 档调度应成功");
    Expect(schedule.stateSlotBytes == 0, "none 档不得为可选输出申请 workspace");
    Expect(schedule.usedCoreNum == 8, "head 数小于核数时 usedCoreNum 应等于 head 数");
    Expect(schedule.headsPerCore == 1, "每核 head 数应为 1");
    Expect(schedule.workspaceBytes == 4096, "none 档 workspace 只含平台预留");
}

void TestSaveModeAllocatesStateSlot()
{
    ExampleScanScheduleContext context;
    context.headNum = 96;
    context.chunksPerSequence = 16;
    context.dim = 128;
    context.coreNum = 20;
    context.libApiWorkspaceBytes = 4096;
    context.outputMode = ExampleScanOutputMode::EXAMPLE_SCAN_OUTPUT_MODE_SAVE;

    ExampleScanSchedule schedule;
    Expect(ExampleScanTilingProcessor(context).Process(schedule), "save 档调度应成功");
    // 20 核 * 5 head/核 * 16 chunk * 128 * 2B 状态 + 20 核 * 16 chunk * 128 * 4B scratch + 预留。
    const uint64_t expectedState = 20ULL * 5ULL * 16ULL * 128ULL * 2ULL;
    const uint64_t expectedScratch = 20ULL * 16ULL * 128ULL * sizeof(float);
    Expect(schedule.headsPerCore == 5, "96 head / 20 核应向上取整为 5");
    Expect(schedule.stateSlotBytes == expectedState, "save 档状态区大小与设计不一致");
    Expect(schedule.workspaceBytes == 4096ULL + expectedState + expectedScratch,
           "save 档 workspace 总量与设计不一致");
}

void TestRejectsEmptyWork()
{
    ExampleScanScheduleContext context;
    context.headNum = 0;
    context.chunksPerSequence = 4;
    context.coreNum = 20;
    ExampleScanSchedule schedule;
    Expect(!ExampleScanTilingProcessor(context).Process(schedule),
           "head 数为 0 时必须返回失败而不是给出默认调度");
}

void TestMaskMappingMatchesMode()
{
    Expect(optiling::EXAMPLE_SCAN_NONE_OUTPUT_MASK == optiling::EXAMPLE_SCAN_REQUIRED_OUTPUT_MASK,
           "none 档只能包含必选输出");
    Expect((optiling::EXAMPLE_SCAN_SAVE_OUTPUT_MASK & optiling::EXAMPLE_SCAN_OPTIONAL_OUTPUT_MASK) ==
               optiling::EXAMPLE_SCAN_OPTIONAL_OUTPUT_MASK,
           "save 档必须包含全部可选输出");
}

} // namespace

int main()
{
    TestNoneModeAllocatesNoStateSlot();
    TestSaveModeAllocatesStateSlot();
    TestRejectsEmptyWork();
    TestMaskMappingMatchesMode();
    if (g_failures != 0) {
        std::printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
