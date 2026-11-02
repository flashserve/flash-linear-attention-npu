// Native CANN failures only: no --wrap, no simulated reporting, no kernels.
#include <dlfcn.h>
#include "opdev/op_executor.h"
#include "fla/ops/ascendc/cp/pre_process_bwd_kernel_merged/op_host/op_api/aclnn_pre_process_bwd_kernel_merged.h"

extern "C" __attribute__((visibility("default"))) const char *cp_native_executor_owner()
{
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void *>(&CommonOpExecutorRun), &info)) return "";
    return info.dli_fname;
}

extern "C" __attribute__((visibility("default"))) int cp_native_executor_case(int scenario, int throughL2)
{
    aclOpExecutor *executor = nullptr;
    if (scenario == 1) {
        UniqueExecutor unique("CP_NATIVE_EXECUTOR_UT");
        unique.ReleaseTo(&executor);
    }
    const uint64_t workspaceSize = scenario == 1 ? 1 : 0;
    // For scenario1 ownership has been transferred. The native run failure
    // path owns cleanup; this probe must not delete the executor again.
    return throughL2 ? aclnnPreProcessBwdKernelMerged(nullptr, workspaceSize, executor, nullptr)
                     : CommonOpExecutorRun(nullptr, workspaceSize, executor, nullptr);
}
