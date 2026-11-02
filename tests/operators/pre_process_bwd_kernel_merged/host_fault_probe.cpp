// Link-time doubles only. Both production L2 and L0 are compiled unchanged.
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "aclnn/acl_meta.h"
#include "base/err_msg.h"
#include "opdev/op_arg_def.h"
#include "opdev/op_executor.h"
#include "fla/ops/ascendc/cp/pre_process_bwd_kernel_merged/op_host/op_api/aclnn_pre_process_bwd_kernel_merged.h"

namespace {
thread_local int mode;
thread_local bool reportLower;
// ctor, contiguous, convert, launcher, run, injected hits, wrong argument,
// workspace published, executor published
thread_local std::array<int, 9> counts;
thread_local std::array<const aclTensor *, 7> expectedInputs;
thread_local std::vector<int> visitedInputs;
thread_local const aclTensor *convertedTensor;
thread_local const aclIntArray *expectedArray;

void SimulatedFirstError(const char *parameter)
{
    if (!reportLower) return;
    const std::vector<const char *> keys = {"param_name", "op_name", "reason"};
    const std::vector<const char *> values = {parameter, "CP_UT_SIMULATED_LOWER",
        "test double reports a simulated lower-layer error; this is boundary evidence only"};
    REPORT_PREDEFINED_ERR_MSG("EZ0037", keys, values);
}

struct Inputs {
    std::vector<aclTensor *> descriptors;
    std::vector<std::vector<uint8_t>> storage;
    aclIntArray *segment = nullptr;
    aclTensor *Make(std::initializer_list<int64_t> shape, aclDataType dtype)
    {
        std::vector<int64_t> dims(shape), strides(dims.size(), 1);
        for (size_t index = dims.size() - 1; index > 0; --index) {
            strides[index - 1] = strides[index] * dims[index];
        }
        const auto elements = static_cast<size_t>(dims[0] * strides[0]);
        const size_t bytes = dtype == ACL_FLOAT ? 4 : dtype == ACL_INT64 ? 8 : 2;
        storage.emplace_back(elements * bytes, 0);
        auto *tensor = aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0,
            ACL_FORMAT_ND, dims.data(), dims.size(), storage.back().data());
        if (!tensor) throw std::runtime_error("UT aclCreateTensor failed");
        descriptors.push_back(tensor);
        return tensor;
    }
    ~Inputs()
    {
        if (segment) aclDestroyIntArray(segment);
        for (auto *tensor : descriptors) aclDestroyTensor(tensor);
    }
};
}

extern "C" void __real__ZN14UniqueExecutorC1EPKc(void *, const char *);
extern "C" void __wrap__ZN14UniqueExecutorC1EPKc(void *object, const char *name)
{
    ++counts[0];
    __real__ZN14UniqueExecutorC1EPKc(object, name);
}

extern "C" const aclTensor *__wrap__ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor(
    const aclTensor *tensor, aclOpExecutor *)
{
    ++counts[1];
    int parameter = -1;
    for (size_t index = 0; index < expectedInputs.size(); ++index) {
        if (tensor == expectedInputs[index]) parameter = static_cast<int>(index);
    }
    visitedInputs.push_back(parameter);
    if (parameter < 0) ++counts[6];
    if (mode >= 10 && mode <= 16 && parameter == mode - 10) {
        ++counts[5];
        const char *names[] = {"q", "k", "w", "d_o", "dv", "g", "gk"};
        SimulatedFirstError(names[parameter]);
        return nullptr;
    }
    // Successful preprocessing is also a double: return the valid descriptor
    // unchanged so no unrelated kernel or real conversion is launched.
    return tensor;
}

extern "C" const aclTensor *__wrap__ZN13aclOpExecutor15ConvertToTensorEPK11aclIntArrayN2ge8DataTypeE(
    aclOpExecutor *, const aclIntArray *array, ge::DataType dtype)
{
    ++counts[2];
    if (array != expectedArray || dtype != ge::DT_INT64) ++counts[6];
    if (mode == 17) {
        ++counts[5];
        SimulatedFirstError("cu_seqlens");
        return nullptr;
    }
    return convertedTensor;
}

extern "C" int __wrap__Z25CreatAiCoreKernelLauncherPKcjP13aclOpExecutorPN2op12OpArgContextE(
    const char *, unsigned int, aclOpExecutor *, op::OpArgContext *arguments)
{
    ++counts[3];
    if (arguments == nullptr) ++counts[6];
    op::DestroyOpArgContext(arguments);
    ++counts[5];
    SimulatedFirstError("launcher");
    return 361001;
}

extern "C" int __wrap__Z19CommonOpExecutorRunPvmP13aclOpExecutorS_(void *, uint64_t, void *, void *)
{
    ++counts[4];
    ++counts[5];
    SimulatedFirstError("executor_run");
    return 361001;
}

extern "C" __attribute__((visibility("default"))) int cp_boundary_case(int requestedMode, int simulatedReport)
{
    mode = 0;
    if (requestedMode != 4 && (requestedMode < 10 || requestedMode > 21)) return -1;
    counts.fill(0);
    expectedInputs.fill(nullptr);
    visitedInputs.clear();
    reportLower = simulatedReport != 0;
    if (requestedMode == 4) {
        UniqueExecutor executor("CP_UT");
        counts.fill(0);
        mode = requestedMode;
        return aclnnPreProcessBwdKernelMerged(nullptr, 0, executor.get(), nullptr);
    }
    Inputs inputs;
    auto *q = inputs.Make({1, 2, 8, 128}, ACL_BF16);
    auto *k = inputs.Make({1, 2, 8, 128}, ACL_BF16);
    auto *w = inputs.Make({1, 4, 8, 128}, ACL_BF16);
    auto *dO = inputs.Make({1, 4, 8, 128}, ACL_BF16);
    auto *dv = inputs.Make({1, 4, 8, 128}, ACL_BF16);
    auto *g = requestedMode == 15 ? inputs.Make({1, 4, 8}, ACL_BF16) : nullptr;
    auto *gk = requestedMode == 16 ? inputs.Make({1, 4, 8, 128}, ACL_BF16) : nullptr;
    auto *dhm = inputs.Make({4, 128, 256}, ACL_FLOAT);
    expectedInputs = {q, k, w, dO, dv, g, gk};
    const bool withSegment = requestedMode == 17 || requestedMode == 19;
    if (withSegment) {
        const int64_t segment[] = {0, 8};
        inputs.segment = aclCreateIntArray(segment, 2);
        if (!inputs.segment) throw std::runtime_error("UT aclCreateIntArray failed");
        expectedArray = inputs.segment;
        convertedTensor = inputs.Make({2}, ACL_INT64);
    }
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    counts.fill(0);
    mode = requestedMode;
    int status = aclnnPreProcessBwdKernelMergedGetWorkspaceSize(q, k, w, dO, dv,
        g, gk, inputs.segment, 1.0, 64, dhm,
        requestedMode == 20 ? nullptr : &workspaceSize, requestedMode == 21 ? nullptr : &executor);
    counts[7] = workspaceSize != 0;
    counts[8] = executor != nullptr;
    mode = 0;
    return status;
}

extern "C" __attribute__((visibility("default"))) int cp_boundary_metric(int index)
{
    return counts.at(static_cast<size_t>(index));
}

extern "C" __attribute__((visibility("default"))) int cp_boundary_visited(int index)
{
    return visitedInputs.at(static_cast<size_t>(index));
}
