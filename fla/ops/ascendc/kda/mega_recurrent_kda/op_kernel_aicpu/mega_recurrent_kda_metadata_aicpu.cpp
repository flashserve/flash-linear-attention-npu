#include "cpu_kernel.h"
#include <algorithm>
#include <cstdint>

namespace aicpu {
namespace {
template <typename Index>
uint32_t BuildPlan(Tensor &prefix, Tensor &plan)
{
    const auto *ends = static_cast<const Index *>(prefix.GetData());
    auto *out = static_cast<int32_t *>(plan.GetData());
    const int64_t batch = prefix.NumElements() - 1;
    const int64_t entries = plan.NumElements() - 2;
    if (!ends || !out || batch < 1 || batch > INT32_MAX || entries < 1) {
        return 1;
    }
    int64_t tile = 0;
    uint32_t longest = 0;
    bool crosses = false;
    for (int64_t b = 0; b < batch; ++b) {
        // The model supplies the validated packed prefix. Bound every store by the
        // output descriptor.
        const uint32_t start = static_cast<uint32_t>(ends[b]);
        const uint32_t end = static_cast<uint32_t>(ends[b + 1]);
        longest = std::max(longest, end - start);
        crosses |= start / 64 != (end - 1) / 64;
        while (tile < entries && tile * 16 < end) {
            out[2 + tile++] = static_cast<int32_t>(b);
        }
    }
    out[0] = crosses;
    out[1] = static_cast<int32_t>(longest);
    return 0;
}
} // namespace

class MegaRecurrentKdaMetadataKernel : public CpuKernel {
public:
    uint32_t Compute(CpuKernelContext &ctx) override
    {
        auto *prefix = ctx.Input(0);
        auto *plan = ctx.Output(0);
        if (!prefix || !plan || plan->GetDataType() != DT_INT32) {
            return 1;
        }
        switch (prefix->GetDataType()) {
            case DT_INT32:
                return BuildPlan<int32_t>(*prefix, *plan);
            case DT_INT64:
                return BuildPlan<int64_t>(*prefix, *plan);
            default:
                return 1;
        }
    }
};
constexpr const char *kMetadataOp = "MegaRecurrentKdaMetadata";
REGISTER_CPU_KERNEL(kMetadataOp, MegaRecurrentKdaMetadataKernel);
} // namespace aicpu
