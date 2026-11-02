// Exercise the real CP host processor and ErrorManager, without mocking reports.
#include "fla/ops/ascendc/cp/pre_process_bwd_kernel_merged/op_host/op_tiling/pre_process_bwd_kernel_merged_tiling_processor.h"

extern "C" __attribute__((visibility("default"))) uint32_t dfx_cp_processor_case(int scenario)
{
    gert::StorageShape q({1, 2, 128, 128}, {1, 2, 128, 128}), k = q;
    gert::StorageShape w({1, 4, 128, 128}, {1, 4, 128, 128}), dO = w, dv = w, gk = w;
    gert::StorageShape g({1, 4, 128}, {1, 4, 128}), cu({2}, {2});
    optiling::PreProcessBwdKernelMergedTilingContext ctx{
        "dfx_probe_cp", &q, &k, &w, &dO, &dv, &g, &gk, nullptr,
        ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, false, false, true, 1.0, 64, 24, 1, 4096};
    if (scenario >= 34 && scenario <= 47) {
        const char *names[] = {"q", "k", "w", "d_o", "dv", "g", "gk"};
        const int index = scenario <= 43 ? (scenario - 34) % 5 : (scenario - 44) % 2 + 5;
        // The tested function is also called from the production tiling entry.
        return optiling::CheckRequiredInputMetadata(ctx.nodeName,
            scenario < 39 || scenario == 44 || scenario == 45 ? nullptr : &q, nullptr, names[index]);
    }
    switch (scenario) {
        case 1: k = gert::StorageShape({1, 2, 127, 128}, {1, 2, 127, 128}); break;
        case 2: w = gert::StorageShape({1, 3, 128, 128}, {1, 3, 128, 128}); break;
        case 3: dO = gert::StorageShape({1, 4, 127, 128}, {1, 4, 127, 128}); break;
        case 4: dv = gert::StorageShape({1, 4, 127, 128}, {1, 4, 127, 128}); break;
        case 5: q = gert::StorageShape({1, 2, 0, 128}, {1, 2, 0, 128}); break;
        case 6: q = gert::StorageShape({1, 3, 128, 128}, {1, 3, 128, 128}); k = q; break;
        case 7:
            q = gert::StorageShape({1, 2, 128, 64}, {1, 2, 128, 64}); k = q;
            w = gert::StorageShape({1, 4, 128, 64}, {1, 4, 128, 64}); break;
        case 8: ctx.chunkSize = 32; break;
        case 9: ctx.hasG = ctx.hasGk = true; break;
        case 10: ctx.hasGk = true; gk = gert::StorageShape({1, 4, 127, 128}, {1, 4, 127, 128}); break;
        case 11: ctx.hasGk = true; ctx.gkDtype = ge::DT_FLOAT; break;
        case 12: ctx.hasG = true; g = gert::StorageShape({1, 3, 128}, {1, 3, 128}); break;
        case 13: ctx.hasG = true; ctx.gDtype = ge::DT_INT32; break;
        case 14: ctx.cuSeqlensShape = &cu; cu = gert::StorageShape({1}, {1}); break;
        case 15: ctx.cuSeqlensShape = &cu; [[fallthrough]];
        case 16:
            q = gert::StorageShape({2, 2, 128, 128}, {2, 2, 128, 128}); k = q;
            w = gert::StorageShape({2, 4, 128, 128}, {2, 4, 128, 128}); dO = dv = w; break;
        case 17: ctx.totalCoreNum = 0; break;
        case 18: ctx.qShape = nullptr; break;
        case 19: q = gert::StorageShape({2, 128, 128}, {2, 128, 128}); break;
        case 20: ctx.hasG = true; ctx.gShape = nullptr; break;
        case 21: ctx.hasGk = true; ctx.gkShape = nullptr; break;
        case 22: ctx.hasG = true; g = gert::StorageShape({1, 4, 128, 1}, {1, 4, 128, 1}); break;
        case 23: ctx.hasGk = true; gk = gert::StorageShape({4, 128, 128}, {4, 128, 128}); break;
        case 24: ctx.cuSeqlensShape = &cu; cu = gert::StorageShape({1, 2}, {1, 2}); break;
        case 25: ctx.qDtype = ge::DT_FLOAT; break;
        case 26: ctx.chunkSize = (int64_t{1} << 32) + 64; break;
        case 27: return optiling::CheckTilingResource(ctx.nodeName, nullptr, "context");
        case 28: return optiling::CheckTilingResource(ctx.nodeName, nullptr, "tilingData");
        case 29: return optiling::CheckTilingResource(ctx.nodeName, nullptr, "attrs");
        case 30: return optiling::CheckTilingResource(ctx.nodeName, nullptr, "platformInfo");
        case 31: return optiling::CheckTilingResource(ctx.nodeName, nullptr, "workspaceSizes");
        case 32:
            q = gert::StorageShape({1, 2, 4294967296LL, 128}, {1, 2, 4294967296LL, 128}); k = q;
            w = gert::StorageShape({1, 4, 4294967296LL, 128}, {1, 4, 4294967296LL, 128}); dO = dv = w;
            break;
        case 33:
            q = gert::StorageShape({1, 2, 4294967295LL, 128}, {1, 2, 4294967295LL, 128}); k = q;
            w = gert::StorageShape({1, 4, 4294967295LL, 128}, {1, 4, 4294967295LL, 128}); dO = dv = w;
            break;
        default: break;
    }
    CP::PreProcessBwdKernelMergedTilingData data{};
    optiling::PreProcessBwdKernelMergedTilingProcessor processor(ctx, data);
    return processor.Process();
}
