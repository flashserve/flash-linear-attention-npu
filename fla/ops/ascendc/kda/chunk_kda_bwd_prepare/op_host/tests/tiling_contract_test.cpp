#include <cassert>
#include <iostream>
#include "tiling_base/tiling_templates_registry.h"
#undef OP_LOGD
#undef OP_LOGE
#define OP_LOGD(...) ((void)0)
#define OP_LOGE(...) ((void)0)
#include "../op_tiling/chunk_kda_bwd_prepare_tiling_processor.h"

int main()
{
    constexpr const char *layouts[] = {"BNSD", "BSND", "NTD", "TND"};
    for (uint32_t layout = 0; layout < 4; ++layout) {
        for (bool variable : {false, true}) {
        const bool packed = layout >= 2, seq = (layout & 1) != 0;
        gert::StorageShape aqk, value, state;
        if (packed) {
            aqk = seq ? gert::StorageShape({65, 5, 64}, {65, 5, 64})
                      : gert::StorageShape({5, 65, 64}, {5, 65, 64});
            value = seq ? gert::StorageShape({65, 5, 128}, {65, 5, 128})
                        : gert::StorageShape({5, 65, 128}, {5, 65, 128});
            state = gert::StorageShape({2, 5, 128, 128}, {2, 5, 128, 128});
        } else {
            aqk = seq ? gert::StorageShape({1, 65, 5, 64}, {1, 65, 5, 64})
                      : gert::StorageShape({1, 5, 65, 64}, {1, 5, 65, 64});
            value = seq ? gert::StorageShape({1, 65, 5, 128}, {1, 65, 5, 128})
                        : gert::StorageShape({1, 5, 65, 128}, {1, 5, 65, 128});
            state = gert::StorageShape({1, 2, 5, 128, 128}, {1, 2, 5, 128, 128});
        }
        gert::StorageShape cu({3}, {3}), indices({4}, {4});
        for (bool a5 : {false, true}) {
            for (bool direction : {false, true}) {
                KDA::ChunkKdaBwdPrepareTilingData data{};
                optiling::ChunkKdaBwdPrepareTilingContext ctx{
                    "contract", &aqk, &value, &value, &state,
                    variable ? &cu : nullptr, variable ? &indices : nullptr,
                    ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
                    0.125, 64, direction, layouts[layout], 24, 4096, a5};
                optiling::ChunkKdaBwdPrepareTilingProcessor processor(ctx, data);
                if (a5 && (seq || variable != packed)) {
                    assert(processor.Process() == ge::GRAPH_FAILED);
                    continue;
                }
                assert(processor.Process() == ge::GRAPH_SUCCESS);
                assert(processor.GetTilingKey() == (layout + (variable ? 4 : 0)));
                assert(processor.GetBlockDim() == 4);
                assert(processor.GetWorkspaceSize() == 4096 + (a5 ? 0 : processor.GetBlockDim() * 65536));
                assert(data.inputLayout == layout && data.isVariable == variable);
                assert(data.stateVFirst == direction);
                assert(data.NV == 5 && data.T == 65 && data.B == 1);
                std::cout << layouts[layout] << " state=" << direction << " key="
                          << processor.GetTilingKey() << " blocks=" << processor.GetBlockDim() << '\n';
                ctx.layout = "INVALID";
                assert(processor.Process() == ge::GRAPH_FAILED);
                ctx.layout = layouts[layout];
                ctx.chunkSize = 32;
                assert(processor.Process() == ge::GRAPH_FAILED);
                ctx.chunkSize = 64;
                ctx.cuSeqlensShape = variable ? nullptr : &cu;
                assert(processor.Process() == ge::GRAPH_FAILED);
            }
        }
    }
    }
}
