#ifndef RECURRENT_GATED_DELTA_RULE_V2_TILING_H
#define RECURRENT_GATED_DELTA_RULE_V2_TILING_H

#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "../op_kernel/recurrent_gated_delta_rule_v2_struct.h"

namespace optiling {
using namespace RecurrentGatedDeltaRuleV2;

struct RecurrentGatedDeltaRuleV2CompileInfo {};

class RecurrentGatedDeltaRuleV2Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit RecurrentGatedDeltaRuleV2Tiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
    }
    ~RecurrentGatedDeltaRuleV2Tiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override
    {
        return tilingKey_;
    }
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus CheckAndFill();
    ge::graphStatus FillStrides();
    RecurrentGatedDeltaRuleV2TilingData tilingData_{};
    uint64_t tilingKey_{0};
};
} // namespace optiling

#endif // RECURRENT_GATED_DELTA_RULE_V2_TILING_H
