/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef COMPACT_CONV_STATE_TILING_H
#define COMPACT_CONV_STATE_TILING_H

#include <cstdint>
#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "err/ops_err.h"
#include "../op_kernel/compact_conv_state_tiling_data.h"

namespace optiling {
using namespace GDN;

struct CompactConvStateCompileInfo {};

class CompactConvStateTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit CompactConvStateTiling(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapesAndStrides();
    ge::graphStatus AnalyzeAttributes();
    ge::graphStatus CheckUb();

    CompactConvStateTilingData tilingData_{};
    uint64_t tilingKey_{0};
    int64_t workspaceSize_{0};
    uint64_t stateDtypeTemplate_{0};
    uint64_t planDtypeTemplate_{0};
    uint64_t stateIndexDtypeTemplate_{0};
    bool alignMode_{false};
    bool sharedDimStride_{false};
    int64_t dimStride_{-1};
};

} // namespace optiling

#endif // COMPACT_CONV_STATE_TILING_H
