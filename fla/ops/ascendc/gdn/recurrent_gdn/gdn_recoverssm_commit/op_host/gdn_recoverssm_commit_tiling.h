/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef GDN_RECOVERSSM_COMMIT_TILING_H
#define GDN_RECOVERSSM_COMMIT_TILING_H

#include <cstdint>
#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "err/ops_err.h"
#include "../op_kernel/gdn_recoverssm_commit_tiling_data.h"

namespace optiling {
using namespace GDN;

struct GdnRecoverssmCommitCompileInfo {};

class GdnRecoverssmCommitTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GdnRecoverssmCommitTiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
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

    GdnRecoverssmCommitTilingData tilingData_{};
    uint64_t tilingKey_{0};
    int64_t workspaceSize_{0};
    uint64_t stateDtypeTemplate_{0};
    uint64_t planDtypeTemplate_{0};
    uint64_t stateIndexDtypeTemplate_{0};
    bool alignMode_{false};
};

} // namespace optiling

#endif // GDN_RECOVERSSM_COMMIT_TILING_H
