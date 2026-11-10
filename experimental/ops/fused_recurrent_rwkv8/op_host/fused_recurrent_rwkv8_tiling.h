/*!
 * \file fused_recurrent_rwkv8_tiling.h
 * \brief
 */
#ifndef __OP_HOST_FUSED_RECURRENT_RWKV8_TILING_H__
#define __OP_HOST_FUSED_RECURRENT_RWKV8_TILING_H__
#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "err/ops_err.h"
#include "../op_kernel/fused_recurrent_rwkv8_tiling_data.h"

namespace optiling {
using namespace FusedRecurrentRwkv8;

struct FusedRecurrentRwkv8CompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

struct FusedRecurrentRwkv8Info {
public:
    const char *opName = "FusedRecurrentRwkv8";
};

class FusedRecurrentRwkv8Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit FusedRecurrentRwkv8Tiling(gert::TilingContext *context) : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
        InitCompileInfo();
    };
    ~FusedRecurrentRwkv8Tiling() override = default;

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

protected:
    void InitCompileInfo();
    void PrintTilingData();

    ge::graphStatus CheckContext();
    ge::graphStatus AnalyzeDtype();
    ge::graphStatus AnalyzeShapes();
    ge::graphStatus GetScale();
    ge::graphStatus GetFlagAttrs();
    ge::graphStatus GetOptionalInput();
    ge::graphStatus AnalyzeFormat();
    bool CheckFormat(ge::Format format, const std::string &desc);

    FusedRecurrentRwkv8CompileInfo compileInfo_;
    FusedRecurrentRwkv8TilingData tilingData_;
    FusedRecurrentRwkv8Info inputParams_;
};

} // namespace optiling
#endif // __OP_HOST_FUSED_RECURRENT_RWKV8_TILING_H__
