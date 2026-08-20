#include "register/op_def_registry.h"

namespace ops {
class ChunkKktSolveTri : public OpDef {
public:
    explicit ChunkKktSolveTri(const char *name) : OpDef(name)
    {
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("g")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL);
        this->Input("chunk_indices")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL);
        this->Output("A")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("chunk_size").AttrType(OPTIONAL).Int(64);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "chunk_kkt_solve_tri")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKktSolveTri);
}  // namespace ops
