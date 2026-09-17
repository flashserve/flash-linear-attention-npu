#include "register/op_def_registry.h"

namespace ops {
class ChunkKdaBwd : public OpDef {
public:
    explicit ChunkKdaBwd(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> data = {
            ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> beta = {
            ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16,
            ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16};
        const std::initializer_list<ge::DataType> gate = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> fp32 = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> i64 = {
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::Format> nd = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("q").ParamType(REQUIRED).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("k").ParamType(REQUIRED).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("v").ParamType(REQUIRED).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("beta").ParamType(REQUIRED).DataType(beta).Format(nd).UnknownShapeFormat(nd);
        this->Input("gk").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("Aqk").ParamType(REQUIRED).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("Akk").ParamType(REQUIRED).DataType(data).Format(nd).UnknownShapeFormat(nd);
        // Forward-saved intermediates. They are required by the currently
        // implemented disable_recompute=true path, but stay optional in the
        // public schema so disable_recompute=false can recompute them later
        // without another ABI change.
        this->Input("w").ParamType(OPTIONAL).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("qg").ParamType(OPTIONAL).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("kg").ParamType(OPTIONAL).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("v_new").ParamType(OPTIONAL).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("h").ParamType(OPTIONAL).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("d_o").ParamType(REQUIRED).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Input("raw_g").ParamType(OPTIONAL).DataType(gate).Format(nd).UnknownShapeFormat(nd);
        this->Input("a_log").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("dt_bias").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(i64).Format(nd).UnknownShapeFormat(nd);
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(i64).Format(nd).UnknownShapeFormat(nd);

        this->Output("dq").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dk").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dv").ParamType(REQUIRED).DataType(data).Format(nd).UnknownShapeFormat(nd);
        this->Output("db").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dg").ParamType(REQUIRED).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dA").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);
        this->Output("dbias").ParamType(OPTIONAL).DataType(fp32).Format(nd).UnknownShapeFormat(nd);

        this->Attr("scale").AttrType(REQUIRED).Float(1.0f);
        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("safe_gate").AttrType(REQUIRED).Bool(true);
        this->Attr("use_gate_in_kernel").AttrType(REQUIRED).Bool(false);
        this->Attr("lower_bound").AttrType(OPTIONAL).Float(-5.0f);
        // Reserved CUDA-compatible controls. The current fused implementation
        // consumes forward-saved intermediates and keeps the exp2 path, so only
        // true/true is accepted until the alternative kernels are implemented.
        this->Attr("disable_recompute").AttrType(OPTIONAL).Bool(true);
        this->Attr("use_exp2").AttrType(OPTIONAL).Bool(true);
        this->Attr("state_v_first").AttrType(OPTIONAL).Bool(false);
        // Private L0 scheduling marker. Public callers never set this
        // directly; KdaChunkBackward uses it to defer reverse-cumsum and raw
        // gate chain rule to the delivery fallback post kernel.
        this->Attr("defer_gate_post").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKdaBwd);
} // namespace ops
