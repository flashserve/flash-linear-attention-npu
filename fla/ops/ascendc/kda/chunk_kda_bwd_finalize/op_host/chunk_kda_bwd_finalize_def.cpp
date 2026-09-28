#include "register/op_def_registry.h"

#include <initializer_list>

namespace ops {

class ChunkKdaBwdFinalize : public OpDef {
public:
    explicit ChunkKdaBwdFinalize(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> bf16 = {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> fp32 = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> betaTypes = {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> aLogTypes = {ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16};
        const std::initializer_list<ge::DataType> int64 = {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::Format> nd = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

#define ADD_INPUT(NAME, PARAM, TYPES) \
        this->Input(NAME).ParamType(PARAM).DataType(TYPES).Format(nd).UnknownShapeFormat(nd).AutoContiguous()
#define ADD_OUTPUT(NAME, TYPES) \
        this->Output(NAME).ParamType(REQUIRED).DataType(TYPES).Format(nd).UnknownShapeFormat(nd)

        ADD_INPUT("q", REQUIRED, bf16);
        ADD_INPUT("k", REQUIRED, bf16);
        ADD_INPUT("v", REQUIRED, bf16);
        ADD_INPUT("gk", REQUIRED, fp32);
        ADD_INPUT("raw_g", REQUIRED, fp32);
        ADD_INPUT("beta", REQUIRED, betaTypes);
        ADD_INPUT("a_log", REQUIRED, aLogTypes);
        ADD_INPUT("dt_bias", REQUIRED, fp32);
        ADD_INPUT("akk", REQUIRED, bf16);
        ADD_INPUT("v_new", REQUIRED, bf16);
        ADD_INPUT("h", REQUIRED, bf16);
        ADD_INPUT("dh", REQUIRED, bf16);
        ADD_INPUT("dv_scan", REQUIRED, bf16);
        ADD_INPUT("d_aqk", REQUIRED, fp32);
        ADD_INPUT("dq_raw", REQUIRED, fp32);
        ADD_INPUT("q_rstd", OPTIONAL, fp32);
        ADD_INPUT("k_rstd", OPTIONAL, fp32);
        ADD_INPUT("cu_seqlens", OPTIONAL, int64).ValueDepend(OPTIONAL);
        ADD_INPUT("chunk_indices", OPTIONAL, int64).ValueDepend(OPTIONAL);

        ADD_OUTPUT("dq", bf16);
        ADD_OUTPUT("dk", bf16);
        ADD_OUTPUT("dv", bf16);
        ADD_OUTPUT("d_beta", betaTypes);
        ADD_OUTPUT("d_g", fp32);
        ADD_OUTPUT("d_a_log", fp32);
        ADD_OUTPUT("d_dt_bias", fp32);

#undef ADD_OUTPUT
#undef ADD_INPUT

        this->Attr("scale").AttrType(REQUIRED).Float();
        this->Attr("lower_bound").AttrType(OPTIONAL).Float(-5.0);
        this->Attr("chunk_size").AttrType(OPTIONAL).Int(64);
        this->Attr("safe_gate").AttrType(OPTIONAL).Bool(true);
        this->Attr("use_gate_in_kernel").AttrType(OPTIONAL).Bool(true);
        this->Attr("use_exp2").AttrType(OPTIONAL).Bool(true);
        this->Attr("state_v_first").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "chunk_kda_bwd_finalize")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKdaBwdFinalize);

} // namespace ops
