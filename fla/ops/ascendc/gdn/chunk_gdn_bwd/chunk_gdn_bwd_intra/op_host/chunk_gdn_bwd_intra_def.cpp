/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#include "register/op_def_registry.h"

#include <initializer_list>

namespace ops {
class ChunkGdnBwdIntra : public OpDef {
public:
    explicit ChunkGdnBwdIntra(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> mainTypes = {
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16
        };
        const std::initializer_list<ge::DataType> gateTypes = {
            ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT
        };
        const std::initializer_list<ge::DataType> betaTypes = {
            ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT,
            ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT
        };
        const std::initializer_list<ge::DataType> int64Types = {
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64
        };
        const std::initializer_list<ge::Format> nd = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
        };

        this->Input("q").ParamType(REQUIRED).DataType(mainTypes).Format(nd).UnknownShapeFormat(nd);
        this->Input("k").ParamType(REQUIRED).DataType(mainTypes).Format(nd).UnknownShapeFormat(nd);
        this->Input("v").ParamType(REQUIRED).DataType(mainTypes).Format(nd).UnknownShapeFormat(nd);
        this->Input("g").ParamType(REQUIRED).DataType(gateTypes).Format(nd).UnknownShapeFormat(nd);
        this->Input("beta").ParamType(REQUIRED).DataType(betaTypes).Format(nd).UnknownShapeFormat(nd);
        this->Input("A").ParamType(REQUIRED).DataType(mainTypes).Format(nd).UnknownShapeFormat(nd);
        this->Input("d_o").ParamType(REQUIRED).DataType(mainTypes).Format(nd).UnknownShapeFormat(nd);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64Types).Format(nd).UnknownShapeFormat(nd);
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(int64Types).Format(nd).UnknownShapeFormat(nd);

        this->Output("w").ParamType(REQUIRED).DataType(mainTypes).Format(nd).UnknownShapeFormat(nd);
        this->Output("u").ParamType(REQUIRED).DataType(mainTypes).Format(nd).UnknownShapeFormat(nd);
        this->Output("dv_local").ParamType(REQUIRED).DataType(mainTypes).Format(nd).UnknownShapeFormat(nd);

        this->Attr("scale").AttrType(REQUIRED).Float(1.0);
        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("use_exp2").AttrType(OPTIONAL).Bool(true);
        this->Attr("stage").AttrType(OPTIONAL).Int(2);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "chunk_gdn_bwd_intra")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkGdnBwdIntra);
} // namespace ops
