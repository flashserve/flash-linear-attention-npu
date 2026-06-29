/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

/*!
 * \file solve_tril_def.cpp
 * \brief SolveTril operator definition: input/output declaration and platform registration
 *
 * 接口收敛说明（MCH 上片后）：
 *   - 采用源仓 MCH 的 6 段 kernel 形参方案：x / cu_seqlens / chunk_indices / x_out / workspace / tiling，
 *     删除原 MBH 调试入参 mch_out / zero_mat / eye_mat。
 *   - 输出沿用本仓命名 x_out（非源仓的 y）。
 *   - cu_seqlens / chunk_indices 沿用本仓 INT64（kernel 端按 int64_t 读取）。
 *   - x / x_out 支持 FP16 + BF16 两个编译变体。
 */

#include "register/op_def_registry.h"

namespace ops {
class SolveTril : public OpDef {
public:
    explicit SolveTril(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // cu_seqlens / chunk_indices 为变长场景的索引（kernel 端 GetValue 读取）。
        // 采用 INT64（与本仓一致）；DataType 数组大小须与各 input/output 的变体数一致（2 个变体）。
        // 不使用 ValueDepend(OPTIONAL)：它会触发框架追加额外编译变体（参见源仓 review 问题 7）。
        this->Input("cu_seqlens")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("chunk_indices")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("x_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("chunk_size").AttrType(OPTIONAL).Int(64);
        this->Attr("layout").AttrType(OPTIONAL).String("bsnd");

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(false)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(SolveTril);
} // namespace ops
