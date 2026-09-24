/*!
 * \file fused_recurrent_rwkv8_def.cpp
 * \brief FusedRecurrentRwkv8 op definition (WKV7 fused recurrent forward, z/b parametrization).
 *
 * dtype 口径（对齐 fla fused_recurrent）：io（q/w/k/v/z/b/o）支持 fp16/bf16/fp32，
 * state 张量（initial_state/s/sa）恒 fp32——递推累加全程 fp32，见 kernel 注释。
 * 注意：DataType/Format 配对列表必须全参数等长（第 i 项跨参数组成一个 kernel variant），
 * 故 state 参数用 3×DT_FLOAT 补齐；列表不等长会导致 opbuild 丢弃 dtype 行、打包失败。
 *
 * 训练预埋（对齐官方 wkv7_cuda.cu 三输出 + fla reverse）：
 * 输出 s（chunk 快照 (B,H,T//chunk_len,K,V)，官方转置布局）与 sa（每 token state@z，
 * (B,H,T,V)）恒 fp32，由 attr output_chunk_state/output_sa 门控（默认 false，
 * 关闭时 infershape 给零尺寸 shape、kernel 跳过写出）；attr reverse 控制 T 维
 * 倒序递推；attr chunk_len 控制快照间隔（默认 16 对齐官方 backward 重建粒度，
 * 非 16 值与官方 backward 不兼容）。io 布局定档 BHTC = (B,H,T,C)（2026-08-17，H 在 T 前）。
 */
#include "register/op_def_registry.h"

namespace ops {
class FusedRecurrentRwkv8 : public OpDef {
public:
    explicit FusedRecurrentRwkv8(const char *name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("w")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("initial_state")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("o")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("s")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("sa")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
        this->Attr("output_chunk_state").AttrType(OPTIONAL).Bool(false);
        this->Attr("output_sa").AttrType(OPTIONAL).Bool(false);
        this->Attr("reverse").AttrType(OPTIONAL).Bool(false);
        this->Attr("chunk_len").AttrType(OPTIONAL).Int(16);

        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("softsync.flag", "true");
        this->AICore().AddConfig("ascend910b", aicConfig);
    }
};

OP_ADD(FusedRecurrentRwkv8);

} // namespace ops
