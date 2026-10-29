// -----------------------------------------------------------------------------------------------------------
// chunk_kda_fwd_fused 算子定义(chunk32 大融合,issue #829)
// kernel 输入 [H,T,K] 头连续 bf16(g 亦 bf16),
// a_log fp32 [H] 原始对数门控(核外 l0op::Exp 预取,核内用 exp 结果),
// dt_bias fp32 [H,K]。输出 o bf16 [H,T,V]。K=V=128,任意 T(尾块核内处理)。
// q/k/v 仅支持 bf16。
// -----------------------------------------------------------------------------------------------------------
#include "register/op_def_registry.h"

namespace ops {
class ChunkKdaFwdFused : public OpDef {
public:
    explicit ChunkKdaFwdFused(const char *name) : OpDef(name)
    {
        // 多 dtype 是 zip 组合语义:所有 io 的 DataType/Format 列表必须等长(组合内
        // dtype 不变也要重复),否则 ini 生成器静默丢 dtype 行。两组合:
        //   0 = 全 bf16;1 = bf16 qkv + fp32 g/beta。
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("g").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("a_log").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dt_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // initial_state:fp32 [H,K,V] ND(state_v_first=false 唯一形态),可选;
        // 缺省时 kernel 按 tiling flag 走清零初始化。
        this->Input("initial_state").ParamType(OPTIONAL).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("o").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // final_state:fp32 [H,K,V] ND,可选;请求时 kernel 在末组把 UB 常驻态写出。
        this->Output("final_state").ParamType(OPTIONAL).DataType({ge::DT_FLOAT, ge::DT_FLOAT}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("scale").AttrType(REQUIRED).Float(1.0);
        this->Attr("lower_bound").AttrType(OPTIONAL).Float(-5.0);
        this->Attr("layout").AttrType(OPTIONAL).String("TND");
        // varlen:cu_seqlens 以 attr 传入,tiling 打包进 tilingdata(kernel
        // 从 tiling GM 直读,免 H2D);缺省 = dense。单启动 varlen 上限 1024 序列。
        this->Attr("cu_seqlens").AttrType(OPTIONAL).ListInt();
        // q/k L2 归一化的 epsilon:q_hat = q/sqrt(sum(q^2)+eps),
        // 与 prepare 同公式同默认值;经 tiling 传到 kernel,核内不硬编码。
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1.0e-6F);

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
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKdaFwdFused);
} // namespace ops
