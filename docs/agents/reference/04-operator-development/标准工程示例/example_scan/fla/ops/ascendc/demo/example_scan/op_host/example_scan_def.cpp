/**
 * 示例文件：fla/ops/ascendc/demo/example_scan/op_host/example_scan_def.cpp
 *
 * 注意事项：
 *   1. 本文件是 ABI 敏感路径（docs/repository-rules.md）：Input/Output/Attr 的数量、顺序、类型、
 *      dtype/format 列表、必选/可选属性一旦发布就不能随意改；改前先按 ABI 流程申请 owner 检视。
 *   2. **输出一律 ParamType(REQUIRED)**：调用方"可以不传某个输出"是 L2 语义，用 aclnn 的可空输出
 *      描述符表达（见 op_api/aclnn_example_scan.h），不要在这里写 ParamType(OPTIONAL)。
 *   3. 可选输入写 ParamType(OPTIONAL)；由值决定行为的元数据（cu_seqlens/chunk_indices）同时写
 *      ValueDepend(OPTIONAL)。
 *   4. 所有输入/输出的 dtype 与 format 列表长度必须一致，且顺序与模板实例顺序对应。
 *   5. 三个 SoC 用同一份 OpAICoreConfig：平台差异放 tiling 与 op_kernel/arch22|arch35，不复制算子定义。
 *   6. output_mode 是 L2 传给 tiling 的内部档位属性，必须在 docs/api.md 注明它不是用户参数。
 */

#include "register/op_def_registry.h"

namespace ops {

class ExampleScan : public OpDef {
public:
    explicit ExampleScan(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> xTypes = {ge::DT_BF16, ge::DT_FLOAT16};
        const std::vector<ge::DataType> gTypes = {ge::DT_FLOAT, ge::DT_BF16};
        const std::vector<ge::Format> ndFormats = {ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(xTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();

        this->Input("g")
            .ParamType(REQUIRED)
            .DataType(gTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();

        this->Input("a_log")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();

        this->Input("initial_state")
            .ParamType(OPTIONAL)
            .DataType(xTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();

        this->Input("cu_seqlens")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();

        this->Input("chunk_indices")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();

        // 输出全部 REQUIRED：state/x_norm 是否需要导出由 L2 的可空输出指针与 output_mode 决定。
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(xTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);

        this->Output("state")
            .ParamType(REQUIRED)
            .DataType(xTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);

        this->Output("x_norm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats);

        this->Attr("layout").AttrType(OPTIONAL).String("BSND");
        this->Attr("scale").AttrType(REQUIRED).Float(1.0F);
        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1.0e-6F);
        // L2 内部档位（0=none, 1=save），不是用户参数。
        this->Attr("output_mode").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");

        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ExampleScan);

} // namespace ops
