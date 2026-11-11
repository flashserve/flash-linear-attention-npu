// Diagnostic tiling for the direct-kernel QKV stride regression.
#include "op_host/recurrent_kda_tiling_processor.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    nlohmann::json spec;
    std::ifstream(argv[1]) >> spec;
    optiling::RecurrentKdaTilingContext ctx;
    ctx.isA5 = spec.value("is_a5", true);
    ctx.aivNum = 48;
    ctx.ubSize = ctx.isA5 ? 262144 : 196608;
    auto shape = [](const nlohmann::json &dims) {
        gert::Shape result;
        result.SetDimNum(dims.size());
        for (size_t axis = 0; axis < dims.size(); ++axis) result.SetDim(axis, dims[axis].get<int64_t>());
        return result;
    };
    ctx.queryShape = shape(spec.at("q_shape"));
    ctx.keyShape = ctx.queryShape;
    ctx.valueShape = shape(spec.at("v_shape"));
    ctx.gateShape = ctx.queryShape;
    ctx.gateShape.SetDim(2, ctx.valueShape.GetDim(2));
    ctx.betaShape = shape({ctx.valueShape.GetDim(0), ctx.valueShape.GetDim(1), ctx.valueShape.GetDim(2)});
    ctx.stateShape = shape(spec.at("state_shape"));
    ctx.queryStrides = spec.at("q_stride").get<std::array<int64_t, 4>>();
    ctx.keyStrides = spec.at("k_stride").get<std::array<int64_t, 4>>();
    ctx.valueStrides = spec.at("v_stride").get<std::array<int64_t, 4>>();
    ctx.hasQueryStrides = ctx.hasKeyStrides = ctx.hasValueStrides = true;
    ctx.hasCuSeqlens = spec.at("has_cu").get<bool>();
    if (ctx.hasCuSeqlens) ctx.cuSeqlensShape = shape({spec.at("sequences").get<int64_t>() + 1});
    ctx.cuSeqlensDtype = ge::DT_INT32;
    ctx.gateDtype = ctx.betaDtype = ge::DT_FLOAT;
    ctx.stateDtype = spec.at("state_bf16").get<bool>() ? ge::DT_BF16 : ge::DT_FLOAT;
    ctx.layout = 0;
    ctx.stateVFirst = spec.at("v_first").get<bool>();
    ctx.scale = 0.08838834764831845f;
    ctx.outputFinalState = 1;
    ctx.inplaceFinalState = 0;
    if (spec.at("fused").get<bool>()) {
        ctx.useQkL2norm = ctx.useGateInKernel = ctx.useBetaSigmoid = 1;
        ctx.hasALog = ctx.hasDtBias = 1;
        ctx.aLogShape = shape({ctx.valueShape.GetDim(2)});
        ctx.dtBiasShape = shape({ctx.valueShape.GetDim(2), 128});
    }
    RecurrentKda::RecurrentKdaTilingData tiling{};
    optiling::RecurrentKdaTilingProcessor processor(ctx);
    auto status = processor.ProcessShapes(tiling);
    if (status == ge::GRAPH_SUCCESS) status = processor.ProcessUb(tiling);
    nlohmann::json result = {{"success", status == ge::GRAPH_SUCCESS}};
    if (status == ge::GRAPH_SUCCESS) {
        std::ofstream(argv[2], std::ios::binary).write(reinterpret_cast<char *>(&tiling), sizeof(tiling));
        result["blocks"] = tiling.vectorCoreNum;
        result["batch_strides"] = {tiling.queryBatchStride, tiling.keyBatchStride, tiling.valueBatchStride};
    }
    std::cout << result << '\n';
}
