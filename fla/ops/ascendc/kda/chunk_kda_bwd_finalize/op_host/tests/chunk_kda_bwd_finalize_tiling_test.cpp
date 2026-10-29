#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "base/runtime/runtime_attrs_def.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "../op_tiling/chunk_kda_bwd_finalize_tiling.h"
#include "../op_tiling/chunk_kda_bwd_finalize_arch22.h"

namespace optiling {
ge::graphStatus Tiling4ChunkKdaBwdFinalize(gert::TilingContext *context);
}

namespace {

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    uint32_t usedCoreNum = 0;
    uint32_t outputGateTokenMajor = UINT32_MAX;
    std::string message;
};

std::unique_ptr<gert::Tensor> MakeTensor(const gert::StorageShape &shape, ge::DataType dtype)
{
    return std::make_unique<gert::Tensor>(
        shape,
        gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
        gert::TensorPlacement::kOnHost, dtype, nullptr);
}

void SetFinalizeAttrs(gert::TilingContext *context, bool gateOutputHeadMajor)
{
    auto *attrs = const_cast<gert::RuntimeAttrs *>(context->GetAttrs());
    auto *def = reinterpret_cast<RuntimeAttrsDef *>(attrs);
    def->attr_num = 11;
    size_t index = 0;
    size_t offset = sizeof(RuntimeAttrsDef) + def->attr_num * sizeof(size_t);
    auto write = [&](const void *value, size_t size) {
        def->offset[index++] = offset;
        std::memcpy(reinterpret_cast<uint8_t *>(def) + offset, value, size);
        offset += size;
    };
    const double scale = 0.125;
    const double lowerBound = -1.0;
    const int64_t chunkSize = 64;
    const bool enabled = true;
    const bool disabled = false;
    const char layout[] = "NTD";
    write(&scale, sizeof(scale));
    write(&lowerBound, sizeof(lowerBound));
    write(&chunkSize, sizeof(chunkSize));
    write(&enabled, sizeof(enabled));
    write(&enabled, sizeof(enabled));
    write(&enabled, sizeof(enabled));
    write(&disabled, sizeof(disabled));
    write(&disabled, sizeof(disabled));
    write(&disabled, sizeof(disabled));
    write(layout, sizeof(layout));
    write(&gateOutputHeadMajor, sizeof(gateOutputHeadMajor));
}

std::string RunWithCapturedLog(gert::TilingContext *context, ge::graphStatus &status)
{
    int pipeFds[2];
    if (pipe(pipeFds) != 0) {
        status = optiling::Tiling4ChunkKdaBwdFinalize(context);
        return {};
    }
    std::fflush(nullptr);
    const int savedStdout = dup(STDOUT_FILENO);
    const int savedStderr = dup(STDERR_FILENO);
    if (savedStdout < 0 || savedStderr < 0 ||
        dup2(pipeFds[1], STDOUT_FILENO) < 0 || dup2(pipeFds[1], STDERR_FILENO) < 0) {
        if (savedStdout >= 0) close(savedStdout);
        if (savedStderr >= 0) close(savedStderr);
        close(pipeFds[0]);
        close(pipeFds[1]);
        status = optiling::Tiling4ChunkKdaBwdFinalize(context);
        return {};
    }
    close(pipeFds[1]);
    status = optiling::Tiling4ChunkKdaBwdFinalize(context);
    std::fflush(nullptr);
    dup2(savedStdout, STDOUT_FILENO);
    dup2(savedStderr, STDERR_FILENO);
    close(savedStdout);
    close(savedStderr);

    std::string log;
    char buffer[4096];
    ssize_t size = 0;
    while ((size = read(pipeFds[0], buffer, sizeof(buffer))) > 0) {
        log.append(buffer, static_cast<size_t>(size));
    }
    close(pipeFds[0]);
    return log;
}

TilingResult RunTiling(const char *socVersion, bool gateOutputHeadMajor)
{
    constexpr int64_t heads = 2;
    constexpr int64_t tokens = 64;
    const gert::StorageShape token128({heads, tokens, 128}, {heads, tokens, 128});
    const gert::StorageShape token64({heads, tokens, 64}, {heads, tokens, 64});
    const gert::StorageShape gateToken128({tokens, heads, 128}, {tokens, heads, 128});
    const gert::StorageShape scalarHead({heads, tokens}, {heads, tokens});
    const gert::StorageShape scalarToken({tokens, heads}, {tokens, heads});
    const gert::StorageShape aLog({heads}, {heads});
    const gert::StorageShape dtBias({heads, 128}, {heads, 128});
    const gert::StorageShape state({1, heads, 128, 128}, {1, heads, 128, 128});

    std::vector<std::unique_ptr<gert::Tensor>> inputKeepAlive;
    std::vector<gert::Tensor *> inputs;
    auto addInput = [&](const gert::StorageShape &shape, ge::DataType dtype) {
        inputKeepAlive.push_back(MakeTensor(shape, dtype));
        inputs.push_back(inputKeepAlive.back().get());
    };
    addInput(token128, ge::DT_BF16);
    addInput(token128, ge::DT_BF16);
    addInput(token128, ge::DT_BF16);
    addInput(token128, ge::DT_FLOAT);
    addInput(token128, ge::DT_FLOAT);
    addInput(scalarHead, ge::DT_FLOAT);
    addInput(aLog, ge::DT_FLOAT);
    addInput(dtBias, ge::DT_FLOAT);
    addInput(token64, ge::DT_BF16);
    addInput(token128, ge::DT_BF16);
    addInput(state, ge::DT_BF16);
    addInput(state, ge::DT_BF16);
    addInput(token128, ge::DT_BF16);
    addInput(token64, ge::DT_FLOAT);
    addInput(token128, ge::DT_FLOAT);

    std::vector<std::unique_ptr<gert::Tensor>> outputKeepAlive;
    std::vector<gert::Tensor *> outputs;
    auto addOutput = [&](const gert::StorageShape &shape, ge::DataType dtype) {
        outputKeepAlive.push_back(MakeTensor(shape, dtype));
        outputs.push_back(outputKeepAlive.back().get());
    };
    addOutput(token128, ge::DT_BF16);
    addOutput(token128, ge::DT_BF16);
    addOutput(token128, ge::DT_BF16);
    addOutput(gateOutputHeadMajor ? scalarHead : scalarToken, ge::DT_FLOAT);
    addOutput(gateOutputHeadMajor ? token128 : gateToken128, ge::DT_FLOAT);
    addOutput(aLog, ge::DT_FLOAT);
    addOutput(dtBias, ge::DT_FLOAT);

    optiling::ChunkKdaBwdFinalizeCompileInfo compileInfo{};
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto *workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());
    const std::vector<uint32_t> inputInstances(15, 1);
    std::vector<uint32_t> allInputInstances = inputInstances;
    allInputInstances.resize(19, 0);
    const std::vector<uint32_t> outputInstances(7, 1);

    gert::TilingContextFaker faker;
    faker.SetOpType("ChunkKdaBwdFinalize")
        .IrInstanceNum(allInputInstances, outputInstances)
        .InputTensors(inputs)
        .OutputTensors(outputs)
        .CompileInfo(&compileInfo)
        .PlatformInfo(&platformInfo)
        .TilingData(tilingData.get())
        .Workspace(workspace);
    faker.Attr("scale", 0.125f)
        .Attr("lower_bound", -1.0f)
        .Attr("chunk_size", static_cast<int64_t>(64))
        .Attr("safe_gate", true)
        .Attr("use_gate_in_kernel", true)
        .Attr("use_exp2", true)
        .Attr("state_v_first", false)
        .Attr("use_beta_sigmoid_in_kernel", false)
        .Attr("allow_neg_eigval", false)
        .Attr("output_layout", ge::AscendString("NTD"))
        .Attr("gate_output_head_major", gateOutputHeadMajor)
        .Attr("reserved_0", static_cast<int64_t>(0))
        .Attr("reserved_1", static_cast<int64_t>(0));
    auto contextHolder = faker.Build();
    auto *context = contextHolder.GetContext();
    TilingResult result;
    if (context == nullptr) {
        result.message = "failed to build TilingContext";
        return result;
    }

    std::map<std::string, std::string> socInfos = {
        {"ai_core_cnt", "8"}, {"core_type_list", "AICore"}};
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> versions = {
        {"Short_SoC_version", socVersion},
        {"NpuArch", std::string(socVersion) == "Ascend950" ? "3510" : "2201"}};
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    context->GetPlatformInfo()->SetPlatformRes("version", versions);

    SetFinalizeAttrs(context, gateOutputHeadMajor);
    result.message = RunWithCapturedLog(context, result.status);
    if (result.status == ge::GRAPH_SUCCESS) {
        result.tilingKey = context->GetTilingKey();
        result.blockDim = context->GetBlockDim();
        if (std::string(socVersion) == "Ascend950") {
            result.outputGateTokenMajor =
                context->GetTilingData<KDA::ChunkKdaBwdFinalizeTilingData>()->outputGateTokenMajor;
        } else {
            result.usedCoreNum = context->GetTilingData<KDA::Arch22::FinalizeTilingData>()->usedCoreNum;
        }
    }
    return result;
}

bool CheckNegative(const char *socVersion)
{
    const auto result = RunTiling(socVersion, true);
    bool ok = result.status == ge::GRAPH_FAILED;
    for (const char *text : {"gate_output_head_major", "true", "false", "A2/A3", "token-major"}) {
        ok = result.message.find(text) != std::string::npos && ok;
    }
    std::cout << socVersion << " gate_output_head_major=true: status=" << result.status
              << ", report=" << result.message << '\n';
    return ok;
}

bool CheckArch22Forward(const char *socVersion)
{
    const auto result = RunTiling(socVersion, false);
    const bool ok = result.status == ge::GRAPH_SUCCESS && result.tilingKey == 101 &&
        result.blockDim == 1 && result.usedCoreNum == 1;
    std::cout << socVersion << " gate_output_head_major=false: status=" << result.status
              << ", key=" << result.tilingKey << ", blocks=" << result.blockDim << '\n';
    return ok;
}

bool CheckA5(bool gateOutputHeadMajor)
{
    const auto result = RunTiling("Ascend950", gateOutputHeadMajor);
    const uint32_t expected = gateOutputHeadMajor ? 0U : 1U;
    const bool ok = result.status == ge::GRAPH_SUCCESS && result.tilingKey == 1 &&
        result.blockDim == 1 && result.outputGateTokenMajor == expected;
    std::cout << "Ascend950 gate_output_head_major=" << (gateOutputHeadMajor ? "true" : "false")
              << ": status=" << result.status << ", key=" << result.tilingKey
              << ", outputGateTokenMajor=" << result.outputGateTokenMajor << '\n';
    return ok;
}

} // namespace

int main()
{
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 0);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "3", 0);
    const std::vector<std::pair<std::string, bool>> cases = {
        {"Ascend910B rejects head-major gate output", CheckNegative("Ascend910B")},
        {"Ascend910B keeps token-major Arch22 forwarding", CheckArch22Forward("Ascend910B")},
        {"Ascend910_93 rejects head-major gate output", CheckNegative("Ascend910_93")},
        {"Ascend910_93 keeps token-major Arch22 forwarding", CheckArch22Forward("Ascend910_93")},
        {"Ascend950 accepts head-major gate output", CheckA5(true)},
        {"Ascend950 accepts token-major gate output", CheckA5(false)},
    };
    size_t passed = 0;
    for (const auto &item : cases) {
        std::cout << (item.second ? "PASS: " : "FAIL: ") << item.first << '\n';
        passed += item.second ? 1U : 0U;
    }
    std::cout << passed << '/' << cases.size() << " tests passed\n";
    return passed == cases.size() ? 0 : 1;
}
