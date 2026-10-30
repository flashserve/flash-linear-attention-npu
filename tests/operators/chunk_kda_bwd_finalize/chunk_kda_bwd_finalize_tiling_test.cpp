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
#include "chunk_kda_bwd_finalize_tiling.h"
#include "chunk_kda_bwd_finalize_arch22.h"

namespace optiling {
ge::graphStatus Tiling4ChunkKdaBwdFinalize(gert::TilingContext *context);
}

static_assert(sizeof(KDA::ChunkKdaBwdFinalizeTilingData) == 184, "Finalize tiling ABI size changed");

namespace {

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    uint32_t usedCoreNum = 0;
    uint32_t betaMode = 0;
    uint32_t outputTokenMajor = UINT32_MAX;
    uint32_t qkTokenMajor = UINT32_MAX;
    uint32_t vTokenMajor = UINT32_MAX;
    uint32_t rawTokenMajor = UINT32_MAX;
    uint32_t gateOutputTokenMajor = UINT32_MAX;
    uint32_t rawDataSize = 0;
    std::string message;
};

std::unique_ptr<gert::Tensor> MakeTensor(const gert::StorageShape &shape, ge::DataType dtype)
{
    return std::make_unique<gert::Tensor>(
        shape,
        gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
        gert::TensorPlacement::kOnHost, dtype, nullptr);
}

void SetFinalizeAttrs(
    gert::TilingContext *context, const char *outputLayout, const char *modelLayout)
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
    write(&scale, sizeof(scale));
    write(&lowerBound, sizeof(lowerBound));
    write(&chunkSize, sizeof(chunkSize));
    write(&enabled, sizeof(enabled));
    write(&enabled, sizeof(enabled));
    write(&enabled, sizeof(enabled));
    write(&disabled, sizeof(disabled));
    write(&disabled, sizeof(disabled));
    write(&disabled, sizeof(disabled));
    write(outputLayout, std::strlen(outputLayout) + 1);
    write(modelLayout, std::strlen(modelLayout) + 1);
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

TilingResult RunTiling(
    const char *socVersion, const char *modelLayout = "", bool square = false,
    const char *outputLayout = "NTD", ge::DataType rawType = ge::DT_FLOAT,
    bool withRstd = false, bool mismatchedRawShape = false)
{
    constexpr int64_t heads = 2;
    const int64_t tokens = square ? heads : 64;
    const bool mixed = modelLayout[0] != '\0';
    const bool modelTokenMajor =
        std::strcmp(modelLayout, "BSND") == 0 || std::strcmp(modelLayout, "TND") == 0;
    const bool outputTokenMajor =
        std::strcmp(outputLayout, "BSND") == 0 || std::strcmp(outputLayout, "TND") == 0;
    const bool qkTokenMajor = modelTokenMajor && !withRstd;
    const bool dense =
        std::strcmp(modelLayout, "BSND") == 0 || std::strcmp(modelLayout, "BNSD") == 0;
    auto shape = [dense](int64_t first, int64_t second, int64_t width = -1) {
        if (dense) {
            return width < 0 ?
                gert::StorageShape({1, first, second}, {1, first, second}) :
                gert::StorageShape({1, first, second, width}, {1, first, second, width});
        }
        return width < 0 ?
            gert::StorageShape({first, second}, {first, second}) :
            gert::StorageShape({first, second, width}, {first, second, width});
    };
    const gert::StorageShape token128 = shape(heads, tokens, 128);
    const gert::StorageShape token64 = shape(heads, tokens, 64);
    const gert::StorageShape gateToken128 = shape(tokens, heads, 128);
    const gert::StorageShape scalarHead = shape(heads, tokens);
    const gert::StorageShape scalarToken = shape(tokens, heads);
    const gert::StorageShape aLog({heads}, {heads});
    const gert::StorageShape dtBias({heads, 128}, {heads, 128});
    const gert::StorageShape state = dense ?
        gert::StorageShape({1, 1, heads, 128, 128}, {1, 1, heads, 128, 128}) :
        gert::StorageShape({1, heads, 128, 128}, {1, heads, 128, 128});

    std::vector<std::unique_ptr<gert::Tensor>> inputKeepAlive;
    std::vector<gert::Tensor *> inputs;
    auto addInput = [&](const gert::StorageShape &shape, ge::DataType dtype) {
        inputKeepAlive.push_back(MakeTensor(shape, dtype));
        inputs.push_back(inputKeepAlive.back().get());
    };
    addInput(qkTokenMajor ? gateToken128 : token128, ge::DT_BF16);
    addInput(qkTokenMajor ? gateToken128 : token128, ge::DT_BF16);
    addInput(modelTokenMajor ? gateToken128 : token128, ge::DT_BF16);
    addInput(token128, ge::DT_FLOAT);
    addInput(mixed != mismatchedRawShape ? gateToken128 : token128, rawType);
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
    if (withRstd) {
        addInput(scalarHead, ge::DT_FLOAT);
        addInput(scalarHead, ge::DT_FLOAT);
    }

    std::vector<std::unique_ptr<gert::Tensor>> outputKeepAlive;
    std::vector<gert::Tensor *> outputs;
    auto addOutput = [&](const gert::StorageShape &shape, ge::DataType dtype) {
        outputKeepAlive.push_back(MakeTensor(shape, dtype));
        outputs.push_back(outputKeepAlive.back().get());
    };
    addOutput(outputTokenMajor ? gateToken128 : token128, ge::DT_BF16);
    addOutput(outputTokenMajor ? gateToken128 : token128, ge::DT_BF16);
    addOutput(outputTokenMajor ? gateToken128 : token128, ge::DT_BF16);
    addOutput(mixed || outputTokenMajor ? scalarToken : scalarHead, ge::DT_FLOAT);
    addOutput(mixed || outputTokenMajor ? gateToken128 : token128, rawType);
    addOutput(aLog, ge::DT_FLOAT);
    addOutput(dtBias, ge::DT_FLOAT);

    optiling::ChunkKdaBwdFinalizeCompileInfo compileInfo{};
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto *workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());
    std::vector<uint32_t> allInputInstances(withRstd ? 17 : 15, 1);
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
        .Attr("output_layout", ge::AscendString(outputLayout))
        .Attr("model_layout", ge::AscendString(modelLayout))
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

    SetFinalizeAttrs(context, outputLayout, modelLayout);
    result.message = RunWithCapturedLog(context, result.status);
    if (result.status == ge::GRAPH_SUCCESS) {
        result.tilingKey = context->GetTilingKey();
        result.blockDim = context->GetBlockDim();
        if (std::string(socVersion) == "Ascend950") {
            result.rawDataSize = context->GetRawTilingData()->GetDataSize();
            const auto *tiling = context->GetTilingData<KDA::ChunkKdaBwdFinalizeTilingData>();
            result.outputTokenMajor = tiling->outputTokenMajor;
            result.qkTokenMajor = tiling->qkTokenMajor;
            result.vTokenMajor = tiling->vTokenMajor;
            result.rawTokenMajor = tiling->rawTokenMajor;
            result.gateOutputTokenMajor = tiling->gateOutputTokenMajor;
            result.betaMode = tiling->betaMode;
        } else {
            const auto *tiling = context->GetTilingData<KDA::Arch22::FinalizeTilingData>();
            result.usedCoreNum = tiling->usedCoreNum;
            result.betaMode = tiling->betaMode;
            result.qkTokenMajor = tiling->qkTokenMajor;
            result.vTokenMajor = tiling->vTokenMajor;
            result.rawTokenMajor = tiling->rawTokenMajor;
            result.gateOutputTokenMajor = tiling->gateOutputTokenMajor;
        }
    }
    return result;
}

bool CheckStandalone(const char *socVersion, bool square, bool tokenOutput)
{
    const bool a5 = std::string(socVersion) == "Ascend950";
    const auto result = RunTiling(
        socVersion, "", square, tokenOutput ? "TND" : "NTD");
    const uint32_t expectedGate = tokenOutput ? 1U : 0U;
    const bool ok = result.status == ge::GRAPH_SUCCESS &&
        result.tilingKey == (a5 ? 1U : 101U) && result.blockDim == 1 &&
        result.qkTokenMajor == 0 && result.vTokenMajor == 0 &&
        result.rawTokenMajor == 0 && result.gateOutputTokenMajor == expectedGate &&
        result.betaMode == 0 && (!a5 || result.rawDataSize == 192);
    std::cout << socVersion << " standalone square=" << square
              << " tokenOutput=" << tokenOutput << ": status=" << result.status
              << ", key=" << result.tilingKey << ", blocks=" << result.blockDim
              << ", gateOutputTokenMajor=" << result.gateOutputTokenMajor
              << ", rawDataSize=" << result.rawDataSize << '\n';
    return ok;
}

bool CheckMixed(
    const char *socVersion, const char *layout, bool withRstd, bool square,
    ge::DataType rawType = ge::DT_FLOAT)
{
    const bool a5 = std::string(socVersion) == "Ascend950";
    const bool tokenMajor = std::strcmp(layout, "BSND") == 0 || std::strcmp(layout, "TND") == 0;
    const auto result = RunTiling(
        socVersion, layout, square, layout, rawType, withRstd);
    const uint32_t expectedQK = tokenMajor && !withRstd ? 1U : 0U;
    const uint32_t expectedV = tokenMajor ? 1U : 0U;
    const uint32_t expectedKey = a5 ? (withRstd ? 3U : 1U) : (withRstd ? 102U : 101U);
    const bool ok = result.status == ge::GRAPH_SUCCESS &&
        result.tilingKey == expectedKey && result.blockDim == 1 &&
        result.qkTokenMajor == expectedQK && result.vTokenMajor == expectedV &&
        result.rawTokenMajor == 1 && result.gateOutputTokenMajor == 1 &&
        result.betaMode == 0 && (!a5 || result.rawDataSize == 192);
    std::cout << socVersion << " mixed=" << layout << " square=" << square
              << " rstd=" << withRstd << " rawType=" << rawType
              << ": status=" << result.status << ", key=" << result.tilingKey
              << ", qkTokenMajor=" << result.qkTokenMajor
              << ", vTokenMajor=" << result.vTokenMajor
              << ", rawTokenMajor=" << result.rawTokenMajor
              << ", gateOutputTokenMajor=" << result.gateOutputTokenMajor
              << ", rawDataSize=" << result.rawDataSize << ", report=" << result.message << '\n';
    return ok;
}

bool CheckInvalidModelLayout()
{
    const auto result = RunTiling("Ascend950", "BTHD");
    std::cout << "Ascend950 invalid model_layout: status=" << result.status
              << ", report=" << result.message << '\n';
    return result.status == ge::GRAPH_FAILED;
}

bool CheckStandaloneRejectsTokenRaw()
{
    const auto result = RunTiling("Ascend950", "", false, "NTD", ge::DT_FLOAT, false, true);
    std::cout << "Ascend950 standalone rejects token-major raw_g: status=" << result.status
              << ", report=" << result.message << '\n';
    return result.status == ge::GRAPH_FAILED;
}

} // namespace

int main()
{
    setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 0);
    setenv("ASCEND_GLOBAL_LOG_LEVEL", "3", 0);
    const std::vector<std::pair<std::string, bool>> cases = {
        {"Ascend950 standalone head-major gradients", CheckStandalone("Ascend950", false, false)},
        {"Ascend950 standalone token-major gradients", CheckStandalone("Ascend950", false, true)},
        {"Ascend950 square standalone remains unambiguous", CheckStandalone("Ascend950", true, false)},
        {"Ascend950 mixed BSND", CheckMixed("Ascend950", "BSND", false, false)},
        {"Ascend950 square TND rstd BF16 raw_g", CheckMixed("Ascend950", "TND", true, true, ge::DT_BF16)},
        {"Ascend950 mixed BNSD keeps head-major qkv gradients", CheckMixed("Ascend950", "BNSD", false, false)},
        {"Ascend950 mixed NTD BF16 raw_g", CheckMixed("Ascend950", "NTD", false, false, ge::DT_BF16)},
        {"Ascend910B standalone", CheckStandalone("Ascend910B", false, false)},
        {"Ascend910B square standalone", CheckStandalone("Ascend910B", true, false)},
        {"Ascend910B mixed BSND BF16 raw_g", CheckMixed("Ascend910B", "BSND", false, false, ge::DT_BF16)},
        {"Ascend910B mixed TND rstd", CheckMixed("Ascend910B", "TND", true, false)},
        {"Ascend910_93 standalone", CheckStandalone("Ascend910_93", false, false)},
        {"Ascend910_93 mixed BSND rstd BF16 raw_g", CheckMixed("Ascend910_93", "BSND", true, false, ge::DT_BF16)},
        {"invalid model_layout is rejected", CheckInvalidModelLayout()},
        {"standalone does not infer raw_g layout from shape", CheckStandaloneRejectsTokenRaw()},
    };
    size_t passed = 0;
    for (const auto &item : cases) {
        std::cout << (item.second ? "PASS: " : "FAIL: ") << item.first << '\n';
        passed += item.second ? 1U : 0U;
    }
    std::cout << passed << '/' << cases.size() << " tests passed\n";
    return passed == cases.size() ? 0 : 1;
}
