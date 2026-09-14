/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_host/chunk_kda_fwd_prepare_tiling_processor.h"
#include "chunk_kda_fwd_prepare_direct_kernel.h"

namespace ascend_ops::ChunkKdaFwdPrepareDirect {
namespace {

constexpr size_t kOutputCount = 13;

enum OutputIndex : size_t {
    kGk = 0,
    kAqk,
    kAkk,
    kW,
    kU,
    kQg,
    kKg,
    kQgScaled,
    kQHat,
    kKHat,
    kQRstd,
    kKRstd,
    kBetaEff,
};

struct DenseShape {
    int64_t batch = 0;
    int64_t qkHeads = 0;
    int64_t valueHeads = 0;
    int64_t tokens = 0;
    int64_t keyDim = 0;
    int64_t valueDim = 0;
    bool sequenceMajor = false;
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) : bytes_(bytes)
    {
        const auto ret = aclrtMalloc(&address_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS,
                    "chunk_kda_fwd_prepare_direct: workspace 分配失败，ret=", ret);
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    ~DeviceBuffer()
    {
        if (address_ != nullptr) {
            aclrtFree(address_);
        }
    }

    void *Address() const
    {
        return address_;
    }

    size_t Bytes() const
    {
        return bytes_;
    }

private:
    void *address_ = nullptr;
    size_t bytes_ = 0;
};

bool SameShape(const at::Tensor &tensor, std::initializer_list<int64_t> shape)
{
    return tensor.sizes().equals(at::IntArrayRef(shape));
}

bool IsStandardNpuFormat(int64_t format)
{
    return format == static_cast<int64_t>(aclFormat::ACL_FORMAT_NCHW) ||
           format == static_cast<int64_t>(aclFormat::ACL_FORMAT_ND) ||
           format == static_cast<int64_t>(aclFormat::ACL_FORMAT_NCDHW) ||
           format == static_cast<int64_t>(aclFormat::ACL_FORMAT_NCL);
}

void CheckStandardNpuFormat(const at::Tensor &tensor, const char *name)
{
    const int64_t actualFormat = at_npu::native::get_npu_format(tensor);
    TORCH_CHECK(IsStandardNpuFormat(actualFormat),
                "chunk_kda_fwd_prepare_direct: ", name,
                " must use a standard contiguous-compatible layout; private NPU format ",
                actualFormat, " is not supported.");
}

DenseShape CheckInputs(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &g, const at::Tensor &beta,
    const c10::optional<at::Tensor> &aLog,
    const c10::optional<at::Tensor> &dtBias,
    const std::string &layout, int64_t chunkSize,
    bool useQkL2normInKernel, bool useGateInKernel,
    bool useBetaSigmoidInKernel, bool allowNegEigval, bool safeGate,
    bool useExp2, int64_t outputMode)
{
    TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1,
                "chunk_kda_fwd_prepare_direct: 输入必须位于 NPU");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device() &&
                    q.device() == g.device() && q.device() == beta.device(),
                "chunk_kda_fwd_prepare_direct: 输入必须位于同一 NPU");
    // 私有格式可能仍报告逻辑连续，必须在普通连续性检查之外显式拦截。
    CheckStandardNpuFormat(q, "q");
    CheckStandardNpuFormat(k, "k");
    CheckStandardNpuFormat(v, "v");
    CheckStandardNpuFormat(g, "g");
    CheckStandardNpuFormat(beta, "beta");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                    g.is_contiguous() && beta.is_contiguous(),
                "chunk_kda_fwd_prepare_direct: 输入必须连续");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
                    k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16,
                "chunk_kda_fwd_prepare_direct: q/k/v 只支持 BF16");
    TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes() && v.dim() == 4 &&
                    g.dim() == 4 && beta.dim() == 3,
                "chunk_kda_fwd_prepare_direct: 只支持 dense rank-4 输入");
    TORCH_CHECK(layout == "BNSD" || layout == "BSND",
                "chunk_kda_fwd_prepare_direct: layout 只支持 BNSD 或 BSND");

    DenseShape shape;
    shape.sequenceMajor = layout == "BSND";
    shape.batch = q.size(0);
    shape.qkHeads = q.size(shape.sequenceMajor ? 2 : 1);
    shape.tokens = q.size(shape.sequenceMajor ? 1 : 2);
    shape.keyDim = q.size(3);
    shape.valueHeads = v.size(shape.sequenceMajor ? 2 : 1);
    shape.valueDim = v.size(3);

    TORCH_CHECK(shape.batch > 0 && shape.qkHeads > 0 && shape.valueHeads > 0 &&
                    shape.tokens > 0,
                "chunk_kda_fwd_prepare_direct: B/H/T 必须大于 0");
    TORCH_CHECK(shape.valueHeads >= shape.qkHeads &&
                    shape.valueHeads % shape.qkHeads == 0,
                "chunk_kda_fwd_prepare_direct: HV 必须是 HK 的整数倍");
    TORCH_CHECK(shape.keyDim == 128 && shape.valueDim == 128 && chunkSize == 64,
                "chunk_kda_fwd_prepare_direct: 当前直调验收只支持 K=V=128、chunk_size=64");

    if (shape.sequenceMajor) {
        TORCH_CHECK(SameShape(v, {shape.batch, shape.tokens, shape.valueHeads,
                                 shape.valueDim}) &&
                        SameShape(g, {shape.batch, shape.tokens, shape.valueHeads,
                                     shape.keyDim}) &&
                        SameShape(beta, {shape.batch, shape.tokens,
                                        shape.valueHeads}),
                    "chunk_kda_fwd_prepare_direct: BSND 输入 shape 不匹配");
    } else {
        TORCH_CHECK(SameShape(v, {shape.batch, shape.valueHeads, shape.tokens,
                                 shape.valueDim}) &&
                        SameShape(g, {shape.batch, shape.valueHeads, shape.tokens,
                                     shape.keyDim}) &&
                        SameShape(beta, {shape.batch, shape.valueHeads,
                                        shape.tokens}),
                    "chunk_kda_fwd_prepare_direct: BNSD 输入 shape 不匹配");
    }

    const bool rawPath =
        layout == "BNSD" && g.scalar_type() == at::kFloat &&
        beta.scalar_type() == at::kFloat && !useQkL2normInKernel &&
        !useGateInKernel && !useBetaSigmoidInKernel && !allowNegEigval &&
        !safeGate && !useExp2 && outputMode == CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE &&
        !aLog.has_value() && !dtBias.has_value();
    const bool fusedPath =
        layout == "BSND" && g.scalar_type() == at::kBFloat16 &&
        beta.scalar_type() == at::kBFloat16 && useQkL2normInKernel &&
        useGateInKernel && useBetaSigmoidInKernel && allowNegEigval &&
        safeGate && useExp2 &&
        (outputMode == CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE ||
         outputMode == CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE) &&
        aLog.has_value() && dtBias.has_value();
    TORCH_CHECK(rawPath || fusedPath,
                "chunk_kda_fwd_prepare_direct: 仅支持 canonical raw BNSD 或 fused BSND 代表路径");

    if (aLog.has_value()) {
        TORCH_CHECK(aLog->device() == q.device(),
                    "chunk_kda_fwd_prepare_direct: a_log 必须与 q 位于同一 NPU");
        CheckStandardNpuFormat(*aLog, "a_log");
        TORCH_CHECK(aLog->is_contiguous() && aLog->scalar_type() == at::kFloat &&
                        SameShape(*aLog, {shape.valueHeads}),
                    "chunk_kda_fwd_prepare_direct: a_log 必须是同设备连续 FP32 [HV]");
    }
    if (dtBias.has_value()) {
        TORCH_CHECK(dtBias->device() == q.device(),
                    "chunk_kda_fwd_prepare_direct: dt_bias 必须与 q 位于同一 NPU");
        CheckStandardNpuFormat(*dtBias, "dt_bias");
        TORCH_CHECK(dtBias->is_contiguous() &&
                        dtBias->scalar_type() == at::kFloat &&
                        dtBias->numel() == shape.valueHeads * shape.keyDim,
                    "chunk_kda_fwd_prepare_direct: dt_bias 必须是同设备连续 FP32 [HV*K]");
    }
    return shape;
}

std::array<at::Tensor, kOutputCount> MakeOutputs(
    const at::Tensor &q, const DenseShape &shape, int64_t outputMode)
{
    std::array<at::Tensor, kOutputCount> outputs;
    const auto bf16 = q.options().dtype(at::kBFloat16);
    const auto fp32 = q.options().dtype(at::kFloat);
    const std::vector<int64_t> keyShape{
        shape.batch, shape.valueHeads, shape.tokens, shape.keyDim};
    const std::vector<int64_t> valueShape{
        shape.batch, shape.valueHeads, shape.tokens, shape.valueDim};
    const std::vector<int64_t> matrixShape{
        shape.batch, shape.valueHeads, shape.tokens, 64};
    const std::vector<int64_t> qkShape{
        shape.batch, shape.qkHeads, shape.tokens, shape.keyDim};
    const std::vector<int64_t> qkScalarShape{
        shape.batch, shape.qkHeads, shape.tokens};
    const std::vector<int64_t> valueScalarShape{
        shape.batch, shape.valueHeads, shape.tokens};

    outputs[kGk] = at::empty(keyShape, fp32);
    outputs[kAqk] = at::empty(matrixShape, bf16);
    outputs[kW] = at::empty(keyShape, bf16);
    outputs[kU] = at::empty(valueShape, bf16);
    outputs[kKg] = at::empty(keyShape, bf16);
    outputs[kQgScaled] = at::empty(keyShape, bf16);
    if (outputMode != CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE) {
        outputs[kAkk] = at::empty(matrixShape, bf16);
        outputs[kQHat] = at::empty(qkShape, bf16);
        outputs[kKHat] = at::empty(qkShape, bf16);
        outputs[kQRstd] = at::empty(qkScalarShape, fp32);
        outputs[kKRstd] = at::empty(qkScalarShape, fp32);
        outputs[kBetaEff] = at::empty(valueScalarShape, fp32);
    }
    if (outputMode == CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE) {
        outputs[kQg] = at::empty(keyShape, bf16);
    }
    return outputs;
}

GM_ADDR TensorAddress(const at::Tensor &tensor)
{
    return tensor.defined() ? reinterpret_cast<GM_ADDR>(tensor.data_ptr()) : nullptr;
}

GM_ADDR OptionalAddress(const c10::optional<at::Tensor> &tensor)
{
    return tensor.has_value() ? TensorAddress(*tensor) : nullptr;
}

std::vector<c10::optional<at::Tensor>> ToFixedSlots(
    const std::array<at::Tensor, kOutputCount> &outputs)
{
    std::vector<c10::optional<at::Tensor>> result;
    result.reserve(kOutputCount);
    for (const auto &output : outputs) {
        if (output.defined()) {
            result.emplace_back(output);
        } else {
            result.emplace_back(c10::nullopt);
        }
    }
    return result;
}

std::vector<c10::optional<at::Tensor>> ChunkKdaFwdPrepareDirectNpu(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &g, const at::Tensor &beta,
    const c10::optional<at::Tensor> &aLog,
    const c10::optional<at::Tensor> &dtBias,
    const std::string &layout, double scale, int64_t chunkSize, double epsilon,
    bool useQkL2normInKernel, bool useGateInKernel,
    bool useBetaSigmoidInKernel, bool allowNegEigval, bool safeGate,
    double lowerBound, bool useExp2, int64_t outputMode)
{
    const c10::OptionalDeviceGuard guard(q.device());
    // tiling 和 kernel 最终只接收 FP32，校验必须针对同一份舍入后的值。
    const float scaleFp32 = static_cast<float>(scale);
    const float epsilonFp32 = static_cast<float>(epsilon);
    const float lowerBoundFp32 = static_cast<float>(lowerBound);
    TORCH_CHECK(std::isfinite(scaleFp32),
                "chunk_kda_fwd_prepare_direct: scale must be finite.");
    TORCH_CHECK(std::isfinite(epsilonFp32) && epsilonFp32 > 0.0F,
                "chunk_kda_fwd_prepare_direct: epsilon must be finite and greater than zero.");
    TORCH_CHECK(std::isfinite(lowerBoundFp32),
                "chunk_kda_fwd_prepare_direct: lower_bound must be finite.");
    if (useGateInKernel && safeGate) {
        TORCH_CHECK(lowerBoundFp32 >= -5.0F && lowerBoundFp32 < 0.0F,
                    "chunk_kda_fwd_prepare_direct: lower_bound must be in [-5, 0) when safe_gate=True.");
    }
    const DenseShape shape = CheckInputs(
        q, k, v, g, beta, aLog, dtBias, layout, chunkSize,
        useQkL2normInKernel, useGateInKernel, useBetaSigmoidInKernel,
        allowNegEigval, safeGate, useExp2, outputMode);
    auto outputs = MakeOutputs(q, shape, outputMode);

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr,
                "chunk_kda_fwd_prepare_direct: PlatformAscendCManager 为空");
    const uint64_t chunksPerSequence =
        (static_cast<uint64_t>(shape.tokens) + chunkSize - 1) / chunkSize;
    optiling::ChunkKdaFwdPrepareScheduleContext scheduleContext;
    scheduleContext.batch = static_cast<uint64_t>(shape.batch);
    scheduleContext.qkHeadNum = static_cast<uint64_t>(shape.qkHeads);
    scheduleContext.valueHeadNum = static_cast<uint64_t>(shape.valueHeads);
    scheduleContext.chunksPerSequence = chunksPerSequence;
    scheduleContext.totalVarLenChunks = chunksPerSequence;
    scheduleContext.aicCoreNum = static_cast<uint64_t>(platform->GetCoreNumAic());
    scheduleContext.libApiWorkspaceBytes = platform->GetLibApiWorkSpaceSize();
    scheduleContext.isVarLen = false;
    optiling::ChunkKdaFwdPrepareSchedule schedule;
    TORCH_CHECK(optiling::ChunkKdaFwdPrepareTilingProcessor(scheduleContext)
                    .Process(schedule),
                "chunk_kda_fwd_prepare_direct: 无法生成调度");
    TORCH_CHECK(schedule.workspaceBytes <= std::numeric_limits<size_t>::max(),
                "chunk_kda_fwd_prepare_direct: workspace 超过 size_t 范围");
    DeviceBuffer workspace(static_cast<size_t>(schedule.workspaceBytes));

    KdaPrepare::ChunkKdaFwdPrepareTilingData tiling{};
    tiling.batch = static_cast<uint32_t>(shape.batch);
    tiling.seqNum = static_cast<uint32_t>(shape.batch);
    tiling.seqLen = static_cast<uint32_t>(shape.tokens);
    tiling.qkHeadNum = static_cast<uint32_t>(shape.qkHeads);
    tiling.valueHeadNum = static_cast<uint32_t>(shape.valueHeads);
    tiling.totalChunks = static_cast<uint32_t>(chunksPerSequence);
    tiling.usedCoreNum = schedule.usedCoreNum;
    tiling.headsPerPartition = schedule.headsPerPartition;
    tiling.epsilon = epsilonFp32;
    tiling.lowerBound = lowerBoundFp32;
    tiling.scale = scaleFp32;
    tiling.isVarLen = false;
    tiling.inputSequenceMajor = shape.sequenceMajor;
    tiling.hasDtBias = dtBias.has_value();

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const auto memsetRet = aclrtMemsetAsync(
        workspace.Address(), workspace.Bytes(), 0, workspace.Bytes(), stream);
    TORCH_CHECK(memsetRet == ACL_SUCCESS,
                "chunk_kda_fwd_prepare_direct: workspace 清零失败，ret=", memsetRet);

    auto launch = [&]() -> int {
        if (outputMode == CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE) {
            KdaPrepareDirect::Launch<
                CHUNK_KDA_FWD_PREPARE_TPL_FP32,
                CHUNK_KDA_FWD_PREPARE_TPL_FP32,
                CHUNK_KDA_FWD_PREPARE_NORM_IDENTITY,
                CHUNK_KDA_FWD_PREPARE_BETA_RAW,
                CHUNK_KDA_FWD_PREPARE_GATE_PRECOMPUTED_STEP,
                false, false, CHUNK_KDA_FWD_PREPARE_OUTPUT_NONE>(
                schedule.usedCoreNum, stream,
                TensorAddress(q), TensorAddress(k), TensorAddress(v),
                TensorAddress(g), TensorAddress(beta), nullptr, nullptr,
                nullptr, nullptr, TensorAddress(outputs[kGk]),
                TensorAddress(outputs[kAqk]), nullptr, TensorAddress(outputs[kW]),
                TensorAddress(outputs[kU]), nullptr, TensorAddress(outputs[kKg]),
                TensorAddress(outputs[kQgScaled]), nullptr, nullptr, nullptr,
                nullptr, nullptr, reinterpret_cast<GM_ADDR>(workspace.Address()),
                tiling);
        } else if (outputMode == CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE) {
            KdaPrepareDirect::Launch<
                CHUNK_KDA_FWD_PREPARE_TPL_BF16,
                CHUNK_KDA_FWD_PREPARE_TPL_BF16,
                CHUNK_KDA_FWD_PREPARE_NORM_L2,
                CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID,
                CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID,
                true, true, CHUNK_KDA_FWD_PREPARE_OUTPUT_RECOMPUTE>(
                schedule.usedCoreNum, stream,
                TensorAddress(q), TensorAddress(k), TensorAddress(v),
                TensorAddress(g), TensorAddress(beta), OptionalAddress(aLog),
                OptionalAddress(dtBias), nullptr, nullptr,
                TensorAddress(outputs[kGk]), TensorAddress(outputs[kAqk]),
                TensorAddress(outputs[kAkk]), TensorAddress(outputs[kW]),
                TensorAddress(outputs[kU]), nullptr, TensorAddress(outputs[kKg]),
                TensorAddress(outputs[kQgScaled]), TensorAddress(outputs[kQHat]),
                TensorAddress(outputs[kKHat]), TensorAddress(outputs[kQRstd]),
                TensorAddress(outputs[kKRstd]), TensorAddress(outputs[kBetaEff]),
                reinterpret_cast<GM_ADDR>(workspace.Address()), tiling);
        } else {
            KdaPrepareDirect::Launch<
                CHUNK_KDA_FWD_PREPARE_TPL_BF16,
                CHUNK_KDA_FWD_PREPARE_TPL_BF16,
                CHUNK_KDA_FWD_PREPARE_NORM_L2,
                CHUNK_KDA_FWD_PREPARE_BETA_TWO_SIGMOID,
                CHUNK_KDA_FWD_PREPARE_GATE_SAFE_SIGMOID,
                true, true, CHUNK_KDA_FWD_PREPARE_OUTPUT_SAVE>(
                schedule.usedCoreNum, stream,
                TensorAddress(q), TensorAddress(k), TensorAddress(v),
                TensorAddress(g), TensorAddress(beta), OptionalAddress(aLog),
                OptionalAddress(dtBias), nullptr, nullptr,
                TensorAddress(outputs[kGk]), TensorAddress(outputs[kAqk]),
                TensorAddress(outputs[kAkk]), TensorAddress(outputs[kW]),
                TensorAddress(outputs[kU]), TensorAddress(outputs[kQg]),
                TensorAddress(outputs[kKg]), TensorAddress(outputs[kQgScaled]),
                TensorAddress(outputs[kQHat]), TensorAddress(outputs[kKHat]),
                TensorAddress(outputs[kQRstd]), TensorAddress(outputs[kKRstd]),
                TensorAddress(outputs[kBetaEff]),
                reinterpret_cast<GM_ADDR>(workspace.Address()), tiling);
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("ChunkKdaFwdPrepareDirect", launch);
    c10_npu::getCurrentNPUStream().synchronize();
    return ToFixedSlots(outputs);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("chunk_kda_fwd_prepare_direct("
          "Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
          "Tensor? a_log, Tensor? dt_bias, str layout, float scale, "
          "int chunk_size, float epsilon, bool use_qk_l2norm_in_kernel, "
          "bool use_gate_in_kernel, bool use_beta_sigmoid_in_kernel, "
          "bool allow_neg_eigval, bool safe_gate, float lower_bound, "
          "bool use_exp2, int output_mode) -> Tensor?[]");
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("chunk_kda_fwd_prepare_direct", ChunkKdaFwdPrepareDirectNpu);
}

} // namespace ascend_ops::ChunkKdaFwdPrepareDirect
