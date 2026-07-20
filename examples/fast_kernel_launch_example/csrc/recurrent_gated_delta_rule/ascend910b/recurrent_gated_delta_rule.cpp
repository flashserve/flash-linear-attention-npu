/**
 * Copyright (c) 2025-2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file recurrent_gated_delta_rule.cpp
 * \brief Recurrent gated delta rule operator with fast kernel launch (<<<>>>).
 */

#include <limits>
#include <tuple>
#include <vector>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include "lib/matmul_intf.h"

#include "common/fast_kernel_launch_workspace.h"
#include "fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule_struct.h"
#include "fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_tiling_processor.h"
#include "fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule_tiling_data.h"
#include "fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule.h"

using namespace RecurrentGatedDeltaRule;
using RecurrentGatedDeltaRuleTilingData = RecurrentGatedDeltaRule::RecurrentGatedDeltaRuleTilingData;

namespace ascend_ops {
namespace RecurrentGatedDeltaRuleOp {

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def(
        "recurrent_gated_delta_rule(Tensor query, Tensor key, Tensor value, Tensor(a!) state, *, Tensor? beta=None, "
        "float? scale=None, Tensor? actual_seq_lengths=None, Tensor? ssm_state_indices=None, "
        "Tensor? num_accepted_tokens=None, Tensor? g=None, Tensor? gk=None) -> Tensor");
    m.def(
        "recurrent_gated_delta_rule_functional(Tensor query, Tensor key, Tensor value, Tensor state, *, "
        "Tensor? beta=None, float? scale=None, Tensor? actual_seq_lengths=None, Tensor? ssm_state_indices=None, "
        "Tensor? num_accepted_tokens=None, Tensor? g=None, Tensor? gk=None) -> (Tensor, Tensor)");
}

at::Tensor recurrent_gated_delta_rule_meta(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                            at::Tensor &state, const c10::optional<at::Tensor> &beta,
                                            const c10::optional<double> &scale,
                                            const c10::optional<at::Tensor> &actual_seq_lengths,
                                            const c10::optional<at::Tensor> &ssm_state_indices,
                                            const c10::optional<at::Tensor> &num_accepted_tokens,
                                            const c10::optional<at::Tensor> &g, const c10::optional<at::Tensor> &gk)
{
    (void)query;
    (void)key;
    (void)state;
    (void)beta;
    (void)scale;
    (void)actual_seq_lengths;
    (void)ssm_state_indices;
    (void)num_accepted_tokens;
    (void)g;
    (void)gk;
    TORCH_CHECK(value.dim() == 3, "value dim should be 3");
    c10::SmallVector<int64_t, 3> outShape = {value.size(0), value.size(1), value.size(2)};
    return at::empty(outShape, value.options().dtype(at::ScalarType::BFloat16));
}

std::tuple<at::Tensor, at::Tensor> recurrent_gated_delta_rule_functional_meta(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &state,
    const c10::optional<at::Tensor> &beta, const c10::optional<double> &scale,
    const c10::optional<at::Tensor> &actual_seq_lengths, const c10::optional<at::Tensor> &ssm_state_indices,
    const c10::optional<at::Tensor> &num_accepted_tokens, const c10::optional<at::Tensor> &g,
    const c10::optional<at::Tensor> &gk)
{
    (void)query;
    (void)key;
    (void)beta;
    (void)scale;
    (void)actual_seq_lengths;
    (void)ssm_state_indices;
    (void)num_accepted_tokens;
    (void)g;
    (void)gk;
    TORCH_CHECK(value.dim() == 3, "value dim should be 3");
    TORCH_CHECK(state.dim() == 4, "initial_state dim should be 4");
    c10::SmallVector<int64_t, 3> outShape = {value.size(0), value.size(1), value.size(2)};
    at::Tensor out = at::empty(outShape, value.options().dtype(at::ScalarType::BFloat16));
    at::Tensor stateOut = at::empty_like(state);
    return std::make_tuple(out, stateOut);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("recurrent_gated_delta_rule", recurrent_gated_delta_rule_meta);
    m.impl("recurrent_gated_delta_rule_functional", recurrent_gated_delta_rule_functional_meta);
}

static ge::DataType ToGeStateDtype(at::ScalarType dtype)
{
    if (dtype == at::kFloat) {
        return ge::DT_FLOAT;
    }
    return ge::DT_BF16;
}

static RecurrentGatedDeltaRuleTilingData CalcTilingParams(const at::Tensor &query, const at::Tensor &key,
                                                          const at::Tensor &value, const at::Tensor &beta,
                                                          const at::Tensor &state, const at::Tensor &cuSeqlens,
                                                          const at::Tensor &ssmStateIndices, float scaleVal,
                                                          bool hasG, bool hasGk, bool hasAcceptedTokens,
                                                          uint32_t &blockDim, size_t &workspaceSize)
{
    auto qSizes = query.sizes();
    auto kSizes = key.sizes();
    auto vSizes = value.sizes();
    auto bSizes = beta.sizes();
    auto sSizes = state.sizes();
    auto cuSizes = cuSeqlens.sizes();
    auto ssmSizes = ssmStateIndices.sizes();

    gert::Shape qShape({qSizes[0], qSizes[1], qSizes[2]});
    gert::Shape kShape({kSizes[0], kSizes[1], kSizes[2]});
    gert::Shape vShape({vSizes[0], vSizes[1], vSizes[2]});
    gert::Shape betaShape({bSizes[0], bSizes[1]});
    gert::Shape stateShape({sSizes[0], sSizes[1], sSizes[2], sSizes[3]});
    gert::Shape cuShape({cuSizes[0]});
    gert::Shape ssmShape({ssmSizes[0]});

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(ascendcPlatform != nullptr, "PlatformAscendCManager is null.");

    optiling::RecurrentGatedDeltaRuleTilingContext ctx;
    ctx.nodeName = "recurrent_gated_delta_rule";
    ctx.queryShape = &qShape;
    ctx.keyShape = &kShape;
    ctx.valueShape = &vShape;
    ctx.betaShape = &betaShape;
    ctx.stateShape = &stateShape;
    ctx.cuSeqlensShape = &cuShape;
    ctx.ssmStateShape = &ssmShape;
    ctx.scale = scaleVal;
    ctx.hasGama = hasG ? 1U : 0U;
    ctx.hasGamaK = hasGk ? 1U : 0U;
    ctx.hasAcceptedTokens = hasAcceptedTokens ? 1U : 0U;
    ctx.stateDtype = ToGeStateDtype(state.scalar_type());
    ctx.aivNum = ascendcPlatform->GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ctx.ubSize = ubSize;

    RecurrentGatedDeltaRuleTilingData tiling{};
    optiling::RecurrentGatedDeltaRuleTilingProcessor processor(ctx);
    TORCH_CHECK(processor.Process(tiling, blockDim, workspaceSize) == ge::GRAPH_SUCCESS,
                "recurrent_gated_delta_rule tiling failed.");

    auto stateStrides = state.strides();
    TORCH_CHECK(stateStrides.size() == 4, "state must have four strides.");
    for (size_t index = 0; index < 3; ++index) {
        TORCH_CHECK(stateStrides[index] >= 0 &&
                        static_cast<uint64_t>(stateStrides[index]) <= std::numeric_limits<uint32_t>::max(),
                    "state stride ", index, " is out of uint32 range: ", stateStrides[index]);
    }
    tiling.stateStride0 = static_cast<uint32_t>(stateStrides[0]);
    tiling.stateStride1 = static_cast<uint32_t>(stateStrides[1]);
    tiling.stateStride2 = static_cast<uint32_t>(stateStrides[2]);
    return tiling;
}

template <typename StateT>
__global__ __aicore__ void recurrent_gated_delta_rule_kernel(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR state, GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices,
    GM_ADDR g, GM_ADDR gk, GM_ADDR numAcceptedTokens, GM_ADDR out, GM_ADDR stateOut, GM_ADDR workspace,
    const RecurrentGatedDeltaRuleTilingData tilingData)
{
    AscendC::SetSysWorkspaceForce(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    RGDR<bfloat16_t, bfloat16_t, StateT> op(&tilingData);
    RGDRInitParams initParams{query, key, value, g, gk, beta, state, cuSeqlens, ssmStateIndices, numAcceptedTokens, out,
                              stateOut};
    op.Init(initParams, &pipe);
    op.Process();
}

template <typename StateT>
void LaunchRecurrentGatedDeltaRule(uint32_t blockDim, aclrtStream stream, GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                   GM_ADDR beta, GM_ADDR state, GM_ADDR cuSeqlens, GM_ADDR ssmStateIndices, GM_ADDR g,
                                   GM_ADDR gk, GM_ADDR numAcceptedTokens, GM_ADDR out, GM_ADDR stateOut,
                                   GM_ADDR workspace, const RecurrentGatedDeltaRuleTilingData &tilingData)
{
    recurrent_gated_delta_rule_kernel<StateT><<<blockDim, nullptr, stream>>>(
        query, key, value, beta, state, cuSeqlens, ssmStateIndices, g, gk, numAcceptedTokens, out, stateOut, workspace,
        tilingData);
}

static void RunRecurrentGatedDeltaRuleKernel(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                             const c10::optional<at::Tensor> &beta, at::Tensor &state,
                                             const c10::optional<at::Tensor> &actual_seq_lengths,
                                             const c10::optional<at::Tensor> &ssm_state_indices,
                                             const c10::optional<at::Tensor> &num_accepted_tokens,
                                             const c10::optional<at::Tensor> &g, const c10::optional<at::Tensor> &gk,
                                             float scaleVal, at::Tensor &out, bool functional)
{
    TORCH_CHECK(value.dim() == 3, "value dim should be 3");
    TORCH_CHECK(actual_seq_lengths.has_value(), "actual_seq_lengths is required");
    TORCH_CHECK(ssm_state_indices.has_value(), "ssm_state_indices is required");
    TORCH_CHECK(beta.has_value(), "beta is required");

    const c10::OptionalDeviceGuard guard(query.device());
    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    at::Tensor betaTensor = beta.value();
    at::Tensor actualSeqLengths = actual_seq_lengths.value();
    at::Tensor ssmStateIndices = ssm_state_indices.value();

    uint32_t blockDim = 0;
    size_t workspaceSize = 0;
    auto tiling = CalcTilingParams(query, key, value, betaTensor, state, actualSeqLengths, ssmStateIndices, scaleVal,
                                   g.has_value(), gk.has_value(), num_accepted_tokens.has_value(), blockDim,
                                   workspaceSize);

    GM_ADDR queryPtr = (GM_ADDR)query.data_ptr();
    GM_ADDR keyPtr = (GM_ADDR)key.data_ptr();
    GM_ADDR valuePtr = (GM_ADDR)value.data_ptr();
    GM_ADDR betaPtr = (GM_ADDR)betaTensor.data_ptr();
    GM_ADDR statePtr = (GM_ADDR)state.data_ptr();
    GM_ADDR cuSeqlensPtr = (GM_ADDR)actualSeqLengths.data_ptr();
    GM_ADDR ssmStateIndicesPtr = (GM_ADDR)ssmStateIndices.data_ptr();
    GM_ADDR outPtr = (GM_ADDR)out.data_ptr();
    GM_ADDR stateOutPtr = (GM_ADDR)state.data_ptr();

    GM_ADDR gPtr = nullptr;
    GM_ADDR gkPtr = nullptr;
    GM_ADDR numAcceptedTokensPtr = nullptr;
    if (g.has_value()) {
        gPtr = (GM_ADDR)g.value().data_ptr();
    }
    if (gk.has_value()) {
        gkPtr = (GM_ADDR)gk.value().data_ptr();
    }
    if (num_accepted_tokens.has_value()) {
        numAcceptedTokensPtr = (GM_ADDR)num_accepted_tokens.value().data_ptr();
    }

    at::Tensor workspaceTensor =
        fast_kernel_launch::AllocateDeviceBuffer(query, workspaceSize, "RecurrentGatedDeltaRule workspace");
    fast_kernel_launch::ZeroDeviceBufferAsync(workspaceTensor, workspaceSize, stream,
                                             "RecurrentGatedDeltaRule workspace");
    GM_ADDR workspaceGm = fast_kernel_launch::TensorGmAddr(workspaceTensor);

    auto stateDtype = state.scalar_type();
    auto aclCall = [=, workspaceTensor = workspaceTensor]() -> int {
        if (stateDtype == at::kBFloat16) {
            LaunchRecurrentGatedDeltaRule<bfloat16_t>(blockDim, stream, queryPtr, keyPtr, valuePtr, betaPtr, statePtr,
                                                      cuSeqlensPtr, ssmStateIndicesPtr, gPtr, gkPtr,
                                                      numAcceptedTokensPtr, outPtr, stateOutPtr, workspaceGm, tiling);
        } else if (stateDtype == at::kFloat) {
            LaunchRecurrentGatedDeltaRule<float>(blockDim, stream, queryPtr, keyPtr, valuePtr, betaPtr, statePtr,
                                               cuSeqlensPtr, ssmStateIndicesPtr, gPtr, gkPtr, numAcceptedTokensPtr,
                                               outPtr, stateOutPtr, workspaceGm, tiling);
        } else {
            TORCH_CHECK(false, "Unsupported state dtype: ", stateDtype);
        }
        return 0;
    };

    at_npu::native::OpCommand::RunOpApi(functional ? "RecurrentGatedDeltaRuleFunctional" : "RecurrentGatedDeltaRule",
                                        aclCall);
}

at::Tensor recurrent_gated_delta_rule_npu(const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
                                          at::Tensor &state, const c10::optional<at::Tensor> &beta,
                                          const c10::optional<double> &scale,
                                          const c10::optional<at::Tensor> &actual_seq_lengths,
                                          const c10::optional<at::Tensor> &ssm_state_indices,
                                          const c10::optional<at::Tensor> &num_accepted_tokens,
                                          const c10::optional<at::Tensor> &g, const c10::optional<at::Tensor> &gk)
{
    TORCH_CHECK(scale.has_value(), "scale cannot be empty");
    auto out = recurrent_gated_delta_rule_meta(query, key, value, state, beta, scale, actual_seq_lengths,
                                               ssm_state_indices, num_accepted_tokens, g, gk);
    RunRecurrentGatedDeltaRuleKernel(query, key, value, beta, state, actual_seq_lengths, ssm_state_indices,
                                     num_accepted_tokens, g, gk, static_cast<float>(scale.value()), out, false);
    return out;
}

std::tuple<at::Tensor, at::Tensor> recurrent_gated_delta_rule_functional_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value, const at::Tensor &state,
    const c10::optional<at::Tensor> &beta, const c10::optional<double> &scale,
    const c10::optional<at::Tensor> &actual_seq_lengths, const c10::optional<at::Tensor> &ssm_state_indices,
    const c10::optional<at::Tensor> &num_accepted_tokens, const c10::optional<at::Tensor> &g,
    const c10::optional<at::Tensor> &gk)
{
    TORCH_CHECK(scale.has_value(), "scale cannot be empty");
    auto outs = recurrent_gated_delta_rule_functional_meta(query, key, value, state, beta, scale, actual_seq_lengths,
                                                           ssm_state_indices, num_accepted_tokens, g, gk);
    at::Tensor out = std::get<0>(outs);
    (void)std::get<1>(outs);
    at::Tensor stateInplace = state.clone();
    RunRecurrentGatedDeltaRuleKernel(query, key, value, beta, stateInplace, actual_seq_lengths, ssm_state_indices,
                                     num_accepted_tokens, g, gk, static_cast<float>(scale.value()), out, true);
    return std::make_tuple(out, stateInplace);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("recurrent_gated_delta_rule", recurrent_gated_delta_rule_npu);
    m.impl("recurrent_gated_delta_rule_functional", recurrent_gated_delta_rule_functional_npu);
}

} // namespace RecurrentGatedDeltaRuleOp
} // namespace ascend_ops
