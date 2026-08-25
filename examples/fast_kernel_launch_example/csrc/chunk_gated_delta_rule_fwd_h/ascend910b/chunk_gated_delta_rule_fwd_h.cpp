/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_gated_delta_rule_fwd_h.cpp
 * \brief chunk_gated_delta_rule_fwd_h operator with fast kernel launch (<<<>>>).
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include <type_traits>
#include <vector>

// Plain tiling struct (global ChunkGatedDeltaRuleFwdHTilingData) must be visible before the kernel header.
#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h_struct.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling_processor.h"
#include "fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp"

#include "lib/matmul_intf.h"

using namespace Catlass;

namespace ascend_ops {
namespace ChunkGatedDeltaRuleFwdH {

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def(
        "chunk_gated_delta_rule_fwd_h(Tensor k, Tensor w, Tensor u, Tensor g, *, Tensor? initial_state=None, "
        "bool output_final_state=False, int chunk_size=64, int[]? cu_seqlens=None, int[]? chunk_indices=None) "
        "-> (Tensor, Tensor, Tensor)");
}

static void CheckInputs(
    const at::Tensor &k, const at::Tensor &w, const at::Tensor &u, const at::Tensor &g,
    const c10::optional<at::Tensor> &initial_state, int64_t chunk_size,
    at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    TORCH_CHECK(k.dim() == 4 && w.dim() == 4 && u.dim() == 4,
                "k, w and u must be rank-4 BNSD tensors");
    TORCH_CHECK(g.dim() == 3, "g must be a rank-3 BNT tensor");
    TORCH_CHECK(chunk_size == 64, "chunk_size must be 64, but got ", chunk_size);

    const int64_t B = k.size(0);
    const int64_t HK = k.size(1);
    const int64_t T = k.size(2);
    const int64_t K = k.size(3);
    const int64_t HV = u.size(1);
    const int64_t V = u.size(3);
    TORCH_CHECK(B > 0 && HK > 0 && T > 0 && K > 0 && HV > 0 && V > 0,
                "all logical dimensions must be positive");
    TORCH_CHECK(HV % HK == 0, "HV must be divisible by HK, but got HV=", HV, " and HK=", HK);
    TORCH_CHECK(w.size(0) == B && w.size(1) == HV && w.size(2) == T && w.size(3) == K,
                "w must have shape [B, HV, T, K]");
    TORCH_CHECK(u.size(0) == B && u.size(2) == T,
                "u must have shape [B, HV, T, V]");
    TORCH_CHECK(g.size(0) == B && g.size(1) == HV && g.size(2) == T,
                "g must have shape [B, HV, T]");
    TORCH_CHECK(V == 128, "the ascend910b fast-launch wrapper only supports V = 128, but got ", V);

    const auto input_dtype = k.scalar_type();
    TORCH_CHECK(input_dtype == at::kHalf || input_dtype == at::kBFloat16,
                "k, w and u must use float16 or bfloat16");
    TORCH_CHECK(w.scalar_type() == input_dtype && u.scalar_type() == input_dtype,
                "k, w and u must have the same dtype");
    TORCH_CHECK(g.scalar_type() == at::kFloat || g.scalar_type() == input_dtype,
                "g must use float32 or match the input dtype");
    TORCH_CHECK(w.device() == k.device() && u.device() == k.device() && g.device() == k.device(),
                "k, w, u and g must be on the same device");
    TORCH_CHECK(k.is_contiguous() && w.is_contiguous() && u.is_contiguous() && g.is_contiguous(),
                "k, w, u and g must be contiguous");

    TORCH_CHECK(cu_seqlens.has_value() == chunk_indices.has_value(),
                "cu_seqlens and chunk_indices must be provided together");
    int64_t state_batch = B;
    if (cu_seqlens.has_value()) {
        const auto cu = cu_seqlens.value();
        TORCH_CHECK(B == 1, "varlen mode requires B == 1, but got ", B);
        TORCH_CHECK(cu.size() >= 2 && cu.front() == 0 && cu.back() == T,
                    "cu_seqlens must start at 0, end at T, and contain at least two elements");
        for (size_t i = 1; i < cu.size(); ++i) {
            TORCH_CHECK(cu[i] > cu[i - 1], "cu_seqlens must be strictly increasing");
        }
        TORCH_CHECK(chunk_indices.value().size() % 2 == 0,
                    "chunk_indices must be a flattened [NT, 2] list");
        int64_t expected_chunks = 0;
        const auto indices = chunk_indices.value();
        for (size_t i = 1; i < cu.size(); ++i) {
            expected_chunks += (cu[i] - cu[i - 1] + chunk_size - 1) / chunk_size;
        }
        TORCH_CHECK(static_cast<int64_t>(indices.size()) == expected_chunks * 2,
                    "chunk_indices must contain one [sequence, local_chunk] pair per chunk");
        size_t chunk_index_offset = 0;
        for (size_t i = 1; i < cu.size(); ++i) {
            const int64_t sequence_chunks =
                (cu[i] - cu[i - 1] + chunk_size - 1) / chunk_size;
            for (int64_t local_chunk = 0; local_chunk < sequence_chunks; ++local_chunk) {
                TORCH_CHECK(indices[chunk_index_offset] == static_cast<int64_t>(i - 1) &&
                                indices[chunk_index_offset + 1] == local_chunk,
                            "chunk_indices must use canonical sequence-major [sequence, local_chunk] pairs");
                chunk_index_offset += 2;
            }
        }
        state_batch = static_cast<int64_t>(cu.size()) - 1;
    }

    if (initial_state.has_value()) {
        const auto &state = initial_state.value();
        TORCH_CHECK(state.dim() == 4 && state.size(0) == state_batch && state.size(1) == HV &&
                        state.size(2) == K && state.size(3) == V,
                    "initial_state must have shape [N, HV, K, V]");
        TORCH_CHECK(state.scalar_type() == at::kFloat || state.scalar_type() == at::kBFloat16,
                    "initial_state must use float32 or bfloat16");
        TORCH_CHECK(state.device() == k.device(), "initial_state must be on the same device as k");
        TORCH_CHECK(state.is_contiguous(), "initial_state must be contiguous");
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_fwd_h_meta(
    const at::Tensor &k, const at::Tensor &w, const at::Tensor &u, const at::Tensor &g,
    const c10::optional<at::Tensor> &initial_state, bool output_final_state, int64_t chunk_size,
    at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    CheckInputs(k, w, u, g, initial_state, chunk_size, cu_seqlens, chunk_indices);
    auto k_sizes = k.sizes();
    auto u_sizes = u.sizes();
    int64_t B = k_sizes[0];
    int64_t T = k_sizes[2];
    int64_t K = k_sizes[3];
    int64_t HV = u_sizes[1];
    int64_t V = u_sizes[3];
    int64_t NT = 0;
    if (chunk_indices.has_value()) {
        NT = static_cast<int64_t>(chunk_indices.value().size()) / 2;
    } else {
        NT = (T + chunk_size - 1) / chunk_size;
    }

    at::Tensor h_out = at::zeros({B, HV, NT, K, V}, k.options());
    at::Tensor v_new_out = at::empty_like(u);
    auto state_options = initial_state.has_value()
                             ? initial_state.value().options()
                             : k.options().dtype(at::kFloat);
    at::Tensor final_state_out;
    if (output_final_state) {
        int64_t N = cu_seqlens.has_value() ? static_cast<int64_t>(cu_seqlens.value().size()) - 1 : B;
        final_state_out = at::empty({N, HV, K, V}, state_options);
    } else {
        final_state_out = at::empty({0}, state_options);
    }
    return std::make_tuple(h_out, v_new_out, final_state_out);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("chunk_gated_delta_rule_fwd_h", chunk_gated_delta_rule_fwd_h_meta);
}

static ::ChunkGatedDeltaRuleFwdHTilingData calc_tiling_params(
    const at::Tensor &k, const at::Tensor &u,
    const c10::optional<at::Tensor> &initial_state, bool output_final_state, int64_t chunk_size,
    at::OptionalIntArrayRef cu_seqlens, uint32_t &blockDim, size_t &workspaceSize)
{
    optiling::ChunkGatedDeltaRuleFwdHTilingContext ctx{};
    ctx.seqlen = k.size(2);
    ctx.kNumHead = k.size(1);
    ctx.kHeadDim = k.size(3);
    ctx.vNumHead = u.size(1);
    ctx.vHeadDim = u.size(3);
    ctx.shapeBatchDim = k.size(0);
    ctx.hasCuSeqlens = cu_seqlens.has_value();
    ctx.cuSeqlensDim0 = cu_seqlens.has_value() ? static_cast<int64_t>(cu_seqlens.value().size()) : 0;
    ctx.useInitialState = initial_state.has_value();
    ctx.storeFinalState = output_final_state;
    ctx.chunkSize = chunk_size;

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    ctx.aicCoreNum = ascendcPlatform->GetCoreNumAic();
    ctx.libApiWorkSpaceSize = ascendcPlatform->GetLibApiWorkSpaceSize();

    ::ChunkGatedDeltaRuleFwdHTilingData tiling{};
    optiling::ChunkGatedDeltaRuleFwdHTilingProcessor processor(ctx);
    processor.Process(tiling, blockDim, workspaceSize);
    TORCH_CHECK(
        blockDim > 0,
        "chunk_gated_delta_rule_fwd_h: cannot resolve head sharding for v_heads=",
        u.size(1), ", available_core_num=", ctx.aicCoreNum);
    return tiling;
}

template <typename INPUT_TYPE, typename G_TYPE, typename STATE_TYPE>
__global__ __aicore__ void chunk_gated_delta_rule_fwd_h_kernel(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
    GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::SetSysWorkspaceForce(workspace);
    GM_ADDR user = AscendC::GetUserWorkspace(workspace);
    if (user == nullptr) {
        return;
    }

    using GDNFwdHKernel = Catlass::Gemm::Kernel::GDNFwdHKernel<
        INPUT_TYPE, G_TYPE, STATE_TYPE, float,
        Catlass::Gemm::Kernel::GDNFwdHTileShapes128,
        GDN_FWD_H_GATE_G, GDN_FWD_H_EXP_E>;
    GDNFwdHKernel gdnFwdH;
    gdnFwdH.Init(
        k, w, u, g, nullptr, inital_state, cu_seqlens, chunk_indices,
        h, v_new, final_state, tiling, user);
    gdnFwdH.Process();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_fwd_h_npu(
    const at::Tensor &k, const at::Tensor &w, const at::Tensor &u, const at::Tensor &g,
    const c10::optional<at::Tensor> &initial_state, bool output_final_state, int64_t chunk_size,
    at::OptionalIntArrayRef cu_seqlens, at::OptionalIntArrayRef chunk_indices)
{
    const c10::OptionalDeviceGuard guard(k.device());

    auto outs = chunk_gated_delta_rule_fwd_h_meta(
        k, w, u, g, initial_state, output_final_state, chunk_size, cu_seqlens, chunk_indices);
    at::Tensor h_out = std::get<0>(outs);
    at::Tensor v_new_out = std::get<1>(outs);
    at::Tensor final_state_out = std::get<2>(outs);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    uint32_t blockDim = 0;
    size_t workspaceSize = 0;
    auto tiling = calc_tiling_params(
        k, u, initial_state, output_final_state, chunk_size, cu_seqlens, blockDim, workspaceSize);

    auto k_ptr = (GM_ADDR)k.data_ptr();
    auto w_ptr = (GM_ADDR)w.data_ptr();
    auto u_ptr = (GM_ADDR)u.data_ptr();
    auto g_ptr = (GM_ADDR)g.data_ptr();
    auto h_ptr = (GM_ADDR)h_out.data_ptr();
    auto v_new_ptr = (GM_ADDR)v_new_out.data_ptr();
    auto final_state_ptr = (GM_ADDR)final_state_out.data_ptr();

    GM_ADDR initial_state_ptr = nullptr;
    if (initial_state.has_value()) {
        initial_state_ptr = (GM_ADDR)initial_state.value().data_ptr();
    }

    GM_ADDR cu_seqlens_ptr = nullptr;
    GM_ADDR chunk_indices_ptr = nullptr;
    std::vector<int64_t> cu_seqlens_vec;
    std::vector<int64_t> chunk_indices_vec;
    at::Tensor cu_seqlens_tensor;
    at::Tensor chunk_indices_tensor;
    if (cu_seqlens.has_value()) {
        cu_seqlens_vec = std::vector<int64_t>(cu_seqlens.value().begin(), cu_seqlens.value().end());
        cu_seqlens_tensor = at::tensor(cu_seqlens_vec, at::dtype(at::kLong).device(k.device()));
        cu_seqlens_ptr = (GM_ADDR)cu_seqlens_tensor.data_ptr();
    }
    if (chunk_indices.has_value()) {
        chunk_indices_vec = std::vector<int64_t>(chunk_indices.value().begin(), chunk_indices.value().end());
        chunk_indices_tensor = at::tensor(chunk_indices_vec, at::dtype(at::kLong).device(k.device()));
        chunk_indices_ptr = (GM_ADDR)chunk_indices_tensor.data_ptr();
    }

    void *workspace_ptr = nullptr;
    if (workspaceSize > 0) {
        auto ret = aclrtMalloc(&workspace_ptr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate workspace failed. ERROR: ", ret);
        // The kernel uses AscendC::SyncAll(), whose cross-core counter lives in the system
        // workspace region. A raw aclrtMalloc buffer is not zero-initialized, so the counter must
        // be cleared before launch, otherwise the cores desync and trigger an aicore exception.
        ret = aclrtMemsetAsync(workspace_ptr, workspaceSize, 0, workspaceSize, stream);
        TORCH_CHECK(ret == ACL_SUCCESS, "memset workspace failed. ERROR: ", ret);
    }
    auto workspace_gm = (GM_ADDR)workspace_ptr;

    // The kernel reads the tiling data from GM, so copy the plain tiling struct to a device buffer.
    void *tiling_ptr = nullptr;
    const size_t tiling_bytes = sizeof(::ChunkGatedDeltaRuleFwdHTilingData);
    {
        auto ret = aclrtMalloc(&tiling_ptr, tiling_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "allocate tiling buffer failed. ERROR: ", ret);
        ret = aclrtMemcpy(tiling_ptr, tiling_bytes, &tiling, tiling_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        TORCH_CHECK(ret == ACL_SUCCESS, "copy tiling to device failed. ERROR: ", ret);
    }
    auto tiling_gm = (GM_ADDR)tiling_ptr;

    auto in_dtype = k.scalar_type();
    auto g_dtype = g.scalar_type();
    auto state_dtype = initial_state.has_value() ? initial_state.value().scalar_type() : at::kFloat;

    auto acl_call = [=]() -> int {
        bool gIsFp32 = (g_dtype == at::kFloat);
        bool stateIsFp32 = (!initial_state.has_value()) || (state_dtype == at::kFloat);
        if (in_dtype == at::kBFloat16) {
            using INPUT_TYPE = bfloat16_t;
            if (stateIsFp32) {
                if (gIsFp32) {
                    chunk_gated_delta_rule_fwd_h_kernel<INPUT_TYPE, float, float><<<blockDim, nullptr, stream>>>(
                        k_ptr, w_ptr, u_ptr, g_ptr, initial_state_ptr, cu_seqlens_ptr, chunk_indices_ptr,
                        h_ptr, v_new_ptr, final_state_ptr, workspace_gm, tiling_gm);
                } else {
                    chunk_gated_delta_rule_fwd_h_kernel<INPUT_TYPE, INPUT_TYPE, float><<<blockDim, nullptr, stream>>>(
                        k_ptr, w_ptr, u_ptr, g_ptr, initial_state_ptr, cu_seqlens_ptr, chunk_indices_ptr,
                        h_ptr, v_new_ptr, final_state_ptr, workspace_gm, tiling_gm);
                }
            } else {
                if (gIsFp32) {
                    chunk_gated_delta_rule_fwd_h_kernel<INPUT_TYPE, float, bfloat16_t><<<blockDim, nullptr, stream>>>(
                        k_ptr, w_ptr, u_ptr, g_ptr, initial_state_ptr, cu_seqlens_ptr, chunk_indices_ptr,
                        h_ptr, v_new_ptr, final_state_ptr, workspace_gm, tiling_gm);
                } else {
                    chunk_gated_delta_rule_fwd_h_kernel<INPUT_TYPE, INPUT_TYPE, bfloat16_t><<<blockDim, nullptr, stream>>>(
                        k_ptr, w_ptr, u_ptr, g_ptr, initial_state_ptr, cu_seqlens_ptr, chunk_indices_ptr,
                        h_ptr, v_new_ptr, final_state_ptr, workspace_gm, tiling_gm);
                }
            }
        } else if (in_dtype == at::kHalf) {
            using INPUT_TYPE = half;
            if (stateIsFp32) {
                if (gIsFp32) {
                    chunk_gated_delta_rule_fwd_h_kernel<INPUT_TYPE, float, float><<<blockDim, nullptr, stream>>>(
                        k_ptr, w_ptr, u_ptr, g_ptr, initial_state_ptr, cu_seqlens_ptr, chunk_indices_ptr,
                        h_ptr, v_new_ptr, final_state_ptr, workspace_gm, tiling_gm);
                } else {
                    chunk_gated_delta_rule_fwd_h_kernel<INPUT_TYPE, INPUT_TYPE, float><<<blockDim, nullptr, stream>>>(
                        k_ptr, w_ptr, u_ptr, g_ptr, initial_state_ptr, cu_seqlens_ptr, chunk_indices_ptr,
                        h_ptr, v_new_ptr, final_state_ptr, workspace_gm, tiling_gm);
                }
            } else {
                if (gIsFp32) {
                    chunk_gated_delta_rule_fwd_h_kernel<INPUT_TYPE, float, bfloat16_t><<<blockDim, nullptr, stream>>>(
                        k_ptr, w_ptr, u_ptr, g_ptr, initial_state_ptr, cu_seqlens_ptr, chunk_indices_ptr,
                        h_ptr, v_new_ptr, final_state_ptr, workspace_gm, tiling_gm);
                } else {
                    chunk_gated_delta_rule_fwd_h_kernel<INPUT_TYPE, INPUT_TYPE, bfloat16_t><<<blockDim, nullptr, stream>>>(
                        k_ptr, w_ptr, u_ptr, g_ptr, initial_state_ptr, cu_seqlens_ptr, chunk_indices_ptr,
                        h_ptr, v_new_ptr, final_state_ptr, workspace_gm, tiling_gm);
                }
            }
        } else {
            TORCH_CHECK(false, "Unsupported input dtype: ", in_dtype);
        }
        return 0;
    };

    at_npu::native::OpCommand::RunOpApi("ChunkGatedDeltaRuleFwdH", acl_call);

    // The tiling/workspace buffers are raw aclrtMalloc (not stream-ordered), so the kernel must
    // finish before freeing them. RunOpApi only ENQUEUES the launch into torch_npu's async task
    // queue; a raw aclrtSynchronizeStream does NOT drain that software queue, so the launch could
    // still be pending and the frees below would race with the (later) kernel reading the tiling
    // from GM. NPUStream::synchronize() empties the task queue first and then syncs the hardware
    // stream, guaranteeing the kernel has run before we free.
    c10_npu::getCurrentNPUStream().synchronize();

    if (workspace_ptr != nullptr) {
        aclrtFree(workspace_ptr);
        workspace_ptr = nullptr;
    }
    if (tiling_ptr != nullptr) {
        aclrtFree(tiling_ptr);
        tiling_ptr = nullptr;
    }

    if (output_final_state) {
        return std::make_tuple(h_out, v_new_out, final_state_out);
    }
    return std::make_tuple(h_out, v_new_out, at::Tensor());
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("chunk_gated_delta_rule_fwd_h", chunk_gated_delta_rule_fwd_h_npu);
}

} // namespace ChunkGatedDeltaRuleFwdH
} // namespace ascend_ops
