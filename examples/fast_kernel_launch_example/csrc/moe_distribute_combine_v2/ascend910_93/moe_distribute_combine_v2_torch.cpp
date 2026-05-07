/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_v2_torch.cpp
 * \brief
 */

#include <ATen/ATen.h>
#include <vector>
#include <torch/all.h>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_operator.h"
#include "moe_distribute_combine_v2_entry.h"
#include "op_kernel/moe_distribute_combine_v2_tiling.h"
#include "../../moe_distribute_dispatch_v2/ascend910_93/moe_distribute_dispatch_v2_torch.h"

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {

namespace MoeDistributeCombineV2 {

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("MoeDistributeCombineV2(Tensor expand_x, Tensor expert_ids, Tensor assist_info_for_combine, " \
            "Tensor ep_send_counts, Tensor expert_scales, Tensor mc2_context, str group_ep, int ep_world_size, " \
            "int ep_rank_id, int moe_expert_num, int total_winsize_ep, *, " \
            "Tensor? x_active_mask=None, Tensor? shared_expert_x=None, Tensor? ori_x=None, " \
            "Tensor? const_expert_alpha_1=None, Tensor? const_expert_alpha_2=None, Tensor? const_expert_v=None, " \
            "Tensor? performance_info=None, " \
            "int expert_shard_type=0, int shared_expert_num=0, int shared_expert_rank_num=0, " \
            "int global_bs=0, int comm_quant_mode=0, " \
            "str comm_alg=\"\", int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0) " \
            "-> Tensor");
}

constexpr uint32_t DIM_ONE = 1UL;
constexpr uint32_t DIM_TWO = 2UL;
constexpr uint32_t TILINGKEY_XTYPE = 10;
constexpr uint32_t WORKSPACESIZE = 16 * 1024 *1024;
constexpr int64_t INT8_COMM_QUANT = 2;
constexpr uint32_t BUFFER_SINGLE = 1;
constexpr uint32_t BUFFER_DOUBLE = 2;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint32_t DTYPE_SIZE_HALF = 2;
constexpr uint32_t ALIGNED_LEN = 256U;
constexpr uint32_t STATE_OFFSET = 32U;

static void calculate_tilingkey(int32_t &tilingKey, at::ScalarType xType, const uint32_t quantMode)
{
    tilingKey += static_cast<uint64_t>(quantMode);
    if (xType == at::kBFloat16) {
        tilingKey += static_cast<uint64_t>(TILINGKEY_XTYPE);
    }
    return;
}

void MoeDistributeCombineV2_api(aclrtStream stream,
                                const at::Tensor &expand_x, const at::Tensor &expert_ids,
                                const at::Tensor &assist_info_for_combine,
                                const at::Tensor &ep_send_counts, const at::Tensor &expert_scales,
                                const at::Tensor &new_workspace, const at::Tensor &mc2_context,
                                const c10::optional<at::Tensor> &x_active_mask,
                                const c10::optional<at::Tensor> &shared_expert_x,
                                const c10::optional<at::Tensor> &ori_x,
                                const c10::optional<at::Tensor> &const_expert_alpha_1,
                                const c10::optional<at::Tensor> &const_expert_alpha_2,
                                const c10::optional<at::Tensor> &const_expert_v,
                                const c10::optional<at::Tensor> &performance_info,
                                at::Tensor &x_out, int64_t comm_quant_mode,
                                MoeDistributeCombineV2Info tilingData)
{
    auto expandX_ptr = get_first_tensor_address<at::Tensor>(expand_x.scalar_type(), expand_x, false);
    auto expertIds_ptr = get_first_tensor_address<at::Tensor>(expert_ids.scalar_type(), expert_ids, false);
    auto expandIdx_ptr = get_first_tensor_address<at::Tensor>(assist_info_for_combine.scalar_type(),
        assist_info_for_combine, false);
    auto epSendCount_ptr = get_first_tensor_address<at::Tensor>(ep_send_counts.scalar_type(), ep_send_counts, false);
    auto expertScales_ptr = get_first_tensor_address<at::Tensor>(expert_scales.scalar_type(), expert_scales, false);
    auto workspace_ptr = get_first_tensor_address<at::Tensor>(new_workspace.scalar_type(), new_workspace, false);
    auto mc2Context_ptr = get_first_tensor_address<at::Tensor>(mc2_context.scalar_type(), mc2_context, false);


    void* xActiveMask_ptr = nullptr;
    if(x_active_mask.has_value()) {
        xActiveMask_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(x_active_mask->scalar_type(),
            x_active_mask, false);
    }

    void* sharedExpertX_ptr = nullptr;
    if(shared_expert_x.has_value()) {
        sharedExpertX_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(shared_expert_x->scalar_type(),
            shared_expert_x, false);
    }

    void* oriX_ptr = nullptr;
    if(ori_x.has_value()) {
        oriX_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(ori_x->scalar_type(), ori_x, false);
    }

    void* constExpertAlpha1_ptr = nullptr;
    if(const_expert_alpha_1.has_value()) {
        constExpertAlpha1_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(const_expert_alpha_1->scalar_type(),
            const_expert_alpha_1, false);
    }

    void* constExpertAlpha2_ptr = nullptr;
    if(const_expert_alpha_2.has_value()) {
        constExpertAlpha2_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(const_expert_alpha_2->scalar_type(),
            const_expert_alpha_2, false);
    }

    void* constExpertV_ptr = nullptr;
    if(const_expert_v.has_value()) {
        constExpertV_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(const_expert_v->scalar_type(),
            const_expert_v, false);
    }

    void* performanceInfo_ptr = nullptr;
    if(performance_info.has_value()) {
        performanceInfo_ptr = get_first_tensor_address<c10::optional<at::Tensor>>(performance_info->scalar_type(),
            performance_info, false);
    }
    void* residualX_ptr = nullptr;
    void* gamma_ptr = nullptr;
    auto XOut_ptr = get_first_tensor_address<at::Tensor>(x_out.scalar_type(), x_out, false);

    
    int32_t tilingKey = 100;
    calculate_tilingkey(tilingKey, expand_x.scalar_type(), comm_quant_mode);

    moe_distribute_combine_v2_entry(tilingKey, tilingData.aivNum, stream, (GM_ADDR)expandX_ptr, (GM_ADDR)expertIds_ptr,
    (GM_ADDR)expandIdx_ptr, (GM_ADDR)epSendCount_ptr, (GM_ADDR)residualX_ptr,
    (GM_ADDR)gamma_ptr, (GM_ADDR)expertScales_ptr, (GM_ADDR)xActiveMask_ptr, (GM_ADDR)sharedExpertX_ptr,
    (GM_ADDR)oriX_ptr, (GM_ADDR)constExpertAlpha1_ptr, (GM_ADDR)constExpertAlpha2_ptr,
    (GM_ADDR)constExpertV_ptr, (GM_ADDR)performanceInfo_ptr, (GM_ADDR)XOut_ptr,
    (GM_ADDR)workspace_ptr, (GM_ADDR)mc2Context_ptr, tilingData);
}

void calculate_buffernum(MoeDistributeCombineV2Info &tilingData, const at::Tensor &expand_x, int64_t comm_quant_mode)
{
    uint32_t axisH = tilingData.h;
    uint32_t axisBS = tilingData.bs;
    uint32_t axisK = tilingData.k;
    uint32_t zeroExpertNum = tilingData.zeroExpertNum;
    uint32_t copyExpertNum = tilingData.copyExpertNum;
    uint32_t constExpertNum = tilingData.constExpertNum;
    bool isInputExpertMaskFlag = tilingData.isExpertMask;
    bool isInputTokenMaskFlag = tilingData.isTokenMask;
    bool enableSpecialExpert = (constExpertNum + zeroExpertNum + copyExpertNum > 0U);
    uint32_t maxSizeTokenBuf = (axisH * expand_x.element_size() + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t hExpandXTypeSize = axisH * expand_x.element_size();
    uint32_t activeMaskAlignSize = axisBS * ((axisK * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
    uint32_t hExpandXAlign32Size = (hExpandXTypeSize + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t hFloatSize = axisH * static_cast<uint32_t>(sizeof(float));
    uint32_t hFloatAlign32Size = (hFloatSize + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size;
    uint32_t flagRcvCount = axisK + tilingData.sharedExpertNum;
    uint32_t hFloatAlign256Size = (hFloatSize + ALIGNED_LEN - 1) / ALIGNED_LEN * ALIGNED_LEN;
    uint32_t bsKNum = axisBS * axisK;
    uint32_t bsKFloatAlign = (bsKNum * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t mulBufSize = hFloatAlign256Size > bsKFloatAlign ? hFloatAlign256Size : bsKFloatAlign;

    if (isInputExpertMaskFlag || enableSpecialExpert) {
        uint32_t activeMaskAlignHalfSize = activeMaskAlignSize * sizeof(DTYPE_SIZE_HALF);
        maxSizeTokenBuf = (activeMaskAlignSize > hExpandXAlign32Size ? activeMaskAlignSize : hExpandXAlign32Size);
        maxSizeRowTmpFloatBuf = (activeMaskAlignHalfSize > hFloatAlign32Size ? activeMaskAlignHalfSize : hFloatAlign32Size);
    }

    // LocalWindowCopy的ub使用总量
    uint32_t totalBufferSize = 0;
    totalBufferSize = maxSizeTokenBuf + maxSizeRowTmpFloatBuf + mulBufSize + hFloatAlign32Size + hExpandXAlign32Size
        * BUFFER_DOUBLE + flagRcvCount * STATE_OFFSET * BUFFER_DOUBLE + UB_ALIGN;
    if (comm_quant_mode == INT8_COMM_QUANT) {
        uint32_t scaleNum = (hExpandXAlign32Size / expand_x.element_size()) / static_cast<uint32_t>(UB_ALIGN / sizeof(float));
        uint32_t scaleNumAlignSize = (scaleNum * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        totalBufferSize += scaleNumAlignSize;
    }
    if (isInputTokenMaskFlag) {
        uint32_t axisBsAlignSize = (axisBS * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        totalBufferSize += axisBsAlignSize + axisBsAlignSize * sizeof(DTYPE_SIZE_HALF) * BUFFER_DOUBLE;
    }
    if (isInputExpertMaskFlag) {
        totalBufferSize += (axisBS * sizeof(DTYPE_SIZE_HALF) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN + (axisBS * sizeof(int32_t) +
            UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN + (axisBS * axisK * sizeof(bool) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    }
    if (enableSpecialExpert && !isInputExpertMaskFlag) {
        totalBufferSize += (axisBS * sizeof(DTYPE_SIZE_HALF) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    }
    tilingData.bufferNum = totalBufferSize > tilingData.totalUbSize ? BUFFER_SINGLE : BUFFER_DOUBLE;
}

void calculate_tilingdata(MoeDistributeCombineV2Info &tilingData, int64_t ep_world_size, int64_t ep_rank_id,
    int64_t moe_expert_num, int64_t total_winsize_ep,
    int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num,
    int64_t global_bs, int64_t bs, int64_t h, int64_t k, int64_t a,
    int64_t zero_expert_num, int64_t copy_expert_num, int64_t const_expert_num, int64_t comm_quant_mode,
    const at::Tensor &expand_x,
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &shared_expert_x,
    const c10::optional<at::Tensor> &performance_info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSizePlatFrom;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatFrom);

    tilingData.epWorldSize = ep_world_size;
    tilingData.epRankId = ep_rank_id;
    tilingData.expertShardType = expert_shard_type;
    tilingData.sharedExpertNum = shared_expert_num;
    tilingData.sharedExpertRankNum = shared_expert_rank_num;
    tilingData.moeExpertNum = moe_expert_num;
    tilingData.moeExpertPerRankNum = moe_expert_num / (ep_world_size - shared_expert_rank_num);
    tilingData.zeroExpertNum = zero_expert_num;
    tilingData.copyExpertNum = copy_expert_num;
    tilingData.constExpertNum = const_expert_num;
    tilingData.globalBs = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    tilingData.bs = bs;
    tilingData.k = k;
    tilingData.h = h;
    tilingData.a = a;
    tilingData.aivNum = ascendcPlatform->GetCoreNumAiv();
    tilingData.isTokenMask = (x_active_mask.has_value() && x_active_mask->dim() == DIM_ONE);
    tilingData.isExpertMask = (x_active_mask.has_value() && x_active_mask->dim() == DIM_TWO);
    tilingData.hasSharedExpertX = shared_expert_x.has_value();  // 或根据实际变量名调整
    tilingData.isPerformance = performance_info.has_value();
    tilingData.reserved0 = false;
    tilingData.reserved1 = false;
    tilingData.reserved2 = false;
    tilingData.totalUbSize = ubSizePlatFrom;
    tilingData.totalWinSizeEp = total_winsize_ep;
    calculate_buffernum(tilingData, expand_x, comm_quant_mode);
}

at::Tensor npu_moe_distribute_combine_v2(
    const at::Tensor &expand_x, const at::Tensor &expert_ids,
    const at::Tensor &assist_info_for_combine,
    const at::Tensor &ep_send_counts, const at::Tensor &expert_scales, const at::Tensor &mc2_context,
    c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
    int64_t moe_expert_num, int64_t total_winsize_ep, 
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &shared_expert_x,
    const c10::optional<at::Tensor> &ori_x,
    const c10::optional<at::Tensor> &const_expert_alpha_1,
    const c10::optional<at::Tensor> &const_expert_alpha_2,
    const c10::optional<at::Tensor> &const_expert_v,
    const c10::optional<at::Tensor> &performance_info,
    int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num,
    int64_t global_bs, int64_t comm_quant_mode,
    c10::string_view comm_alg, int64_t zero_expert_num, int64_t copy_expert_num, int64_t const_expert_num)
{
    auto expand_x_size = expand_x.sizes();
    auto expert_ids_size = expert_ids.sizes();
    
    int64_t bs = expert_ids_size[0];
    int64_t h = expand_x_size[1];
    int64_t k = expert_ids_size[1];

    bool is_shared_default = ((shared_expert_num == 1) && (shared_expert_rank_num == 0));
    bool is_no_shared = ((shared_expert_num == 0) && (shared_expert_rank_num == 0));
    
    bool shared_front = (expert_shard_type == 0);
    int64_t local_moe_expert_num = 1;
    int64_t global_bs_real = (global_bs == 0) ? (bs * ep_world_size) : global_bs;
    int64_t a = 0;
    if (shared_front) {
        if (ep_rank_id < shared_expert_rank_num) {
            local_moe_expert_num = 1;
            int64_t max_bs = global_bs_real / ep_world_size;  // 前面已有拦截，保证ep_world_size > 0
            int64_t rank_num_per_shared_expert = shared_expert_rank_num / shared_expert_num;  // 前面已有拦截, 保证进入该分支时shared_expert_num > 0
            int64_t max_shared_group_num = (ep_world_size + rank_num_per_shared_expert - 1) / rank_num_per_shared_expert;
            a = max_bs * max_shared_group_num;
        } else {
            local_moe_expert_num = moe_expert_num / (ep_world_size - shared_expert_rank_num);
            a = global_bs_real * std::min(local_moe_expert_num, k);
        }
    }
    TORCH_CHECK((expand_x.dim() == DIM_TWO) && (expert_ids.dim() == DIM_TWO), "The x and expert_ids should be 2D");
    TORCH_CHECK((expand_x.scalar_type() == at::kBFloat16) || (expand_x.scalar_type() == at::kHalf)
        || (expand_x.scalar_type() == at::kInt),
        "dtype of expand_x should be BFloat16, Float16 or Int, but got " + std::string(c10::toString(expand_x.scalar_type())));
    TORCH_CHECK(expert_ids.scalar_type() == at::kInt,
                "dtype of expert_ids should be Int, but got " + std::string(c10::toString(expert_ids.scalar_type())));

    char *group_ep_ptr = const_cast<char *>(group_ep.data());

    at::Tensor output;
    at::Tensor new_workspace = at::empty({WORKSPACESIZE / 4}, expert_ids.options().dtype(at::kInt));
    if (expand_x.scalar_type() != at::kInt) {
        output = at::empty({bs, h}, expert_ids.options().dtype(expand_x.scalar_type()));
    } else {
        output = at::empty({bs, h}, expert_ids.options().dtype(at::kBFloat16));
    }

    c10::optional<at::Tensor> nulltensor = c10::nullopt;
    int64_t out_dtype = 0;
    int64_t group_list_type = 0;

    std::string comm_alg_str = std::string(comm_alg);
    char *comm_alg_ptr = const_cast<char *>(comm_alg_str.c_str());

    MoeDistributeCombineV2Info tilingData;
    calculate_tilingdata(tilingData, ep_world_size, ep_rank_id, moe_expert_num, total_winsize_ep,
        expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, bs, h, k, a, zero_expert_num,
        copy_expert_num, const_expert_num, comm_quant_mode, expand_x, x_active_mask, shared_expert_x, performance_info);

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() mutable -> int {
        MoeDistributeCombineV2_api(stream, expand_x, expert_ids, assist_info_for_combine, ep_send_counts,
            expert_scales, new_workspace, mc2_context, x_active_mask,
            shared_expert_x, ori_x, const_expert_alpha_1, const_expert_alpha_2,
            const_expert_v, performance_info, output, comm_quant_mode, tilingData);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("moeDistributeCombineV2", acl_call);
    
    return output;
}

at::Tensor npu_moe_distribute_combine_v2_meta(
    const at::Tensor &expand_x, const at::Tensor &expert_ids,
    const at::Tensor &assist_info_for_combine,
    const at::Tensor &ep_send_counts, const at::Tensor &expert_scales, const at::Tensor &mc2_context,
    c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id,
    int64_t moe_expert_num, int64_t total_winsize_ep, 
    const c10::optional<at::Tensor> &x_active_mask,
    const c10::optional<at::Tensor> &shared_expert_x,
    const c10::optional<at::Tensor> &ori_x,
    const c10::optional<at::Tensor> &const_expert_alpha_1,
    const c10::optional<at::Tensor> &const_expert_alpha_2,
    const c10::optional<at::Tensor> &const_expert_v,
    const c10::optional<at::Tensor> &performance_info,
    int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num,
    int64_t global_bs, int64_t comm_quant_mode,
    c10::string_view comm_alg, int64_t zero_expert_num, int64_t copy_expert_num, int64_t const_expert_num)
{
    auto expert_ids_size = expert_ids.sizes();
    auto expand_x_size = expand_x.sizes();
    
    int64_t bs = expert_ids_size[0];
    int64_t h = expand_x_size[1];
    
    at::Tensor output;

    if (expand_x.scalar_type() != at::kInt) {
        output = at::empty({bs, h}, expert_ids.options().dtype(expand_x.scalar_type()));
    } else {
        output = at::empty({bs, h}, expert_ids.options().dtype(at::kBFloat16));
    }
    return output;
}

TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("MoeDistributeCombineV2", TORCH_FN(npu_moe_distribute_combine_v2));
}

TORCH_LIBRARY_IMPL(ascend_ops, Meta, m)
{
    m.impl("MoeDistributeCombineV2", &npu_moe_distribute_combine_v2_meta);
}

} // namespace MoeDistributeCombineV2
} // namespace ascend_ops
