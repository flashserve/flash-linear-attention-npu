/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_tiling.h
 * \brief
 */
#ifndef ASCEND_OPS_GROUPED_MATMUL_TILING_H
#define ASCEND_OPS_GROUPED_MATMUL_TILING_H
#include "kernel_tiling/kernel_tiling.h"

#pragma once

class GMMBaseParams
{
public:
    uint32_t groupNum = 0;
    uint32_t coreNum = 0;
    uint32_t activeType = 0;
    uint32_t ubBaseK = 0;
    uint32_t ubBaseN = 0;
    uint32_t ubCalSize = 0;
    uint32_t ubRestBytes = 0;
    uint32_t singleWeight = 0;
    uint32_t singleX = 0;
    uint32_t singleY = 0;
    int32_t groupType = 0;
    uint32_t singleN = 0;
    uint32_t quantParam = 0;
    uint32_t groupListType = 0;
    uint32_t m = 0;
    uint32_t hasBias = 0;
    uint64_t workspaceSize = 0;
    uint64_t totalInGroup = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    uint64_t vBaseM = 0;
    uint64_t parallNum = 0;
    uint64_t quantGroupNum = 0;
    uint64_t isPreTiling = 0;
    uint32_t withOffset = 0;
    uint32_t isOutputDisableL2Cache=0;
public:
    // uint32_t 类型变量
    uint32_t get_groupNum() { return this->groupNum; }
    void set_groupNum(uint32_t value) { this->groupNum = value; }

    uint32_t get_coreNum() { return this->coreNum; }
    void set_coreNum(uint32_t value) { this->coreNum = value; }

    uint32_t get_activeType() { return this->activeType; }
    void set_activeType(uint32_t value) { this->activeType = value; }

    uint32_t get_ubBaseK() { return this->ubBaseK; }
    void set_ubBaseK(uint32_t value) { this->ubBaseK = value; }

    uint32_t get_ubBaseN() { return this->ubBaseN; }
    void set_ubBaseN(uint32_t value) { this->ubBaseN = value; }

    uint32_t get_ubCalSize() { return this->ubCalSize; }
    void set_ubCalSize(uint32_t value) { this->ubCalSize = value; }

    uint32_t get_ubRestBytes() { return this->ubRestBytes; }
    void set_ubRestBytes(uint32_t value) { this->ubRestBytes = value; }

    uint32_t get_singleWeight() { return this->singleWeight; }
    void set_singleWeight(uint32_t value) { this->singleWeight = value; }

    uint32_t get_singleX() { return this->singleX; }
    void set_singleX(uint32_t value) { this->singleX = value; }

    uint32_t get_singleY() { return this->singleY; }
    void set_singleY(uint32_t value) { this->singleY = value; }

    uint32_t get_singleN() { return this->singleN; }
    void set_singleN(uint32_t value) { this->singleN = value; }

    uint32_t get_quantParam() { return this->quantParam; }
    void set_quantParam(uint32_t value) { this->quantParam = value; }

    uint32_t get_groupListType() { return this->groupListType; }
    void set_groupListType(uint32_t value) { this->groupListType = value; }

    uint32_t get_m() { return this->m; }
    void set_m(uint32_t value) { this->m = value; }

    uint32_t get_hasBias() { return this->hasBias; }
    void set_hasBias(uint32_t value) { this->hasBias = value; }

    uint32_t get_withOffset() { return this->withOffset; }
    void set_withOffset(uint32_t value) { this->withOffset = value; }

    uint32_t get_isOutputDisableL2Cache() { return this->isOutputDisableL2Cache; }
    void set_isOutputDisableL2Cache(uint32_t value) { this->isOutputDisableL2Cache = value; }

    // int32_t 类型变量
    int32_t get_groupType() { return this->groupType; }
    void set_groupType(int32_t value) { this->groupType = value; }

    // uint64_t 类型变量
    uint64_t get_workspaceSize() { return this->workspaceSize; }
    void set_workspaceSize(uint64_t value) { this->workspaceSize = value; }

    uint64_t get_totalInGroup() { return this->totalInGroup; }
    void set_totalInGroup(uint64_t value) { this->totalInGroup = value; }

    uint64_t get_k() { return this->k; }
    void set_k(uint64_t value) { this->k = value; }

    uint64_t get_n() { return this->n; }
    void set_n(uint64_t value) { this->n = value; }

    uint64_t get_vBaseM() { return this->vBaseM; }
    void set_vBaseM(uint64_t value) { this->vBaseM = value; }

    uint64_t get_parallNum() { return this->parallNum; }
    void set_parallNum(uint64_t value) { this->parallNum = value; }

    uint64_t get_quantGroupNum() { return this->quantGroupNum; }
    void set_quantGroupNum(uint64_t value) { this->quantGroupNum = value; }

    uint64_t get_isPreTiling() { return this->isPreTiling; }
    void set_isPreTiling(uint64_t value) { this->isPreTiling = value; }
};

class GroupedMatmulArray
{
public:
    int32_t mList[128] = {};
    int32_t kList[128] = {};
    int32_t nList[128] = {};

    // GroupedMatmulArray 的 getter 和 setter
    int32_t* get_mList() { return this->mList; }
    void set_mList(const int32_t* value) {
        for (int i = 0; i < 128; i++) {
            mList[i] = value[i];
        }
    }

    int32_t* get_kList() { return this->kList; }
    void set_kList(const int32_t* value) {
        for (int i = 0; i < 128; i++) {
            kList[i] = value[i];
        }
    }

    int32_t* get_nList() { return this->nList; }
    void set_nList(const int32_t* value) {
        for (int i = 0; i < 128; i++) {
            nList[i] = value[i];
        }
    }
};

class A8W4HPTiling
{
public:
    uint32_t group_num = 0;
    int8_t group_type = 0;
    uint8_t required_core_numPH[3] = {};
    uint32_t required_core_num = 0;
    float format_in = 0;
    float format_out = 0;
    uint32_t numAic = 0;
    uint32_t numAiv = 0;
    uint8_t szUbPH[4] = {};
    uint64_t szUb = 0;
    uint64_t szL0A = 0;
    uint64_t szL0C = 0;
    uint8_t pattern = 0;
    uint8_t kernel_index = 0;
    uint8_t splitTimesPH[2] = {};
    uint32_t splitTimes = 0;
    int8_t output_type = 0;
    uint8_t ori_in0_shapePH[3] = {};
    uint32_t ori_in0_shape[2] = {};
    uint32_t ori_in1_shape[2] = {};
    uint32_t ori_out_shape[2] = {};
    uint32_t single_core_tiling[3] = {};
    uint32_t single_core_base_tiling[2] = {};
    uint32_t splitRecord[3] = {};
    uint8_t workspaceOffsetPH[4] = {};
    uint64_t workspaceOffset = 0;

    // A8W4HPTiling 的 getter 和 setter
    uint32_t get_group_num() { return this->group_num; }
    void set_group_num(uint32_t value) { this->group_num = value; }

    int8_t get_group_type() { return this->group_type; }
    void set_group_type(int8_t value) { this->group_type = value; }

    uint8_t* get_required_core_numPH() { return this->required_core_numPH; }
    void set_required_core_numPH(const uint8_t* value) {
        for(int i = 0; i < 3; i++) {
            required_core_numPH[i] = value[i];
        }
    }

    uint32_t get_required_core_num() { return this->required_core_num; }
    void set_required_core_num(uint32_t value) { this->required_core_num = value; }

    float get_format_in() { return this->format_in; }
    void set_format_in(float value) { this->format_in = value; }

    float get_format_out() { return this->format_out; }
    void set_format_out(float value) { this->format_out = value; }

    uint32_t get_numAic() { return this->numAic; }
    void set_numAic(uint32_t value) { this->numAic = value; }

    uint32_t get_numAiv() { return this->numAiv; }
    void set_numAiv(uint32_t value) { this->numAiv = value; }

    uint8_t* get_szUbPH() { return this->szUbPH; }
    void set_szUbPH(const uint8_t* value) {
        for(int i = 0; i < 4; i++) {
            szUbPH[i] = value[i];
        }
    }

    uint64_t get_szUb() { return this->szUb; }
    void set_szUb(uint64_t value) { this->szUb = value; }

    uint64_t get_szL0A() { return this->szL0A; }
    void set_szL0A(uint64_t value) { this->szL0A = value; }

    uint64_t get_szL0C() { return this->szL0C; }
    void set_szL0C(uint64_t value) { this->szL0C = value; }

    uint8_t get_pattern() { return this->pattern; }
    void set_pattern(uint8_t value) { this->pattern = value; }

    uint8_t get_kernel_index() { return this->kernel_index; }
    void set_kernel_index(uint8_t value) { this->kernel_index = value; }

    uint8_t* get_splitTimesPH() { return this->splitTimesPH; }
    void set_splitTimesPH(const uint8_t* value) {
        for(int i = 0; i < 2; i++) {
            splitTimesPH[i] = value[i];
        }
    }

    uint32_t get_splitTimes() { return this->splitTimes; }
    void set_splitTimes(uint32_t value) { this->splitTimes = value; }

    int8_t get_output_type() { return this->output_type; }
    void set_output_type(int8_t value) { this->output_type = value; }

    uint8_t* get_ori_in0_shapePH() { return this->ori_in0_shapePH; }
    void set_ori_in0_shapePH(const uint8_t* value) {
        for(int i = 0; i < 3; i++) {
            ori_in0_shapePH[i] = value[i];
        }
    }

    uint32_t* get_ori_in0_shape() { return this->ori_in0_shape; }
    void set_ori_in0_shape(const uint32_t* value) {
        for(int i = 0; i < 2; i++) {
            ori_in0_shape[i] = value[i];
        }
    }

    uint32_t* get_ori_in1_shape() { return this->ori_in1_shape; }
    void set_ori_in1_shape(const uint32_t* value) {
        for(int i = 0; i < 2; i++) {
            ori_in1_shape[i] = value[i];
        }
    }

    uint32_t* get_ori_out_shape() { return this->ori_out_shape; }
    void set_ori_out_shape(const uint32_t* value) {
        for(int i = 0; i < 2; i++) {
            ori_out_shape[i] = value[i];
        }
    }

    uint32_t* get_single_core_tiling() { return this->single_core_tiling; }
    void set_single_core_tiling(const uint32_t* value) {
        for(int i = 0; i < 3; i++) {
            single_core_tiling[i] = value[i];
        }
    }

    uint32_t* get_single_core_base_tiling() { return this->single_core_base_tiling; }
    void set_single_core_base_tiling(const uint32_t* value) {
        for(int i = 0; i < 2; i++) {
            single_core_base_tiling[i] = value[i];
        }
    }

    uint32_t* get_splitRecord() { return this->splitRecord; }
    void set_splitRecord(const uint32_t* value) {
        for(int i = 0; i < 3; i++) {
            splitRecord[i] = value[i];
        }
    }

    uint8_t* get_workspaceOffsetPH() { return this->workspaceOffsetPH; }
    void set_workspaceOffsetPH(const uint8_t* value) {
        for(int i = 0; i < 4; i++) {
            workspaceOffsetPH[i] = value[i];
        }
    }

    uint64_t get_workspaceOffset() { return this->workspaceOffset; }
    void set_workspaceOffset(uint64_t value) { this->workspaceOffset = value; }
};

class GroupedMatmulTilingData
{
public:
    GMMBaseParams gmmBaseParams;
    GroupedMatmulArray gmmArray;
    AscendC::tiling::TCubeTiling mmTilingData;
    A8W4HPTiling hpTilingData;
};

#endif