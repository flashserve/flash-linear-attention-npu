#include "op_common/log/log.h"
#pragma once

// aclnnChunkKdaFwdV2 chunk_size=32 分支的路由实现（issue #829），由独立 L0 算子
// ChunkKdaFwdFused 承载；不满足融合 kernel 契约即报错。
// 本文件是 aclnn_chunk_kda_fwd.cpp 的 body-include（在其匿名命名空间内、共享声明
// 之后 include 一次），不供其它编译单元使用。
// kernel 契约:[B*H,T,K] 头连续 bf16(batch 拍平进头维);K=V=128;任意 T>0
// (尾块核内钳制);HV % H == 0;dense batch 使用单启动任务循环;
// 原始 g + a_log(fp32 [H]) + dt_bias(fp32 [H*K]);beta 核内 sigmoid、q/k 核内 L2 归一;
// 产出 attn_out + 可选 final_state;可选 initial_state(fp32 [1,H,128,128],state_v_first=false);
// varlen 支持 TND/NTD;不支持中间态导出。
aclnnStatus RunChunkKdaFwd32FusedVarlen(ChunkKdaFwdParams &params, aclOpExecutor *executor,
                                        const KdaShapeInfo &info);

aclnnStatus CheckChunkKdaFwd32Layout(const ChunkKdaFwdParams &params, KdaFwdLayout &layout)
{
    if (params.layout == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnChunkKdaFwd", "layout");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (std::strcmp(params.layout, "BSND") == 0) layout = KdaFwdLayout::BSND;
    else if (std::strcmp(params.layout, "BNSD") == 0) layout = KdaFwdLayout::BNSD;
    else if (std::strcmp(params.layout, "TND") == 0) layout = KdaFwdLayout::TND;
    else if (std::strcmp(params.layout, "NTD") == 0) layout = KdaFwdLayout::NTD;
    else {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "layout", params.layout, "BSND, BNSD, TND or NTD");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckChunkKdaFwd32Shapes(const ChunkKdaFwdParams &params, KdaFwdLayout layout,
                                     KdaShapeInfo &info)
{
    info.isRank3 = layout == KdaFwdLayout::TND || layout == KdaFwdLayout::NTD;
    const size_t rank = info.isRank3 ? 3 : 4;
    const size_t betaRank = info.isRank3 ? 2 : 3;
    const std::string qExpected = info.isRank3 ? "[T,H,K] or [H,T,K]" : "[B,T,H,K] or [B,H,T,K]";
    const aclTensor *tokenTensors[] = {params.q, params.k, params.v, params.g};
    const char *tokenNames[] = {"q", "k", "v", "g"};
    for (size_t i = 0; i < 4; ++i) {
        if (Rank(tokenTensors[i]) != rank) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", tokenNames[i], op::ToString(tokenTensors[i]->GetViewShape()).GetString(), qExpected);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (Rank(params.beta) != betaRank) {
        OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "beta", op::ToString(params.beta->GetViewShape()).GetString(), info.isRank3 ? "[T,HV] or [HV,T]" : "[B,T,HV] or [B,HV,T]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!SameShape(params.q, params.k)) {
        OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "k", op::ToString(params.k->GetViewShape()).GetString(), op::ToString(params.q->GetViewShape()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (layout == KdaFwdLayout::TND) {
        info.batch = 1;
        info.seqlen = Dim(params.q, 0);
        info.hNum = Dim(params.q, 1);
        info.kDim = Dim(params.q, 2);
        info.hvNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 2);
        if (!HasShape(params.v, {info.seqlen, info.hvNum, info.vDim})) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "v", op::ToString(params.v->GetViewShape()).GetString(), "[T,HV,V]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.g, {info.seqlen, info.hvNum, info.kDim})) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "g", op::ToString(params.g->GetViewShape()).GetString(), "[T,HV,K]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.beta, {info.seqlen, info.hvNum})) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "beta", op::ToString(params.beta->GetViewShape()).GetString(), "[T,HV]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (layout == KdaFwdLayout::NTD) {
        info.batch = 1;
        info.seqlen = Dim(params.q, 1);
        info.hNum = Dim(params.q, 0);
        info.kDim = Dim(params.q, 2);
        info.hvNum = Dim(params.v, 0);
        info.vDim = Dim(params.v, 2);
        if (!HasShape(params.v, {info.hvNum, info.seqlen, info.vDim})) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "v", op::ToString(params.v->GetViewShape()).GetString(), "[HV,T,V]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.g, {info.hvNum, info.seqlen, info.kDim})) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "g", op::ToString(params.g->GetViewShape()).GetString(), "[HV,T,K]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.beta, {info.hvNum, info.seqlen})) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "beta", op::ToString(params.beta->GetViewShape()).GetString(), "[HV,T]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (layout == KdaFwdLayout::BSND) {
        info.batch = Dim(params.q, 0);
        info.seqlen = Dim(params.q, 1);
        info.hNum = Dim(params.q, 2);
        info.kDim = Dim(params.q, 3);
        info.hvNum = Dim(params.v, 2);
        info.vDim = Dim(params.v, 3);
        if (!HasShape(params.v, {info.batch, info.seqlen, info.hvNum, info.vDim})) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "v", op::ToString(params.v->GetViewShape()).GetString(), "[B,T,HV,V]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.g, {info.batch, info.seqlen, info.hvNum, info.kDim})) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "g", op::ToString(params.g->GetViewShape()).GetString(), "[B,T,HV,K]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.beta, {info.batch, info.seqlen, info.hvNum})) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "beta", op::ToString(params.beta->GetViewShape()).GetString(), "[B,T,HV]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        info.batch = Dim(params.q, 0);
        info.seqlen = Dim(params.q, 2);
        info.hNum = Dim(params.q, 1);
        info.kDim = Dim(params.q, 3);
        info.hvNum = Dim(params.v, 1);
        info.vDim = Dim(params.v, 3);
        if (!HasShape(params.v, {info.batch, info.hvNum, info.seqlen, info.vDim})) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "v", op::ToString(params.v->GetViewShape()).GetString(), "[B,HV,T,V]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.g, {info.batch, info.hvNum, info.seqlen, info.kDim})) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "g", op::ToString(params.g->GetViewShape()).GetString(), "[B,HV,T,K]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!HasShape(params.beta, {info.batch, info.hvNum, info.seqlen})) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "beta", op::ToString(params.beta->GetViewShape()).GetString(), "[B,HV,T]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckChunkKdaFwd32Metadata(const ChunkKdaFwdParams &params, KdaShapeInfo &info)
{
    const aclIntArray *cu = params.cuSeqlensOptional;
    if (cu == nullptr) {
        info.seqNum = info.batch;
        if (params.chunkIndicesOptional != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "chunkIndicesOptional", "provided", "requires cuSeqlensOptional");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (cu->Size() < 2) {
            OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "cuSeqlensOptional.size", std::to_string(cu->Size()), ">=2");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if ((*cu)[0] != 0) {
            OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "cuSeqlensOptional[0]", std::to_string((*cu)[0]), "0");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if ((*cu)[cu->Size() - 1] != info.seqlen) {
            OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "cuSeqlensOptional[-1]", std::to_string((*cu)[cu->Size() - 1]), std::to_string(info.seqlen));
            return ACLNN_ERR_PARAM_INVALID;
        }
        for (size_t i = 0; i + 1 < cu->Size(); ++i) {
            if ((*cu)[i] > (*cu)[i + 1]) {
                OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "cuSeqlensOptional", "decreasing at index " + std::to_string(i), "must be nondecreasing");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        info.seqNum = static_cast<int64_t>(cu->Size()) - 1;
        if (info.seqNum > MAX_KDA_VARLEN_SEQUENCES) {
            OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "varlen sequence count", std::to_string(info.seqNum), "<=1024");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    info.totalChunks = CountChunks(cu, info.seqlen, params.chunkSize);
    const aclIntArray *indices = params.chunkIndicesOptional;
    if (indices != nullptr) {
        if (indices->Size() != static_cast<size_t>(info.totalChunks) * 2) {
            OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "chunkIndicesOptional.size", std::to_string(indices->Size()), std::to_string(info.totalChunks * 2));
            return ACLNN_ERR_PARAM_INVALID;
        }
        size_t offset = 0;
        for (size_t seq = 0; seq + 1 < cu->Size(); ++seq) {
            const int64_t chunks = ((*cu)[seq + 1] - (*cu)[seq] + params.chunkSize - 1) / params.chunkSize;
            for (int64_t chunk = 0; chunk < chunks; ++chunk) {
                if ((*indices)[offset] != static_cast<int64_t>(seq) || (*indices)[offset + 1] != chunk) {
                    OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "chunkIndicesOptional", "non-canonical pair at " + std::to_string(offset / 2), "canonical sequence-major order");
                    return ACLNN_ERR_PARAM_INVALID;
                }
                offset += 2;
            }
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckChunkKdaFwd32Dtypes(const ChunkKdaFwdParams &params)
{
    const aclTensor *qkvs[] = {params.q, params.k, params.v};
    const char *names[] = {"q", "k", "v"};
    for (size_t i = 0; i < 3; ++i) {
        if (qkvs[i]->GetDataType() != DataType::DT_BF16) {
            OP_LOGE_FOR_INVALID_DTYPE("aclnnChunkKdaFwd", names[i], ge::TypeUtils::DataTypeToSerialString(qkvs[i]->GetDataType()), "BF16");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const aclTensor *gates[] = {params.g, params.beta};
    const char *gateNames[] = {"g", "beta"};
    for (size_t i = 0; i < 2; ++i) {
        const auto dtype = gates[i]->GetDataType();
        if (dtype != DataType::DT_BF16 && dtype != DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE("aclnnChunkKdaFwd", gateNames[i], ge::TypeUtils::DataTypeToSerialString(gates[i]->GetDataType()), "BF16 or FP32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (params.aLogOptional->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE("aclnnChunkKdaFwd", "aLogOptional", ge::TypeUtils::DataTypeToSerialString(params.aLogOptional->GetDataType()), "FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.dtBiasOptional->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE("aclnnChunkKdaFwd", "dtBiasOptional", ge::TypeUtils::DataTypeToSerialString(params.dtBiasOptional->GetDataType()), "FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.initialStateOptional != nullptr && params.initialStateOptional->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE("aclnnChunkKdaFwd", "initialStateOptional", ge::TypeUtils::DataTypeToSerialString(params.initialStateOptional->GetDataType()), "FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.finalStateOut != nullptr && params.finalStateOut->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE("aclnnChunkKdaFwd", "finalStateOut", ge::TypeUtils::DataTypeToSerialString(params.finalStateOut->GetDataType()), "FP32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckChunkKdaFwd32Outputs(const ChunkKdaFwdParams &params, const KdaShapeInfo &info)
{
    const bool attnValid = info.isRank3
        ? HasShape(params.attnOut, {info.seqlen, info.hvNum, info.vDim})
        : HasShape(params.attnOut, {info.batch, info.seqlen, info.hvNum, info.vDim});
    if (!attnValid) {
        OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "attnOut", op::ToString(params.attnOut->GetViewShape()).GetString(), info.isRank3 ? "[T,HV,V]" : "[B,T,HV,V]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.attnOut->GetDataType() != params.q->GetDataType()) {
        OP_LOGE_FOR_INVALID_DTYPE("aclnnChunkKdaFwd", "attnOut", ge::TypeUtils::DataTypeToSerialString(params.attnOut->GetDataType()), "same as q (BF16)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclTensor *matrices[] = {params.aqkOut, params.akkOut};
    const char *names[] = {"Aqk", "Akk"};
    for (size_t i = 0; i < 2; ++i) {
        if (matrices[i] == nullptr) continue;
        const bool valid = info.isRank3
            ? HasShape(matrices[i], {info.hvNum, info.seqlen, params.chunkSize})
            : HasShape(matrices[i], {info.batch, info.hvNum, info.seqlen, params.chunkSize});
        if (!valid) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", names[i], op::ToString(matrices[i]->GetViewShape()).GetString(), info.isRank3 ? "[HV,T,32]" : "[B,HV,T,32]");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (matrices[i]->GetDataType() != params.q->GetDataType()) {
            OP_LOGE_FOR_INVALID_DTYPE("aclnnChunkKdaFwd", names[i], ge::TypeUtils::DataTypeToSerialString(matrices[i]->GetDataType()), "same as q (BF16)");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const aclTensor *states[] = {params.initialStateOptional, params.finalStateOut};
    const char *stateNames[] = {"initialStateOptional", "finalStateOut"};
    for (size_t i = 0; i < 2; ++i) {
        if (states[i] != nullptr && !HasShape(states[i], {info.seqNum, info.hvNum, info.kDim, info.vDim})) {
            OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", stateNames[i], op::ToString(states[i]->GetViewShape()).GetString(), "[N,HV,128,128] with stateVFirst=false");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckChunkKdaFwd32Params(const ChunkKdaFwdParams &params, KdaFwdLayout &layout,
                                     KdaShapeInfo &info)
{
    const aclTensor *required[] = {params.q, params.k, params.v, params.g, params.beta, params.attnOut};
    const char *names[] = {"q", "k", "v", "g", "beta", "attnOut"};
    for (size_t i = 0; i < 6; ++i) {
        if (required[i] == nullptr) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnChunkKdaFwd", names[i], "input argument must not be nullptr");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    if (!params.safeGate) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "safeGate", "false", "true for chunkSize=32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto status = CheckChunkKdaFwd32Layout(params, layout);
    if (status != ACLNN_SUCCESS) return status;
    status = CheckChunkKdaFwd32Shapes(params, layout, info);
    if (status != ACLNN_SUCCESS) return status;
    status = CheckChunkKdaFwd32Metadata(params, info);
    if (status != ACLNN_SUCCESS) return status;
    if (params.cuSeqlensOptional != nullptr && !info.isRank3 && info.batch != 1) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "B", std::to_string(info.batch), "1 for rank4 varlen");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclTensor *unsupported[] = {params.gkOut, params.aqkOut, params.akkOut, params.wOut,
        params.uOut, params.qgOut, params.kgOut, params.vNewOut, params.hOut};
    const char *unsupportedNames[] = {"gkOut", "aqkOut", "akkOut", "wOut", "uOut", "qgOut",
        "kgOut", "vNewOut", "hOut"};
    for (size_t i = 0; i < 9; ++i) {
        if (unsupported[i] != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", unsupportedNames[i], "provided", "chunkSize=32 only exports attnOut/finalStateOut");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if ((params.initialStateOptional != nullptr || params.finalStateOut != nullptr) && params.stateVFirst) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "stateVFirst", "true", "false for chunkSize=32 state IO");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!params.useGateInKernel) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "useGateInKernel", "false", "true for chunkSize=32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.aLogOptional == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "aLogOptional", "null", "non-null when useGateInKernel=true");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.dtBiasOptional == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "dtBiasOptional", "null", "non-null for chunkSize=32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!HasShape(params.aLogOptional, {info.hvNum})) {
        OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "aLogOptional", op::ToString(params.aLogOptional->GetViewShape()).GetString(), "[HV]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!HasShape(params.dtBiasOptional, {info.hvNum * info.kDim})) {
        OP_LOGE_FOR_INVALID_SHAPE("aclnnChunkKdaFwd", "dtBiasOptional", op::ToString(params.dtBiasOptional->GetViewShape()).GetString(), "[HV*K]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    status = CheckChunkKdaFwd32Dtypes(params);
    if (status != ACLNN_SUCCESS) return status;
    if (info.batch < 1) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "B", std::to_string(info.batch), ">=1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (info.hNum <= 0 || info.hvNum < info.hNum || info.hvNum % info.hNum != 0) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "H/HV", std::to_string(info.hNum) + "/" + std::to_string(info.hvNum), "H>0 and HV a positive multiple of H");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 与通用路径同一上限:超出模型符号表约束的 H/HV 必须拒绝,不能放行进 kernel。
    if (info.hNum > MAX_KDA_HEAD_NUM || info.hvNum > MAX_KDA_HEAD_NUM) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "H/HV", std::to_string(info.hNum) + "/" + std::to_string(info.hvNum), "<=128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // safeGate=true 已在前置强制;lowerBound 必须有限且落在 [-5, 0),
    // 正下界会翻转 safe gate 符号并可能造成核内指数溢出。
    if (!std::isfinite(params.lowerBound) || params.lowerBound < -5.0 || params.lowerBound >= 0.0) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "lowerBound", std::to_string(params.lowerBound), "a finite value in [-5, 0)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (info.kDim != KDA_FWD_KV_DIM_LARGE || info.vDim != KDA_FWD_KV_DIM_LARGE) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "K/V", std::to_string(info.kDim) + "/" + std::to_string(info.vDim), "128/128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (info.seqlen <= 0) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "T", std::to_string(info.seqlen), ">0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return CheckChunkKdaFwd32Outputs(params, info);
}

aclnnStatus RunChunkKdaFwd32FusedDense(ChunkKdaFwdParams &params, aclOpExecutor *executor)
{
    KdaFwdLayout layout = KdaFwdLayout::BSND;
    KdaShapeInfo info;
    // 参数校验状态原样返回(PARAM_NULLPTR/PARAM_INVALID);ContiguousInputs 属于
    // 执行阶段,内部失败不映射为参数错误。
    aclnnStatus checkRet = CheckChunkKdaFwd32Params(params, layout, info);
    if (!(checkRet == ACLNN_SUCCESS)) {
        return checkRet;
    }
    aclnnStatus contRet = ContiguousInputs(params, executor);
    if (!(contRet == ACLNN_SUCCESS)) {
        return contRet;
    }

    // dense BSND:等长 batch 可表示为整齐 varlen,合成
    // cu=[0,T,2T,...],[B,S,H,*] 连续视图零成本压成 packed [B*S,H,*],走 varlen
    // 核内直读路径,省掉 q/k/v/g 的 host 大转置(实测占端到端 ~28%)与回拷。
    // B=1 同样成立(cu=[0,T]);BNSD 本就头连续零转置,均不受此影响。
    if (!info.isRank3 && layout == KdaFwdLayout::BSND) {
        std::vector<int64_t> cuVals(info.batch + 1);
        for (int64_t b = 0; b <= info.batch; ++b) {
            cuVals[b] = b * info.seqlen;
        }
        aclIntArray *cuArr = aclCreateIntArray(cuVals.data(), cuVals.size());
        if (!(cuArr != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: cuArr != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        ChunkKdaFwdParams ps = params;
        const int64_t totalTok = info.batch * info.seqlen;
        ps.q = l0op::Reshape(params.q, MakeShape({totalTok, info.hNum, info.kDim}), executor);
        ps.k = l0op::Reshape(params.k, MakeShape({totalTok, info.hNum, info.kDim}), executor);
        ps.v = l0op::Reshape(params.v, MakeShape({totalTok, info.hvNum, info.vDim}), executor);
        ps.g = l0op::Reshape(params.g, MakeShape({totalTok, info.hvNum, info.kDim}), executor);
        ps.beta = l0op::Reshape(params.beta, MakeShape({totalTok, info.hvNum}), executor);
        ps.attnOut = l0op::Reshape(params.attnOut, MakeShape({totalTok, info.hvNum, info.vDim}), executor);
        ps.cuSeqlensOptional = cuArr;
        aclnnStatus ret = ACLNN_ERR_INNER;
        if (ps.q != nullptr && ps.k != nullptr && ps.v != nullptr && ps.g != nullptr &&
            ps.beta != nullptr && ps.attnOut != nullptr) {
            KdaShapeInfo vinfo;   // packed 形态:TND 同构(batch=1, seqlen=B*T)
            vinfo.isRank3 = true;
            vinfo.batch = 1;
            vinfo.seqlen = totalTok;
            vinfo.hNum = info.hNum;
            vinfo.hvNum = info.hvNum;
            vinfo.kDim = info.kDim;
            vinfo.vDim = info.vDim;
            vinfo.seqNum = info.batch;
            ret = RunChunkKdaFwd32FusedVarlen(ps, executor, vinfo);
        }
        aclDestroyIntArray(cuArr);
        return ret;
    }

    // 布局规整到 [H,T,K] 头连续:NTD rank3 直达,其余 host 侧转置
    const aclTensor *qNtd = params.q;
    const aclTensor *kNtd = params.k;
    const aclTensor *vNtd = params.v;
    const aclTensor *gNtd = params.g;
    const aclTensor *betaNtd = params.beta;
    const bool rank3Ntd = info.isRank3 && layout == KdaFwdLayout::NTD;
    if (!rank3Ntd) {
        if (info.isRank3 && layout == KdaFwdLayout::TND) {
            qNtd = Transpose(params.q, {1, 0, 2}, executor);
            kNtd = Transpose(params.k, {1, 0, 2}, executor);
            vNtd = Transpose(params.v, {1, 0, 2}, executor);
            gNtd = Transpose(params.g, {1, 0, 2}, executor);
            betaNtd = Transpose(params.beta, {1, 0}, executor);
        } else if (!info.isRank3 && layout == KdaFwdLayout::BSND) {
            qNtd = Transpose(params.q, {0, 2, 1, 3}, executor);
            kNtd = Transpose(params.k, {0, 2, 1, 3}, executor);
            vNtd = Transpose(params.v, {0, 2, 1, 3}, executor);
            gNtd = Transpose(params.g, {0, 2, 1, 3}, executor);
            betaNtd = Transpose(params.beta, {0, 2, 1}, executor);
        } else if (!info.isRank3 && layout == KdaFwdLayout::BNSD) {
            // [B,H,S,*] 已是头连续,去 batch 维即可
        } else {
            OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "layout/rank", params.layout, "a supported chunkSize=32 layout/rank combination");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (!(qNtd != nullptr && kNtd != nullptr && vNtd != nullptr && gNtd != nullptr && betaNtd != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: qNtd != nullptr && kNtd != nullptr && vNtd != nullptr && gNtd != nullptr && betaNtd != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        if (!info.isRank3) {
            // [B,H,T,*] → [B*H,T,*](batch 拍平进头维;转置后物理布局 batch-outer 头连续,
            // flatten 合法;kernel 任务号 fh = batch*HV + head 与此布局一一对应)
            qNtd = l0op::Reshape(qNtd, MakeShape({info.batch * info.hNum, info.seqlen, info.kDim}), executor);
            kNtd = l0op::Reshape(kNtd, MakeShape({info.batch * info.hNum, info.seqlen, info.kDim}), executor);
            vNtd = l0op::Reshape(vNtd, MakeShape({info.batch * info.hvNum, info.seqlen, info.vDim}), executor);
            gNtd = l0op::Reshape(gNtd, MakeShape({info.batch * info.hvNum, info.seqlen, info.kDim}), executor);
            betaNtd = l0op::Reshape(betaNtd, MakeShape({info.batch * info.hvNum, info.seqlen}), executor);
            if (!(qNtd != nullptr && kNtd != nullptr && vNtd != nullptr && gNtd != nullptr && betaNtd != nullptr)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: qNtd != nullptr && kNtd != nullptr && vNtd != nullptr && gNtd != nullptr && betaNtd != nullptr failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
    }

    // g/beta dtype 规则:任一为 fp32 → 两者都无损上转 fp32
    // (bf16→fp32 逐位保真),走 def 组合 1(kernel 核内 fp32 原生,全程无量化);
    // 全 bf16 → 组合 0。q 恒 bf16,不再以 q 的 dtype 为锚。
    const bool gateFp32 = (gNtd->GetDataType() == DataType::DT_FLOAT) ||
                          (betaNtd->GetDataType() == DataType::DT_FLOAT);
    const DataType gateDt = gateFp32 ? DataType::DT_FLOAT : DataType::DT_BF16;
    if (gNtd->GetDataType() != gateDt) {
        gNtd = l0op::Cast(gNtd, gateDt, executor);
        if (!(gNtd != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: gNtd != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    if (betaNtd->GetDataType() != gateDt) {
        betaNtd = l0op::Cast(betaNtd, gateDt, executor);
        if (!(betaNtd != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: betaNtd != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    // 输出:attnOut 恒定序列主序(TND/BSND,见 CheckOutputShapes),kernel 产出
    // [H,T,V] 头连续,统一写临时张量再转置回写。
    const aclTensor *oTmp = AllocTensor(executor, MakeShape({info.batch * info.hvNum, info.seqlen, info.vDim}),
                                        params.q->GetDataType());
    if (!(oTmp != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: oTmp != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // initial_state:[N_seq,H,K,V] 拍平成 [N_seq*H,K,V](kernel 按 (seq*H+head) 寻址);
    // final_state 先写 [N_seq*H,K,V] 临时张量,结束后 ViewCopy 回 [N_seq,H,K,V] 输出槽。
    const int64_t stateHeads = info.seqNum * info.hvNum;
    const aclTensor *initialStateHead = nullptr;
    if (params.initialStateOptional != nullptr) {
        initialStateHead = l0op::Reshape(params.initialStateOptional,
                                         MakeShape({stateHeads, info.kDim, info.vDim}), executor);
        if (!(initialStateHead != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: initialStateHead != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *finalStateTmp = nullptr;
    if (params.finalStateOut != nullptr) {
        finalStateTmp = AllocTensor(executor, MakeShape({stateHeads, info.kDim, info.vDim}),
                                    DataType::DT_FLOAT);
        if (!(finalStateTmp != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: finalStateTmp != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    } else {
        // optional 输出缺省时本框架不生成 placeholder,kernel 参数
        // 序列少一个指针,tiling 指针错位 → AICore 异常(507015)。始终给一个真实张量:
        // [1,1,1] 在 tiling 侧按"未请求"处理(ShapeSize==1),kernel 不会写它。
        finalStateTmp = AllocTensor(executor, MakeShape({1, 1, 1}), DataType::DT_FLOAT);
        if (!(finalStateTmp != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: finalStateTmp != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    const aclTensor *fused = l0op::KdaChunkForwardFused(
        qNtd, kNtd, vNtd, gNtd, betaNtd, params.aLogOptional, params.dtBiasOptional,
        initialStateHead, params.scale, params.lowerBound, oTmp, finalStateTmp,
        nullptr /* dense 无 cu_seqlens */, executor);
    if (!(fused != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: fused != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }

    const aclTensor *oBack = nullptr;
    if (info.isRank3) {  // TND/NTD:[H,T,V] → [T,H,V]
        oBack = Transpose(oTmp, {1, 0, 2}, executor);
    } else {             // BSND/BNSD:[B*HV,S,V] → [B,HV,S,V] → [B,S,HV,V]
        const aclTensor *oRank4 =
            l0op::Reshape(oTmp, MakeShape({info.batch, info.hvNum, info.seqlen, info.vDim}), executor);
        if (!(oRank4 != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: oRank4 != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        oBack = Transpose(oRank4, {0, 2, 1, 3}, executor);
    }
    if (!(oBack != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: oBack != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto copied = l0op::ViewCopy(oBack, params.attnOut, executor);
    if (!(copied != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: copied != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (params.finalStateOut != nullptr) {
        const aclTensor *fsBack = l0op::Reshape(
            finalStateTmp, MakeShape({info.seqNum, info.hvNum, info.kDim, info.vDim}), executor);
        if (!(fsBack != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: fsBack != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        auto fsCopied = l0op::ViewCopy(fsBack, params.finalStateOut, executor);
        if (!(fsCopied != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: fsCopied != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}

// ==================== chunk32 varlen:单 kernel 启动====================
// 任务 = seq*HV + value_head 拍平,一次融合 kernel 启动处理全部序列:序列边界经
// cu_seqlens attr 打包进 tiling,kernel 逐任务解析;packed token-major 布局核内
// 跨步直读(不再逐序列转置/启动);空序列 kernel 内写 final=initial(或 0)。
aclnnStatus RunChunkKdaFwd32FusedVarlen(ChunkKdaFwdParams &params, aclOpExecutor *executor,
                                        const KdaShapeInfo &info)
{
    const int64_t nSeq = static_cast<int64_t>(params.cuSeqlensOptional->Size()) - 1;

    aclnnStatus contRet = ContiguousInputs(params, executor);
    if (!(contRet == ACLNN_SUCCESS)) {
        return contRet;
    }

    // token-major 视图:BSND [1,T,*] 去 B 维;TND 直通
    const aclTensor *qTok = params.q;
    const aclTensor *kTok = params.k;
    const aclTensor *vTok = params.v;
    const aclTensor *gTok = params.g;
    const aclTensor *betaTok = params.beta;
    const aclTensor *oTok = params.attnOut;
    if (!info.isRank3) {   // BSND:[1,T,HV,*] → [T,HV,*](B==1 由外层契约卡住)
        qTok = l0op::Reshape(params.q, MakeShape({info.seqlen, info.hNum, info.kDim}), executor);
        kTok = l0op::Reshape(params.k, MakeShape({info.seqlen, info.hNum, info.kDim}), executor);
        vTok = l0op::Reshape(params.v, MakeShape({info.seqlen, info.hvNum, info.vDim}), executor);
        gTok = l0op::Reshape(params.g, MakeShape({info.seqlen, info.hvNum, info.kDim}), executor);
        betaTok = l0op::Reshape(params.beta, MakeShape({info.seqlen, info.hvNum}), executor);
        oTok = l0op::Reshape(params.attnOut, MakeShape({info.seqlen, info.hvNum, info.vDim}), executor);
        if (!(qTok != nullptr && kTok != nullptr && vTok != nullptr && gTok != nullptr && betaTok != nullptr && oTok != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: qTok != nullptr && kTok != nullptr && vTok != nullptr && gTok != nullptr && betaTok != nullptr && oTok != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    // g/beta dtype 规则同 dense 路径:任一 fp32 → 两者无损上转 fp32(组合 1),否则 bf16
    const bool gateFp32 = (gTok->GetDataType() == DataType::DT_FLOAT) ||
                          (betaTok->GetDataType() == DataType::DT_FLOAT);
    const DataType gateDt = gateFp32 ? DataType::DT_FLOAT : DataType::DT_BF16;
    if (gTok->GetDataType() != gateDt) {
        gTok = l0op::Cast(gTok, gateDt, executor);
        if (!(gTok != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: gTok != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    if (betaTok->GetDataType() != gateDt) {
        betaTok = l0op::Cast(betaTok, gateDt, executor);
        if (!(betaTok != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: betaTok != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    // beta 仅 token-major 需要转置为头连续 [HV,T]；NTD 输入已是该布局。
    if (std::strcmp(params.layout, "NTD") != 0) {
        betaTok = Transpose(betaTok, {1, 0}, executor);
        if (!(betaTok != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: betaTok != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    // state:[nSeq,HV,K,V] 拍平 [nSeq*HV,K,V](kernel 按任务号 fh = seq*HV+head 寻址)
    const int64_t stateHeads = nSeq * info.hvNum;
    const aclTensor *initialStateHead = nullptr;
    if (params.initialStateOptional != nullptr) {
        initialStateHead = l0op::Reshape(params.initialStateOptional,
                                         MakeShape({stateHeads, info.kDim, info.vDim}), executor);
        if (!(initialStateHead != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: initialStateHead != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    // final_state:直写调用方输出的拍平视图(少一次回拷);空序列契约 final=initial
    // 由整体预填兜住(kernel 只写非空序列;无 initial 时空序列由 kernel 清零)。
    const aclTensor *finalStateTmp = nullptr;
    if (params.finalStateOut != nullptr) {
        finalStateTmp = l0op::Reshape(params.finalStateOut,
                                      MakeShape({stateHeads, info.kDim, info.vDim}), executor);
        if (!(finalStateTmp != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: finalStateTmp != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
        if (initialStateHead != nullptr) {
            // 核内 istate 直通实测不稳定(见 kernel WriteEmptyFinalState 注释);
            // ViewCopy 的 dst 不能是 workspace 张量,故预填直接落在调用方输出上
            auto pre = l0op::ViewCopy(params.initialStateOptional, params.finalStateOut, executor);
            if (!(pre != nullptr)) {
                OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: pre != nullptr failed", __func__);
                return ACLNN_ERR_INNER_NULLPTR;
            }
        }
    } else {
        finalStateTmp = AllocTensor(executor, MakeShape({1, 1, 1}), DataType::DT_FLOAT);
        if (!(finalStateTmp != nullptr)) {
            OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: finalStateTmp != nullptr failed", __func__);
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    const aclTensor *fused = l0op::KdaChunkForwardFused(
        qTok, kTok, vTok, gTok, betaTok, params.aLogOptional, params.dtBiasOptional,
        initialStateHead, params.scale, params.lowerBound, oTok, finalStateTmp,
        params.cuSeqlensOptional, executor,
        std::strcmp(params.layout, "NTD") == 0 ? "NTD" : "TND");
    if (!(fused != nullptr)) {
        OP_LOGE_WITHOUT_REPORT(OP_NAME, "%s: fused != nullptr failed", __func__);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus RunChunkKdaFwd32Fused(ChunkKdaFwdParams &params, aclOpExecutor *executor)
{
    // chunk32 单 kernel 只在 Ascend950(DAV_3510) 编译(op_kernel 由
    // __NPU_ARCH__ == 3510 门控);其它 SoC 若只按 chunkSize==32 路由,会在无实现
    // 的情况下成功返回但输出未初始化,必须在 host 侧明确拒绝。
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "chunkSize", "32",
                                  "chunkSize=32 requires Ascend950 (DAV_3510)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // chunk32 kernel 的 epsilon 由 tiling 固定写入 1e-6(ChunkKdaFwdFused 的
    // def 无此 attr);调用方提供其它值会被静默丢弃,
    // 因此显式拒绝,与文档的"chunk32 epsilon 固定 1e-6"契约一致。
    constexpr float KDA32_FIXED_EPSILON = 1.0e-6F;
    const float epsilonFp32 = static_cast<float>(params.epsilon);
    if (!std::isfinite(epsilonFp32) || epsilonFp32 != KDA32_FIXED_EPSILON) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "epsilon", std::to_string(params.epsilon), "exactly 1e-6 for chunkSize=32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.cuSeqlensOptional == nullptr) {
        return RunChunkKdaFwd32FusedDense(params, executor);
    }
    KdaFwdLayout layout = KdaFwdLayout::BSND;
    KdaShapeInfo info;
    aclnnStatus checkRet = CheckChunkKdaFwd32Params(params, layout, info);
    if (!(checkRet == ACLNN_SUCCESS)) {
        return checkRet;
    }
    // varlen 支持 BSND/TND 的 token-major 布局和 NTD 的 head-major 布局;BNSD 使用 chunk64 路径
    if (layout == KdaFwdLayout::BNSD) {
        OP_LOGE_FOR_INVALID_VALUE("aclnnChunkKdaFwd", "layout", "BNSD with cuSeqlensOptional", "BSND, TND or NTD for chunkSize=32 varlen");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return RunChunkKdaFwd32FusedVarlen(params, executor, info);
}
