/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */
#include "fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_host/chunk_kda_fwd_prepare_tiling_processor.h"

#include <cstdlib>

int main()
{
    using namespace KdaPrepare::Arch22TargetConfig;
    const Contract target{1, 8192, 96, 96, 128, false, false, false,
                          true, true, true, true, true, true};
    if (!Matches(target)) {
        return EXIT_FAILURE;
    }
    bool Contract::*const disallowed[] = {
        &Contract::arch35, &Contract::varLen, &Contract::sequenceMajor};
    for (auto field : disallowed) {
        auto candidate = target;
        candidate.*field = true;
        if (Matches(candidate)) {
            return EXIT_FAILURE;
        }
    }
    bool Contract::*const required[] = {
        &Contract::betaBf16, &Contract::normL2, &Contract::betaSigmoid,
        &Contract::precomputedGate, &Contract::exp2, &Contract::saveOutputs};
    for (auto field : required) {
        auto candidate = target;
        candidate.*field = false;
        if (Matches(candidate)) {
            return EXIT_FAILURE;
        }
    }
    uint64_t Contract::*const dimensions[] = {
        &Contract::batch, &Contract::seqLen, &Contract::qkHeads,
        &Contract::valueHeads, &Contract::totalChunks};
    for (auto field : dimensions) {
        auto candidate = target;
        ++(candidate.*field);
        if (Matches(candidate)) {
            return EXIT_FAILURE;
        }
    }

    optiling::ChunkKdaFwdPrepareScheduleContext context{
        1, 96, 96, 128, 0, 24, 1024, false};
    optiling::ChunkKdaFwdPrepareSchedule fallback;
    if (!optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(fallback) ||
        fallback.headsPerPartition != 96 || fallback.totalWorkItems != 128) {
        return EXIT_FAILURE;
    }
    context.targetPipeline = true;
    context.workspaceSlotBytes = kSlotBytes;
    optiling::ChunkKdaFwdPrepareSchedule optimized;
    if (!optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(optimized) ||
        optimized.usedCoreNum != 24 || optimized.headsPerPartition != 32 ||
        optimized.totalWorkItems != 384 ||
        optimized.workspaceBytes != 1024 + 24 * kSlotCount * kSlotBytes) {
        return EXIT_FAILURE;
    }
    context.aicCoreNum = 20;
    if (!optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(optimized) ||
        optimized.usedCoreNum != 20 || optimized.headsPerPartition != 48 ||
        optimized.totalWorkItems != 256 ||
        optimized.workspaceBytes != 1024 + 20 * kSlotCount * kSlotBytes) {
        return EXIT_FAILURE;
    }
    // A target key and a fallback-sized slot must never be mixed.
    context.workspaceSlotBytes =
        optiling::CHUNK_KDA_FWD_PREPARE_ARCH22_SLOT_BYTES;
    if (optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(optimized)) {
        return EXIT_FAILURE;
    }
    context.targetPipeline = false;
    context.workspaceSlotBytes =
        optiling::CHUNK_KDA_FWD_PREPARE_ARCH35_SLOT_BYTES;
    return optiling::ChunkKdaFwdPrepareTilingProcessor(context).Process(fallback) &&
                   fallback.headsPerPartition == 96 &&
                   fallback.totalWorkItems == 128
               ? EXIT_SUCCESS : EXIT_FAILURE;
}
