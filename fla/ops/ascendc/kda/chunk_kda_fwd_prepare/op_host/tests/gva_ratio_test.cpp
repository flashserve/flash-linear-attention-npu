/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../chunk_kda_fwd_prepare_tiling_processor.h"

#include <cassert>
#include <cstdint>

int main()
{
    using namespace optiling;

    for (uint64_t ratio = 1;
         ratio <= CHUNK_KDA_FWD_PREPARE_MAX_GVA_RATIO; ++ratio) {
        assert(IsValidChunkKdaFwdPrepareGva(3, 3 * ratio));

        const ChunkKdaFwdPrepareScheduleContext context{
            1, 3, 3 * ratio, 1, 0, 24, 0, false, true};
        ChunkKdaFwdPrepareSchedule schedule;
        assert(ChunkKdaFwdPrepareTilingProcessor(context).Process(schedule));
    }

    assert(!IsValidChunkKdaFwdPrepareGva(0, 0));
    assert(!IsValidChunkKdaFwdPrepareGva(3, 2));
    assert(!IsValidChunkKdaFwdPrepareGva(3, 8));
    assert(!IsValidChunkKdaFwdPrepareGva(3, 51));
    assert(IsValidChunkKdaFwdPrepareLegacyHeads(3, 51));
    assert(!IsValidChunkKdaFwdPrepareLegacyHeads(3, 2));
    const ChunkKdaFwdPrepareScheduleContext a5Context{
        1, 3, 51, 1, 0, 24, 0, false, false};
    ChunkKdaFwdPrepareSchedule a5Schedule;
    assert(ChunkKdaFwdPrepareTilingProcessor(a5Context).Process(a5Schedule));
}
