#ifndef MEGA_RECURRENT_KDA_V2_T02_H
#define MEGA_RECURRENT_KDA_V2_T02_H

#include "catlass/catlass.hpp"

#include "mega_recurrent_kda_v2_sync.h"
#include "mega_recurrent_kda_v2_tiling.h"
#include "mega_recurrent_kda_v2_vector.h"

namespace MegaRecurrentKdaV2 {

/// Process each owned sequence/head in prefix order and publish its completed tiles.
CATLASS_DEVICE void RunStageR02(V2Params const &p, AscendC::LocalTensor<uint8_t> const &ub)
{
    const uint32_t rank = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t aivNum = static_cast<uint32_t>(AscendC::GetBlockNum()) * 2u;
    AscendC::GlobalTensor<int32_t> prefix;
    prefix.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(p.prefix));
    HeadConstants cached;
    for (uint32_t group = rank; group < p.batch * p.n; group += aivNum) {
        const uint32_t b = group / p.n, head = group % p.n;
        const uint32_t first = p.sequence ? b * p.sequence : static_cast<uint32_t>(prefix.GetValue(b));
        const uint32_t end = p.sequence ? first + p.sequence : static_cast<uint32_t>(prefix.GetValue(b + 1u));
        ProcessSequence<true>(p, ub, b, head, first, end, cached);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace MegaRecurrentKdaV2

#endif // MEGA_RECURRENT_KDA_V2_T02_H
