#ifndef CHUNK_SCALED_DOT_KKT_TILING_H
#define CHUNK_SCALED_DOT_KKT_TILING_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ChunkScaledDotKktTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, B);
    TILING_DATA_FIELD_DEF(uint64_t, H);
    TILING_DATA_FIELD_DEF(uint64_t, T);
    TILING_DATA_FIELD_DEF(uint64_t, K);
    TILING_DATA_FIELD_DEF(uint64_t, BT);
    TILING_DATA_FIELD_DEF(uint64_t, NT);
    TILING_DATA_FIELD_DEF(uint64_t, taskNum);
    TILING_DATA_FIELD_DEF(uint64_t, usedAicNum);
    TILING_DATA_FIELD_DEF(uint64_t, usedAivNum);
    TILING_DATA_FIELD_DEF(uint64_t, btAlign);
    TILING_DATA_FIELD_DEF(uint64_t, scoreWorkspaceBytes);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkScaledDotKkt, ChunkScaledDotKktTilingData)

ge::graphStatus TilingFunc(gert::TilingContext *context);
}  // namespace optiling

#endif  // CHUNK_SCALED_DOT_KKT_TILING_H
