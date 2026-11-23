#pragma once
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/tensor.hpp"

namespace kda {
using Arch = Catlass::Arch::Ascend950;
using Row = Catlass::layout::RowMajor;
using Col = Catlass::layout::ColumnMajor;
using Bf = bfloat16_t;
using L1Tile = tla::Shape<tla::Int<16>, tla::Int<128>, tla::Int<128>>;
using L0Tile = tla::Shape<tla::Int<16>, tla::Int<128>, tla::Int<64>>;
using Policy = Catlass::Gemm::MmadPingpong<Arch, true, false>;
using Copy = Catlass::Gemm::Tile::PackedTileCopyTlaToUB<Arch, Bf, Row, Bf, Col, Bf, Row, void,
    Catlass::Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
using TileMmad = Catlass::Gemm::Tile::TileMmadTla<Arch, Bf, Copy::LayoutTagL1A>;
using Projection = Catlass::Gemm::Block::BlockMmadTla<Policy, L1Tile, L0Tile, Bf, Bf, Bf, void, Copy, TileMmad>;
using ObserveProjection = Catlass::Epilogue::Block::BlockEpilogue<
    Catlass::Epilogue::EpilogueAscend950Fixpipe<2, false>, L0Tile, Bf, Bf>;

} // namespace kda
