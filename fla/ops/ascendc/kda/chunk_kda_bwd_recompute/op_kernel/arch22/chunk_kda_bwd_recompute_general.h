// Copyright (c) 2026 Tianjin University, Ltd. SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "../../../chunk_kda_fwd_prepare/op_kernel/arch22/chunk_kda_fwd_prepare_general.h"
namespace KDA::Arch22 {
// V retains the original FP16/BF16 dtype; normalized Q/K, effective beta and
// Akk are already FP32. Forward and recompute share the exact derived graph.
template<class ValueT, class GateT>
using GeneralRecompute = KdaPrepare::Arch22::GeneralPrepare<ValueT, GateT, float>;
} // namespace KDA::Arch22
