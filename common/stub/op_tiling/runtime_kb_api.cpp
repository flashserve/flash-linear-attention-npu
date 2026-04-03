/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "runtime_kb_api.h"

namespace RuntimeKb {
uint32_t QueryBank(
    const void* /*src*/, size_t /*src_len*/, const std::string& /*op_type*/, const std::string& /*soc_version*/,
    uint32_t /*core_num*/, tuningtiling::TuningTilingDefPtr& /*tiling*/)
{
    return 0;
}
} // namespace RuntimeKb