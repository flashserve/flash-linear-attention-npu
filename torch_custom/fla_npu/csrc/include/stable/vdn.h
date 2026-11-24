// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "stable/acl_meta.h"

namespace fla_npu_stable::vdn {
inline void check_same_npu(const stable::TensorMeta& first, const stable::TensorMeta& other) {
  if (first.device_type != aoti_torch_device_type_privateuse1() ||
      other.device_type != first.device_type || other.device_index != first.device_index) {
    throw std::invalid_argument("VDN requires tensors on the same NPU");
  }
}
}  // namespace fla_npu_stable::vdn
