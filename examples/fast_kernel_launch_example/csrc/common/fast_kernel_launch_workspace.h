#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include <torch/all.h>

#include "acl/acl.h"

namespace ascend_ops {
namespace fast_kernel_launch {

inline at::Tensor AllocateDeviceBuffer(const at::Tensor &reference, size_t sizeBytes, const char *name)
{
    if (sizeBytes == 0) {
        return at::Tensor();
    }
    TORCH_CHECK(sizeBytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                name, " device buffer is too large: ", sizeBytes);
    return at::empty({static_cast<int64_t>(sizeBytes)}, reference.options().dtype(at::kByte));
}

inline GM_ADDR TensorGmAddr(const at::Tensor &tensor)
{
    return tensor.defined() ? (GM_ADDR)tensor.data_ptr() : nullptr;
}

inline void ZeroDeviceBufferAsync(const at::Tensor &buffer, size_t sizeBytes, aclrtStream stream, const char *name)
{
    if (!buffer.defined()) {
        return;
    }
    auto ret = aclrtMemsetAsync(buffer.data_ptr(), sizeBytes, 0, sizeBytes, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "memset ", name, " device buffer failed. ERROR: ", ret);
}

inline at::Tensor CopyHostStructToDevice(const at::Tensor &reference, const void *hostData, size_t sizeBytes,
                                         const char *name)
{
    at::Tensor buffer = AllocateDeviceBuffer(reference, sizeBytes, name);
    auto ret = aclrtMemcpy(buffer.data_ptr(), sizeBytes, hostData, sizeBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(ret == ACL_SUCCESS, "copy ", name, " to device failed. ERROR: ", ret);
    return buffer;
}

} // namespace fast_kernel_launch
} // namespace ascend_ops
