// Minimal ND tensor descriptor builder mirroring fla_npu's Python nd_tensor.
#pragma once

#include <ATen/Tensor.h>

#include <cstdint>
#include <vector>

namespace fla_npu_thin {

// Opaque CANN types used by the opapi ABI. Forward-declared on purpose so this
// translation unit only needs <acl/acl_base.h>-free declarations.
typedef struct aclTensor aclTensor;

// RAII wrapper around aclCreateTensor / aclDestroyTensor.
class AclTensorView {
 public:
  AclTensorView(const at::Tensor& t, bool force_nd = true,
                bool storage_numel_1d = false);
  ~AclTensorView();

  AclTensorView(const AclTensorView&) = delete;
  AclTensorView& operator=(const AclTensorView&) = delete;

  aclTensor* get() const { return ptr_; }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  aclTensor* ptr_ = nullptr;
  void destroy();
};

// Build an aclIntArray from a host int64 vector. RAII-managed like above.
typedef struct aclIntArray aclIntArray;

class AclIntArrayView {
 public:
  explicit AclIntArrayView(const std::vector<int64_t>& values);
  ~AclIntArrayView();

  AclIntArrayView(const AclIntArrayView&) = delete;
  AclIntArrayView& operator=(const AclIntArrayView&) = delete;

  aclIntArray* get() const { return ptr_; }

 private:
  aclIntArray* ptr_ = nullptr;
};

}  // namespace fla_npu_thin
