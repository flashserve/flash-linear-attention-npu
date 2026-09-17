#include "thin_launcher/tensor_desc.h"

#include "thin_launcher/runtime.h"

#include <ATen/ATen.h>

#include <cstring>
#include <stdexcept>

namespace fla_npu_thin {

namespace {

// aclDataType / aclFormat values used by fla_npu's ctypes runtime.
constexpr int32_t kAclFloat = 0;
constexpr int32_t kAclFloat16 = 1;
constexpr int32_t kAclInt8 = 2;
constexpr int32_t kAclInt32 = 3;
constexpr int32_t kAclInt64 = 9;
constexpr int32_t kAclDouble = 11;
constexpr int32_t kAclBool = 12;
constexpr int32_t kAclBf16 = 27;
constexpr int32_t kAclFormatNd = 2;

using AclCreateTensorFn = aclTensor* (*)(const int64_t*, uint64_t, int32_t,
                                         const int64_t*, int64_t, int32_t,
                                         const int64_t*, uint64_t, void*);
using AclDestroyTensorFn = int (*)(aclTensor*);
using AclCreateIntArrayFn = aclIntArray* (*)(const int64_t*, uint64_t);
using AclDestroyIntArrayFn = int (*)(aclIntArray*);

int32_t acl_dtype(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return kAclFloat;
    case at::kHalf:
      return kAclFloat16;
    case at::kChar:
      return kAclInt8;
    case at::kInt:
      return kAclInt32;
    case at::kLong:
      return kAclInt64;
    case at::kDouble:
      return kAclDouble;
    case at::kBool:
      return kAclBool;
    case at::kBFloat16:
      return kAclBf16;
    default:
      throw std::runtime_error("thin launcher: unsupported torch dtype");
  }
}

int64_t storage_numel(const at::Tensor& t) {
  const int64_t elem_size = t.element_size();
  const int64_t nbytes = static_cast<int64_t>(t.storage().nbytes());
  return elem_size > 0 ? nbytes / elem_size : 0;
}

}  // namespace

AclTensorView::AclTensorView(const at::Tensor& t, bool force_nd,
                             bool storage_numel_1d) {
  if (!t.defined()) {
    return;
  }
  const auto& sizes = t.sizes();
  const auto& strides = t.strides();
  const int64_t ndim = static_cast<int64_t>(sizes.size());
  const int64_t offset = t.storage_offset();
  const bool contiguous = t.is_contiguous();

  std::vector<int64_t> view_dims(sizes.begin(), sizes.end());
  std::vector<int64_t> view_strides(strides.begin(), strides.end());
  std::vector<int64_t> storage_dims;
  if (contiguous && force_nd && !storage_numel_1d) {
    // Mirrors fla_npu nd_tensor(): contiguous views describe storage with the
    // logical shape.
    storage_dims = view_dims;
  } else {
    storage_dims.assign(1, storage_numel(t));
  }

  auto fn = reinterpret_cast<AclCreateTensorFn>(
      Runtime::instance().symbol("aclCreateTensor"));
  // aclCreateTensor expects the storage base address; storage_offset is passed
  // separately (Tensor::data_ptr() already includes the offset and would double
  // it for strided views).
  void* data = t.storage().data_ptr().get();
  ptr_ = fn(view_dims.data(), static_cast<uint64_t>(ndim),
            acl_dtype(t.scalar_type()), view_strides.data(), offset,
            force_nd ? kAclFormatNd : kAclFormatNd, storage_dims.data(),
            static_cast<uint64_t>(storage_dims.size()), data);
  if (ptr_ == nullptr) {
    throw std::runtime_error("thin launcher: aclCreateTensor returned nullptr");
  }
}

void AclTensorView::destroy() {
  if (ptr_ != nullptr) {
    auto fn = reinterpret_cast<AclDestroyTensorFn>(
        Runtime::instance().symbol("aclDestroyTensor"));
    fn(ptr_);
    ptr_ = nullptr;
  }
}

AclTensorView::~AclTensorView() {
  destroy();
}

AclIntArrayView::AclIntArrayView(const std::vector<int64_t>& values) {
  if (values.empty()) {
    return;
  }
  auto fn = reinterpret_cast<AclCreateIntArrayFn>(
      Runtime::instance().symbol("aclCreateIntArray"));
  ptr_ = fn(values.data(), static_cast<uint64_t>(values.size()));
  if (ptr_ == nullptr) {
    throw std::runtime_error("thin launcher: aclCreateIntArray returned nullptr");
  }
}

AclIntArrayView::~AclIntArrayView() {
  if (ptr_ != nullptr) {
    auto fn = reinterpret_cast<AclDestroyIntArrayFn>(
        Runtime::instance().symbol("aclDestroyIntArray"));
    fn(ptr_);
    ptr_ = nullptr;
  }
}

}  // namespace fla_npu_thin
