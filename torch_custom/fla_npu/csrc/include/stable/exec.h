// Stable-ABI launch helper: the ABI-free counterpart of vLLM's EXEC_NPU_CMD.
//
// Every adapter is the same three steps -- build an acl descriptor per argument,
// ask the operator for its workspace size, launch it on the caller's stream --
// and only the argument list differs.  This header owns the mechanical part so
// an adapter stays ~3 lines plus its own shape logic.
//
// Two properties matter and are easy to lose:
//
//   * Descriptors must outlive the launch, not just the GetWorkspaceSize call.
//     The holders are therefore collected in a tuple that lives for the whole
//     helper, never as temporaries inside one expression.
//   * The GetWorkspaceSize pointer has an operator-specific signature, so it is
//     reinterpret_cast to the type derived from the tuple's holder types.  A
//     `const aclTensor*` parameter is spelled `aclTensor*` here; the pointer
//     representation is identical, which is what makes the cast safe.
//
// `FLA_STABLE_EXEC` must be called with arguments in exactly the order the
// aclnn entry point declares them; tools/op_abi_parity.py checks that order
// against both the schema and the ctypes reference.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "stable/acl_meta.h"

namespace fla_npu_stable {
namespace stable {

// --- argument holders -------------------------------------------------------

class TensorArg {
 public:
  // `format` defaults to the rank-inferred one; pass `kAclFormatNd` for the
  // call sites whose reference asks for ND explicitly (see nd_tensor below).
  explicit TensorArg(const TensorMeta& meta,
                     int32_t format = kAclFormatAuto,
                     bool logical_storage = false)
      : view_(meta, format, logical_storage) {}
  aclTensor* get() const { return view_.get(); }

 private:
  AclTensorView view_;
};

// Same storage as an input view: aclnn declares outputs `const aclTensor*` in
// most operators and the cast below does not care, but naming it keeps the
// adapter readable.
using OutTensorArg = TensorArg;

class IntArrayArg {
 public:
  explicit IntArrayArg(std::vector<int64_t> values) : view_(values) {}
  aclIntArray* get() const { return view_.get(); }

 private:
  AclIntArrayView view_;
};

class CStrArg {
 public:
  explicit CStrArg(std::string name) : name_(std::move(name)) {}
  const char* get() const { return name_.c_str(); }

 private:
  std::string name_;
};

template <class T>
class ScalarArg {
 public:
  explicit ScalarArg(T value) : value_(value) {}
  T get() const { return value_; }

 private:
  T value_;
};

// --- constructors used at the call site ------------------------------------

inline TensorArg tensor(const TensorMeta& meta) { return TensorArg(meta); }
inline TensorArg tensor(const torch::stable::Tensor& value) {
  return TensorArg(meta_of(value));
}

// The same holders again, with the descriptor forced to ND.  A handful of
// operators' references pass `acl_format_override=ACL_FORMAT_ND` (they say the
// tensors are row-major and the format tag should not be read from the tensor):
// `tools/op_abi_validate.py` cannot see this, so the list is kept explicit --
// chunk_fwd_h, chunk_gated_delta_rule_fwd_prepare, chunk_gdn_bwd_intra,
// chunk_kda_bwd, chunk_kda_bwd_intra and the two recurrent adapters.
inline TensorArg nd_tensor(const TensorMeta& meta) {
  return TensorArg(meta, kAclFormatNd, /*logical_storage=*/true);
}
inline TensorArg nd_tensor(const torch::stable::Tensor& value) {
  return TensorArg(meta_of(value), kAclFormatNd, /*logical_storage=*/true);
}
inline TensorArg nd_optional_tensor(const std::optional<TensorMeta>& meta) {
  return TensorArg(meta.value_or(TensorMeta()), kAclFormatNd,
                   /*logical_storage=*/true);
}
inline TensorArg nd_optional_tensor(std::nullopt_t) {
  return TensorArg(TensorMeta(), kAclFormatNd, /*logical_storage=*/true);
}
inline TensorArg nd_optional_tensor(
    const std::optional<torch::stable::Tensor>& value) {
  if (!value.has_value()) {
    return TensorArg(TensorMeta(), kAclFormatNd, /*logical_storage=*/true);
  }
  const TensorMeta meta = meta_of(*value);
  return TensorArg(meta.defined ? meta : TensorMeta(), kAclFormatNd,
                   /*logical_storage=*/true);
}

// The reference's third spelling: `storage_shape_override=_shape(tensor)` with
// the format left to the tensor.  Used by the operators that hand the tiling a
// logical storage shape without forcing ND: chunk_gated_delta_rule_bwd,
// chunk_gated_delta_rule_bwd_dhu and chunk_gated_delta_rule_bwd_finalize.
inline TensorArg logical_tensor(const TensorMeta& meta) {
  return TensorArg(meta, kAclFormatAuto, /*logical_storage=*/true);
}
inline TensorArg logical_tensor(const torch::stable::Tensor& value) {
  return TensorArg(meta_of(value), kAclFormatAuto, /*logical_storage=*/true);
}
inline TensorArg logical_optional_tensor(
    const std::optional<torch::stable::Tensor>& value) {
  if (!value.has_value()) {
    return TensorArg(TensorMeta(), kAclFormatAuto, /*logical_storage=*/true);
  }
  const TensorMeta meta = meta_of(*value);
  return TensorArg(meta.defined ? meta : TensorMeta(), kAclFormatAuto,
                   /*logical_storage=*/true);
}
inline TensorArg logical_optional_tensor(std::nullopt_t) {
  return TensorArg(TensorMeta(), kAclFormatAuto, /*logical_storage=*/true);
}
inline TensorArg logical_optional_tensor(
    const std::optional<TensorMeta>& meta) {
  return TensorArg(meta.value_or(TensorMeta()), kAclFormatAuto,
                   /*logical_storage=*/true);
}
inline OutTensorArg logical_out_tensor(const TensorMeta& meta) {
  return OutTensorArg(meta, kAclFormatAuto, /*logical_storage=*/true);
}

inline TensorArg optional_tensor(std::optional<TensorMeta> meta) {
  return TensorArg(meta.value_or(TensorMeta()));
}
// `std::nullopt` converts to both optional types, so the plain spelling is
// ambiguous; the adapters mean "no descriptor" when they pass it.
inline TensorArg optional_tensor(std::nullopt_t) {
  return TensorArg(TensorMeta());
}
inline TensorArg optional_tensor(
    const std::optional<torch::stable::Tensor>& value) {
  if (!value.has_value()) {
    return TensorArg(TensorMeta());
  }
  const TensorMeta meta = meta_of(*value);
  return TensorArg(meta.defined ? meta : TensorMeta());
}

inline OutTensorArg out_tensor(const TensorMeta& meta) {
  return OutTensorArg(meta);
}
inline OutTensorArg nd_out_tensor(const TensorMeta& meta) {
  return OutTensorArg(meta, kAclFormatNd);
}
inline OutTensorArg out_tensor(const torch::stable::Tensor& value) {
  return OutTensorArg(meta_of(value));
}

// `int_array` arguments cross the Python boundary as a host int64 tensor (the
// stable conversions have no int[]), so the adapter copies the values out of
// it.  An absent/empty value becomes a null aclIntArray, exactly like the
// ctypes path passing nullptr.
//
// Adapters that need the metadata *before* the call -- an output shape that
// counts segments, for instance -- take the values with `int_values` and hand
// the same vector to `int_array`, so the values are copied once.
inline std::vector<int64_t> int_values(
    const std::optional<torch::stable::Tensor>& value) {
  std::vector<int64_t> values;
  if (value.has_value()) {
    const TensorMeta meta = meta_of(*value);
    if (meta.defined) {
      const int64_t count = meta.storage_numel > 0 ? meta.storage_numel : 0;
      if (meta.scalar_type == 4) {  // kLong
        const auto* data = static_cast<const int64_t*>(meta.data);
        values.assign(data, data + count);
      } else if (meta.scalar_type == 3) {  // kInt
        const auto* data = static_cast<const int32_t*>(meta.data);
        values.reserve(static_cast<size_t>(count));
        for (int64_t index = 0; index < count; ++index) {
          values.push_back(static_cast<int64_t>(data[index]));
        }
      } else {
        throw std::runtime_error(
            "fla_npu(stable): int[] argument must be an int32/int64 "
            "tensor");
      }
    }
  }
  return values;
}

inline IntArrayArg int_array(const std::optional<torch::stable::Tensor>& value) {
  return IntArrayArg(int_values(value));
}
inline IntArrayArg int_array(std::vector<int64_t> values) {
  return IntArrayArg(std::move(values));
}

// Enum-coded `char*` argument: the name table is the single source of the legal
// values, so the same array feeds `cstr` and the coverage gate.
template <size_t N>
inline const char* enum_name(const char* const (&names)[N], int64_t code) {
  if (code >= 0 && static_cast<size_t>(code) < N) {
    return names[code];
  }
  throw std::runtime_error("fla_npu(stable): bad enum code " +
                           std::to_string(code));
}

template <size_t N>
inline CStrArg cstr(const char* const (&names)[N], int64_t code) {
  return CStrArg(enum_name(names, code));
}

template <class T>
inline ScalarArg<T> scalar(T value) {
  return ScalarArg<T>(value);
}

// --- the launch itself ------------------------------------------------------

namespace detail {

// The GetWorkspaceSize pointer type this argument list implies.  Elements may
// be references (the macro forwards a tuple of them), so every step strips the
// reference before asking a holder for its C value.
template <class Element>
using ArgCType = decltype(
    std::declval<const std::remove_reference_t<Element>&>().get());

template <class Tuple, size_t... I>
inline int call_get_workspace(void* address, Tuple& args,
                              std::index_sequence<I...>, uint64_t* workspace,
                              aclOpExecutor** executor) {
  using GetWorkspaceFn = int (*)(ArgCType<std::tuple_element_t<I, Tuple>>...,
                                 uint64_t*, aclOpExecutor**);
  auto fn = reinterpret_cast<GetWorkspaceFn>(address);
  return fn(std::get<I>(args).get()..., workspace, executor);
}

}  // namespace detail

// `Tuple` is whatever the macro's std::forward_as_tuple produced; the holders
// inside it live until the end of the calling full-expression, which is what
// keeps every descriptor alive across both aclnn calls.
template <class Tuple>
inline void exec(const char* api, const TensorMeta& workspace_meta,
                 int64_t stream, Tuple&& args) {
  using Elements = std::remove_reference_t<Tuple>;
  const std::string base(api);
  auto& runtime = Runtime::instance();
  void* get_workspace = runtime.symbol(base + "GetWorkspaceSize");
  auto launch = reinterpret_cast<LaunchFn>(runtime.symbol(base));

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  const int get_ret =
      detail::call_get_workspace(get_workspace, args,
                                 std::make_index_sequence<
                                     std::tuple_size<Elements>::value>{},
                                 &workspace_size, &executor);
  if (get_ret != 0) {
    throw std::runtime_error("fla_npu(stable): " + base +
                             "GetWorkspaceSize failed: " +
                             std::to_string(get_ret));
  }

  torch::stable::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    // The workspace must be allocated on the operator's device, which is why
    // the caller passes a meta rather than letting this guess.
    workspace = allocate_bytes(static_cast<int64_t>(workspace_size),
                               workspace_meta);
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_data_ptr(workspace.get(), &workspace_ptr));
  }

  const int launch_ret = launch(workspace_ptr, workspace_size, executor,
                                reinterpret_cast<void*>(stream));
  if (launch_ret != 0) {
    throw std::runtime_error("fla_npu(stable): " + base + " failed: " +
                             std::to_string(launch_ret));
  }
}

}  // namespace stable
}  // namespace fla_npu_stable

// Arguments go in aclnn order; see the header comment for why that is checked
// offline rather than at run time.
#define FLA_STABLE_EXEC(api, workspace_meta, stream, ...)         \
  ::fla_npu_stable::stable::exec(                                   \
      (api), (workspace_meta), (stream),                          \
      std::forward_as_tuple(__VA_ARGS__))
