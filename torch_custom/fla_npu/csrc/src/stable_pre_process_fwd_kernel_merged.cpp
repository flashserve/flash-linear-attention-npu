// Stable-ABI adapter for npu_pre_process_fwd_kernel_merged.
// aclnn: aclnnPreProcessFwdKernelMerged
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.
//
// 语义：把 token 窗口压成仿射链 (h | m)。一次调用可以带多段（cu_seqlens），
// 输出 hm[Nseq, HV, K, V+K] fp32 —— Nseq = len(cu_seqlens) - 1。
// 形状契约（与 aclnn 头文件逐参对齐）：
//   k  [1, HK, T, K] BF16  （gk/KDA 路径下 HK == HV，k 是预 gate 的 kg）
//   w  [1, HV, T, K] BF16
//   u  [1, HV, T, V] BF16
//   g? [1, HV, T]      FP32   —— 与 gk 二选一
//   gk?[1, HV, T, K]   FP32   —— 与 g 二选一
//   bg?[1, HK, T, K]   BF16   —— DPLR 专用位；本版本不支持 DPLR，必须传 None
//   v? [1, HV, T, V]   BF16   —— DPLR 专用位；本版本不支持 DPLR，必须传 None
//   （两者留在 schema 里只为保持参数顺序；aclnn 与 tiling 都会拒绝非空值）
//   cu_seqlens?  host int64 数组（严格递增，0 <= cu[0] < cu[-1] <= T）
//   chunk_size   int（本算子固定 64）

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::at_shim::kFloat;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::int_values;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;

// ---------------------------------------------------------------------------
// npu_pre_process_fwd_kernel_merged
// ---------------------------------------------------------------------------

constexpr const char* kSchema_pre_process_fwd_kernel_merged =
    "npu_pre_process_fwd_kernel_merged(Tensor k, Tensor w, Tensor u, "
    "Tensor? g, Tensor? gk, Tensor? bg, Tensor? v, Tensor? cu_seqlens, "
    "int chunk_size, int stream) -> Tensor";

Tensor run_npu_pre_process_fwd_kernel_merged(
    Tensor k, Tensor w, Tensor u, std::optional<Tensor> g,
    std::optional<Tensor> gk, std::optional<Tensor> bg,
    std::optional<Tensor> v, std::optional<Tensor> cu_seqlens,
    int64_t chunk_size, int64_t stream) {
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta u_meta = meta_of(u);
  const std::vector<int64_t> cu = int_values(cu_seqlens);
  if (cu.size() < 2) {
    throw std::runtime_error(
        "fla_npu(stable): pre_process_fwd_kernel_merged requires cu_seqlens "
        "with at least two entries (varlen packed windows only)");
  }

  // hm[Nseq, HV, K, V+K] fp32。K/V 从输入末维取（rank 固定 4），HV 取 u。
  // 注：这一版头文件提供的是 size_of(meta, dim)（带越界检查）；
  //     若换到"一算子一文件"的新版头（提供 SIZE_OF 宏，报错带张量名/行号），
  //     把下面四行换成 SIZE_OF(u_meta, 1) 之类即可。
  const int64_t nseq = static_cast<int64_t>(cu.size()) - 1;
  const int64_t hv = size_of(u_meta, 1);
  const int64_t kdim = size_of(k_meta, 3);
  const int64_t vdim = size_of(u_meta, 3);
  const std::vector<int64_t> hm_sizes = {nseq, hv, kdim, vdim + kdim};
  Tensor hm = allocate_sizes(hm_sizes, kFloat, k_meta);

  FLA_STABLE_EXEC("aclnnPreProcessFwdKernelMerged", k_meta, stream,
                  tensor(k_meta), tensor(meta_of(w)), tensor(u_meta),
                  optional_tensor(g), optional_tensor(gk), optional_tensor(bg),
                  optional_tensor(v), int_array(cu), scalar(chunk_size),
                  out_tensor(meta_of(hm)));
  return hm;
}

}  // namespace
