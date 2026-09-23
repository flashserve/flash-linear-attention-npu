/**
 * 示例文件（形态 C）：torch_custom/fla_npu/csrc/src/stable_example_scan_fused.cpp
 *
 * 交付布局依据 torch_custom/fla_npu/README.md §1.1：组合算子没有 def/kernel，但对外仍然是一个
 * 算子入口，所以交付件与普通算子完全相同（4 个文件）：
 *
 *   csrc/src/stable_example_scan_fused.cpp   # 本文件（新建，文件名 = 算子名去掉 npu_）
 *   csrc/src/stable_ops.cpp                  # include 一行 + m.def/m.impl 两行
 *   fla_npu/ops/ascendc/_stable.py            # 真签名 wrapper
 *   fla_npu/ops/ascendc/__init__.py           # _ASCENDC_OPS 加一行 public 名
 *
 * 注意事项：
 *   1. 一算子一文件、用宏写：`kSchema_example_scan_fused` 形参 === `run_` 形参 ===
 *      `FLA_STABLE_EXEC` 实参 === aclnn 头文件顺序（`stream` 固定在最后）。
 *   2. 三处参数顺序一致：kSchema_ 声明 = run_ 形参 = FLA_STABLE_EXEC 下发的 aclnn 实参顺序。
 *   3. 可选输出用 std::optional<Tensor>；缺席返回 std::nullopt（走 boxed optional 打包）。
 *   4. 档位与"回落入口"的选择在 Python wrapper 完成，适配层只做拆栈与下发，不判断场景。
 *   5. 不要在这个文件里调用被组合算子的适配函数：组合发生在 L2（aclnn）内部，适配层只调本算子的 aclnn。
 *   6. 组合入口的依赖闭包（构建侧 + 打包侧）见同目录 README「依赖闭包与过滤构建」一节。
 */

// Stable-ABI adapter for npu_example_scan_fused.
// aclnn: aclnnExampleScanFused

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"
#include "stable/layout_math.h"

#include <cstdint>
#include <optional>
#include <tuple>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::cstr;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::tensor;

// 名表顺序必须与 _stable._ENUM["npu_example_scan_fused"]["layout"] 一致。
constexpr const char* kExampleScanFusedLayoutNames[] = {"BSND", "BNSD", "TND", "NTD"};

constexpr const char* kSchema_example_scan_fused =
    "npu_example_scan_fused(Tensor x, Tensor g, Tensor? a_log, Tensor? initial_state, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, int layout, float scale, "
    "int chunk_size, float epsilon, bool return_saved, bool with_tail, int stream) "
    "-> (Tensor, Tensor?, Tensor?, Tensor?)";

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>, std::optional<Tensor>>
run_npu_example_scan_fused(Tensor x, Tensor g, std::optional<Tensor> a_log,
                           std::optional<Tensor> initial_state,
                           std::optional<Tensor> cu_seqlens,
                           std::optional<Tensor> chunk_indices, int64_t layout, double scale,
                           int64_t chunk_size, double epsilon, bool return_saved,
                           bool with_tail, int64_t stream) {
  const TensorMeta x_meta = meta_of(x);
  Tensor y = allocate_like(x);

  // 档位只在这里解释成"传哪些输出指针"：state/x_norm 需要同时给出或同时为空（L2 会校验组合）。
  std::optional<Tensor> state = std::nullopt;
  std::optional<Tensor> x_norm = std::nullopt;
  if (return_saved) {
    state = allocate_like(x);
    x_norm = allocate_like(x);
  }
  // tail 只能与 save 档同时给出，非法组合由 L2 返回 ACLNN_ERR_PARAM_INVALID。
  std::optional<Tensor> tail = std::nullopt;
  if (with_tail) {
    tail = allocate_like(x);
  }

  FLA_STABLE_EXEC("aclnnExampleScanFused", x_meta, stream, tensor(x_meta),
                  tensor(meta_of(g)), optional_tensor(a_log), optional_tensor(initial_state),
                  int_array(cu_seqlens), int_array(chunk_indices),
                  cstr(kExampleScanFusedLayoutNames, layout), scalar(scale),
                  scalar(chunk_size), scalar(epsilon), out_tensor(meta_of(y)),
                  optional_tensor(state), optional_tensor(x_norm), optional_tensor(tail));
  return {y, state, x_norm, tail};
}

}  // namespace
