/**
 * 示例文件：torch_custom/fla_npu/csrc/src/stable_example_scan.cpp
 *
 * 注意事项：
 *   1. 一算子一个文件：文件名 = 算子名去掉 npu_ 前缀（npu_example_scan -> stable_example_scan.cpp）。
 *      新建后必须在 stable_ops.cpp 加 `#include "stable_example_scan.cpp"` 一行 + 注册两行，
 *      否则文件不参与编译（stable_coverage.py 会报 "never compiled"）。
 *   2. 三处形参顺序必须完全一致：kSchema_ 声明顺序 = run_ 形参顺序 = FLA_STABLE_EXEC 下发的
 *      aclnn 实参顺序（由 op_abi_parity.py / op_abi_validate.py 检查）。
 *   3. 新算子一律用宏写；不要照抄 pre-macro 的历史手写入口（两个 recurrent 适配）。
 *   4. 可选输出用 std::optional<Tensor>，缺席时返回 std::nullopt；boxed.h 会按 boxed optional 打包，
 *      不能直接塞 tensor handle。
 *   5. 不缓存 stream：每次调用现取 `_current_stream_ptr()`（Python 侧）并把 stream 作为实参传入。
 *   6. 适配层只做元数据搬运与下发：不判定布局能力、不补 dense 拷贝。
 *   7. 名表（layout 等枚举）命名要带算子前缀，且顺序与 _stable._ENUM 一致。
 */

// Stable-ABI adapter for npu_example_scan.
// aclnn: aclnnExampleScan
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into the
// single translation unit and registers it there.

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
using fla_npu_stable::stable::SIZE_OF;
using fla_npu_stable::stable::tensor;

// layout 名表：顺序必须与 fla_npu/ops/ascendc/_stable._ENUM 完全一致（stable_coverage.py 检查）。
constexpr const char* kExampleScanLayoutNames[] = {"BSND", "BNSD", "TND", "NTD"};

constexpr const char* kSchema_example_scan =
    "npu_example_scan(Tensor x, Tensor g, Tensor? a_log, Tensor? initial_state, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, int layout, float scale, "
    "int chunk_size, float epsilon, bool return_saved, int stream) "
    "-> (Tensor, Tensor?, Tensor?)";

std::tuple<Tensor, std::optional<Tensor>, std::optional<Tensor>> run_npu_example_scan(
    Tensor x, Tensor g, std::optional<Tensor> a_log, std::optional<Tensor> initial_state,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices, int64_t layout,
    double scale, int64_t chunk_size, double epsilon, bool return_saved, int64_t stream) {
  const TensorMeta x_meta = meta_of(x);
  Tensor y = allocate_like(x);

  // 保留策略只在 Python/适配层解释：只有调用方要中间量时才申请公开输出。
  std::optional<Tensor> state = std::nullopt;
  std::optional<Tensor> x_norm = std::nullopt;
  if (return_saved) {
    // 状态 shape 由 layout_math.h 的轴 helper 决定：SIZE_OF 会把张量名与调用行号一起带进异常。
    const int64_t head = SIZE_OF(x_meta, layout_math::head_axis(layout));
    const int64_t dim = SIZE_OF(x_meta, layout_math::dim_axis(layout));
    state = allocate_sizes({head, dim}, x_meta.dtype, x_meta);
    x_norm = allocate_like(x);
  }

  FLA_STABLE_EXEC("aclnnExampleScan", x_meta, stream, tensor(x_meta),
                  tensor(meta_of(g)), optional_tensor(a_log), optional_tensor(initial_state),
                  int_array(cu_seqlens), int_array(chunk_indices),
                  cstr(kExampleScanLayoutNames, layout), scalar(scale), scalar(chunk_size),
                  scalar(epsilon), out_tensor(meta_of(y)),
                  optional_tensor(state), optional_tensor(x_norm));
  return {y, state, x_norm};
}

}  // namespace
