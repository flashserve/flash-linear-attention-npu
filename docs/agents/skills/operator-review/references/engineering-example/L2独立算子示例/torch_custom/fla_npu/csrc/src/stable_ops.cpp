/**
 * 示例文件：torch_custom/fla_npu/csrc/src/stable_ops.cpp
 *
 * 交付布局依据 torch_custom/fla_npu/README.md §1.1（一次适配 = 新建 1 个算子文件 + 1 行 include
 * + 2 行注册 + 1 个 Python wrapper + 1 行 public 名）：
 *
 *   torch_custom/fla_npu/
 *   |-- csrc/src/stable_<op>.cpp        # 新建：文件名 = 算子名去掉 npu_
 *   |-- csrc/src/stable_ops.cpp         # 改：include 一行 + 注册两行  ← 本文件
 *   `-- fla_npu/ops/ascendc/
 *       |-- _stable.py                  # 改：加一个真签名 wrapper
 *       `-- __init__.py                 # 改：public 名加一行
 *
 * 注意事项：
 *   1. stable_ops.cpp 是**唯一编译单元**：所有 stable_<op>.cpp 靠它 include 进来，
 *      漏 include 的文件等于没编译（运行期 "dispatcher 找不到实现"，stable_coverage.py 报 "never compiled"）。
 *   2. include 顺序：共享 helper（stable_<前缀>_common.cpp、stable_stream_probe.cpp）在前，
 *      其余按算子名排序；共享 helper 是"被 ≥2 个算子共用"的才建，文件名带共享前缀。
 *   3. 注册分两处，两处都要加：
 *      - m.def(kSchema_<op>) 在 STABLE_TORCH_LIBRARY(fla_npu_stable, m) 里；
 *      - m.impl("<op>", &fla_npu_stable::stable::boxed_adapter<run_<op>>) 在
 *        STABLE_TORCH_LIBRARY_IMPL(fla_npu_stable, CompositeExplicitAutograd, m) 里。
 *   4. 本示例只展示本次新增的行；实际交付时直接改仓库中的这两个段落，不要另建注册表。
 */

// ---- 1) include：共享 helper 在前，其余按算子名排序 --------------------------
#include "stable_causal_conv1d_common.cpp"
#include "stable_fwd_h_common.cpp"
#include "stable_stream_probe.cpp"

#include "stable_op_name.cpp"
// ... 其余算子按算子名排序 ...

// ---- 2) 注册：schema 与实现分别落在两个宏里 ---------------------------------
STABLE_TORCH_LIBRARY(fla_npu_stable, m) {
  // ... 既有 m.def(...) ...
  m.def(kSchema_op_name);
}

STABLE_TORCH_LIBRARY_IMPL(fla_npu_stable, CompositeExplicitAutograd, m) {
  // ... 既有 m.impl(...) ...
  m.impl("npu_op_name",
         &fla_npu_stable::stable::boxed_adapter<run_npu_op_name>);
}
