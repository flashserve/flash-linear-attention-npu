/**
 * 示例文件（形态 B/C）：torch_custom/fla_npu/csrc/src/stable_ops.cpp
 *
 * 交付布局依据 torch_custom/fla_npu/README.md §1.1：组合算子同样走"1 个算子文件 + 1 行 include
 * + 2 行注册 + 1 个 Python wrapper + 1 行 public 名"这套流程。
 *
 * 注意事项：
 *   1. include 顺序：共享 helper 在前，其余按算子名排序：
 *      `stable_example_scan.cpp`（形态 B：在已有适配里加 V2 分支）→ `stable_example_scan_fused.cpp`
 *      （形态 C：新的组合算子）。
 *   2. 形态 B 不新增 schema：V1/V2 是同一个 `npu_example_scan` 的两条 aclnn 入口，由 `run_` 内部选择
 *      （参考 `stable_chunk_kda_fwd.cpp` 对 `aclnnChunkKdaFwdV2` / `aclnnChunkKdaFwd` 的分支），
 *      所以这里**不改注册**。
 *   3. 形态 C 新增一个算子：`m.def(kSchema_example_scan_fused)` +
 *      `m.impl("npu_example_scan_fused", &boxed_adapter<run_npu_example_scan_fused>)` 两行都要加。
 *   4. 组合算子的依赖闭包在 `fla/ops/ascendc/...` 侧解决（构建 + 打包），不在适配层做；
 *      漏依赖时这里只会看到运行期 `561103` / `EZ1013`，按 §3.1/§5.4 排查。
 */

#include "stable_causal_conv1d_common.cpp"
#include "stable_fwd_h_common.cpp"
#include "stable_stream_probe.cpp"

#include "stable_example_scan.cpp"        // 形态 B：同一算子加了 V2 分支
#include "stable_example_scan_fused.cpp"  // 形态 C：新的组合算子
// ... 其余算子按算子名排序 ...

STABLE_TORCH_LIBRARY(fla_npu_stable, m) {
  // ... 既有 m.def(...) ...
  // 形态 B 不新增 schema；形态 C 新增一行：
  m.def(kSchema_example_scan_fused);
}

STABLE_TORCH_LIBRARY_IMPL(fla_npu_stable, CompositeExplicitAutograd, m) {
  // ... 既有 m.impl(...) ...
  m.impl("npu_example_scan_fused",
         &fla_npu_stable::stable::boxed_adapter<run_npu_example_scan_fused>);
}
