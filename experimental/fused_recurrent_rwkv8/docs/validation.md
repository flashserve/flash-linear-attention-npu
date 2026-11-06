# 2026-10-23 实验迁移历史记录

> 本文中双算子构建结果包含现已移除的 recompute 样例，只用于历史追溯。

以下只描述旧 PR #954 基线，当前实现已改用独立 Stable ABI launcher。
2026-11-03 的新 main 验证见 [本轮记录](../../../validation-20261103.md)，旧通过数不累计到本轮。

本记录保留开发中状态；尚未完成 `docs/agents/05-算子测试.md` 的正式验收。

来源：PR #446 `049eb0d` 的 `experiment/fla/ops/ascendc/rwkv8/fused_recurrent_rwkv8`。
目标：PR #954 实验框架下的 `experimental/fused_recurrent_rwkv8`。
用户于 2026-10-23 指定：当前只有 A5，本轮先适配并验证编译，A2 环境稍后提供。

| 项目 | 实现 / 证据 | 状态 |
| --- | --- | --- |
| 接口、数学语义、golden | 逐文件核对原 README、aclnn、host、kernel、PTA；golden 原样复制 | 静态核对完成 |
| 构建发现 | 根层 CMake 与 op_host CMake 位于一级算子目录 | 最终单算子及全实验树 A5 构建通过 |
| 架构配置 | OpDef 保留 ascend910b，新增 ascend950 | A5 生成 FP16/BF16/FP32 三个 dtype 的 .o/.json |
| UB 容量 | host 按实际 state+mat+tmp8+向量+对齐 staging 计算 | 已修正；最大 142080 B |
| Python 隔离 | `experimental/_python/fla_npu_experimental`，独立 ctypes runtime，无稳定注册 | 已实现 |
| loader 与标量契约测试 | A5 服务器 conda zzy：19/19 PASS，包含符号链接越界拒绝 | 通过；测试不加载 CANN、不运行 NPU |
| CPU golden 闭式检查 | A5 服务器纯 CPU：零初态、非零初态、reverse 共 3/3 PASS | 通过；golden 本体与来源 SHA256 相同 |
| C++ chunk_len 边界 | `g++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only experimental/fused_recurrent_rwkv8/tests/test_chunk_len_limits.cpp` | 通过，无 CANN/NPU 依赖 |
| Python 语法 | A5 服务器 compileall 覆盖本目录和实验 Python | 通过 |
| A5 host/kernel/OPP | `bash build.sh --pkg --experimental --soc=ascend950 --ops=fused_recurrent_rwkv8 -j4` | chunk_len 边界修复后最终构建、独立安装及包审计通过 |
| A5 NPU 运行、精度、性能 | 本轮不运行 | 未验证 |
| A2 构建与 PTA/ATK 回归 | 等用户提供 A2 环境 | 未重新验证 |

原 PR README 中的 PTA 13/13、ATK 200+9 条 PASS 为原提交历史结果，不能计入本次迁移。
此工作目录保留 PTA 13 条、CPU golden 和 aclnn example；ATK 完整用例包、GPU fixture
及原始测量证据尚未搬入，A2 正式验证前应按仓库 ATK 流程补齐。

后续应覆盖 FP16/BF16/FP32、K≠V、K 非二次幂但按 8 对齐、K/V=8/128、T=1/尾序列/
长序列、initial_state、reverse、s/sa 开关、T<chunk_len、B*H 接近上限、非法 shape/dtype。
唯一 TilingKey=0；所有原分支都需要实际选择与精度证据，才能声明该平台的运行支持。

本轮未提出性能目标或性能优化，未运行 profiling。

首轮编译表明 CANN 9.2.0 可以编译原 Brcb/stride Vector 路径，未添加 MicroAPI
替代分支。随后发现并修复 `chunk_len=2^32` 在 uint32 tiling 字段中变 0 的风险；
`op_host/fused_recurrent_rwkv8_limits.h` 被 aclnn 和 tiling 共用，Python 也按相同上下界拒绝。
`tests/test_chunk_len_limits.cpp` 的纯 CPU static_assert 覆盖 -1、0、1、16、UINT32_MAX、
2^32、2^32+1、INT64_MAX；Python `test_scalar_contract.py` 覆盖同样的边界。
`test_cpu_golden.py` 用零 rank-one 项的闭式递推检查零/非零初态与 reverse，独立于 NPU。

2026-10-23 在 `/home/zzy1205/experimental-integration-20261023` 的 conda `zzy` 中执行：

```bash
FLA_NPU_DISABLE_PTH=1 python -m unittest discover -s experimental/_python/tests -v
FLA_NPU_DISABLE_PTH=1 python experimental/fused_recurrent_rwkv8/tests/pta/test_cpu_golden.py -v
python -m compileall -q experimental/_python experimental/fused_recurrent_rwkv8
g++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only experimental/fused_recurrent_rwkv8/tests/test_chunk_len_limits.cpp
```

返回码均为 0。CPU golden SHA256：
`DC495A807801EF47451A954E631E86A7A06BE2C764E89E74943131C0E5CB22CB`，与 PR #446
缓存源文件逐字节一致。这些 CPU 检查不替代 A2/A5 NPU 精度和正式 ATK 验收。

最终构建入口：`CI_SOC=ascend950 CI_EXPERIMENTAL_OPS=fused_recurrent_rwkv8 CI_JOBS=4 bash ci/run_experimental_validation.sh`；随后将 `CI_EXPERIMENTAL_OPS` 改为 `all`，两个命令均退出 0。全实验包包含 RWKV8 的 3 个 kernel 变体和 recompute 样例的 4 个变体。独立安装后的真实清单支持 `import fla_npu_experimental`，导入时没有加载稳定 `fla_npu`，没有执行 NPU。

证据保存在上述服务器目录的 `validation-artifacts/`：`rwkv8/` 和 `all/` 各含 `.run`、构建日志、清单与审计 JSON；`summary.json` 明确记录 `a2_validated=false` 和 `npu_runtime_executed=false`。基础提交是 PR #954 的 `0edc82f` 加未提交修改，构建源码归档 SHA256 为 `eca3e4350fefa7f4409dfeb6ccb509f875810e088ddf2c680bc3bdd46b1956ff`，不把基础提交本身标记为通过本次验证。
