# AGENTS.md

本文件是给 AI coding agent 的仓库级工作说明，适用于整个 `flash-linear-attention-npu` 仓库。若子目录后续出现更近的 `AGENTS.md`，以更近文件为准。

## 项目定位

`flash-linear-attention-npu` 是面向昇腾 NPU 的高性能线性注意力算子库，核心工作包括 Ascend C 算子、Tiling/InferShape/op_host、aclnn op_api、PyTorch 适配、Triton 适配、单算子测试和端到端 Example/ST。

优先阅读：

- `README.md`：构建、安装、调用、测试入口和目录结构。
- `CONTRIBUTING.md`：贡献流程和新增算子交付要求。
- `docs/repository-rules.md`：分支、ABI、NPU CI 和合入规则。
- `docs/agents/README.md`：面向 AI agent 的开发原理、方法论、验证和经验总结索引。
- `.github/pull_request_template.md`：PR 必填信息和验证矩阵。
- 当前修改算子的 `README.md`、`docs/aclnn*.md`、测试脚本和相邻算子实现。

## 工作原则

- 开始前先看 `git status --short`，不要回滚或覆盖用户已有改动。
- 先用 `rg` / `rg --files` 找代码和文档，再修改；不要凭记忆猜目录。
- 改动保持聚焦，避免无关格式化、批量重排和生成物噪声。
- 公共接口、shape/dtype/layout/range、预留参数、平台差异、返回码或报错文本变化，必须同步检查代码、README、aclnn 文档、PyTorch API 文档、测试和示例。
- 公开 PR、issue、评论和总结中不要暴露内网地址、机器名、用户名、绝对路径、临时目录、日志路径、token 或本地调测环境细节。
- 构建和测试默认面向 Linux + CANN + NPU 环境；其他平台只做静态阅读、文本编辑或格式检查，不把未验证命令写成已验证结论。

## 关键目录

- `fla/ops/ascendc/`：Ascend C 算子实现。
- `fla/ops/ascendc/common/`：公共 Ascend C 组件。
- `fla/ops/ascendc/gdn/`：GDN 相关算子。
- `torch_custom/fla_npu/`：PyTorch 自定义算子适配、YAML schema、Python 包和测试。
- `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py`：推荐 Python 稳定入口导出。
- `examples/`：端到端调用示例。
- `ci/`：NPU CI、Example/ST case 和本地 CI 脚本。
- `scripts/`：构建、打包、环境检查和代码生成辅助脚本。
- `tests/`：工程级 UT。

## 调用约定

安装后的 wheel 公开 Python import 面只使用 `fla_npu`。不要让新代码依赖顶层
`fla` 包；`fla/` 目录主要作为源码树内实现来源存在。

Ascend C 新代码优先使用稳定 Python 入口：

```python
from fla_npu.ops.ascendc import chunk_bwd_dv_local
```

Triton 算子同样使用 `fla_npu` 下的稳定入口：

```python
from fla_npu.ops.triton import chunk_local_cumsum
```

`torch.ops.npu.*` 是兼容旧调用的过渡路径，仅在兼容性测试或旧 API 验证中使用。需要旧路径时先确认 `fla_npu.load_legacy_torch_ops()` 的加载逻辑。

## 算子开发交付 checklist

新增或修改 Ascend C 算子时，交付前逐项核对：

- [ ] `fla/ops/ascendc/**/op_host/*_def.cpp` 已同步输入、输出、属性、dtype、format、required/optional 定义。
- [ ] `fla/ops/ascendc/**/op_host/*tiling*` / `*infershape*` 已同步参数校验、shape 推导和 tiling。
- [ ] `fla/ops/ascendc/**/op_host/op_api/aclnn_*.h/.cpp` 已同步 aclnn 接口签名、返回和执行器逻辑。
- [ ] `fla/ops/ascendc/**/op_kernel/` 已同步 Kernel 实现、tiling data 和 tiling key。
- [ ] `torch_custom/fla_npu/npu_custom.yaml`、`test_native_functions.yaml`、`deprecated.yaml` 已同步 PyTorch schema 和适配生成输入。
- [ ] `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py` 已同步 Python 稳定导出。
- [ ] `torch_custom/fla_npu/test/test_npu_<op>.py` 已补充或更新单算子测试和参考实现。
- [ ] 当前算子的 `README.md`、`docs/aclnn*.md`、示例和 CI case 已同步更新。
- [ ] 参数校验、shape/dtype/layout/range、平台差异、预留参数语义和报错文本保持一致。
- [ ] 正向、反向、边界、异常、dense/varlen、关键 SOC 和目标 dtype/layout 场景已按改动风险覆盖。

ABI 敏感路径包括 `*_def.cpp`、`aclnn_*.h/.cpp`、`torch_custom/fla_npu/*.yaml` 和 `torch_custom/fla_npu/op_plugin/ops/opapi/**`。修改这些文件时，PR 需要明确说明 ABI 影响，并按 `.github/CODEOWNERS` 请求对应 owner 检视。

## 构建命令

先准备环境：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install -r requirements.txt
python scripts/check_npu_env.py --build-only
```

推荐的一体化 wheel 构建：

```sh
FLA_NPU_SOC=ascend910b python -m pip wheel --no-build-isolation --no-deps . -w dist
```

常用 SOC：

- A2：`ascend910b`
- A3：`ascend910_93`
- A5：`ascend950`

本地增量调试可以使用：

```sh
FLA_NPU_SOC=ascend910b FLA_NPU_INCREMENTAL_BUILD=1 python -m pip wheel --no-build-isolation --no-deps . -w dist
```

只构建部分算子用于定位时使用 `FLA_NPU_OPS`，不要和 `FLA_NPU_INCREMENTAL_BUILD` 同时使用：

```sh
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_fwd_o python -m pip wheel --no-build-isolation --no-deps . -w dist
```

分开编 OPP run 包和 `torch_custom` 适配时：

```sh
bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu
cd torch_custom/fla_npu
bash build.sh
```

一体化 wheel 和 `torch_custom/fla_npu` 单独编出的 wheel 使用相同 pip 项目名
`flash-linear-attention-npu`，Python import 名均为 `fla_npu`。单独编出的
standalone wheel 只提供 Python 适配和 OPP 骨架；配套 run 包使用 `--full` 或
`--install` 安装时，会把 run 包里的 `packages/vendors/fla_npu_transformer`
合并覆盖到当前 Python 环境的 `site-packages/fla_npu/opp/vendors/fla_npu_transformer`。

## 安装和验证

安装一体化 wheel：

```sh
python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
python scripts/check_packaged_wheel_api.py
```

运行 GDN 单算子测试：

```sh
cd torch_custom/fla_npu/test
bash test.sh --device 0
bash test.sh --device 0 --op causal_conv1d
```

`test.sh` 只覆盖已接入脚本的 GDN 算子。未纳入 `test.sh` 的算子按对应测试脚本直接运行，例如：

```sh
cd torch_custom/fla_npu/test
python3 test_npu_<op>.py
```

端到端 Example/ST：

```sh
python examples/flash_gated_delta_rule.py
python3 ci/run_example_st_cases.py --device 0 --cases-file ci/example_st_cases.json
```

本地复现 CI 主流程：

```sh
CI_MODE=quick CI_SOC=ascend910b CI_OPS=<op_name> bash ci/run_checks.sh
CI_MODE=full bash ci/run_checks.sh
```

如果缺少 CANN、NPU、`torch_npu`、`torchnpugen` 或 `triton-ascend`，不要伪造验证结论；在回复中说明未执行的命令和阻塞原因。

## 测试要求

- 修改参数校验、shape、dtype、layout、range、平台差异或预留参数语义时，补充反向测试和边界测试。
- 修改 Kernel 时至少覆盖对应单算子测试；涉及 GDN 端到端链路时跑 Example/ST。
- 修改 ABI、公共模块或共享路径时扩大回归范围，至少说明影响到的算子。
- 精度失败不能通过收窄输入 range、跳过 case、降低覆盖强度或放宽阈值来制造通过结论；应先定位误差来源，再修 kernel、标杆或后处理语义。
- 性能结论以合适的 profiling/CI 结果为准，不用 Python wall time 直接下结论。

## 构建产物与提交规范

不要提交构建、安装、调测和性能分析产物。重点检查：

- `build/`、`build_out/`、`output/`、`dist/`、`.ci-cache/`、`third_party/`
- `torch_custom/fla_npu/build/`、`torch_custom/fla_npu/dist/`、`torch_custom/fla_npu/torch_npu/`
- `torch_custom/fla_npu/test/test_output/`、`torch_custom/fla_npu/test/data/`
- `__pycache__/`、`*.pyc`
- `.tmp*`、`outputs/`、`PROF_*`、`OPPROF_*`、`extra-info`

提交前至少运行：

```sh
git status --short
git diff --check
```

## PR 和 CI

- PR 描述使用 `.github/pull_request_template.md`，不要自创栏目替代模板。
- PR 应关联 Issue，或在模板中说明无需 Issue 的原因。
- NPU CI 不会在每次 push 后自动跑；需要仓库 Admin 在 Actions 手动触发，或在 PR 评论 `/run-npu-ci quick` / `/run-npu-ci full`。
- 当前 head commit 需要通过 `NPU CI / 手动验证` 和 `NPU CI / 精度检查`，并满足 2 个 approval。
- push 新 commit 后旧 commit 的 CI 结果失效，需要重新触发。

## 给 AI 的最后检查

结束任务前确认：

- 改动是否只覆盖本次任务需要的文件。
- 文档、报错、返回码、schema、Python 导出和测试是否与代码语义一致。
- 是否遗漏对应 SOC、layout、dtype、varlen/dense、边界 case。
- 是否有未跟踪生成物或敏感信息混入。
- 是否清楚说明已执行和未执行的验证。
