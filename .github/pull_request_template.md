# Pull Request 模板

## 合入门禁提醒

> **NPU CI 不会在 PR 新建、重开或 push 新 commit 时自动执行。**
>
> PR 新建、重开或 push 新 commit 后，GitHub 会自动把当前 head commit 标记为 `NPU CI / 手动验证 pending`，说明该 commit 暂未执行 NPU CI。机器人评论会提示可请求触发的维护账号和触发命令。
>
> 合入前，当前 head commit 必须具备成功的 `NPU CI / 手动验证` 状态；如果本 PR 后续更新了 commit，旧 commit 的 CI 结果不再有效，需要维护者重新触发。即使已有 2 个维护账号检视通过，只要当前 commit 未完成 NPU CI，仍不可合入（`weinachuan` 可按仓库保护规则 bypass）。
>
> 触发方式：
>
> - GitHub `Actions` 页面选择 `NPU CI`，点击 `Run workflow`，填写本 PR 编号。
> - 在 PR 评论区发送 `/run-npu-ci quick` 或 `/run-npu-ci full`。
> - 同一 PR 的同一 commit 如果已有 NPU CI 在排队或运行，重复评论只会更新机器人评论，不会再次占用 NPU。
> - NPU CI 会固定执行 `examples/flash_gated_delta_rule.py`，默认保持该用例原始 shape；通常不要填写 `example_args`，除非维护者明确要求追加已批准的泛化场景参数。
> - Example ST 必须使用 Ascend PyTorch `v26.1.0-beta.1` release family 的配套 `torch_npu` wheel（已包含 `torchnpugen` 并修复 GDN stream 同步问题）；PyTorch 小版本可按环境选择，但不要拉取 `op-plugin` 重新编译或安装不属于该 release family 的旧版 `torch-npu`。
> - 修改算子 `def`、`aclnn` 接口入参类型，或修改 `torch` 接口入参类型等导致不满足 ABI 一致性的改动，必须由 `weinachuan` 在当前 head commit 上检视通过。
> - NPU CI 部署与排障教程见 [`docs/Fla-npu仓CI部署教程.md`](docs/Fla-npu仓CI部署教程.md)。

## 关联 Issue / 背景

- 关联 Issue:
- 背景与目标:
- 本 PR 解决的问题:

## 变更类型

- [ ] Bug 修复
- [ ] 新功能 / 新算子
- [ ] 算子性能优化
- [ ] 算子精度 / 稳定性修复
- [ ] Tiling / InferShape / OpApi 调整
- [ ] 构建 / CI / 脚本变更
- [ ] 文档 / 示例 / 测试更新

## 算子责任范围

请明确本 PR 修改的算子责任边界，便于审查、验收和后续维护。

- 涉及算子名称:
- 涉及目录 / 文件:
- 责任人 / 提交人: <!-- pr-author -->
- 修改范围:
  - [ ] `op_host` / 算子定义
  - [ ] `InferShape` / 参数校验
  - [ ] `Tiling` 策略
  - [ ] `op_kernel` / Ascend C Kernel
  - [ ] `op_api` / aclnn 接口
  - [ ] `torch_custom` 适配
  - [ ] `Triton` 算子
  - [ ] 单算子测试
  - [ ] `example / ST` 测试
  - [ ] 文档
- 输入 / 输出 / shape / dtype / format 支持范围:
- 明确不包含的范围:
- 是否影响公共模块或其他算子:
  - [ ] 否
  - [ ] 是，请说明影响面、兼容策略和回归范围:
- ABI / 接口兼容性:
  - [ ] 不涉及算子 `def`、`aclnn` 接口或 `torch` 接口入参 / 返回值 / 必选可选属性变化
  - [ ] 涉及 ABI 不兼容风险，已说明原因，并需要 `weinachuan` 检视通过
- ABI 不兼容风险说明:

<details>
<summary>版本与产品编译矩阵</summary>

本 PR 必须保证配套版本满足以下编译要求。当前工程中产品与 `--soc` 参数的对应关系如下:

| 产品 | `--soc` |
| --- | --- |
| A2 | `ascend910b` |
| A3 | `ascend910_93` |
| A5 | `ascend950` |

</details>

<details>
<summary>验证方法</summary>

请按本节方法执行验证，并把实际命令、结果和日志填写到后续矩阵中。`<op_name>` 替换为本 PR 修改的算子名；多个算子用逗号分隔，或逐个填写。

### 1. 环境确认

在每套 CANN / 产品环境中先确认基础信息。

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cat /usr/local/Ascend/ascend-toolkit/latest/opp/version.info
npu-smi info
```

记录项:

- CANN 版本:
- 产品 / 芯片类型:
- 机器或 CI 链接:

### 2. 编译验证

CANN 8.5 及以上需要验证 A2 / A3:

```sh
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=fla_npu
bash build.sh --pkg --soc=ascend910_93 --ops=<op_name> --vendor_name=fla_npu
```

CANN 9.0 及以上需要验证 A2 / A3 / A5:

```sh
bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=fla_npu
bash build.sh --pkg --soc=ascend910_93 --ops=<op_name> --vendor_name=fla_npu
bash build.sh --pkg --soc=ascend950 --ops=<op_name> --vendor_name=fla_npu
```

通过标准:

- 命令退出码为 0
- `build_out/` 下生成对应 `.run` 包
- 编译日志无 `ERROR` / `FAILED` / 关键 warning

### 3. 修改算子 test 验证

根据本 PR 修改范围选择对应测试入口；涉及多个入口时均需执行。

`torch_custom` 适配验证:

```sh
cd torch_custom/fla_npu
bash build.sh
cd test
python3 test_npu_<op_name>.py
# 或执行全部 GDN 单算子测试
bash test.sh
```

`examples/fast_kernel_launch_example` 验证:

```sh
cd examples/fast_kernel_launch_example
bash build_and_test.sh <op_name>
# 不传 <op_name> 时执行该 example 下全部测试
bash build_and_test.sh
```

如果对应算子目录下已有专用测试脚本或 ATK 用例，请填写实际脚本命令，并说明覆盖的 shape / dtype / format / 边界场景。

通过标准:

- 命令退出码为 0
- 测试日志显示 `PASS` / `passed` / `execute samples success`
- 精度误差满足该算子 README、测试脚本或需求说明中的阈值

### 4. 整体 example ST 验证

整体 example ST 用于确认修改没有破坏 GDN 端到端调用链。请在 A2 / A3 / A5 对应环境分别执行。

```sh
python examples/flash_gated_delta_rule.py
```

也可以由维护者触发 CI 验证：`/run-npu-ci quick`。CI 固定执行 `examples/flash_gated_delta_rule.py`，默认保持该用例原始 shape，仅覆盖容器内逻辑设备号 `--device`。

通过标准:

- 命令退出码为 0
- 端到端结果与 golden / PyTorch 参考实现一致
- 日志无精度异常、运行时异常、fallback 异常或 device error

</details>

<details>
<summary>修改算子测试</summary>

本 PR 修改到的算子必须完成对应 test 测试，并在 A2 / A3 / A5 覆盖验证结果。请填写实际执行命令，避免填写当前工程不支持的占位命令。

### CANN 8.5 及以上

要求: 可以编译出 A2 / A3 目标产物。

| 产品 | `--soc` | CANN 实际版本 | 实际执行命令 | 结果 | 产物 / 日志 |
| --- | --- | --- | --- | --- | --- |
| A2 | `ascend910b` |  | `bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=fla_npu` | 通过 / 未通过 / 未执行 |  |
| A3 | `ascend910_93` |  | `bash build.sh --pkg --soc=ascend910_93 --ops=<op_name> --vendor_name=fla_npu` | 通过 / 未通过 / 未执行 |  |

### CANN 9.0 及以上

要求: 可以编译出 A2 / A3 / A5 目标产物。

| 产品 | `--soc` | CANN 实际版本 | 实际执行命令 | 结果 | 产物 / 日志 |
| --- | --- | --- | --- | --- | --- |
| A2 | `ascend910b` |  | `bash build.sh --pkg --soc=ascend910b --ops=<op_name> --vendor_name=fla_npu` | 通过 / 未通过 / 未执行 |  |
| A3 | `ascend910_93` |  | `bash build.sh --pkg --soc=ascend910_93 --ops=<op_name> --vendor_name=fla_npu` | 通过 / 未通过 / 未执行 |  |
| A5 | `ascend950` |  | `bash build.sh --pkg --soc=ascend950 --ops=<op_name> --vendor_name=fla_npu` | 通过 / 未通过 / 未执行 |  |

如结果为“未通过”或“未执行”，请在日志列填写原因、当前阻塞点、责任人和预计补齐时间。

### 单算子测试结果

| 产品 | `--soc` | 实际执行命令 | 结果 | 日志 / 精度结论 |
| --- | --- | --- | --- | --- |
| A2 | `ascend910b` |  | 通过 / 未通过 / 未执行 |  |
| A3 | `ascend910_93` |  | 通过 / 未通过 / 未执行 |  |
| A5 | `ascend950` |  | 通过 / 未通过 / 未执行 |  |

- 精度对比:
- 性能对比:
- 边界 / 异常场景:
- 未覆盖风险:

</details>

<details>
<summary>整体 Example ST 验证</summary>

整体 example 的 ST 测试必须在 A2 / A3 / A5 通过。

| 产品 | `--soc` | 实际执行命令 | 结果 | 日志 / 结论 |
| --- | --- | --- | --- | --- |
| A2 | `ascend910b` | `python examples/flash_gated_delta_rule.py` | 通过 / 未通过 / 未执行 |  |
| A3 | `ascend910_93` | `python examples/flash_gated_delta_rule.py` | 通过 / 未通过 / 未执行 |  |
| A5 | `ascend950` | `python examples/flash_gated_delta_rule.py` | 通过 / 未通过 / 未执行 |  |

- 端到端场景说明:
- 与本 PR 修改算子的关联:
- 未覆盖原因及补充计划:

</details>

## 兼容性、风险与回退

- 兼容性说明:
- 风险点:
- 回退方案:
- 回退责任确认:
  - [ ] 若本 PR 未满足上述编译 / 测试要求，或引入阻塞性 bug，PR 提交人承诺第一时间回退或配合维护者完成回退
  - [ ] 回退后会同步说明影响范围、恢复版本、后续修复计划

## Checklist

- [ ] 已关联 Issue，或说明无需 Issue 的原因
- [ ] 已明确本 PR 的算子责任范围和不包含范围
- [ ] CANN 8.5 及以上已验证 A2 / A3 可以编译通过
- [ ] CANN 9.0 及以上已验证 A2 / A3 / A5 可以编译通过
- [ ] 已验证修改算子的对应 test 在 A2 / A3 / A5 通过
- [ ] 已验证整体 example ST 在 A2 / A3 / A5 通过
- [ ] 已完成必要的精度、性能或回归验证
- [ ] 已确认没有引入与本 PR 无关的格式化或大规模重构
- [ ] 已更新相关 README、算子说明或变更记录，如适用
- [ ] PR 标题已使用合适类型标签，如 `feat:`, `fix:`, `perf:`, `docs:`
- [ ] 已确认如不满足准入要求或引入 bug，将第一时间回退

## 其他信息

在这里补充评审人需要关注的实现细节、限制条件或后续计划。
