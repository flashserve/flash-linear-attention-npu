# experimental 构建适配过程记录

> 本文记录调研、试错、实测和方案收敛过程，不作为最终行为规范。
>
> 最终目标设计见 [experimental 构建适配设计.md](experimental%20构建适配设计.md)。验收口径以 [Issue #931](https://github.com/flashserve/flash-linear-attention-npu/issues/931) 为准。

## 1. 初始问题

仓库已有从 `cann/ops-transformer` 带入的 `ENABLE_EXPERIMENTAL` 骨架，但它没有适配本仓目录：

- 上游按 `ffn/gmm/moe/posembedding` 等域扫描，本仓实际为 `cp/gdn/kda`；
- 上游存在 `experimental/attention` 聚合器，本仓没有该目录；
- 本仓算子位于 `fla/ops/ascendc/`，层级为 2～4 层且不固定；
- 本仓 `<op>_depends` 使用仓根相对路径；
- kernel common 是构建级 include 根，不在 `<op>_depends` 中；
- Python/wheel 和 CI 均不感知 experimental。

未修改代码时启用 `--experimental`，首先遇到的错误是：

```text
add_subdirectory given source "experimental/attention"
which is not an existing directory
```

## 2. 第一阶段：打通算子发现

### 2.1 目录方案调研

最初比较了扁平实验目录与仓库相对路径镜像。由于 depends 写法类似：

```text
fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h
```

第一版为了尽量不改算子 CMake，选择了：

```text
experimental/fla/ops/ascendc/<域>/<算子>/
```

这样迁移算子时大多数相对路径无需修改，因此适合快速验证构建链。

后续进一步明确了一个更强的目录契约：`experimental/` 后的第一个子目录可以保证是全局唯一算子名，并且该目录一定包含根层 `CMakeLists.txt`。在这个前提下，镜像稳定树不再必要。最终方案收敛为：

```text
experimental/<算子名>/CMakeLists.txt
```

扁平结构需要迁移时显式整理依赖声明和跨目录相对引用，但能显著简化算子发现、CI 定位、依赖解析和产物来源审计。因此镜像树保留为中间验证结构，不再是目标设计。

### 2.2 首轮修改

首轮实现包括：

1. 将实验分支的域白名单改成递归 glob；
2. 删除不存在的 `add_subdirectory(experimental/attention)`；
3. 为 `ENABLE_EXPERIMENTAL` 补默认值；
4. 调整 `parse_changed_ops.py`，支持本仓多层目录；
5. 删除不存在算子 `fused_infer_attention_score` 的 CI 兜底；
6. 增加 experimental README。

该阶段证明了“同一 `if/else` 选择根算子树”可以完成直接算子集合的互斥，但尚未证明依赖、公共代码和产物互斥。

## 3. 第二阶段：A5 实测暴露依赖问题

使用 `kda_gate_bwd_post` 进行 A5 验证时观察到：

| 模式 | 依赖处理 | `chunk_kda_bwd_tiling.cpp` | 结果 |
| --- | --- | --- | --- |
| 稳定模式 | 依赖被补挂 | 编译 4 次 | 打包成功 |
| experimental，依赖未复制 | 依赖静默丢弃 | 编译 0 次 | 链接失败 |

失败信息为：

```text
cannot find -lophost_transformer_tiling_obj
```

这说明依赖缺失不能继续沿用“静默跳过”。无论最终采用哪种隔离策略，都必须在 configure 阶段给出明确结果。

## 4. 中间尝试：稳定依赖回退

为了验证最小迁移能否出包，曾实现 `fla_resolve_depend_path()`：

```text
experimental/<dep> 存在 -> 使用实验依赖
experimental/<dep> 不存在且稳定依赖存在 -> 回退稳定依赖并打印消息
```

该尝试在 A5 上达到了工程闭环：实验树只放 `kda_gate_bwd_post` 时能够完成打包，依赖 tiling 源恢复编译。

随后对 Issue #931 的验收条件重新核对，确认该路线不能作为最终方案：

- Issue 要求实验依赖必须来自 experimental；
- 缺失依赖必须失败，禁止回退；
- 实验产物不得包含稳定算子；
- 公共代码也必须隔离。

因此，稳定依赖回退被定性为一次定位问题和验证构建链的中间实验。最终实现需要删除回退，改为严格失败。

## 5. 审计中发现的其他缺口

### 5.1 依赖路径不止两个消费者

首轮回退只覆盖依赖 `op_host` 补挂和 kernel 编译源复制。后续审计发现 `custom_build.cmake` 在安装依赖 kernel 源时仍直接拼稳定路径。即使编译选择了实验依赖，包内源码仍可能来自稳定树或为空。最终设计因此要求所有消费者共享一个解析结果。

### 5.2 算子级 CMake 可绕过框架

`chunk_kda_bwd/op_host/CMakeLists.txt` 在 A5 分支直接使用：

```cmake
add_subdirectory("${PROJECT_SOURCE_DIR}/${DEP_PATH}/op_host" ...)
```

将该算子复制到 experimental 后仍会挂载稳定依赖。三个其他算子也有自己的依赖挂载逻辑。仅修改公共 helper 不能保证隔离，必须增加规范化路径断言并清理算子级旁路。

### 5.3 `--ops` 前置校验只认识稳定树

`scripts/check_build_ops.py` 的搜索根固定为 `fla/ops/ascendc`。因此实验算子若在稳定树没有同名项，会在进入 CMake 前被判为“不支持”。早期验证使用的是稳定树已有名字，没有覆盖这一场景。目标实现改为枚举 `experimental/<算子名>/CMakeLists.txt`。

### 5.4 common 未隔离

本仓 26 个算子共 68 处引用 `fla/ops/ascendc/common` 下的头文件。控制点是 `cmake/config.cmake` 中的 `OPS_ADV_UTILS_KERNEL_INC`，experimental 开关不影响它。

曾比较：

| 方案 | 结果 |
| --- | --- |
| 继续共享稳定 common | 实验改动污染稳定构建，不满足验收 |
| 稳定 common 与实验 common 叠加 | 实验构建仍读取稳定代码，不满足严格隔离 |
| 独立 `experimental/_common`，整体切换并做完整性门禁 | 满足隔离，最终采用 |

### 5.5 CI 解析范围不足

实验 PR 解析器硬编码 `cp/gdn/kda`，会忽略未来变化；它还会忽略 `cmake/`、构建脚本和当前共享 common 的改动。最终设计直接使用 `experimental/` 后第一个路径段作为算子名，并规定框架或 `_common` 变更触发 experimental 全量构建。

### 5.6 产物身份未区分

稳定包与实验包当前使用相同 `.run` 文件名和 vendor 路径。虽然分两次构建可以分别得到文件，但容易被覆盖，也无法从产物身份证明隔离。最终设计要求独立命名和包内来源清单。

## 6. 已执行的验证

以下结果用于理解构建链，不等同于 Issue #931 已验收：

| # | 验证项 | 结果 |
| --- | --- | --- |
| 1 | Shell/Python 基础语法检查 | 通过 |
| 2 | 未打补丁启用 experimental | 复现 `experimental/attention` 不存在 |
| 3 | 递归发现多层实验算子 | configure 成功 |
| 4 | OFF 时实验树放非法 CMake | 稳定 configure 不受影响 |
| 5 | ON 时实验树放非法 CMake | 能确认实验树被扫描 |
| 6 | `--PR_PKG --experimental` 识别多层路径 | 能识别已有同名算子 |
| 7 | PR 空清单 | 返回“无可编算子” |
| 8 | 只修改稳定算子路径 | 实验解析器忽略 |
| 9 | 无依赖实验算子打包 | 成功生成 `.run` |
| 10 | 有依赖但未复制依赖 | 链接失败，暴露静默丢依赖问题 |
| 11 | 复制完整依赖闭包 | 打包成功 |
| 12 | 显式下发开关 TRUE/FALSE | 状态可见且可切换 |
| 13 | 稳定依赖回退实验 | 打包成功，但不符合 Issue #931 |
| 14 | OFF 模式回归 | configure 成功 |

尚未完成的关键验证：

- experimental-only 新算子通过 `--ops` 前置校验；
- 缺少直接或传递实验依赖时 configure 明确失败；
- 实验 common 缺失门禁；
- 包内不存在稳定算子或稳定公共代码；
- RWKV recurrent 独立实验包；
- Python/wheel experimental 注册和调用；
- experimental CI；
- A2/A3/A5 稳定完整回归。

## 7. 过程中的工程问题

### 7.1 空实验树

glob 空集本身不会报错，但后续 opbuild 不生成 `aic-*-ops-info.ini`，最终 adapt Python 阶段失败。最终实现应在更早位置报告“实验树没有可构建算子”。

### 7.2 复制目录嵌套

执行 `cp -r <父目录> <已存在的目标父目录>` 会产生重复嵌套目录和同名算子。实验时使用的安全方式是：

```bash
mkdir -p "$DST"
cp -a "$SRC/." "$DST/"
```

### 7.3 构建并发

验证机内存较小而 CPU 数较多，默认 `nproc*2` 曾触发 OOM。该问题不决定隔离方案，但 CI 应显式限制并发。

## 8. 方案收敛

| 维度 | 中间实现 | Issue #931 目标设计 |
| --- | --- | --- |
| 目录结构 | 镜像 `fla/ops/ascendc/<域>/<算子>` | 扁平 `experimental/<算子名>` |
| 根算子发现 | 递归 glob 推断边界 | 只枚举一级目录及其根 `CMakeLists.txt` |
| 实验依赖缺失 | 回退稳定树 | 明确失败 |
| 传递依赖 | 未统一闭包 | 全部来自实验树并递归校验 |
| kernel common | 固定稳定树 | 独立 `experimental/_common` 并切换 |
| `--ops` 校验 | 只扫描稳定树 | 按模式扫描对应树 |
| 入包依赖源码 | 仍可能取稳定路径 | 使用统一解析结果 |
| CI | 无 experimental job | 独立构建和产物审计 |
| Python/wheel | 不感知开关 | 独立实验注册与制品 |
| 产物名称 | 与稳定包相同 | 带 experimental 标识 |

这次过程的主要价值是确认了真实构建链及其旁路，也证明“只把直接算子放进 experimental”不足以完成严格隔离。一级算子目录契约降低了发现和审计复杂度，但最终实现仍必须以完整依赖闭包、公共代码切换和产物审计为边界。

## 9. 稳定路径零影响与模式互斥复审（2026-10-22）

本节记录精简样例阶段的复审；第 10 节记录随后恢复完整实验算子的结果。不把第 5 节的目标设计视为已完成能力。“互斥”分别检查构建输入、包内容、安装目标、运行时发现路径和 CI 执行路径；其中任何一层未隔离，都不能声称 Issue #931 已满足。

| 维度 | 当前代码与证据 | 结论 |
| --- | --- | --- |
| 默认构建参数 | `build.sh` 默认 `ENABLE_EXPERIMENTAL=FALSE`，只有显式 `--experimental` 才增加 `--experimental` 校验参数与 CMake `TRUE`；普通运行先清理 `build/`，继续使用 CMake 默认 `OFF` | 普通命令行参数保持原样 |
| 算子发现与依赖 | OFF 保留稳定树 glob、原单轮依赖挂载及原 kernel 依赖安装路径；ON 只发现 `experimental/<算子>/CMakeLists.txt`，缺失或路径形式实验依赖报错，不回退稳定树 | A5 单算子及依赖算子已验证；尚需 A2/A3 与全量覆盖 |
| 公共代码与编译输入 | OFF 仍使用 `fla/ops/ascendc/common`；ON 改用 manifest 校验的 `experimental/_common`，并跳过 `common/src/framework` 的稳定 ONNX plugin 源。根 `common/` 的通用 tiling/framework 基础设施仍共享 | A5 ON 构建仅有实验 kernel，稳定 ONNX plugin `.o` 为 0；任意 symlink/CMake 跨树引用尚无完整门禁 |
| 包内容 | A5 ON 隔离安装的 Ascend 950 kernel 目录仅有 `recompute_w_u_fwd_experimental`，aclnn 导出只有实验名；OFF 稳定包没有实验文件 | 当前两个单算子包的内容互斥，不代表全量包已审计 |
| 安装目标 | 两种模式当前仍生成同名 `.run`、同名 `fla_npu_transformer` vendor。安装脚本的 wheel overlay 会向该 vendor 合并文件并覆盖共享库；只用不同 `--install-path` 才能人为隔离 | **未保证**。禁止把实验包安装到稳定 wheel、稳定 CANN OPP 或其相同根目录；独立制品/vendor 和安装器拒绝混装是合入阻断项 |
| 运行时 | `torch_custom/fla_npu` 未修改，仍固定解析稳定 wheel 内的 `fla_npu_transformer`。若实验包覆盖同一 vendor，稳定进程可能加载被替换的 `libcust_opapi.so`；当前 fixture 也不是可运行的真实算子 | **未验证且未保证**。独立 runtime/wheel、加载路径与稳定执行回归仍需完成 |
| CI 与 Release | 默认 workflow 未设置 `CI_EXPERIMENTAL`，普通容器不传此变量；显式实验模式仅允许 OPP 打包阶段，禁止继续进入稳定 wheel、安装和运行阶段。当前没有独立 experimental job 或实验制品上传链路 | 稳定 CI 默认入口未切换；双平台正式 NPU CI、独立实验 CI 和 Release 产物审计仍待完成 |

A5 / Ascend 950 / CANN 9.2.0 对照记录：

- 同一 `--ops=kda_gate_bwd_post` 稳定包，修正前递归依赖逻辑带入 7 个 kernel 目录、453 个安装文件；恢复旧单轮逻辑后仅有选定算子的 kernel 目录、364 个安装文件，依赖 host 的 aclnn 导出集合相同。该差异证明原 PR 曾改变 OFF 包内容，不能以“OFF 单算子编译通过”替代回归。
- 修正后的 `--experimental --ops=recompute_w_u_fwd_experimental` 编译、打包和隔离安装通过；kernel 目录仅有实验名，构建目录无稳定 ONNX plugin `.o`。缺少实验 common 文件、缺少实验依赖、用稳定路径声明实验依赖均在 configure 阶段失败。
- 修正后的 OFF 依赖算子编译、打包和隔离安装通过；未运行算子，也未验证稳定 Python 调用、精度、性能或 A2/A3 的构建和安装。
- A5 `zzy` 环境的 Python CI 契约测试 156 项通过，其中 `ci.tests.test_experimental_build_contract` 的 6 项发现/解析和阶段隔离测试通过。`ci/tests/test_run_checks_stage_pipeline.sh` 在 A5 宿主机上使用假 `ASCEND_OPP_PATH`，被本机 toolkit `set_env.sh` 覆盖而失败；这不能作为真实容器或 GitHub-hosted CI 的通过证据，需要在其目标环境重跑。

合入前必须补齐：实验包独立文件名和 vendor、安装器拒绝合并到稳定 wheel/OPP、实验 runtime/wheel 策略与稳定调用回归、独立 experimental CI 与包来源审计、A2/A3/A5 稳定全量 NPU CI。当前只能确认上述 A5 隔离安装场景；**不能宣称构建、安装、运行、CI 四个维度已全面互斥，也不能据此关闭 Issue #931**。

## 10. 完整实验算子恢复与 A5 运行验证（2026-10-22）

先前把 `recompute_w_u_fwd_experimental` 精简成单 tile 编译 fixture 的步骤过早：它不能证明原始移植的运行链。现从 PR 提交 `a0e0e31f` 恢复 `experimental/recompute_w_u_fwd_experimental/` 及其七个 `_common` 头文件，保留独立的 `RecomputeWUFwdExperimental` 算子类型、kernel 入口、`opFile`、aclnn 导出和源码文件名。稳定 `recompute_w_u_fwd` 目录未修改，也不再继续精简该移植。

A5（Ascend950PR、CANN 9.2.0、conda `zzy`）在独立源码和 OPP 根目录执行：

1. `bash build.sh --pkg --experimental --soc=ascend950 --ops=recompute_w_u_fwd_experimental -j4` 成功，生成四个实验 Ascend 950 kernel 变体和 `.run`；稳定 ONNX plugin 编译对象为 0。
2. `.run --install-path=/home/zzy1205/fla-restored-recompute-install-20261022 --quiet` 成功；kernel 目录仅有实验名，`libcust_opapi.so` 只导出 `aclnnRecomputeWUFwdExperimental` 及其 `GetWorkspaceSize`，七个 manifest 公共头齐全。
3. `ci/run_experimental_validation.sh` 串行执行发现测试、构建、隔离安装、kernel/config/dynamic/proto/aclnn/common 清单审计和 ACLNN 实际运行，退出码为 0。测试进程只设置隔离 OPP 的 `ASCEND_CUSTOM_OPP_PATH`，不安装或覆盖稳定 wheel/OPP。
4. 两个 FP16、`A=I`、`k=v=beta=1`、`g=0` 的可核对用例通过：`V=128/chunk=64` 与 `V=256/chunk=128` 的全部 `w/u` 输出均为 1。该 smoke 覆盖实际 kernel 启动及两个形状分支，不等同于 ATK 全量精度、BF16、varlen 或性能验收。

本节所称 CI 是在 A5 直接运行仓内独立 CI 入口脚本，**不是 GitHub 自托管 NPU 检查已通过**。新 workflow 只在 PR 分支新增时不能单靠 `workflow_dispatch` 在主仓触发；正式独立实验 CI 仍需仓库管理员接入现有受控触发链或在目标分支部署后运行。第 9 节同名 `.run` / 同 vendor 安装风险仍在；独立 vendor、安装器拒绝混装、实验 wheel/runtime 及 A2/A3 稳定回归尚未实现，Issue #931 仍不可关闭。
