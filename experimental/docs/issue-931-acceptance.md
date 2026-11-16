# Issue #931：合并后主干编译工程验收核对

核对日期：2026-11-16。需求来源：[issue #931](https://github.com/flashserve/flash-linear-attention-npu/issues/931)，验收口径采用2026-11-16 需求方确认的补充说明。
源码固定为主干 [`134524a6`](https://github.com/flashserve/flash-linear-attention-npu/commit/134524a6aae55bc94101575146e88511856716e5)，包含 #1097 的合并提交 `4a9643ee`。本文记录该基线的验收结论和复现命令，不涉及产品代码或 CI 配置变更。

## 结论

**按本轮确认的验收口径，本项验收通过。**

- 实验算子相关改动参与现有正式 CI，只守护其不影响正式算子的构建、打包和运行。实验算子自身的功能正确性不由 CI 守护，使用手动验证；不要求独立实验 suite/job。
- A2 验证通过；按需求方确认，将该验收结论扩展认可至 A3/A5，三平台均标为通过。A3/A5 的通过属于本次验收口径认可，本轮未在这两个平台实际执行验证。
- “尚未完成转测、不能作为正式商用能力”的措辞由需求方后续补充，不作为本次验收阻断项。

目录边界：`experimental/ops/<op>/op_host/op_api/...` 等算子内部深层文件可以识别；`experimental/ops/<group>/<op>/` 这种新增分类层不受支持。当前设计只发现 `experimental/ops/` 的直接子目录，按这一既定布局验收。变更路径解析用于识别算子，不代表 CI 会构建或验证该实验算子的功能。

本次 A2 实包验证通过：实验 RWKV8 run/wheel 编译、包审计、3 项安装拒绝、17 项装载隔离、队列与 inline 各 3 条精度对拍均成功；随后在同一源码目录切回正式模式，`recompute_w_u_fwd` run/wheel 编译成功，产物未混入实验算子。框架回归合计 372 项，370 通过、2 跳过。

## 期望功能逐项映射（采用本轮确认口径）

| issue 期望功能 | 当前实现与验收范围 | 下文条目 |
| --- | --- | --- |
| 1. 新增独立 experimental 目录 | 已实现；真实算子在 `experimental/ops/fused_recurrent_rwkv8/`，实验框架、安装和测试也在 experimental 内 | 2、4、5 |
| 2. 两种构建模式严格互斥 | 已实现默认正式、显式实验；列表与交叉算子拒绝已验证 | 1、2、3 |
| 3. 正常构建、打包、验证，独立产物 | 有 run、分步 wheel、一键 wheel 和手动验证入口；实包结果见本次记录 | 4、5 |
| 4. PR 变更识别及 CI 回归 | 通过；当前布局可识别变更算子，实验相关改动运行现有正式 CI，只守护对正式算子的影响，不要求实验功能 CI | 6 |
| 5. 不影响稳定构建、打包和发布 | 通过；依据默认路径检查、共享回归及 A2 实编，按需求方确认扩展认可至 A3/A5；本次未执行真实发布 | 1、7、8 |

## 命令约定

以下命令在该主干版本的仓库根目录执行。先激活匹配 Python 环境、source 当前 CANN 的 `set_env.sh`，并准备与仓库固定版本一致的 third-party 依赖。

```bash
# 禁止正式包自动注入，确保验证进程的 NPU 初始化顺序可控。
export FLA_NPU_DISABLE_PTH=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
# 创建本轮输出父目录；各验证子目录须尚不存在。
OUT="$(mktemp -d /tmp/fla-issue931.XXXXXX)"
```

`--ops=all` 或不传 `--ops` 表示当前模式内全选，没有独立的 `--all` 参数。`--experimental` 须显式传入；不能由实验算子名或旧缓存自动启用。SoC 使用 `--soc`；同时传入历史 `-c/--compute-unit` 时，调用者保证二者一致。

### 1. 关闭 experimental：正式发现、编译、打包及默认发布路径

功能：默认仅扫描正式算子树；正式一键编包输出正式发行包，不带实验 Python、vendor 或算子实现。

```bash
# 列出正式算子；本次为 37 个，不含 fused_recurrent_rwkv8。
bash build.sh --list-ops
# 编译正式单算子 run；本次 A2 正式 wheel 构建内部覆盖了这一构建。
bash build.sh --pkg --soc=ascend910b --ops=recompute_w_u_fwd -j8
# 一键生成所选正式算子的 wheel；本次在实验构建后、同一源码目录中执行。
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=recompute_w_u_fwd \
  python scripts/build_wheel.py --build-args=-j8 --wheel-dir "$OUT/formal-wheel" --no-progress
# 负例：正式模式指定实验算子，应在 CMake 前报错并返回非零。
bash build.sh --pkg --soc=ascend910b --ops=fused_recurrent_rwkv8 -j8
```

默认正式 CI 在 `ci/run_checks.sh` 调用未带 `--experimental` 的构建入口；Package 工作流的 `ci/pkg/make_packages.py` 清除外部算子过滤参数后调用正式全量 wheel 入口。本项采用默认路径检查、共享回归及 A2 实编作为验收依据；本次不执行真实发布。

### 2. 开启 experimental：只发现和编译实验集合

```bash
# 列出实验算子；本次只有 fused_recurrent_rwkv8，参数顺序按此写。
bash build.sh --experimental --list-ops
# 全编实验集合，生成独立实验 run；本次手动验证内部执行此命令。
bash build.sh --pkg --experimental --soc=ascend910b --ops=all -j8
# 只选 RWKV8，产物不应含正式算子；当前实验集合只有它，因此与 all 等价。
bash build.sh --pkg --experimental --soc=ascend910b --ops=fused_recurrent_rwkv8 -j8
# 负例：实验模式指定正式算子，应明确失败。
bash build.sh --pkg --experimental --soc=ascend910b --ops=recompute_w_u_fwd -j8
```

功能：模式先选树，`--ops` 再在该树内筛选。单选、多选、全选和依赖闭包有临时算子夹具测试；真实多算子实编尚无样本，因为目前只迁入 RWKV8。实验入口支持 custom OPP/run/wheel，未开放的 static/JIT 等模式报错，不回退编正式集合。

### 3. 依赖隔离：缺失必须失败

```bash
# 执行真实 CMake configure 夹具，验证依赖闭包、缺失、环路、跨树路径和缓存切换。
python -m unittest discover -s experimental/_tests -p test_experimental_cmake_isolation.py -v
```

功能：测试自动建立 `experimental/ops/<op>/` 临时工程并执行 `cmake -S <源码夹具> -B <构建夹具>`。缺失传递依赖、依赖指向正式树、符号链接跨树均应失败；合法菱形依赖只挂载一次。测试不修改真实算子源码。主干代码在 `cmake/func.cmake` 与 `cmake/experimental.cmake` 实现检查，找不到依赖时不回退到正式目录。

### 4. 实验 run/wheel：独立产物及算子清单一致

```bash
# 一键完成实验 run、临时安装、launcher 编译、包审计和 wheel 输出。
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=fused_recurrent_rwkv8 \
  python scripts/build_wheel.py --experimental --build-args=-j8 --wheel-dir "$OUT/experimental-wheel"
```

也可分步构建，以下命令与本次手动验证内部链路对应：

```bash
# 编译全体实验算子。
bash build.sh --pkg --experimental --soc=ascend910b --ops=all -j8
# 将 run 安装到独立新目录，便于审计和打 wheel。
bash "build_out/fla_npu_experimental_linux-$(uname -m).run" --install-path="$OUT/experimental-opp" --quiet
# 对照本次构建清单，检查算子、源码快照和实际 kernel 文件。
python experimental/scripts/experimental_manifest.py audit \
  --install-root "$OUT/experimental-opp" --soc ascend910b --manifest build/experimental-build-manifest.json
# 编译匹配的实验 launcher，并将实验 Python 与 OPP 打成独立 wheel。
python experimental/_framework/build.py wheel --manifest build/experimental-build-manifest.json \
  --install-root "$OUT/experimental-opp" --output "$OUT/experimental-wheel"
```

预期身份：`fla_npu_experimental_linux-<arch>.run`、`flash_linear_attention_npu_experimental_a2-*.whl`、Python 包 `fla_npu_experimental`、vendor `fla_npu_experimental_transformer`。实验版本覆盖完整安装载荷，kernel 内容变化也改变版本。不要把普通 pip 更新 base 与管理器安装增量混为一谈。

本次实机执行分步链路；一键入口本次验证参数分发和包边界单测，不将其列为额外一轮独立实编。

### 5. RWKV recurrent 真实场景与后续测试

```bash
# 编译全部实验算子、安装 run/wheel、审计包、检查 host 装载和隔离，并执行队列/inline 精度对拍。
python experimental/_tests/validate.py --soc ascend910b --mode quick --device 0 -j8 \
  --output "$OUT/experimental-validation"
```

功能：使用 #446 迁入的真实 `fused_recurrent_rwkv8`，不是占位或 recompute 样例。quick 计划执行默认队列和 inline 各 3 条 NPU/CPU reference 对拍，并检查 3 种安装拒绝、17 种装载隔离；结果以输出 `summary.json` 为准。`--mode full` 才是各 209 条全量用例，本次未执行 full。

### 6. PR 变更、多层路径及正式影响回归 CI

```bash
# 列出深层算子文件变更，供旧路径解析工具读取。
printf '%s\n' \
  experimental/ops/fused_recurrent_rwkv8/op_kernel/fused_recurrent_rwkv8.cpp \
  experimental/ops/fused_recurrent_rwkv8/op_host/op_api/aclnn_fused_recurrent_rwkv8.cpp \
  > "$OUT/changed-files.txt"
# 只解析当前目录布局的变更算子，预期 fused_recurrent_rwkv8；此命令不会构建或触发 CI。
python scripts/ci/parse_changed_ops.py "$OUT/changed-files.txt" TRUE
```

本次调用解析函数验证，上述两类路径均得到 RWKV8。临时夹具 `experimental/ops/group/rwkv/` 的发现报缺少 `group/CMakeLists.txt`，变更解析为空。本次按既定 `experimental/ops/<op>/` 布局及算子内部深层文件验收，不扩展任意层级的算子分类。

**验收通过：实验算子相关改动跑 CI，只守护不对正式算子造成影响；实验算子自身功能正确性不通过 CI 守护。** 复用现有正式 CI，不新增独立实验 suite/job，也不要求在 CI 中编译或对拍实验算子。

由具备触发资格的账号在相关 PR 中发送：

```text
/run-npu-ci quick
```

功能：运行现有正式构建、打包、安装及正式用例回归，验证实验相关改动不破坏正式路径。实验 RWKV8 的构建、精度及隔离按第 4、5 节手动验证，不能将这些手动结果记为 CI 执行结果。本次核对确认了 CI 调用范围，未重新发送触发评论或启动 CI。

### 7. 稳定版本回归：A2/A3/A5 与共享构建流程

**验收通过。** 本轮实际执行 A2 验证，A3/A5 按需求方确认由 A2 结果扩展认可：

| 平台 | 验收状态 | 依据 |
| --- | --- | --- |
| A2 | 通过 | 本轮实编、包审计、安装/装载隔离及精度验证，共享框架回归通过 |
| A3 | 通过（扩展认可） | 基于 A2 结果，按需求方确认的验收口径认可；本轮未实测 |
| A5 | 通过（扩展认可） | 基于 A2 结果，按需求方确认的验收口径认可；本轮未实测 |

以下全量命令供复现或后续补充验证，不再作为本次新增的阻断条件：

```bash
# 在对应 SoC 配套环境执行正式全量 run 编译。
bash build.sh --pkg --soc=ascend910b --ops=all -j8
bash build.sh --pkg --soc=ascend910_93 --ops=all -j8
bash build.sh --pkg --soc=ascend950 --ops=all -j8
# 在各配套环境分别生成正式全量 wheel，不设置实验开关或算子过滤。
env -u FLA_NPU_OPS -u FLA_NPU_BUILD_ARGS FLA_NPU_SOC=ascend910b python scripts/build_wheel.py
env -u FLA_NPU_OPS -u FLA_NPU_BUILD_ARGS FLA_NPU_SOC=ascend910_93 python scripts/build_wheel.py
env -u FLA_NPU_OPS -u FLA_NPU_BUILD_ARGS FLA_NPU_SOC=ascend950 python scripts/build_wheel.py
# 回归共享管理器、ABI、launcher 和正式编包环境。
for name in test_wheel_overlay test_opp_abi_contract test_stable_gates \
            test_stable_build_policy test_stable_only_runtime test_wheel_environment; do
  python -m unittest discover -s tests -p "$name.py" -b || exit 1
done
```

本次共享回归 199 项：198 通过、1 跳过；另执行 A2 正式单算子 wheel 抽验。上述三平台全量命令未在本轮全部执行，也未做真实 Release 发布验证；这些执行范围记录保持不变，平台验收状态按上表确认。

### 8. 公共代码隔离与实验公共文件缺失

```bash
# CMake 负例包含公共目录链接到正式树、必需文件缺失、manifest 越界和正式模式忽略坏实验输入。
python -m unittest discover -s experimental/_tests -p test_experimental_cmake_isolation.py -v
# 产物审计负例检查混入额外源码、公共文件变化、空 kernel 和清单不一致。
python -m unittest discover -s experimental/_tests -p test_experimental_manifest.py -v
```

功能：实验专属公共文件在 `experimental/_common/`，`manifest.txt` 列出必须存在的文件；缺失时配置失败。当前 RWKV8 不需要额外算法公共头，清单仅有注释是合法情况。共享框架、SDK 和 ABI 工具仍被复用，修改这些公共基础设施仍需正式回归，不能理解为任意修改仓库公共代码都天然不影响正式构建。

### 9. 文档说明

无需编译命令。现有 [README](https://github.com/flashserve/flash-linear-attention-npu/blob/134524a6aae55bc94101575146e88511856716e5/experimental/README.md) 和[设计文档](https://github.com/flashserve/flash-linear-attention-npu/blob/134524a6aae55bc94101575146e88511856716e5/experimental/docs/architecture/experimental%20%E6%9E%84%E5%BB%BA%E9%80%82%E9%85%8D%E8%AE%BE%E8%AE%A1.md) 已说明用途、模式互斥、构建命令、不进入正式发布、手动验证及支持边界。

**验收通过。** 需求方确认“实验算子尚未完成转测，不能作为正式商用能力使用”的措辞不作为本次关键验收条件，由需求方后续在文档说明中补充。本报告保留该后续事项，不据此判定当前文档验收未通过。

## 本次实测与证据

环境：A2/910B2、aarch64，CANN 9.2.1、Python 3.12.16、torch 2.7.1+cpu / torch_npu 2.7.1.post5；复用已有容器运行手动命令，不触发 GitHub CI。opbase 固定 `c8cfc45e`，Catlass 使用当前主干指定 `86acf1db`，没有沿用缓存中的旧 Catlass。源码由固定提交导出；本文记录测试计数、产物清单及 SHA-256，原始日志保存在本轮手动验证输出目录，未随文档入库。导出源码不含 `.git`，因此正式测试 wheel 的版本带 `+unknown`；源码身份由固定提交归档确定，这不是 Release 发布验证。

| 项目 | 本次结果 |
| --- | --- |
| 实验构建、安装、清单与隔离框架 | 153 项：152 通过、1 跳过 |
| 实验 Python loader | 20/20 |
| 正式共享框架 | 199 项：198 通过、1 跳过 |
| 发现范围 | 正式 37、实验 1；集合不相交 |
| 交叉选择负例 | 两种方向都在 CMake 前返回 1，符合预期 |
| 当前布局的深层文件识别 | 两种路径均识别为 RWKV8 |
| 额外算子分类层夹具 | 不支持；明确记录边界 |
| A2 实验 run/wheel、安装与执行 | 通过；只有 FusedRecurrentRwkv8，安装拒绝 3/3、隔离 17/17、队列 3/3、inline 3/3 |
| 切回正式 A2 wheel 与产物隔离 | 通过；OPP 清单只有 RecomputeWUFwd，无实验 Python/vendor/RWKV8 文件，旧实验 run 已被清除 |

产物记录：

| 产物 | 文件名 | SHA-256 |
| --- | --- | --- |
| 实验 wheel | `flash_linear_attention_npu_experimental_a2-0.1.0.dev0+a723618235c5-py3-none-linux_aarch64.whl` | `14d07970b9ec709b7233ed4ef85ffdf4df27aff410f00c218f83ae86462827a7` |
| 正式抽验 wheel | `flash_linear_attention_npu_a2-26.10.0+unknown-py3-none-manylinux_2_34_aarch64.whl` | `56f22c120b0a1ef652a93c99fcebafb4492914c9ceaf984d8ac8195e4d0acc3f` |

复验时，实验验证入口生成 `summary.json`、`experimental-build-manifest.json`、阶段日志及精度明细；实验包清单可用第 4 节的审计命令核对。两个交叉选择负例的退出码 1 表示按预期拒绝，不是构建回归失败。

按本轮补充说明，本次验收通过。实际执行记录、A3/A5 的扩展认可依据及文档后续补充事项分别保留；每项命令仍标明用途和执行范围，不把扩展认可写成新增实机测试。本文不直接修改 issue 在线正文，后续同步时应采用以上验收口径。
