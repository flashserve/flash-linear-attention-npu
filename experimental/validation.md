# experimental 验证记录

当前基线为 main `7065b0cd`（包含已合入的 #1177、#1148）。experimental 不跑 CI，以下区分本轮手动检查和历史实机结果。
日志和数值明细保存在各次验证指定的输出目录；历史结果不累计为当前提交的通过数。

## 2026-11-13 增量验证目标与完整制品版本

手动增量验证删除 `--package`，自动定位当前 Python 环境管理器对应的实验 base；管理命令、
精度子进程和清理固定使用这个目录。实验 wheel 版本由已组装的完整包内容生成，包含 OPP，
不再仅取 launcher 源码指纹；适配层 ABI 指纹和通用增量安装规则不变。

| 检查 | 结果 |
| --- | --- |
| A2 Linux 容器：实验框架回归 | 153 项：152 通过，1 项跳过 |
| 当前环境定位与目标一致性 | 不导入实验 runtime；安装、执行及异常清理使用同一个 base；旧 `--package` 参数被拒绝 |
| 完整制品版本回归 | kernel、launcher、Python、清单或管理入口内容变化均改变包版本；launcher ABI 指纹不因计算包版本而改写 |
| 普通 pip 安装更新 | 临时 wheel 夹具保持适配层不变、只改 kernel 文件内容，版本变化后普通 `pip install --no-deps` 安装了新内容；同版本强制重装也通过 |
| A2 实验 wheel 打包 | 复用已有 RWKV8 run，重编 launcher，ABI/ELF 审计和打包通过；未重编 kernel |
| A2 手动增量验证 | wheel、wheel+run、禁用、恢复、仅 run 五种场景均通过；每种新进程完成 3 条 NPU/CPU reference 对拍，共 15 条 |
| base 与清理 | 原始 base 文件未变，最终 active/pending 均清空，恢复 base 选择 |

初始化 CANN，并在已安装本轮实验 wheel 的 Python 环境执行：

```bash
# 关闭正式包启动注入及 NPU 自动导入，保持实验进程初始化顺序。
export FLA_NPU_DISABLE_PTH=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
# 自动定位当前环境的实验 base，验证五种增量装载方式及恢复。
python experimental/_tests/incremental.py --wheel "${UPDATE_WHEEL}" --run "${UPDATE_RUN}" \
    --device 0 --output "${VALIDATION_OUTPUT}/incremental"
```

实机环境为 CANN 9.2.1、Python 3.12.16、torch 2.7.1+cpu / torch_npu 2.7.1.post5。
本轮真实 wheel 的内容版本为 `0.1.0.dev0+cfc448bd6f74`，launcher 源码指纹为 `352d4c57492b`，二者职责独立。
153 项回归及 15 条对拍属于本轮结果；普通 pip 更新验证使用一次性 Python 环境和临时制品夹具，
不把它当作另一轮 NPU kernel 执行。未重跑全量精度、MSS、性能或 A5；未增加实验 CI。

## 2026-11-13 一键实验 wheel 入口

`scripts/build_wheel.py --experimental` 在正式 launcher 预编译之前分流至实验编包器，
自动完成 run 构建、临时安装、launcher 编译和 wheel 输出。公共入口沿用参数转换、日志和
本次产物选择；实验编排仍在 `experimental/_framework/build.py`，未修改正式 `setup.py` 或 runtime。

| 检查 | 结果 |
| --- | --- |
| A2 Linux 容器：实验框架回归 | 149 项：148 通过，1 项跳过 |
| 新增一键入口与编包回归 | 12/12；覆盖默认正式路径、实验分流、参数透传、失败不发布、旧产物拒绝及前置依赖检查 |
| Python loader/参数、CPU golden | 20/20、3/3 |
| 正式 wheel/OPP、Stable gates、ABI/runtime/build policy 及 wheel environment | 211 项：210 通过，1 项跳过 |
| A2 一键构建实验 wheel | 通过；单条命令完成 run、临时安装、launcher、ABI/ELF 审计和 wheel 输出 |
| 实际 wheel 内容与 host 装载 | 只有 RWKV8 实验算子；无正式包/vendor/启动 `.pth`，仅注册实验 namespace，未初始化 NPU |

多算子发现、选择及缺 adapter/wrapper 使用临时夹具；真实实验算子仍只有 RWKV8。
实机命令如下，先初始化 CANN 并设置 `FLA_NPU_DISABLE_PTH=1`：

```bash
# 一键构建 A2 实验 RWKV8，使用 8 个编译任务，输出到本轮验证目录。
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=fused_recurrent_rwkv8 \
    python scripts/build_wheel.py --experimental --build-args='-j8' --wheel-dir "${VALIDATION_OUTPUT}/wheels"
```

环境为 CANN 9.2.1、Python 3.12.16、torch 2.7.1+cpu / torch_npu 2.7.1.post5、opbase `c8cfc45e`。
生成 `flash_linear_attention_npu_experimental_a2-0.1.0.dev0+a7e71e999574-py3-none-linux_aarch64.whl`，
launcher 源码指纹 `a7e71e999574`。日志为 `${VALIDATION_OUTPUT}` 下的 `framework.log`、`formal.log`、
`loader.log`、`golden.log`、`oneclick.log`、`wheels/fla-npu-experimental-build.log`、`artifact.json` 和 `host.log`。
本轮未重跑 NPU 精度、实机隔离或增量矩阵；experimental 仍不接入 CI，以上均为手动验证。

## 2026-11-13 算子目录收敛到 ops

RWKV8 整体移至 `experimental/ops/fused_recurrent_rwkv8/`。同步修改 CMake 发现与依赖解析、
构建清单和 wheel adapter 路径；共享框架与 `_common` 不迁移。算子 kernel、host、adapter 内容不变。
实验发现与依赖不再查找旧的 `experimental/<op>/` 目录，正式模式仍不扫描实验算子。

| 检查 | 结果 |
| --- | --- |
| A2 Linux 容器：实验构建、安装与隔离框架 | 137 项：136 通过，1 项跳过；含新目录发现、旧路径拒绝和目录链接边界 |
| Python loader/参数、迁移后的 CPU golden | 20/20、3/3 |
| 正式 wheel/OPP、Stable gates、ABI/runtime/build policy | 167 项：166 通过，1 项跳过 |
| CMake 3.16.9 / Windows | 28 项：18 通过，9 项因符号链接权限跳过，1 项新版私有属性用例按版本跳过 |
| CLI 算子列表 | 实验列表仅含 RWKV8，正式列表不含 RWKV8 |
| A2 实验 `.run`、wheel 构建、安装、包/ABI 审计及 host 装载 | 通过；产物仅含 RWKV8，host 仅注册实验 namespace，未初始化 NPU |
| 文档链接、算子源码迁移检查 | 30/30 本地链接有效；26 个文件随目录迁移，实现文件内容不变 |

构建使用 `fla-npu-ci:9.1.0-910b` 镜像，实际 CANN 9.2.1、torch 2.7.1+cpu / torch_npu 2.7.1.post5，
opbase 为主干指定的 `c8cfc45e`。首次未挂载完整驱动环境时依赖检查失败，补齐挂载后通过；
构建和包审计使用独立工作目录，没有修改现有安装。日志保存在 `${VALIDATION_OUTPUT}` 的
`framework.log`、`loader.log`、`formal.log`、`golden.log` 及 `result-with-driver/summary.json`、`result-with-driver/build-host.log`。
以上为手动验证，本轮未重跑 NPU 精度、实机装载隔离或增量矩阵，未触发 CI；实验算子仍不接入 CI。

## 2026-11-13 同步主干

同步 main `7065b0cd`，包含 #1148 固定 opbase 版本及最新正式 GDN/KDA 修改，rebase 无冲突。
保留“#446 源码基线”和“experimental 适配”两次提交；同步后的补充修改仅更新文档基线与验证记录。

| 检查 | 结果 |
| --- | --- |
| A2 Linux 容器：实验构建、安装与隔离框架 | 134 项：133 通过，1 项跳过 |
| Python loader/参数、CPU golden | 20/20、3/3 |
| 正式 wheel/OPP、Stable gates、ABI/runtime/build policy | 167 项：166 通过，1 项跳过 |
| CMake 3.16.9 / Windows，完整 CMake 隔离测试 | 26 项：17 通过，8 项因符号链接权限跳过，1 项新版私有属性用例按版本跳过 |
| `git diff --check` | 通过 |

以上均为手动检查，未使用 NPU 设备，未重跑实验 NPU 构建或精度。
日志为 `${VALIDATION_OUTPUT}` 下的 `framework.log`、`loader.log`、`formal.log`、`golden.log` 和 `cmake-3.16.9.log`。
experimental 仍不接入 CI；正式算子的 quick CI 通过 PR 评论单独触发，结果以对应 Actions 运行记录为准。

## 2026-11-12 CMake 3.16 兼容性修复

`fla_audit_experimental_targets` 对 CMake 3.19 以前的 interface target 只读取允许的接口属性，
以目标所属目录解析相对路径；较新版本及其他 target 保留完整属性检查。最低版本仍为 3.16。

| 检查 | 结果 |
| --- | --- |
| CMake 3.16.9 / Windows，完整 CMake 隔离测试 | 26 项：17 通过，8 项因符号链接权限跳过，1 项新版 interface 私有属性用例按版本跳过 |
| CMake 3.22.1 / A2 Linux 容器，完整 CMake 隔离测试 | 26/26，通过合法目标、正式路径拒绝、嵌套目录相对链接和新版私有属性检查 |
| Linux 强制旧版兼容分支补测 | 4/4；实际 CMake 为 3.22.1，测试中将版本变量设为 3.16.9，补验目录解析和符号链接隔离 |
| `git diff --check` | 通过 |

日志为 `${VALIDATION_OUTPUT}` 下的 `cmake-3.16.9.log`、`cmake-3.22.1.log`、`legacy-branch-linux.log`。
本轮只修改 CMake 审计分支并增加回归用例，未重跑 NPU 构建或精度，未触发 CI；下方为此前实机结果。

## 2026-11-12 检视修复与 A2 复验

修复部分增量更新的加载误拦截：base 只校验、绑定和注册未被覆盖的算子，
增量适配层负责被覆盖的算子；只更新 OPP 时仍检查原适配层的完整 ABI。
`chunk_len` 非法值改为属性级结构化错误上报，保留原返回码；同时修复三处文档断链，
将公开记录中的个人服务器路径替换为输出目录变量。本轮所有改动位于 `experimental/`，未触发 CI。

环境同下方目录重构阶段。测试制品的 launcher 源码指纹为 `b35edaf514d2`，已与本地待提交源码核对一致。
CPU reference、209 条精度用例及其阈值未改变。

| 检查 | 结果 |
| --- | --- |
| 实验框架测试 | 131 项：130 通过，1 项备选 ATK 检查跳过 |
| 部分更新回归 | 两算子 base 只更新一个算子的 ABI 后，另一算子通过检查；仍拒绝未更新算子的 ABI 错误及仅 OPP 的不兼容更新 |
| 真实编译注册筛选 | 缺少被覆盖旧算子的符号时，剩余算子仍可注册和调用；旧入口未注册，全量选择仍报缺符号 |
| Python loader/参数、CPU golden | 20/20、3/3 |
| 正式 wheel/OPP、Stable gates、ABI/runtime/build policy | 167 项：166 通过，1 项跳过 |
| 实验 run/wheel 编译、ABI/ELF 审计与 host 装载 | 通过 |
| 安装拒绝正式目录 / 装载隔离 | 3/3、17/17 |
| NPU 与 CPU reference：队列 / inline | 209/209、209/209 |
| wheel、wheel+run、禁用、恢复、仅 run 增量 | 5/5，每种 3/3 NPU 对拍，base 文件不变 |
| 直接 ACLNN 非法 `chunk_len` | -1、0、2^32 均返回 161002；RecentErrMsg 含 EZ0002、参数名、实际值和合法范围 |
| experimental 文档本地链接 | 30/30 有效 |

报告保存在本次 `${VALIDATION_OUTPUT}` 下的 `result/summary.json`、`incremental/summary.json`、
`dfx.json`；两算子边界使用临时夹具，真实 NPU 算子仍为 RWKV8。
本轮未重跑正式算子 run/wheel 构建，下方对应结果属于目录重构阶段。

当时的已知问题：手动增量验证脚本未向管理器透传 `--package-dir`，指定目录与 Python 默认位置不同时
会操作错误目标。现已删除验证脚本的 `--package` 参数，修复与复验见本页最新记录。
本节历史验证使用二者一致的新建隔离 base，不覆盖目标位置不一致或已有增量的环境。

## 2026-11-12 目录重构、运行时精简与通用增量接入

以下为检视修复前 `09c9a726` 的目录重构阶段结果。实验专属资产已迁入 `experimental/`，
正式 builder、runtime 和构建校验脚本恢复本次 main 基线；通用 layer 管理器仅增加布局配置。
本轮没有修改或触发 CI。

环境：A2 / Ascend 910B2，物理卡 5；现有镜像标签 `fla-npu-ci:9.1.0-910b`，
**实际 toolkit 为 CANN 9.2.1**，Python 3.12.16、torch 2.7.1+cpu、torch_npu 2.7.1.post5。
CPU reference、输入值域、209 条 JSON 和混合容差未改变；本轮不声明完整正式算子验收。

| 检查 | 结果 |
| --- | --- |
| 构建/安装/通用增量/手动工具单测 | 129 项：128 通过，1 项备选 ATK 检查跳过 |
| Python loader 与参数单测 | 20/20 |
| RWKV8 CPU golden | 3/3 |
| 正式 wheel/OPP 增量、Stable gates、ABI、runtime、build policy 回归 | 167 项：166 通过，1 项跳过 |
| 实验 run、wheel、ABI/ELF 审计与 host 装载 | 通过；只注册 RWKV8 实验 namespace |
| 普通 run 安装拒绝正式包根、其 opp 及新建子目录 | 3/3，受保护目录不变 |
| 首次装载与正式/实验混用隔离 | 17/17，无 FAIL/BLOCKED |
| NPU 与 CPU reference，队列模式 | 209/209 |
| NPU 与 CPU reference，inline 模式 | 209/209 |
| wheel、wheel+run、禁用增量、移除恢复、仅 run 增量 | 5/5 场景，各 3/3 NPU 对拍；base 文件不变 |
| 同一工作目录从实验构建切换到正式构建 | 正式 `recompute_w_u_fwd` run/wheel 编译成功，无实验残留 |
| 正式与实验 wheel 内容检查 | 包路径、vendor、layer 算子清单、host 符号集合隔离；正式不含 RWKV8，实验只有 RWKV8 |

多算子与依赖闭包使用临时夹具验证，仓内真实实验算子仍只有 RWKV8。
保留首次 ABI、launcher 指纹和所选符号来源检查；不再把加载后的文件篡改监控作为验收条件。
跨集合误选增量 base 属于违反使用前提，不提供实验专属拦截保证。

实机日志根目录使用运行时指定的 `${VALIDATION_OUTPUT}`。
主要记录为 `framework-final.log`、`loader.log`、`golden.log`、`formal.log`、
`result/summary.json`、`incremental/summary.json`、`package-boundary.json`。
`result/accuracy-{queue,inline}.json` 保存逐用例数值结果；`incremental/*-trace/` 保存实际 wrapper、
dispatcher、OPP 与 launcher 来源。所有测试均由手动命令启动，没有发布 GitHub CI 状态。

完整精度使用 base launcher 指纹 `13e109cc1969`；随后清理私有编译 helper 文件末尾空行，
重建得到 `58087a0a9e9d`，用于增量验证。运行时代码、生成注册单元和 kernel 未改变；
最终源码指纹已核对与增量 wheel 一致。

当前执行命令见 [README](README.md)，设计与迁移映射见
[设计文档](docs/architecture/experimental%20构建适配设计.md)。

## 2026-11-12 同步 #1177 与更新待确认设计

同步实际已合入的 #1177 `f87e0b38`，未采用其 PR 正文中“本地待推送”的后续内容。
解决 `build_stable.py` 编译 helper 冲突，保留主干 `-fno-gnu-unique`、layer 协议标记，
以及当前实验构建的显式编译单元。runtime 保留主干显式多路径处理和当前实验隔离分支。
新增手动回归用例检查合并后的编译参数与源码选择。

在 A2 主机原生 Python 3.11 环境手动执行：

| 检查 | 本轮结果 |
| --- | --- |
| `experimental/_tests` | 127 项：126 通过，1 项备选 ATK 检查默认跳过 |
| `experimental/_python/tests` | 21/21 |
| 正式 wheel/OPP overlay、Stable gates、ABI、runtime 与 build policy | 168 项：167 通过，1 项跳过 |
| `.github/`、`ci/`、`scripts/ci/` 和 `scripts/github/` | 与本次主干内容一致，无新增实验接入 |

日志在该次 `${VALIDATION_OUTPUT}` 的 `manual.log`、
`loader.log`、`overlay.log`。正式回归命令如下，不启动 NPU 构建或精度任务：

```bash
# 检查同步后的正式增量安装、ABI 和运行时兼容规则。
python -m unittest tests.test_wheel_overlay tests.test_opp_overlay tests.test_stable_gates tests.test_opp_abi_contract tests.test_stable_only_runtime tests.test_stable_build_policy -b
```

本轮未触发 NPU CI，未重建 NPU 制品或重跑精度。目录收敛、运行时精简和转正式流程仅更新设计，
待确认后实施；当前执行命令仍以 [README](README.md) 为准。

## 2026-11-11 移除 CI 接入

已移除实验 CI 入口、必需状态、触发设计和变更算子调度；CI 文件恢复主干版本。
构建、安装/运行隔离、RWKV8 实现及数值规则保持不变。验证脚本和规则测试移至
`experimental/_tests/`，仅通过显式命令手动执行；不再读取 CI 参数或使用 CI 缓存/容器入口。
当前使用方式见 [手动验证](README.md#手动验证)。

在 A2 主机原生 Python 3.11 环境手动执行：

| 检查 | 本轮结果 |
| --- | --- |
| 构建/安装规则及手动验证工具单元测试 | 126 项：125 通过，1 项备选 ATK 检查默认跳过 |
| Python loader 单元测试 | 21/21 |
| RWKV8 CPU golden 单元测试 | 3/3 |
| 手动入口参数解析、依赖预检及 Python 语法检查 | 通过 |
| `.github/`、`ci/`、`scripts/ci/` 和分支保护脚本 | 与主干内容一致，无实验接入 |

日志在该次 `${VALIDATION_OUTPUT}` 的 `manual.log`、`loader.log`、
`golden.log`、`dependencies.log` 及两个 `*-help.log`。本轮未触发 NPU CI，未重建 NPU 制品，
也未重跑 NPU 精度；下方结果属于原基线。

## 2026-11-10 A2 手动构建与数值验证（main `959d6b19`）

本轮在 A2 上从精简后的工作树重新构建并运行完整入口，不复用旧 wheel。
环境：aarch64 / 910B，现有 `fla-npu-ci:9.1.0-910b` 镜像，CANN 9.1.0、Python 3.12.13、torch 2.7.1+cpu / torch_npu 2.7.1.post5，无 ATK/Kombu。

| 检查 | 本轮结果 |
| --- | --- |
| 依赖预检 | 通过；缺 ATK 只 warning，后续构建正常执行 |
| 实验 `.run`、独立安装、包/ABI 审计、wheel 和 host 装载 | 全部通过；算子集合只有 RWKV8 |
| 真实 `.run` 拒绝正式包根、其 `opp` 和新建子目录 | 3/3，受保护目录不变 |
| 装载隔离 | 22/22，无 FAIL/BLOCKED，受保护目录不变 |
| NPU / CPU reference，默认队列 | 209/209；执行证据确认 `queue` |
| NPU / CPU reference，强制 inline | 209/209；执行证据确认 `inline` |
| 共用 reference 后的 PTA 原有边界/负例 | 13/13，使用同一新 wheel |
| 实验 Python loader / CPU golden | 21/21、3/3 |

上述结果来自在 A2 环境中手动执行验证入口。

构建清单记录 `source_commit=f2dd8b93b59c2428f7c01a30bf14d49ecc70ba8b`、`source_dirty=true`，表示在主干同步提交上验证待提交的精简工作树；16 个算子/公共头输入文件哈希与提交输入一致。算子、launcher 和 Python runtime 本轮未改。证据保存在 A2 工作目录 `experimental-minimal-20261110-SrrP3X` 的 `result/` 及 `contracts.log`、`loader.log`、`stages.log`、`golden.log`、`pta.log`。

| 证据/制品 | SHA256 |
| --- | --- |
| `result/summary.json` | `5e46e6854e4e55754d038a2a918a535f6c30c5a9e2e50d13232f6326cbee15a8` |
| `fla_npu_experimental_linux-aarch64.run` | `75abdb4e7dc099bb7391e401aa2d47f056b51edc258109124580fdd256b1bf04` |
| `flash_linear_attention_npu_experimental_a2-0.1.0.dev0+799b4432c42a-py3-none-linux_aarch64.whl` | `baa9ae98a36f72ecaeb2ed77d59f81439c0f83ea88846d1ded94440d31c2b78c` |

本轮没有重跑 A5；这些结果属于当时基线，不计为后续提交已重建和重测。

## 2026-11-10 直接 aclnn 输出布局补测

针对当前实现 `ef34d887`，复用上节同实现的实验 OPP，在 A2 / CANN 9.1.0 镜像内用 C++
直接调用 `aclnnFusedRecurrentRwkv8GetWorkspaceSize` 和执行接口，不经过 Python adapter。
这是一组入口布局补测，不计入原 209 条 Python 精度用例，也没有重跑全量用例。

固定输入：`B=H=1`、`T=2`、`K=V=8`，六个连续 FP32 输入全部为零；无初态，`scale=1`、
`reverse=false`、`outputChunkState=false`、`outputSa=false`、`chunkLen=16`。
两组输出逻辑 shape 都是 `(1,1,2,8)`，存储预填 7，数学期望为 16 个零。

| 输出布局 | 存储与 strides（元素为单位） | 结果 |
| --- | --- | --- |
| 连续 | 分配 16 个 FP32；strides=`(16,16,8,1)` | workspace 查询返回 0，大小 1536 B；执行并同步后 16/16 输出为零 |
| 非连续 view | 分配 32 个 FP32；strides=`(32,32,16,2)`，逻辑元素占用存储下标 `0,2,...,30`，无重叠、无越界 | workspace 查询返回 `561103`；消息 `Output tensor should not from workspace`；未调用执行接口 |

这证明当前直接 aclnn 的合法非连续输出无法执行，不证明已发生静默结果错误。
`sOut/saOut` 的同类处理仅做了代码检查，尚未分别复现。Python adapter 自行分配连续输出，
没有调用方提供的 `out=`，因此已有 Python 精度结果与本限制并不矛盾。
该写回逻辑在来源 PR #446 的 `049eb0d` 中已经存在，尚未修复；原因与修复方向见
[设计 §6.2](docs/architecture/experimental%20构建适配设计.md#62-rwkv8-直接-aclnn-的非连续输出限制)。

## 已有参考证据（不计为本轮重跑）

| 版本/日期 | 环境 | 已证实范围 |
| --- | --- | --- |
| `ee344941` / 2026-11-09 | A2，CANN 9.1.0 镜像，torch 2.7.1 / torch_npu 2.7.1.post5，无 ATK/Kombu | 复用旧 wheel，直接 NPU / CPU reference 默认和 inline 各 209/209；依赖预检缺 ATK 只 warning |
| 2026-11-09 | A2 原生，CANN 9.0.0，torch/torch_npu 2.12.0 | 旧 28 项矩阵及 quick 3×2，通过；Linux 契约测试 314 项，含备选 ATK 比较器对齐 |
| 2026-11-06 | A2，910B2，CANN 9.0.0，ATK 26.9.8 | RWKV8-only `.run` / wheel，`--ops=all` 和单选构建；旧矩阵 28/28；原 ATK 默认/inline 各 3/3 |
| 2026-11-03 | A5，CANN 9.2.0 | 当时版本的编译、包内容与 host 装载；未验证 A5 数值 |

2026-11-06 卡 2 曾出现基础 `isfinite` 异常，在不加载 FLA 的全 1 张量上也可复现；当轮最终 ATK 数值结论来自卡 1。该异常不作为算子精度通过证据。


## 通过标准与限制

- 正式/实验集合分别构建，实验制品只含 RWKV8。
- 实验 `.run` 拒绝正式包目录，wheel 使用独立身份；装载负例必须因指定原因被拒绝。
- A2 默认与 inline 对拍，逐输出数值和实际库来源符合契约；A5 只计构建/host。
- 未完成 reverse、并发、多 stream、确定性、MSS、性能或 A3/A5 数值全量验收；不声明正式发布验收完成。
