# 2026-11-03：A2 experimental 功能与隔离验证

> 历史记录：本轮制品包含现已移除的 recompute 样例。保留原始结果以供追溯，不能作为 2026-11-06 清理后制品的验证结论。

结论：本轮定向集成验证通过。单算子和全量实验制品可构建并正确执行已测用例；不同安装、
装载顺序和错误混用场景符合隔离预期，正式对照算子的文件及数值回归通过。
这是 [验证方案](_tests/a2_isolation/README.md) 的已执行子集，不代表完整算子验收或 Issue #931 关闭。
架构与兼容边界见 [设计文档](../docs/architecture/experimental%20构建适配设计.md)。

## 环境与源码

| 项目 | 实际值 |
| --- | --- |
| 服务器 | `z00879345@179.178.1.2`，Ascend 910B2，aarch64，卡 5 |
| CANN 初始化 | `source /home/z00879345/Ascend/9.0.0/ascend-toolkit/set_env.sh` |
| 实际 CANN 版本 | **9.0.0**，由 `share/info/asc-devkit/version.info` 确认；本轮不是 CANN 9.2.0 验证 |
| Python 环境 | conda `zzy-cann900`；Python 3.11.18；一次性 venv 共享该环境依赖 |
| torch / torch_npu / ATK | 2.12.0+cpu / 2.12.0 / 26.9.8 |
| 构建工具 | GCC/G++ 11.4.0；任务私有 PATH 引用已有 CMake 4.4.4 |
| main 基线 | `fb3d47dd54a3724dbcaba11d1996ad6d0227eae1`，叠加当前未提交的 experimental 适配和测试 |
| 测试工作根 | `/home/z00879345/experimental-a2-20261103-9R2eAj` |
| RWKV8 ATK 来源 | PR #446，`103490da248c81b6a77824fc13c153ad3a0931d2` |
| recompute ATK 来源 | main 基线正式工程中的 6 条现有用例，原 case ID 为 52、8、0、53、9、1 |

原始环境没有已安装的 FLA 包；正式包保护对象为本轮建立的正式/共存/overlay 测试环境。
所有安装、篡改和卸载操作使用任务私有目录，不覆盖母 conda 环境或其他用户任务。
源码导出不含 `.git`，因此正式对照 wheel 版本带 `+unknown`；源码基线与快照另行留档。

## 实测结果

证据路径相对于服务器工作根的 `evidence/`。各行子集可能重复，不累计成独立用例总数。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 构建/包契约 | 117/117，通过且无跳过；包含 runner 自测 15 项 | `unit-build.log` |
| 实验 Python CPU 契约 | 24/24 | `unit-python.log` |
| RWKV8 单算子 `.run` + wheel | 构建、安装、清单、ABI/ELF、host-only 加载通过 | `single/` |
| RWKV8 + recompute 全量 `.run` + wheel | 同上 | `all/` |
| 正式 chunk_fwd_o `.run` + wheel | 默认模式定向构建通过 | `stable/` |
| 装载矩阵 L01–L28 | 28/28；正式目录快照不变 | `load-isolation-001/summary.json` |
| 真实安装/卸载检查 | 19/19；正式目录快照不变 | `installation/summary.json` |
| 反向 pip 顺序及合法 overlay 移除补测 | 5/5；移除 overlay 后 base 哈希保留 | `extra-isolation/summary.json` |
| RWKV8 单算子精度 | **209/209**，原精度 JSON 不变 | `rwkv-accuracy/` |
| 全量 wheel 的 recompute 精度 | **6/6**，固定长度、g 开启、gk 关闭 | `recompute-accuracy/` |
| 全量 wheel 的 RWKV8 默认队列 | 3/3 子集 | `rwkv-full-wheel/` |
| 全量 wheel 的 RWKV8 显式 inline | 同一子集 3/3 | `rwkv-inline/` |
| 正式 chunk_fwd_o 三阶段数值回归 | 安装实验前、共存、卸载实验后各 **2/2**；FP16/BF16 fixed；正式包哈希不变 | `stable-lifecycle/summary.json` 及三阶段报告 |

RWKV8 的 CPU 标杆、输入构造和 run_cpu AST 与来源提交保持一致。recompute 复用正式工程的
相同 CPU 标杆和输入，仅改变 DUT、executor/case 标识及目标 SoC 元信息；不放宽容差。
两个实验 executor 在 ATK 过滤可选输出前检查输出槽位、shape、dtype、device。

ATK 专用环境使用测试 bootstrap 保证实验配置早于 torch_npu；装载矩阵的环境没有该 bootstrap。
真实 NPU 调用后同步并采集 `/proc/self/maps`、dispatcher、host 哈希及实际 launch mode。
三条实验执行轨迹显示仅使用全量实验 wheel 的自有 host 和实验 dispatcher，模式为 queue、queue、inline；
三阶段正式轨迹显示正式 host，未出现实验库或实验 dispatcher。加载通过与真实数值通过分别留证。

## 问题、处理与保留的失败证据

1. CANN 9.0 bisheng 发现不完整的 GCC 12 安装，首次构建缺 `cstdint`。
   在测试 shell 中指定 GCC/G++ 11，并通过 `CPLUS_INCLUDE_PATH` 指向对应三个 C++ 头目录后通过。
   未修改系统编译器；首次失败保留在 `single/build.log`。续建曾被残留 `compile_stop.flag` 拦截，
   确认上一构建结束后只清理本任务标志再成功续建，日志为 `single/build-resumed.log`。
2. 测试启动脚本在 source CANN 后清空 PYTHONPATH，导致 ATK 缺 acl/tbe。
   改为 source 前重置，装载 runner 只保留 CANN 根下的 Python 路径，并加入单元测试。
3. main 的 chunk_fwd_o ATK executor 固定 `use_exp2=True`、`BSND`，首次 A2 正式对照失败。
   仅在本任务暂存副本使用 A2 支持的 `use_exp2=False`、`BNSD`，CPU 自然指数参考不变；
   正式仓库 executor 未修改。初次失败位于 `stable-lifecycle-initial-a5-attrs/`，修正后重跑全部三阶段通过。
4. 安装器 4 项负例最初预期 argparse 文本，真实 `.run` 在 makeself 层先拒绝。
   改为核对该层明确的不支持参数原因、非零退出码和目标未创建，再逐项复跑通过。
   `installation/summary-initial-expectation.json` 保留初始判定，复测日志位于 `extra-isolation/`。

本轮修正的是测试环境、对照调用参数和测试判定，没有为通过验证放宽实验 loader 或修改生产算子。

## 覆盖边界

- 两种 pip 顺序均验证装载；数值三阶段使用正式→实验的安装顺序，反向顺序未重复数值矩阵。
- 正式 overlay 使用同 ABI、同内容副本，验证安装、优先路径、移除和 base 保留；未测试不同 kernel 的数值替换。
- 单算子 209 条覆盖以原 JSON 为准：不包含 reverse；不是设计表每个参数点和每分支三种种子的完整验收。
- 未执行 recompute gk/varlen、正式控制算子 varlen、非默认 stream、正式与实验同时并行 NPU、
  全部非法参数 NPU 矩阵、determinism、MSS 或性能测试。
- 独立 `.run` 验证了构建、安装和审计，数值测试使用匹配 wheel 内嵌 OPP，未做外部 `.run` 原生 ACLNN 数值对拍。
- 未覆盖 A2 CANN 9.2、A5 NPU、全量正式算子或正式发布兼容矩阵。此前 A5 编译结果单列于
  [A5 构建记录](validation-20261103.md)。

## 复跑与归档

装载矩阵的通用入口为 `_tests/a2_isolation/run.py`，配置模板和 28 项清单随仓库保存。
RWKV8 与 recompute 的 ATK 工程分别位于 `tests/atk/fused_recurrent_rwkv8/` 和
`tests/atk/recompute_w_u_fwd_experimental/`。准备顺序及环境要求见 [验证方案](_tests/a2_isolation/README.md)。

本次环境准备、构建、安装生命周期和 ATK 执行脚本归档在 `a2-evidence.tar.gz` 根目录，包括
`env.sh`、`build-a2.sh`、`prepare-profiles.py`、`run-rwkv.sh`、`run-all-smoke.sh`、
`stable-lifecycle.py`、`install-checks.py`、`extra-checks.py`、`collect-results.py`。
这些脚本记录本次绝对路径，复跑需新建独立工作根并替换路径，不能直接覆盖原证据目录。

`evidence/result-summary.json` 汇总环境、制品 SHA256、测试文件 SHA256、执行轨迹和通过数。
归档内包含 `.run`、wheel、日志、ATK Excel 报告、隔离摘要及初次失败证据；文档结果回填发生在归档之后。
归档含有用于负例的软链副本，查阅时可只提取常规报告文件。

| 归档 | SHA256 |
| --- | --- |
| `a2-validation-source.tar.gz`（初始源码及测试快照） | `025140d8e324271352822ea87dc68546a1b7514f40d494e306b6481d1945a7e2` |
| 源文件 LF 规范化排序哈希表摘要 | `7d3b5732f04c614faca3167488281552f678e16333e23a99ba74a08cb6f0c06d` |
| `a2-evidence.tar.gz`（回传后已复核） | `f9493de0ec429d90775c84dd92a19c0a763831426f2789d0fef07bced10439b5` |

服务器归档位于测试工作根；本地归档位于仓库外层 `.review-a2-isolation-20261103/`。
后续测试代码变化的最终摘要保存在结果 JSON 中，不能用初始源码归档代替最终测试文件身份。
