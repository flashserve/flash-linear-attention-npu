# 2026-11-04：最新主干同步与最小适配

> 历史记录：本轮制品包含现已移除的 recompute 样例。保留原始结果以供追溯，不能作为 2026-11-06 清理后制品的验证结论。

主干同步到 `a13871e8121d0c7b18eef9862e20dfce7ccb11be`，沿用现有 experimental 独立构建、
独立 wheel/dispatcher、同进程混用拒绝的设计。此前基线为 `fb3d47dd54a3724dbcaba11d1996ad6d0227eae1`。
通过显式拉取 `refs/heads/main`、fast-forward、恢复工作区完成同步，没有冲突。

## 修改范围

主干新增 39 个可达提交、29 个变更文件，主要是正式 chunk_fwd_h、causal_conv1d 和安装文档，
没有改动实验依赖的公共构建或 C++ runtime。保留这些主干更新，不另行修改正式算子。

在此前已实现的适配上，只收窄以下共享入口：

- `scripts/check_build_ops.py` 保留主干 `OP_SEARCH_ROOTS` 和正式发现循环，实验发现单独提前分支。
- `scripts/ci/parse_changed_ops.py` 恢复主干常量及原 helper，实验解析提前进入独立分支。
- `build.sh` 仅为实验模式禁用空变更集合的正式算子兜底，保留原条件块其他行为。
- 根 `setup.py` 恢复为主干原样；两个 setup 入口复用 `stable_build_policy.validate_build_options()`
  拒绝实验参数。保留 wheel CLI 自己的前置检查，因为 CLI 参数在此时尚未进入子进程环境。
- `build_stable.py` 一次构建只计算一次源码摘要，编译库和写入 Python 摘要使用同一值，
  避免编译期间源码变化导致两者不一致；新增对应回归测试。

未扩展实验接口、kernel 功能或正式发布流程。共享 runtime 的实验行为仍只在
`FLA_STABLE_EXPERIMENTAL` 宏启用时生效。此前 A2 209/6 条精度结果保留为
[2026-11-03 历史记录](validation-a2-20261103.md)，不当作本次重新构建后的结果。

## 静态与行为对比

- 正式 parser 对比主干：616 个路径/标志组合、9 组变更清单结果一致。
- 三个原 helper AST 与主干一致；去掉实验提前分支后，正式提取和发现函数 AST 与主干一致。
- 正式发现的 36 个算子及参数错误提示与主干一致；根 `setup.py` 与主干内容一致。
- 新指纹回归测试在旧代码下失败，修复后通过。

## A2 回归

本次工作根：`/home/z00879345/experimental-main-20261104-uTyhLy`。
使用既有 conda `zzy-cann900`、CANN 9.0.0、torch/torch_npu 2.12.0、ATK 26.9.8、GCC 11。
构建沿用上轮验证的任务私有 CMake 路径和 GCC 11 头文件设置。测试制品和环境均另建目录。

回归结论：定向集成验证通过。真实 NPU 测试使用卡 2；单算子、全量实验和正式对照均重新构建，
未复用前一轮 wheel。以下证据路径相对于本次工作根的 `evidence/`。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 全量 CI 契约 | 297/297，无跳过 | `unit-build.log` |
| 实验 Python 契约 | 24/24 | `unit-python.log` |
| 分阶段调度 | 通过 | `stage-pipeline.log` |
| Stable 契约 | 71 项中 70 通过，1 项因源码目录没有生成指纹跳过；从真实新 wheel 提取指纹后补测通过 | `unit-stable.log`、`unit-stamp-wheel.log` |
| OPP 契约 | 28/28 | `unit-opp.log` |
| 两个真实 setup 入口前置拒绝 | 4/4，输出哨兵不变 | `entry-rejections.json` |
| 单算子/全量实验 `.run` 与 wheel | 构建、安装、清单和 ABI/ELF 审计通过 | `single/`、`all/` |
| 正式 chunk_fwd_o wheel | 构建通过 | `stable/` |
| 装载矩阵 L01–L28 | 28/28，正式目录哈希不变 | `load-isolation-001/summary.json` |
| recompute 精度 | 6/6 | `recompute-accuracy/` |
| RWKV8 默认队列 / inline | 同一子集分别 3/3 | `rwkv-full-wheel/`、`rwkv-inline/` |
| 正式算子安装前 / 共存 / 卸载后 | 各 2/2，FP16/BF16 fixed，正式包哈希不变 | `stable-lifecycle/summary.json` |

真实执行轨迹确认实验调用使用新 wheel 的实验 host/dispatcher，实际模式覆盖 queue 和 inline；
正式算子三阶段只使用正式 host，未加载实验库。正式 ATK 对照沿用上轮任务副本中的 A2 参数
`use_exp2=False`、`BNSD`，CPU 参考和仓库正式 executor 未修改。

本轮没有重跑 RWKV8 209 条全量精度、19 项安装负例矩阵或完整 reverse/varlen/MSS/性能验收，
也没有执行 A5 或 A2 CANN 9.2 测试；不扩大本次结果的适用范围。

首次全量 CI 契约测试缺少 PATH 中的 `node`。补入服务器已有 Node 24.22.2 的任务私有软链后重跑，
原始失败日志单独保留；不改变系统或 conda 配置。
新测试环境安装的 Kombu 5.7.0 拒绝 ATK 的队列名称，初次任务尚未进入算子执行。
停止本任务进程树后，将两个私有 ATK 环境固定到前轮已验证的 Kombu 5.6.2，再复跑通过。
初始日志保存在 `recompute-accuracy.log.initial-kombu-5.7.0` 及对应目录，未为此修改生产代码。

## 追溯

本地同步前文件、暂存区和工作区补丁保存于仓库外层 `.review-main-20261104/`；
Git stash `ab5023454b99d895bb8e9cdd81ed0b760a740429` 同样保留作为同步前备份。
验证源码归档的 LF 规范化文件哈希表摘要为
`1e3006927f204cd58247772399e3936433b0359b1f17f7a8bdecb9c32181b4ca`。
本记录和设计文档的结果回填发生在验证源码导出之后，不影响被编译的代码身份。
收尾时再次通过 GitHub 核实主干仍为 `a13871e8`。测试导出后的代码文件哈希保持一致，
仅更新了设计、README 和本记录。

完整结果、制品 SHA256 和实际库轨迹见 `evidence/result-summary.json`；
`a2-evidence.tar.gz` 保存构建/测试日志、制品及复跑脚本。本地副本位于 `.review-main-20261104/`。
回传后归档 SHA256 已核对为
`2cdd57a542294fe49085037e48f1e338ad5d658b9624c92ee6b63cbf3610ace5`；
两个实验 wheel 的 `source_commit` 和全部实际源码输入摘要均与当前工作区匹配，见本地
`local-verification.json`。
