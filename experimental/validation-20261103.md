# 2026-11-03：新 main 实验构建适配验证

> 历史记录：本轮制品包含现已移除的 recompute 样例。保留原始结果以供追溯，不能作为 2026-11-06 清理后制品的验证结论。

本文保留先前 A5 构建阶段记录；随后完成的 A2 真实 NPU 功能与隔离测试见
[A2 验证记录](validation-a2-20261103.md)，不回填到以下 A5 结果中。

本轮按用户确认范围完成构建与打包适配。A5 host/kernel 编译、包审计、独立安装、
Stable ABI launcher/wheel 与 host-only 加载通过；没有执行 A5 或 A2 NPU 算子。
这不是 `docs/agents/05-算子测试.md` 的正式算子验收，也不据此关闭 Issue #931。

## 代码与环境

- main 基线：`fb3d47dd54a3724dbcaba11d1996ad6d0227eae1`。
- 本地分支：`feat/experimental-stable-abi`；工作区尚未提交、推送。
- RWKV8 来源：PR #446；旧迁移的 UB 容量和 chunk_len 边界修复保留。#954 的 recompute 仅为框架样例。
- A5：`zzy1205@179.228.2.2`，工作目录 `/home/zzy1205/experimental-stable-abi-20261103`。
- 原始 main 对照目录：`/home/zzy1205/main-baseline-fb3d47dd-20261103`。
- CANN：`/home/zzy1205/Ascend/cann-9.2.0/ascend-toolkit/set_env.sh`；conda `zzy`。
- 实测环境：x86_64、Python 3.12.15、torch 2.13.0+cpu、torch_npu 2.13.0、GCC 13.3.0；未做旧 torch 版本加载矩阵。
- 构建命令、环境与日志位于服务器 `evidence/`；所有 Python 构建/验证入口关闭稳定 `.pth`。

## 实现范围

实验模式按一级目录发现算子，仅展开实验依赖；检测缺失、环路、链接越界及稳定源码引用。
实验 kernel common 使用自身快照。所选/解析集合统一驱动 host、kernel、adapter 和产物审计。

实验 `.run` 使用独立 vendor 和安装器。安装必须指向显式绝对空目录，拒绝稳定 OPP、
稳定 wheel、默认升级/overlay 与越界链接；校验 main `abi_manifest.json` 的库/头摘要。
额外来源清单记录源码、选中闭包、SoC、架构与文件摘要，包审计验证实际内核和源码集合。

独立 Stable launcher 只生成所选实验注册，使用独立 dispatcher、库名与查询符号。
沿用 main 的当前流、队列与 descriptor 生命周期设施；修复 #446 零长度快照 backing
buffer 在调用前提前释放的问题。移除 ctypes descriptor 执行后端，ctypes 只读取库的指纹/能力。

实验 Python 校验双清单、launcher/Python 摘要、源码指纹、进程与配置一致性；自有 ACLNN
符号不能回退稳定库或 CANN。recompute 的可变长元数据先转为紧凑 CPU int64 存储；
数据 tensor 仍必须处于同一 NPU。

独立开发 wheel 内嵌匹配 OPP，不允许替换为外部 `.run`。名称按 a2/a3/a5 分档，
本轮 x86_64 开发包使用 `py3-none-linux_x86_64`，未声明 manylinux 发布资格。
稳定 wheel 入口提前拒绝实验参数；实验验证使用独立 CI 入口，没有接入正式发布。

## 实际验证

| 检查 | 结果与证据 |
| --- | --- |
| RWKV8 单算子 | `.run` 编译/安装/双清单与来源审计通过；3 个 dtype 内核变体；`final-fused_recurrent_rwkv8.log` |
| 全量实验树 | RWKV8 + recompute 共 7 个内核变体通过；`final-all.log` |
| ACLNN gate | 单算子 1、全量 2 个实际调用点检查；无零覆盖通过；输入为所选实验 adapters 与真实 OPP 头 |
| 实验 launcher | 两种选择都通过源码/ELF/aoti 审计，GLIBCXX 上限实际为 3.4.21；`final-wheel-refresh.log` |
| wheel 加载 | 新进程只注册对应实验 namespace，NPU `is_initialized()` 为 false；两个目录的 `experimental-load-audit.json` |
| 实验契约 | 102 项通过，包含闭包、路径、安装拒绝、ABI 摘要、源码选择、wheel 混装和真实 C++ 符号来源测试 |
| Python CPU 契约 | 24 项通过，包含 loader 隔离、chunk_len、recompute 紧凑 host int-array；`final-wheel-refresh.log` |
| CPU golden | 3 项通过；`final-cpu-golden.log`。未对比 NPU 输出 |
| chunk_len C++ | `-Wall -Wextra -Werror -fsyntax-only` 通过 |
| 稳定构建/运行入口单元测试 | 70 项中 69 通过、1 项因尚无生成指纹跳过；实际 launcher 编译后该指纹项单独补跑通过；另 10 项 OPP ABI 测试通过 |
| CI 契约测试 | 282 项全部通过（包含上述 102 项实验契约，不重复累计）；`final-ci-tests.log` |
| 稳定 OFF 对照 | 修改版与原始 main 都构建 `chunk_fwd_o`：346 个 vendor 文件路径集合、ACLNN 合约一致，4 个内核二进制逐字节相同；`stable-comparison.json` |
| 稳定 launcher | 默认入口编译和 ELF/ABI 审计通过；`final-stable-launcher.log` |
| 已安装稳定包 | 原有 `fla_npu` 的 543 个非缓存文件前后 SHA256 相同；`stable-before.json` / `stable-after.json` |

重跑入口（初始化环境后）：

```bash
export FLA_NPU_DISABLE_PTH=1
CI_SOC=ascend950 CI_EXPERIMENTAL_OPS=fused_recurrent_rwkv8 CI_JOBS=4 bash ci/run_experimental_validation.sh
CI_SOC=ascend950 CI_EXPERIMENTAL_OPS=all CI_JOBS=4 bash ci/run_experimental_validation.sh
```

本轮先完成两种 OPP 编译，再补充 recompute 的 CPU 参数编组，随后对两种选择重新生成并
安装 wheel、重跑全部 24 项 Python 契约和 host-only 加载；最终 wheel 见各
`evidence/final-*/wheel/`，不能使用前序 `final-work/` 中的旧 wheel。

## 后续验收边界

A2 可用后须重新构建 A2 包并补 PTA/ATK、精度、确定性、异常输入、零快照、队列/stream
生命周期与内存检查。A5 NPU 运行/精度/性能仍未执行。本轮也没有覆盖稳定全量算子构建、
所有设备架构或正式发布矩阵；稳定回归是同环境定向构建和 CPU 工程契约检查。

## 制品与源码追溯

最终单算子 wheel：`flash_linear_attention_npu_experimental_a5-0.1.0.dev0+aa9412b536d7-py3-none-linux_x86_64.whl`。
最终全量 wheel：`flash_linear_attention_npu_experimental_a5-0.1.0.dev0+dbbe49a3219b-py3-none-linux_x86_64.whl`。
两者已回传本地，复核包中全部 Python/launcher 摘要和编译源码指纹，与当前工作区的
launcher 输入逐项匹配；没有稳定 `fla_npu/` 或 `.pth` 文件。

本地归档目录为工作区外层的 `.review-main-20261102/a5-evidence-20261103/`，
包含 `summary.json`、`artifact-sha256.json`、两组 `.run`/wheel、日志与双清单证据。
其中 `summary.json` 在回传后生成，记录当前源码匹配检查。

| 来源归档 | SHA256 | 说明 |
| --- | --- | --- |
| `opp-validated-source-20261103.tar.gz` | `4de72692df7fdb48c8ddc3cbf57a6c9adb8e62a4ceb3e7d576c114049959f76b` | OPP 编译时源码；文件摘要清单指纹 daa04587… |
| `implementation-20261103.tar.gz` | `d98b609fed9576b6b16aa73ecae639ed5990fadaadfe6ebfd2576c88ea7fbb7d` | 最终 launcher 与 Python 源码；文件摘要清单指纹 fcbd3656…；结果文档之后补写 |
