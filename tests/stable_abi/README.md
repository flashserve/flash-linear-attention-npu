# Stable-ABI 设备回归

这一目录放**需要 NPU** 的验证。它们不进默认 CI（上游 CI 是 NPU CI，跑的是它自己
拉取的脚本），用途是让评审能复现 PR 里声称的结论：逐位一致、合法域全覆盖、
多 stream 安全、客户可见面不变。

跑之前需要三样东西：

1. 一个装好 OPP 的 `fla_npu` 包（wheel 安装态，或源码树加
   `ASCEND_CUSTOM_OPP_PATH` 指向 `fla_npu/opp/vendors/fla_npu_transformer`）；
2. 编译好的 launcher：`python csrc/build_stable.py --out <path>`，
   然后用 `FLA_NPU_STABLE_LIB=<path>` 指过去（wheel 安装态会自动用包里那份）；
3. 一张可用的卡：`ASCEND_RT_VISIBLE_DEVICES=<n>`。

## 主驱动

```bash
cd <repo>
source <cann>/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=4 \
FLA_NPU_STABLE_LIB=<libfla_npu_stable.so> \
PYTHONPATH=<env> ASCEND_CUSTOM_OPP_PATH=<opp>:<opp>/op_api/lib \
python tests/stable_abi/regression_stable_full.py
```

- 每个场景都做 ctypes↔launcher 的逐位对比，并和 `stable_scenarios.json` 里本机
  对应那份基线比对：少了场景会报 `lost`，多了会报 `scenario added`。
- `--group hot|recurrent|conv1d|kda|chunk|smoke|a5` 只跑一组（编辑循环用，几十秒）；
  `--list-groups` 列出分组。
- `FLA_NPU_BASELINE_WRITE=1` 重录本机基线（**只改本机那一份**，另一台不动）。

`Ascend950` 专属的四个场景在 `regression_950_ops.py`，由 `--group a5` 选中；
在 910B 上跑全量时会打印对应的 SKIP 及原因。

## 其余

| 文件 | 覆盖什么 | 运行方式 |
| --- | --- | --- |
| `test_stable_stream_interleaving.py` | 多线程 × 各自 stream × recurrent+conv1d 交替；每此调用必须读当次 stream（带自检负例） | `python tests/stable_abi/test_stable_stream_interleaving.py --threads 8 --rounds 3` |
| `regression_mutation_contract.py` | 原地更新算子的 version/grad 契约与数值 | `python tests/stable_abi/regression_mutation_contract.py` |
| `customer_switch_compat.py` | 客户可见面：31 个算子的公开签名 + 若干真实调用的结果与原地契约，`FLA_NPU_STABLE_ABI=ctypes` 与默认后端对照 | `python tests/stable_abi/customer_switch_compat.py` |
| `test_stable_fallback_warning.py` | 加载不到 launcher 时必须告警：库缺失、库不是 launcher、以及正常加载三种情形，并检查降级原因写成"launcher 不可用"而不是"算子没带" | `python tests/stable_abi/test_stable_fallback_warning.py --lib <so>` |
| `bench_stable_host.py` | 逐算子 host A/B（对比 ctypes 与 launcher） | `python tests/stable_abi/bench_stable_host.py [--rounds 5]` |

`regression_ops.py` 是场景库，不单独运行；`regression_stable_full.py` 从这里取场景。
基线文件 `stable_scenarios.json` 有 `Ascend910B3` 与 `Ascend950PR_9579` 两份，改动
场景集时必须两台都重录。

## 离线门禁（不需要 NPU）

在 `torch_custom/fla_npu/tools/`：`stable_coverage.py`、`op_abi_parity.py`、
`op_api_parity.py`、`stable_ctypes_fallbacks.py`、`op_abi_validate.py`（需要 OPP 头）、
`stable_abi_audit.py`；它们的自测在 `tests/test_stable_gates.py`，跑
`python tests/test_stable_gates.py` 即可。
