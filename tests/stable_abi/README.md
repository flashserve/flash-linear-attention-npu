# Stable-ABI 设备回归

这一目录放**需要 NPU** 的验证。它们不进默认 CI（上游 CI 是 NPU CI，跑的是它自己
拉取的脚本），用途是让评审能复现 PR 里声称的结论：多 stream 安全、输入张量
生命周期、mutation 契约、适配层加载失败时的告警。

跑之前需要三样东西：

1. 一个装好 OPP 的 `fla_npu` 包（wheel 安装态，或源码树加
   `ASCEND_CUSTOM_OPP_PATH` 指向 `fla_npu/opp/vendors/fla_npu_transformer`）；
2. 编译好的 launcher：`python csrc/build_stable.py --out <path>`，
   然后用 `FLA_NPU_STABLE_LIB=<path>` 指过去（wheel 安装态会自动用包里那份）；
3. 一张可用的卡：`ASCEND_RT_VISIBLE_DEVICES=<n>`。

## 脚本

公共前置：

```bash
cd <repo>
source <cann>/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=<n>
export FLA_NPU_STABLE_LIB=<libfla_npu_stable.so>
export PYTHONPATH=<env>
```

| 文件 | 覆盖什么 | 运行方式 |
| --- | --- | --- |
| `test_stable_stream_interleaving.py` | 多线程 × 各自 stream × recurrent+conv1d 交替；每次调用必须读当次 stream（带自检负例） | `python tests/stable_abi/test_stable_stream_interleaving.py --threads 8 --rounds 3` |
| `test_input_lifetime.py` | boxed kernel 消费栈引用之后，调用方手里的输入张量必须完好、可继续读写；每轮新建输入不涨内存 | `python tests/stable_abi/test_input_lifetime.py` |
| `regression_mutation_contract.py` | 原地更新算子的 version/grad 契约；同时提供 `gdr_inputs` / `kda_inputs` 供上一条复用 | `python tests/stable_abi/regression_mutation_contract.py` |
| `test_stable_fallback_warning.py` | 加载不到适配层时必须告警：库缺失、库不是本适配层产物、以及正常加载三种情形，并检查降级原因写成"适配层不可用"而不是"算子没带" | `python tests/stable_abi/test_stable_fallback_warning.py --lib <so>` |

> **ctypes 对照已删除**：逐算子 ctypes↔适配层逐位 parity、场景基线
> （`stable_scenarios.json`）、客户可见面切换（`customer_switch_compat.py`）和
> host A/B（`bench_stable_host.py`）都已完成使命。原有算子与 ctypes 的一致性在
> 合并前验证过，此后新增算子**不再要求写 ctypes 适配**。`_aclnn_ctypes.py` 本身
> 保留为回退后端，`stable_ctypes_fallbacks.py` 继续禁止适配器回流到它。

## 离线门禁（不需要 NPU）

在 `torch_custom/fla_npu/tools/`：`stable_coverage.py`、`op_abi_parity.py`、
`stable_ctypes_fallbacks.py`、`op_abi_validate.py`（需要 OPP 头）、
`stable_abi_audit.py`；它们的自测在 `tests/test_stable_gates.py`，跑
`python tests/test_stable_gates.py` 即可。
