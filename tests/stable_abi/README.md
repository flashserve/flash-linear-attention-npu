# Stable-ABI 设备回归

这一目录放**需要 NPU** 的验证。它们不进默认 CI（上游 CI 是 NPU CI，跑的是它自己
拉取的脚本），用途是让评审能复现 PR 里声称的结论：多 stream 安全、输入张量
生命周期、mutation 契约、适配层加载失败时拒绝执行。

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
| `test_stable_fallback_warning.py` | 库缺失、外部错误库时必须抛出可定位错误，正常库必须选择 Stable；没有降级路径 | `python tests/stable_abi/test_stable_fallback_warning.py --lib <so>` |

所有公开 AscendC 算子只使用 Stable ABI。`stable_ctypes_fallbacks.py` 保留历史命令名，
现在扫描产品包，拒绝旧后端文件、导入和调用；标准库 `ctypes.CDLL` 的库加载仍可使用。
`tests/atk/common/aclnn_error_probe.py` 是仅限测试的原始 C ABI 负例探针：
只调用 `GetWorkspaceSize` 验证空指针、错误输出 buffer、错误码与错误消息，
绝不执行第二阶段或分配 workspace，不打入 wheel。正常 ATK 精度和性能用例经 Stable 调用。

迁移后的 Python 参数验证还包括：

```bash
python torch_custom/fla_npu/test/test_aclnn_ctypes_abi.py --npu
FLA_NPU_TEST_NPU=1 python torch_custom/fla_npu/test/test_pre_process_fwd_kernel_merged.py
FLA_NPU_RUN_CHUNK_KDA_BWD_INTRA=1 python -m pytest torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_intra.py
```

前两个文件保留历史名称以兼容 CI；分配、dtype、输出 shape 等 C++ 负责的契约在真实设备上检查，
离线 mock 仅验证 Python 参数转发和拒绝逻辑。未启用设备测试时显示 SKIP，不能作为设备验收通过。

## 离线门禁（不需要 NPU）

在 `torch_custom/fla_npu/tools/`：`stable_coverage.py`、`op_abi_parity.py`、
`stable_ctypes_fallbacks.py`、`op_abi_validate.py`（需要 OPP 头）、
`stable_abi_audit.py`；它们的自测在 `tests/test_stable_gates.py`，跑
`python tests/test_stable_gates.py` 即可。
