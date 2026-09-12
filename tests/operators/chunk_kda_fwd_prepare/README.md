# ChunkKdaFwdPrepare 测试归档

## 用例来源

`tests/op_cases/chunk_kda_fwd_prepare.json` 是 shape、dtype、layout、属性、SOC、调用通路和
随机种子的唯一用例来源。该清单覆盖四种 layout、BF16 输入、GVA、尾块、变长序列、
`use_exp2=false/true` 以及融合 norm/gate/beta 分支。

## 测试内容

| 路径 | 内容 |
| --- | --- |
| `smoke/test_chunk_kda_fwd_prepare.py` | 读取 JSON，真实调用 `fla_npu.ops.ascendc.chunk_kda_fwd_prepare`，检查三种反向模式下固定 13 槽位的 `None`/tensor 分布，以及实际 tensor 的 shape、dtype 和有限值；该项是设备烟测，不作为数值精度结论 |
| `negative/test_invalid_contract.py` | 从统一 JSON 读取负向 case，验证公开参数错误会在 aclnn 启动前被稳定入口拒绝 |
| `routes/test_ctypes_aclnn_chunk_kda_fwd_prepare.py` | 通过稳定 Python ctypes 入口执行 aclnn 两段式调用 |
| `routes/test_aclnn_negative_status.py` | 绕过 Python 预校验，直接验证 C++ `GetWorkspaceSize` 的 `ACLNN_ERR_PARAM_NULLPTR`/`ACLNN_ERR_PARAM_INVALID` 返回码 |
| `routes/test_aclnn_chunk_kda_fwd_prepare.cpp` | 编译期锁定 `GetWorkspaceSize` 和执行接口符号 |
| `routes/test_direct_chunk_kda_fwd_prepare.cpp` | 锁定 host tiling、workspace、`<<<>>>` 和 raw/fused 模板实例化的源码合同；当前没有独立的编译或设备执行目标 |
| `common/` | JSON 筛选、输入构造和 13 输出合同检查 |
| `ut/op_host/test_contract.py` | 检查调度测试已接入 `ENABLE_TEST` CMake |
| `ut/op_kernel/test_contract.py` | 检查 direct launch 源码合同与清单中的代表 case 一致 |
| `ut/test_atk_generation.py` | 检查 ATK 生成声明只来自统一 JSON，且三份冻结 JSON 没有过期 |

完整数值标杆继续由
`tests/atk/chunk_kda_fwd_prepare/executor_chunk_kda_fwd_prepare.py` 提供；本目录的 route 测试
只验证公开调用链和 ABI，不替代 ATK 精度测试。

## 执行

不带 NPU 开关时可先检查 JSON 和公开 ABI 清单：

```bash
python -m pytest -q \
  tests/operators/chunk_kda_fwd_prepare/routes/test_ctypes_aclnn_chunk_kda_fwd_prepare.py \
  tests/operators/chunk_kda_fwd_prepare/negative/test_invalid_contract.py \
  tests/operators/chunk_kda_fwd_prepare/ut
```

安装当前 wheel 和 custom OPP 后，执行稳定 ctypes/aclnn 与四种 layout 的设备烟测：

```bash
FLA_NPU_RUN_OPERATOR_TESTS=1 \
python -m pytest -q \
  tests/operators/chunk_kda_fwd_prepare/routes/test_ctypes_aclnn_chunk_kda_fwd_prepare.py \
  tests/operators/chunk_kda_fwd_prepare/routes/test_aclnn_negative_status.py \
  tests/operators/chunk_kda_fwd_prepare/smoke/test_chunk_kda_fwd_prepare.py
```

可用 `FLA_NPU_SOC` 和 `FLA_NPU_CASE_IDS` 筛选平台及 case。A2、A3、A5 分别使用
`ascend910b`、`ascend910_93`、`ascend950`。

host 调度测试在 `ENABLE_TEST=ON` 时生成目标
`chunk_kda_fwd_prepare_tiling_processor_test`，可通过 CTest 或直接执行该目标验证。
