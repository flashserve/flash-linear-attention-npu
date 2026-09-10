# ChunkKdaFwdPrepare 测试归档

## 用例来源

`tests/op_cases/chunk_kda_fwd_prepare.json` 是 shape、dtype、layout、属性、SOC、调用通路和
随机种子的唯一用例来源。该清单覆盖四种 layout、BF16 输入、GVA、尾块、变长序列、
`use_exp2=false/true` 以及融合 norm/gate/beta 分支。

## 测试内容

| 路径 | 内容 |
| --- | --- |
| `accuracy/test_chunk_kda_fwd_prepare.py` | 读取 JSON，真实调用 `fla_npu.ops.ascendc.chunk_kda_fwd_prepare`，检查 13 个公开输出的顺序、shape、dtype 和有限值 |
| `routes/test_ctypes_aclnn_chunk_kda_fwd_prepare.py` | 通过稳定 Python ctypes 入口执行 aclnn 两段式调用 |
| `routes/test_aclnn_chunk_kda_fwd_prepare.cpp` | 编译期锁定 `GetWorkspaceSize` 和执行接口符号 |
| `routes/test_direct_chunk_kda_fwd_prepare.cpp` | 使用 host tiling 结构和 `<<<>>>` 发射 raw/fused 两个代表模板 |
| `common/` | JSON 筛选、输入构造和 13 输出合同检查 |
| `ut/op_host/test_contract.py` | 检查调度测试已接入 `ENABLE_TEST` CMake |
| `ut/op_kernel/test_contract.py` | 检查 direct launch case、tiling、workspace 和生产计算入口一致 |

完整数值标杆继续由
`tests/atk/chunk_kda_fwd_prepare/executor_chunk_kda_fwd_prepare.py` 提供；本目录的 route 测试
只验证公开调用链和 ABI，不替代 ATK 精度测试。

## 执行

不带 NPU 开关时可先检查 JSON 和公开 ABI 清单：

```bash
python -m pytest -q \
  tests/operators/chunk_kda_fwd_prepare/routes/test_ctypes_aclnn_chunk_kda_fwd_prepare.py \
  tests/operators/chunk_kda_fwd_prepare/ut
```

安装当前 wheel 和 custom OPP 后，执行稳定 ctypes/aclnn 与四种 layout 的设备烟测：

```bash
FLA_NPU_RUN_OPERATOR_TESTS=1 \
python -m pytest -q \
  tests/operators/chunk_kda_fwd_prepare/routes/test_ctypes_aclnn_chunk_kda_fwd_prepare.py \
  tests/operators/chunk_kda_fwd_prepare/accuracy/test_chunk_kda_fwd_prepare.py
```

可用 `FLA_NPU_SOC` 和 `FLA_NPU_CASE_IDS` 筛选平台及 case。A2、A3、A5 分别使用
`ascend910b`、`ascend910_93`、`ascend950`。

host 调度测试在 `ENABLE_TEST=ON` 时生成目标
`chunk_kda_fwd_prepare_tiling_processor_test`，可通过 CTest 或直接执行该目标验证。
