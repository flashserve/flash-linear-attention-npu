# ChunkKdaFwdPrepare 测试归档

## 用例来源

`tests/op_cases/chunk_kda_fwd_prepare.json` 是 shape、dtype、layout、属性、SOC、调用通路和
随机种子的唯一用例来源。该清单覆盖四种 layout、BF16 输入、GVA、尾块、变长序列、
`use_exp2=false/true` 以及融合 norm/gate/beta 分支。

## 测试内容

| 路径 | 内容 |
| --- | --- |
| `routes/test_aclnn_chunk_kda_fwd_prepare.cpp` | 编译期锁定 `GetWorkspaceSize` 和执行接口符号 |
| `ut/op_host/test_contract.py` | 检查调度测试已接入 `ENABLE_TEST` CMake |
| `ut/op_kernel/test_contract.py` | 检查 A2/A3/A5 kernel 的模板输出、同步、静态内存与伪代码一致性合同 |
| `ut/test_atk_generation.py` | 检查 21 个公开输入、200 条精度、10 条性能和 432 个 TilingKey 用例合同 |

完整数值标杆由
`tests/atk/chunk_kda_fwd_prepare/executor_chunk_kda_fwd_prepare.py` 提供；本目录保留算子私有的
C++ 符号检查、host/kernel 合同检查，不替代 ATK 精度测试。稳定 `fla_npu` Python/ctypes
入口和设备烟测位于 companion PR，避免在本 PR 修改公共文件。

## 执行

不带 NPU 开关时可先检查算子私有合同：

```bash
python -m pytest -q tests/operators/chunk_kda_fwd_prepare/ut
```

host 调度测试在 `ENABLE_TEST=ON` 时生成目标
`chunk_kda_fwd_prepare_tiling_processor_test`，可通过 CTest 或直接执行该目标验证。
