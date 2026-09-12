# ChunkBwdDqkwg ATK 测试说明

本目录归档 `chunk_bwd_dqkwg` 单算子的 ATK 资产，形式与
`fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/tests/ATK` 保持一致：
JSON/YAML 使用显式算子输入，executor 直接 import `scripts/chunk_bwd_dqkwg_cpu.py` 作为 CPU 标杆，不在运行时按源码路径查找。

## 文件

```text
atk_chunk_bwd_dqkwg.json
chunk_bwd_dqkwg.yaml
executor_chunk_bwd_dqkwg.py
gen_chunk_bwd_dqkwg.py
scripts/
`-- chunk_bwd_dqkwg_cpu.py
README.md
```

## 输入与约束

输入顺序为 `q, k, v, g, h, do, dh, dv, cu_seqlens, chunk_indices, w,
g_gamma, scale, chunk_size, is_mix, is_fix, use_exp2, transpose_state_layout,
qkv_type`。

YAML 已按算子限制收敛：
- `q/k=[B,HK,T,K]`，且二者 shape 必须一致；`v/do/dv=[B,HV,T,V]`。
- `g=[B,HV,T]`，需要为非正且沿 `T` 维单调递减，`g` 支持 `fp32/bf16/fp16`。
- `h/dh=[B,HV,num_chunks,K,V]`，`num_chunks=ceil(T/chunk_size)` 或由 `chunk_indices` 推导。
- `q/k/v/h/do/dh/dv` 支持 `bf16/fp16`，输入的 `B/T` 维必须对齐。
- `HV` 必须是 `HK` 的整数倍。
- `K=128`，`V=128/256`，`chunk_size=64/128`。
- `use_exp2=false`，`transpose_state_layout=false`。
- `w` 和 `g_gamma` 当前实现未启用，必须按可选输入传 `None`。
- 变长模式中 `cu_seqlens` 与 `chunk_indices` 必须同时提供，`chunk_indices` 为 `[seq_id, chunk_id]` 扁平化数组，且 `B=1`。

当前评审 JSON 保留 1 条评审验证 case：

| case 序号 | B | HK | HV | T | K | V | chunk_size | qkv_type | g dtype | scale | is_mix | is_fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1 | 1 | 1 | 128 | 128 | 64 | fp16 | fp32 | 0.088 | true | true |

`w` 和 `g_gamma` 在 YAML/JSON 中均按可选输入标记；executor 会按当前源 ATK
行为传入 `None`，保持与算子 optional 接口一致。

## Executor 行为

`executor_chunk_bwd_dqkwg.py` 注册 `executor_chunk_bwd_dqkwg`：
- NPU 路径调用 `fla_npu.ops.ascendc.npu_chunk_bwd_dqkwg`，若该路径不可用则回退到
  `torch.ops.npu.npu_chunk_bwd_dqkwg`。
- CPU golden 在 ATK benchmark task 中使用 `float64` 完成高精度计算，并转换为 ATK 混合容差比较支持的输出 dtype。

## 一键执行

在仓库根目录执行：

```bash
bash tests/atk/run_test_cpu.sh \
  -op=chunk_bwd_dqkwg \
  -npu_device_id=<physical_npu_device>
```

常用范围变量：

```bash
CASE_START=0 CASE_END=1 bash tests/atk/run_test_cpu.sh \
  -op=chunk_bwd_dqkwg \
  -npu_device_id=<physical_npu_device>
```

脚本会通过 ATK 覆盖混合容差精度、性能、确定性和 mssanitizer；精度检查以 CPU 高精度
结果作为唯一 golden、以 NPU 输出作为 DUT。所有范围均使用
`-s <start> -e <end>` 表示 JSON 顺序中的第几个 case。脚本不会导出 `PYTHONPATH`，需要在调用前准备好 ATK、CANN、OPP 和 Python 包路径。

## 重新生成

只在需要扩展 case 矩阵时使用 ATK 生成器。默认 `bf16/fp16` 两类 q dtype，`-dt 100 -en 0` 生成 200 条普通泛化用例：

```bash
bash tests/atk/run_test_cpu.sh \
  -op=chunk_bwd_dqkwg \
  -scope=gen_cases
```

等价 ATK 命令：

```bash
cd "$REPO_ROOT/tests/atk/chunk_bwd_dqkwg"
atk case -f ./chunk_bwd_dqkwg.yaml -p ./gen_chunk_bwd_dqkwg.py -dt 100 -en 0 -s 20260813
```

生成后需要人工确认 JSON 输入顺序、dtype、shape、`scale`、`chunk_size`、optional
输入语义和算子限制，再决定是否覆盖 `atk_chunk_bwd_dqkwg.json`。
