# ChunkFwdO ATK 工程

本目录提供 `chunk_fwd_o` 的 ATK 单算子工程，包含 `executor_chunk_fwd_o.py`、`gen_chunk_fwd_o.py`、`chunk_fwd_o.yaml`、`atk_chunk_fwd_o.json`。

## 输入约束

- `q/k` 必须为 `[B,HK,T,K]`，且二者形状完全一致。
- `v/o` 必须为 `[B,HV,T,V]`，`g` 必须为 `[B,HV,T]`。
- `h` 必须为 `[B,HV,num_chunks,K,V]`，其中 `num_chunks` 由 `T/chunk_size` 或变长 `chunk_indices` 推导。
- `q/k` 与 `v/g/o` 的 `B`、`T` 必须一致；`HV % HK == 0`。
- `K` 固定为 `128`，`V` 支持 `128/256`，`chunk_size` 仅支持 `64/128`；`scale` 建议按 `1 / sqrt(K)` 设置。
- `q/k/v/h/o` 支持 `BFLOAT16/FLOAT16`；`g` 支持 `FLOAT/FLOAT16/BFLOAT16`。
- 变长模式下 `cu_seqlens` 与 `chunk_indices` 必须同时提供，`chunk_indices` 为 `[seq_id, chunk_id]` 扁平化数组，且 `B=1`。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

torch_custom/fla_npu/test/test_fwd_o.py; fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_chunk_fwd_o.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 200 条评审用例矩阵

`gen_chunk_fwd_o.py` 是 200 条评审 profile 与 JSON 校验的唯一来源；`atk_chunk_fwd_o.json` 是其物化后纳入版本控制的执行用例。不要用手工缩减的 JSON 替换它。

| 维度 | 覆盖 |
| --- | --- |
| 精度 | BF16 100 条，FP16 100 条；`g` 覆盖 FP32 与输入同精度 |
| 模式 | 定长 100 条；变长 100 条（`B=1`，覆盖单序列、多序列、chunk 边界及尾块） |
| 形状 | `T=1` 到 `384`；`B=1/2`；MHA 和 GVA (`HK/HV=1/1,1/2,2/2,2/4,4/4`) |
| 资源分支 | `chunk_size=64/128` 各 100 条；`V=128/256`；标准 scale 与半 scale |

CPU 双标杆会使用同一份量化输入：高精度分支以 FP64 计算真值，同精度分支保留 BF16/FP16 输出量化。标杆实现严格对应算子语义：`((q @ k^T) * exp(g_i-g_j) * tril) @ v + (q * exp(g)) @ h`，随后整体乘 `scale`。每个变长 case 同时生成匹配的 `cu_seqlens` 与扁平化 `[seq_id, chunk_id]`。

修改矩阵后必须重新生成并检查执行 JSON：

```bash
python3 tests/atk/chunk_fwd_o/gen_chunk_fwd_o.py
python3 tests/atk/chunk_fwd_o/gen_chunk_fwd_o.py --check
```

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=<device>
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=<device> -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=<device> -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=<device> -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -scope=gen_cases
```

默认 `scope=all` 顺序执行 200 条精度/NaN 检测、确定性和内存检测（另含性能阶段）。在目标 SOC 上必须看到三个阶段均为 `success 200, failed 0`；精度阶段还必须为 `acc_pass_result: Pass`，才可将该 SOC 标记为 **OK**。`gen_cases` 默认传入 `-dt 100 -en 0`，会重建相同的 200 条矩阵；提交前仍须用上面的 builder 检查已评审 JSON。
