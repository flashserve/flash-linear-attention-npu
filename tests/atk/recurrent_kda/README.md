# RecurrentKda ATK 工程

本目录提供 `recurrent_kda` 的 ATK 单算子工程，包含执行器、200-case 生成器、YAML 和根部用例 JSON。

## 算子约束

- `layout` 支持 `BSND/TND`；`BSND` 下 `q/k=[B,T,H,K]`，`v/out=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`；`TND` 下去掉物理 batch 维。
- `q/k/v/out` 仅支持 BF16；`gate/beta` 各支持 FP16、BF16、FP32；state 支持 BF16、FP32。
- `K=128`，`V=128/256`，`H/HV <= 256`，且 `HV % H == 0`。
- dense 与 varlen 的每条有效序列长度均不超过 8。
- `state_v_first=true` 时 state 为 `[state_capacity,HV,V,K]`；否则为 `[state_capacity,HV,K,V]`。
- 未传 `ssm_state_indices` 时 `state_capacity=seq_num`；传入后支持 packed `[T]` 和 speculative `[seq_num,max_step]` 两种索引。
- `cu_seqlens`、`ssm_state_indices`、`num_accepted_tokens` 支持 INT32/INT64。
- `use_gate_in_kernel=true` 时必须传 FP32 `A_log=[HV]`，`dt_bias` 支持 `[HV*K]` 或 `[HV,K]`；`safe_gate=true` 时 `lower_bound` 必须位于 `[-5,0)`。
- initial/final state 的最后二维矩阵必须稠密；泛化用例同时覆盖连续 state 和仅在 slot/head 外层带间隔的非连续 state。

## tilingKey 与用例覆盖

当前 host tiling 的 `DoLibApiTiling()` 无条件设置 `tilingKey=0`，kernel 入口使用默认 tiling 数据且没有其他 `TILING_KEY_IS` 分派，因此当前完整 tilingKey 集合为 `{0}`。

`gen_recurrent_kda.py` 固定生成 200 条用例，并在 import 时校验：

- tilingKey 0：200 条；
- 小 shape 190 条，大 shape 10 条；
- BSND/TND；
- gate/beta 的 3×3 dtype 组合；
- FP32/BF16 state、V-first/K-first state；
- V=128/256 与多组 H/HV 比例；
- 无 cu、uniform、varlen、padding tail、零长度序列；
- 无索引、packed 1D、speculative 2D state indices，以及 accepted tokens；
- q/k L2 normalize、raw gate、两种 dt_bias shape、safe gate、beta sigmoid、allow negative eigenvalue；
- `output_final_state` 与 `inplace_final_state` 的四种组合；
- 连续/非连续输入与 state，以及 wrapper 自动创建 initial state 的非原位场景。

生成器将第二路 marker 也重写为合法 BF16 q/k/v；marker 的双 dtype 仅用于配合统一脚本的 `-dt 100` 产生恰好 200 条 case。

## 标杆语义

CPU reference、输入构造、NPU 调用和 ATK `FunctionApi` 均在 `executor_recurrent_kda.py` 中实现。CPU 同精度节点按原始输入 dtype 量化后使用 FP32 递归计算；CPU benchmark 节点使用相同量化输入执行 FP64 递归计算。

对于 `cu_seqlens[-1] < token_capacity` 的 padding tail，文档规定输出无语义；executor 仅在 ATK 比较前清零该无效区域，不改变有效 token 或 state 的比较结果。

对于带 `ssm_state_indices` 的 state pool，kernel 只写有效 token 实际引用的 final-state slot；非原地输出中未被引用的 slot 无输出语义。executor 在 ATK 比较前仅清零这些未引用 slot，所有被引用 slot 仍参与完整精度与 NaN/Inf 检查。

## 生成与执行

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -scope=gen_cases
cp tests/atk/recurrent_kda/result/recurrent_kda/json/all_recurrent_kda.json \
   tests/atk/recurrent_kda/atk_recurrent_kda.json

bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=accuracy
```

其他统一入口：

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=mssanitizer
```

`atk_recurrent_kda_perf.json` 和 `atk_recurrent_kda_mss.json` 保留为性能、Sanitizer 的单 case 专用入口，不随 200 条精度泛化 JSON 扩展。
