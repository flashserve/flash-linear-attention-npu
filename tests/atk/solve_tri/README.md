# SolveTri 本机混合容差 ATK 工程

本目录验证公开接口 `fla_npu.ops.ascendc.solve_tri`，**只在本机跑**，不连远程 GPU。

精度标准为 `mixed_tolerance_bm`（参考 `chunk_gated_delta_rule_fwd_prepare`）：

- **dut** = 本机 NPU `solve_tri`
- **golden** = 本机 CPU FP64 `linalg.inv`（`--bm_device cpu`）

`run_test_cpu.sh` 默认要求 ATK ≥ 26.8.8（该版本原生提供 `mixed_tolerance_bm`）。本机若仍是 26.7.8，可设 `REQUIRED_ATK_VERSION=26.7.8`；executor 会把缺失的 `mixed_tolerance_bm` 别名到同角色的 `single_bm`，不必改 JSON。全量 200 条建议加 `--single_process`，避免 Celery 对比卡住。

## 输入约束

- `layout` 支持 `bsnd/bnsd/tnd/ntd`。
- `x/out` 仅 `FLOAT16/BFLOAT16`；`chunk_size` 支持 `16/32/64/128`。
- `bsnd` / `bnsd` 一定长；`tnd` / `ntd` 含 1-seq 定长和 2+ seq 变长。
- 变长用 `seqlens` 固定，executor 转成 `cu_seqlens` / `chunk_indices`。
- CPU golden 与 `fwd_prepare` 里 solve-tri 一段相同：块内 `(I + L)^{-1}`，尾块 padding 列清零。

## 默认用例

`atk_solve_tri.json` 为 **200** 条中小 shape（`scripts/generate_solve_tri_case.py` 的 `iter_profiles`），标准为 `mixed_tolerance_bm`。

| 维度 | 条数 |
|---|---|
| chunk 16 / 32 / 64 / 128 | 各 50 |
| bf16 / fp16 | 各 100 |
| bsnd / bnsd | 各 34 |
| tnd / ntd | 各 66 |
| align / unalign | 98 / 102 |
| fixed / varlen | 132 / 68 |

`atk_solve_tri_perf.json`：6 条对齐定长、chunk 64/128。  
`atk_solve_tri_mss.json`：6 条，覆盖 4 种 chunk、密/打包、尾块和 3-seq packed。

重生成：

```bash
python3 tests/atk/solve_tri/gen_solve_tri.py
```

## 执行

走统一入口，**不要**改 `run_test_cpu.sh`。精度建议关 GM init。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

ATK_GM_INIT_MODE=off \
bash tests/atk/run_test_cpu.sh -op=solve_tri -npu_device_id=6 -soc=ascend950 -scope=accuracy

ATK_GM_INIT_MODE=off ACCURACY_START=0 ACCURACY_END=1 REQUIRED_ATK_VERSION=26.7.8 \
bash tests/atk/run_test_cpu.sh -op=solve_tri -npu_device_id=6 -soc=ascend950 -scope=accuracy

bash tests/atk/run_test_cpu.sh -op=solve_tri -npu_device_id=6 -soc=ascend950 -scope=performance
bash tests/atk/run_test_cpu.sh -op=solve_tri -npu_device_id=6 -soc=ascend950 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=solve_tri -npu_device_id=6 -soc=ascend950 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=solve_tri -scope=gen_cases
```
