# MergeFwdBwdKernel ATK 工程

本目录提供 `merge_fwd_bwd_kernel` 的 ATK 单算子工程：`executor_merge_fwd_bwd_kernel.py`、
`gen_merge_fwd_bwd_kernel.py`、`merge_fwd_bwd_kernel.yaml`、三份用例 JSON 和 `scripts/`。

公开入口是 `fla_npu.ops.ascendc.npu_merge_fwd_bwd_kernel`。接口和 shape 以算子
`fla/ops/ascendc/cp/merge_fwd_bwd_kernel/docs/api.md` 为准。

## 算子做什么

一次调用把若干 rank 上的 `(M, He)` 收成一块状态 `h`。`h` 原地写回，计算不读 `h` 的旧值。

```text
h <- He_0
h <- M_i @ h + He_i    i = 1 .. N-1
```

`ag_hm` 是 `[S, HV, 128, 256]`：每个 head 的前 128 列是 `He`，后 128 列是 `M`。`h` 是 `[HV, 128, 128]`。

源 rank：

```text
forward:  src(i) = rank - N + i          # rank-N .. rank-1
backward: src(i) = rank + N - i          # rank+N .. rank+1
S = 1 且 N = 1 且 rank = 0: src = 0
```

`N = pre_or_post_num_ranks`。`N = 1` 没有矩阵乘，结果就是 `He_0`。`state_v_first=true` 时，内部仍按 `[K, V]` 累加，写出前转成 `[V, K]`。

## 输入约束

- `ag_hm` / `h` 的 dtype 是 BF16 或 FP32，两者相同。`K = V = 128`。
- `S ∈ [1, 1024]`，`HV ∈ [1, 256]`，`N ∈ [1, S]`，`rank ∈ [0, S)`。
- 除 `S = N = 1, rank = 0` 外：前向要求 `rank >= N`，反向要求 `rank + N < S`。
- `h` 必须连续。`ag_hm` 可以不连续，host 会先收成连续再读。
- 存储格式是 ND 一类公开格式。
- SoC：`ascend910b`、`ascend910_93`、`ascend950`。

## TilingKey

Host 只按 dtype 选择模板参数 `D_T`，再交给 `GET_TPL_TILING_KEY`。可达集合是：

| TilingKey | dtype | 常量 |
| --- | --- | --- |
| 10 | BF16 | `MERGE_FWD_BWD_KERNEL_TPL_BF16` |
| 30 | FP32 | `MERGE_FWD_BWD_KERNEL_TPL_FP32` |

`forward`、`N`、`state_v_first` 和 head 切分写在 tiling data 里，不另开 TilingKey。`N = 1` 不起 Cube、不写 scratch；`N = 2` 做一步 GEMM；`N >= 3` 会来回使用两块 M bank。

## 标杆与值域

CPU 标杆在 `scripts/merge_fwd_bwd_kernel_cpu.py`。Executor 忽略 ATK 对 `h` / `ag_hm` 的默认物化，按 `case_spec` 自己构造输入，避免 YAML 值域把长链 GEMM 撑爆。

| 角色 | 计算 |
| --- | --- |
| CPU 同精度 | 输入先量化到用例 dtype，再按 FP32 做链。BF16 写出用 round-to-nearest-even，对应 kernel 的 `CAST_RINT` |
| CPU 高精度 | 同一份量化输入升到 FP64 再做链 |
| NPU DUT | `npu_merge_fwd_bwd_kernel`，结果写回 `h` |

输入：`He ~ N(0, 0.05)`，`M ~ N(0, 0.02)`，先在 FP32 采样再量化。`h` 填常数 `3`，标杆不使用这个旧值。`128 × 0.02 < 1`，长链状态幅度保持在 O(0.2)，不会按 `128^{N-1}` 爆炸。

精度标准用 `cv_fused_double_benchmark`：`max_re_ratio = 5`，`avg_re_ratio = 1.5`，`root_mean_squared_ratio = 1.5`。Cube 把 K 切成两段 64 累加，和 `torch.matmul` 顺序不同；缩放后的值域使正确 FP32 链通过混合容差，代表性错误（全零、漏最后一步 `He`、`M` 错 scale、反方向合并）会被混合容差抓住。旧 YAML `[-5, 5]` 在 `N=7` FP32 上，即使完美 `torch.matmul(FP32)` 也会对 FP64 超差，因此不再作为精度基线。

值域校准命令：`python scripts/check_shapes.py`（含解析检查和值域对/错分离）。2026-11-26 在 CPU 上对 `N=7` FP32 前向链确认：缩放后 mixed_fail=0、幅度约 0.25；`[-5, 5]` 长链 mixed_fail=1819、幅度约 `1.4e10`。全零、漏最后一步 `He`、`M` 错 scale、反方向合并均被混合容差抓住。

2026-11-27 在 A2（910B3）+ ATK 26.11.11 + CANN 9.2.1 上跑完全量 84 条
`scripts/run_accuracy.sh`（约 50 min）。ATK 汇总表 `cpu_cpu_lowp` 84/84 Pass；
`npu_npu_dut` 精度列仍是 `-`。DUT 以 `--save_data output` 的三路 `output_0.pt` 为准：
`python scripts/compare_saved_outputs.py <output/>` 得到 84/84 通过，幅度约 0.04。
N=1 拷贝与 FP64 一致；带 Cube 的 FP32 链对 FP64 的误差不大于 `torch.matmul` 同精度链。
结果目录：`atk_output/double_benchmark/20261127_111615`。

守护复测包 `ATK复测-20261125` 里那 13 条 FP32 `N≥4` 失败用例仍是 `[-5, 5]` + `mixed_tolerance_bm`，不在本目录已固化值域内，不能当作 kernel 公式缺陷。

## 用例

`python gen_merge_fwd_bwd_kernel.py` 会先跑标杆自检，再写出三份 JSON。不要用 `atk case -dt 100` 覆盖已经冻结的 JSON。

| 文件 | 条数 | 来源 |
| --- | ---: | --- |
| `atk_merge_fwd_bwd_kernel.json` | 84 | 28 条逻辑分支 × 3 个固定种子，种子从 `20260817` 起连续编号 |
| `atk_merge_fwd_bwd_kernel_perf.json` | 8 | CP 模型档。设计测量形态是 `N = cp - 1`、`rank = cp - 1`、FP32。没有单独达标线 |
| `atk_merge_fwd_bwd_kernel_mss.json` | 6 | 两个 TilingKey，以及 N=1、一步 GEMM、ping-pong、HV=256、S=1024 |

YAML 的 marker 保留 BF16 / FP16 两路，生成器把 FP16 marker 改回 BF16。算子自身的 dtype 只有 BF16 和 FP32。

### 精度用例

同一逻辑分支的 3 个 case id 只改种子。

| case id | 名称 | dtype | S | HV | N | rank | forward | v_first | 不连续 ag_hm | key | 覆盖 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| 0-2 | MFB-01 | bf16 | 1 | 1 | 1 | 0 | true | false | false | 10 | N=1 单世界，只拷贝 He |
| 3-5 | MFB-02 | fp32 | 1 | 1 | 1 | 0 | true | true | false | 30 | N=1 转置写出 |
| 6-8 | MFB-03 | bf16 | 4 | 2 | 1 | 1 | true | false | false | 10 | forward 且 rank=N |
| 9-11 | MFB-04 | fp32 | 4 | 3 | 1 | 0 | false | true | false | 30 | backward N=1，奇数 HV |
| 12-14 | MFB-05 | bf16 | 8 | 4 | 1 | 7 | true | false | false | 10 | 最后一个 rank，只拷贝 |
| 15-17 | MFB-06 | bf16 | 4 | 4 | 2 | 2 | true | false | false | 10 | 一步 GEMM |
| 18-20 | MFB-07 | fp32 | 4 | 5 | 2 | 1 | false | true | false | 30 | 一步 GEMM + 转置 + 奇数 HV |
| 21-23 | MFB-08 | bf16 | 8 | 8 | 3 | 5 | true | false | false | 10 | ping-pong 两块 M bank |
| 24-26 | MFB-09 | fp32 | 8 | 7 | 3 | 2 | false | true | false | 30 | ping-pong + 反向 + 转置 |
| 27-29 | MFB-10 | bf16 | 16 | 16 | 8 | 15 | true | false | false | 10 | 较长前向链 |
| 30-32 | MFB-11 | fp32 | 16 | 16 | 8 | 0 | false | true | false | 30 | 较长反向链 |
| 33-35 | MFB-12 | bf16 | 8 | 1 | 4 | 4 | true | false | false | 10 | 单 head，另一 AIV 不拥有 head |
| 36-38 | MFB-13 | fp32 | 8 | 1 | 4 | 0 | false | true | false | 30 | 单 head 反向并转置 |
| 39-41 | MFB-14 | fp32 | 8 | 32 | 7 | 7 | true | false | false | 30 | 设计测量档 cp=8 |
| 42-44 | MFB-15 | bf16 | 8 | 32 | 7 | 0 | false | true | false | 10 | cp=8 最坏反向 |
| 45-47 | MFB-16 | fp32 | 8 | 32 | 7 | 7 | true | true | false | 30 | 设计测量档 + 转置 |
| 48-50 | MFB-17 | fp32 | 1024 | 1 | 1 | 1023 | true | false | false | 30 | S 上界，只拷贝 |
| 51-53 | MFB-18 | bf16 | 1024 | 1 | 1 | 0 | false | true | false | 10 | S 上界反向 N=1 |
| 54-56 | MFB-19 | fp32 | 1024 | 2 | 2 | 1023 | true | false | false | 30 | S 上界一步 GEMM |
| 57-59 | MFB-20 | bf16 | 4 | 256 | 2 | 3 | true | false | false | 10 | HV 上界 |
| 60-62 | MFB-21 | fp32 | 6 | 256 | 3 | 0 | false | true | false | 30 | HV 上界 + ping-pong + 转置 |
| 63-65 | MFB-22 | bf16 | 4 | 4 | 3 | 3 | true | false | true | 10 | ag_hm 最后一维不连续 |
| 66-68 | MFB-23 | fp32 | 4 | 6 | 3 | 0 | false | true | true | 30 | 不连续 ag_hm + 反向转置 |
| 69-71 | MFB-24 | fp32 | 8 | 2 | 7 | 7 | true | false | false | 30 | N=S-1 前向 |
| 72-74 | MFB-25 | bf16 | 8 | 2 | 7 | 0 | false | true | false | 10 | N=S-1 反向 |
| 75-77 | MFB-26 | bf16 | 4 | 2 | 2 | 2 | true | false | false | 10 | 两个 AIV 各拥有一个 head |
| 78-80 | MFB-27 | fp32 | 2 | 32 | 1 | 1 | true | false | false | 30 | cp=2，N=1 前向 |
| 81-83 | MFB-28 | bf16 | 2 | 32 | 1 | 0 | false | true | false | 10 | cp=2，N=1 反向 |

### 性能用例

| case id | 名称 | 形状 |
| --- | --- | --- |
| 0 | PERF-cp8-hv32-fp32-fwd | S=8, HV=32, N=7, rank=7, FP32, forward |
| 1 | PERF-cp8-hv32-bf16-fwd | 同形状，BF16 |
| 2 | PERF-cp8-hv32-fp32-bwd | S=8, HV=32, N=7, rank=0, FP32, backward |
| 3 | PERF-cp8-hv32-fp32-vfirst | 设计测量档，`state_v_first=true` |
| 4 | PERF-cp16-hv32-fp32-fwd | S=16, HV=32, N=15, rank=15, FP32 |
| 5 | PERF-cp64-hv32-fp32-fwd | S=64, HV=32, N=63, rank=63, FP32 |
| 6 | PERF-cp8-hv64-fp32-fwd | S=8, HV=64, N=7, rank=7, FP32 |
| 7 | PERF-cp8-hv8-fp32-fwd | S=8, HV=8, N=7, rank=7, FP32 |

### 内存 / 确定性用例

| case id | TilingKey | 名称 | 路径 |
| --- | ---: | --- | --- |
| 0 | 10 | MSS-bf16-n1 | N=1，不写 scratch |
| 1 | 30 | MSS-fp32-n1-vfirst | N=1，转置写出 |
| 2 | 10 | MSS-bf16-n2 | 一步 GEMM，两个 AIV 都是 owner |
| 3 | 30 | MSS-fp32-n3-odd-vfirst | ping-pong、奇数 HV、转置 |
| 4 | 10 | MSS-bf16-hv256-n3 | HV=256，每核多个 head |
| 5 | 30 | MSS-fp32-s1024-n2 | S=1024，单 head GEMM |

## 执行

```bash
cd tests/atk/merge_fwd_bwd_kernel
python gen_merge_fwd_bwd_kernel.py --summary
python scripts/check_shapes.py
python scripts/merge_fwd_bwd_kernel_cpu.py

bash tests/atk/merge_fwd_bwd_kernel/scripts/run_accuracy.sh 0
bash tests/atk/run_test_cpu.sh -op=merge_fwd_bwd_kernel -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=merge_fwd_bwd_kernel -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=merge_fwd_bwd_kernel -npu_device_id=0 -scope=mssanitizer
```

精度必须走 `scripts/run_accuracy.sh`：`npu_dut` + `cpu_lowp` + `--bm_device cpu` 三路
`cv_fused_double_benchmark`，并 `--save_data output`。`run_test_cpu.sh -scope=accuracy`
只有 DUT 和 `cpu_golden`，会退化成 golden 的 `nan_check`，不能当作验收。ATK 汇总表即使
在三路运行里也常常只显示 `cpu_cpu_lowp` 通过、DUT 为 `-`；用
`python scripts/compare_saved_outputs.py <atk_output/.../output>` 核 NPU。
`ACCURACY_START` / `ACCURACY_END` 可截一段 JSON。`performance` 用 `_perf.json`。
`determinism` 和 `mssanitizer` 用 `_mss.json`。YAML 的 `atk case` 生成只用于 marker 校验，
不要覆盖冻结 JSON。
