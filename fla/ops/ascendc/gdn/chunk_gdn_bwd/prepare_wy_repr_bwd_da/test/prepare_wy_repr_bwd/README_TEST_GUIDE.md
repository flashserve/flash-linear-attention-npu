# prepare_wy_repr_bwd 测试指南

本目录用于验证 `prepare_wy_repr_bwd` 二阶段反向链路。这里的 `prepare_wy_repr_bwd = prepare_wy_repr_bwd_da + prepare_wy_repr_bwd_full`：先用 `npu_prepare_wy_repr_bwd_da` 计算 `dA`，再用 `npu_prepare_wy_repr_bwd_full` 计算 `dk/dv/dbeta/dg`。

## 目录结构

```text
prepare_wy_repr_bwd/
├── README_TEST_GUIDE.md                       # 本测试指南
├── run_prepare_wy_repr_bwd.sh                 # CPU 双标杆精度/性能测试统一入口
├── run_prepare_wy_repr_bwd_gpu_dump_dual.sh   # GPU dump 双标杆测试入口
├── test_prepare_wy_repr_bwd.json              # 普通精度/性能测试用例
├── test_prepare_wy_repr_bwd.py                # CPU 低精度 + CPU fp64 双标杆精度测试
├── test_prepare_wy_repr_bwd_performance.py    # 性能测试脚本，配合 msprof 使用
├── test_prepare_wy_repr_bwd_gpu_dump_dual.py  # NPU + CPU fp64 + GPU dump 双标杆测试
└── gen_perf_report.py                         # 生成 perf_report.csv
```

## 1. CPU 双标杆精度测试

```bash
cd fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da/test/prepare_wy_repr_bwd
bash run_prepare_wy_repr_bwd.sh --precision
```

常用参数：

```bash
bash run_prepare_wy_repr_bwd.sh --precision --device 0
bash run_prepare_wy_repr_bwd.sh --precision --json /path/to/custom_cases.json
```

说明：`--precision` 执行 `test_prepare_wy_repr_bwd.py`，对比 NPU 输出、CPU 低精度标杆和 CPU fp64 高精度标杆。

## 2. 性能测试

```bash
cd fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da/test/prepare_wy_repr_bwd
bash run_prepare_wy_repr_bwd.sh --performance
```

常用参数：

```bash
bash run_prepare_wy_repr_bwd.sh --performance --device 1
bash run_prepare_wy_repr_bwd.sh --performance --json /path/to/custom_cases.json
```

说明：`--performance` 使用 `msprof` 执行 `test_prepare_wy_repr_bwd_performance.py`，随后调用 `gen_perf_report.py` 生成 `perf_report.csv`。性能测试输出包括：

- `prof_output/`：`msprof` 原始输出目录，每次运行前会清理重建。
- `perf_report.csv`：性能汇总报告，包含 shape、数据类型、定长/变长标识和算子耗时。

## 3. 普通测试用例配置

普通精度/性能测试用例配置在 `test_prepare_wy_repr_bwd.json`。核心字段：

| 字段 | 说明 |
| :--- | :--- |
| `enabled` | 是否执行该用例 |
| `B` | batch size |
| `query_head` | Key 头数，记作 `KH` |
| `value_head` | Value 头数，记作 `VH`，需满足 `VH = group_size * KH` |
| `T` | 序列长度 |
| `Kdim` / `Vdim` | Key/Value 维度 |
| `chunk_size` | 分块大小 |
| `dtype` / `gtype` | 数据类型配置 |
| `varlen` / `mean_len` | 变长模式及 `cu_seqlens` 生成配置 |

## 4. GPU dump 双标杆测试

GPU dump 双标杆测试对比 NPU 输出、CPU fp64 高精度标杆和 GPU dump 输出。dump 文件使用 `008_prepare_wy_repr_bwd.pt`，其中：

- 输入：`k`、`v`、`beta`、`g`、`A`、`dw`、`du`
- 输出：`dk2`、`dv`、`db`、`dg2`
- meta：`cu_seqlens`、`chunk_indices` 等变长信息

注意：`dA` 不从 pt 读取，脚本会现场调用 `npu_prepare_wy_repr_bwd_da` 计算。

```bash
# 按 dump 根目录批量执行
bash run_prepare_wy_repr_bwd_gpu_dump_dual.sh /path/to/GPU_DUMP

# 直接执行单个 dump 文件
bash run_prepare_wy_repr_bwd_gpu_dump_dual.sh /path/to/008_prepare_wy_repr_bwd.pt
bash run_prepare_wy_repr_bwd_gpu_dump_dual.sh --pt /path/to/008_prepare_wy_repr_bwd.pt

# 指定 NPU device
TEST_DEVICE_ID=1 bash run_prepare_wy_repr_bwd_gpu_dump_dual.sh /path/to/GPU_DUMP
```

常用参数：

```bash
bash run_prepare_wy_repr_bwd_gpu_dump_dual.sh /path/to/GPU_DUMP --case phase_1_fix_1
bash run_prepare_wy_repr_bwd_gpu_dump_dual.sh /path/to/GPU_DUMP --phase prefix:phase_1_
bash run_prepare_wy_repr_bwd_gpu_dump_dual.sh /path/to/008_prepare_wy_repr_bwd.pt --no-viz
```

GPU dump 测试输出：

- 日志：`<DUMP_ROOT>/logs/prepare_wy_repr_bwd_gpu_dump_dual_<timestamp>.log`
- 最新日志：`<DUMP_ROOT>/logs/prepare_wy_repr_bwd_gpu_dump_dual_latest.log`
- 报告：`<DUMP_ROOT>/prepare_wy_repr_bwd_gpu_dump_dual_report.json`，单 pt 模式下写到 pt 所在目录
- 可视化：默认写到 case 或 pt 所在目录的 `viz` 子目录，可用 `--viz-dir` 指定