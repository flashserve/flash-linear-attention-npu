# ChunkKdaFwd 测试归档

## 1. 唯一用例规格

`tests/op_cases/chunk_kda_fwd.json` 统一保存 shape、dtype、layout、属性、可选输入、SOC、运行通路、随机种子、
参考实现、容差和预期返回码。新 case 必须先进入 JSON，再由执行端按 case ID 消费。

## 2. 归档内容

| 路径 | 内容 |
| --- | --- |
| `common/case_matrix.py` | 本算子 JSON 加载、tag/route 筛选和 case ID 环境变量 |
| `accuracy/test_chunk_kda_fwd.py` | `fla_npu.ops.ascendc` 主精度、泛化、边界和回归入口 |
| `routes/test_aclnn_chunk_kda_fwd.cpp` | aclnn 两段式接口签名、workspace/executor/stream 契约 |
| `examples/fast_kernel_launch_example/csrc/chunk_kda_fwd/` | 完整 KDA 语义的真实 `<<<>>>` 发射实现 |
| `examples/fast_kernel_launch_example/tests/chunk_kda_fwd/` | 直调与稳定 aclnn/ctypes 路径的逐输出精度对比 |
| `ut/op_host/test_contract.py` | manifest、SOC、返回码、host 负向用例静态契约 |
| `ut/op_host/test_optional_output_policy.py` | 固定 fla-org 提交并穷举 16 种可选输出策略 |
| `ut/op_kernel/test_contract.py` | kernel 入口、tiling key 说明和 direct launch 静态契约 |
| `performance/profile.py` | 读取 performance tag 并通过 msopprof 运行设备侧 profiling |
| `st/test_example.py` | example tag 与仓内数值执行后端的 ST 入口 |
| `integration/validate_triton_ascend_adapter.py` | AscendC 正向接入模型现有 Triton 反向，包含 H=96 长序列契约、精度和确定性验证 |

- legacy 通路：`torch.ops.npu.npu_chunk_kda_fwd`，由主 route case 验证显式加载。

现有数值/reference 后端：`tests/operators/_shared/chunk_kda_backend.py`。该后端由 canonical 入口传入
`FLA_NPU_CASE_MANIFEST`、`FLA_NPU_CASE_IDS` 和 `FLA_NPU_OPERATOR`；关键 shape、dtype、属性组合不在
canonical 脚本中重复定义。

## 3. 历史资产迁移

torch_custom 适配工程中原有的主线测试矩阵和诊断脚本已迁入本目录，非标准目录已删除。case 数据只保留在唯一 manifest；共享执行器从 JSON 按 case ID 加载，不再维护第二份 shape 表。

| 迁移集合 | 数量 | 唯一规格 |
| --- | ---: | --- |
| chunk128_backend | 3 | tests/op_cases/chunk_kda_fwd.json |
| model_shape | 1 | tests/op_cases/chunk_kda_fwd.json |

使用 python3 -m tests.operators._shared.legacy_cases list --op chunk_kda_fwd 可列出迁移 case。

## 4. 执行命令

```bash
pytest -q tests/operators/chunk_kda_fwd/accuracy/test_chunk_kda_fwd.py
FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/chunk_kda_fwd/accuracy/test_chunk_kda_fwd.py
FLA_NPU_CASE_TAGS=generalization FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/chunk_kda_fwd/accuracy/test_chunk_kda_fwd.py
pytest -q --import-mode=importlib tests/operators/chunk_kda_fwd/ut
python tests/operators/chunk_kda_fwd/performance/profile.py --dry-run
python tests/operators/chunk_kda_fwd/performance/profile.py --case-id chunk_kda_fwd_h96_t8k_model_performance
python tests/operators/chunk_kda_fwd/performance/profile.py --case-id chunk_kda_fwd_h96_t16k_model_performance
python tests/operators/chunk_kda_fwd/st/probe_a5_tail.py --device 0
python tests/operators/chunk_kda_fwd/st/probe_a5_tail.py --device 0 --long-seq
FLA_NPU_RUN_OPERATOR_TESTS=1 pytest -q tests/operators/chunk_kda_fwd/st/test_example.py
cd examples/fast_kernel_launch_example && FAST_KERNEL_OP_NAME=chunk_kda_fwd pytest -q tests/chunk_kda_fwd
```

`scripts/benchmark_kda_main.sh` 提供独立 wheel 构建、安装、打包 API 自检和 `msopprof`
报告聚合入口。默认直接使用当前
checkout，不访问远端仓库；需要验证远端 ref 时才同时传入 `--repo-url` 和 `--ref`：

```bash
bash scripts/benchmark_kda_main.sh \
  --soc ascend950 \
  --device 0 \
  --work-root "$PWD/outputs/kda-main-benchmark"
```

该入口固定运行 8 条 A5 dense 正向用例（case ID 250 至 257）。所有用例固定为
BF16、`H=96`、`K=V=128`、`chunk_size=64`、
`initial_state=None`、`output_final_state=False`、`use_gate_in_kernel=True`、
`safe_gate=True`、`return_intermediate_states=False`、`state_v_first=True`、
`cu_seqlens=None`，并覆盖：

- dense 序列长度：1024、8192、16384、65536；
- 每种长度覆盖启用重计算和关闭重计算两种模式，即底层
  `disable_recompute=False/True`。

可先查看固定矩阵，或只运行部分 ATK case ID / case key：

```bash
python scripts/benchmark_kda_matrix.py --list-cases

bash scripts/benchmark_kda_main.sh \
  --soc ascend950 \
  --device 0 \
  --work-root "$PWD/outputs/kda-main-benchmark" \
  --cases 252,253,254,255
```

为兼容旧的一键命令，`--decode-step 1` 会被接受但不参与 KDA 正向测试；旧 case 名
`prefill_fwd_b1_s1024`、`prefill_fwd_b1_s8192`、`prefill_fwd_b1_s16384`
分别映射到 case 250、252、254；`prefill_fwd_b1_s65536` 映射到 case 256。
新增测试应直接使用 case ID 或 case key。
脚本会优先使用当前已加载的 CANN 环境；未加载时自动尝试标准安装位置
`/usr/local/Ascend/cann/set_env.sh` 和 `/usr/local/Ascend/ascend-toolkit/set_env.sh`。
每次运行会将 CANN 应用日志重定向到当前 `run_*/ascend_logs/`，避免继续写入默认
`$HOME/ascend/log`；preflight 和 profiler 输出同时实时显示在终端并保存到结果日志。

报告使用 `msopprof` 设备侧耗时，默认 `--aic-metrics Default`。不要改为
`BasicInfo`，否则只能得到基础耗时，不能生成完整资源明细。结果目录包含：

每个 ATK case 只启动一次 application，固定使用 `--launch-count 1` 和
`--replay-mode=kernel`；`--warm-up 5` 保持开启。预热以及 Default 全指标所需的
kernel replay 均由 msopprof 在同一个 application 进程内完成，不会重复调用 case worker，
每个 case 日志中应只出现一次 `BENCH_OK`。

| 文件 | 内容 |
| --- | --- |
| `case_matrix.csv` | 8 条固定 dense 用例及序列长度、chunk 数、重计算开关、随机种子和功能属性 |
| `results.csv` / `results.md` | 每条用例的端到端 kernel 聚合耗时和执行状态 |
| `kernel_detail.xlsx` | 8 个 case sheet；每个 sheet 保存该用例的 kernel、replay、block/sub-block 完整性能明细 |
| `results.json` | 环境元数据、用例汇总和 kernel 耗时分解 |

`kernel_detail.xlsx` 固定包含 `case_250` 至 `case_257` 共 8 个 sheet，每个 sheet
提供 Cube、MAC、Vector、MTE1、MTE2、MTE3、Fixpipe、
Scalar 的时间、占比和带宽列，并保留 `PipeUtilization`、`ArithmeticUtilization`、
`Memory`、`MemoryL0`、`MemoryUB`、`L2Cache`、`ResourceConflictRatio` 和
`OpBasicInfo` 中出现的全部原始字段。`mac_time_us` 是 profiler
`aic_cube_time(us)` 的易读别名，原始字段仍会保留。AIC 与 AIV 的 MTE 指标分别记录，
不合并为单一数值。使用 `Default` 时若缺少任一资源表，该 case 会判为 `ERROR` 并保留诊断。

若已有运行目录只需重新生成 `kernel_detail.xlsx`，无需重跑 NPU profiling：

```bash
python scripts/benchmark_kda_matrix.py \
  --repair-workbook /path/to/run_YYYYMMDD_HHMMSS_PID/results
```

该命令读取原有 `results.json` 及其中记录的 msopprof CSV，覆盖结果目录内的
`kernel_detail.xlsx`，不会启动算子或修改其他报告文件。

任一 case 出现 `ERROR`、`OOM` 或 `TIMEOUT` 时，入口返回非零状态并保留逐 case 诊断。

### B200 FLA Triton 性能对比

`scripts/benchmark_kda_b200_triton.py` 在单张 NVIDIA B200 上直接调用当前 Python
环境已经安装的 `flash-linear-attention==0.5.2`，不会拉取 fla-org 仓库，也不会执行
`pip install`。脚本使用与 A5 完全相同的 case 250 至 257、输入 shape、随机种子和功能属性，
并强制关闭 FlashKDA、TileLang 等可选后端分发，确保执行 FLA 默认 Triton 实现。

在已经安装 FLA 0.5.2 的 B200 环境中一键执行：

```bash
python scripts/benchmark_kda_b200_triton.py --device 0
```

默认每条 case 预热 5 次，随后正式执行 10 次。每次使用 CUDA event 记录完整低层
`chunk_kda_fwd` 前向耗时并同步，最终取 10 次的算术平均值。输入生成、首次 Triton 编译、
自动调优和预热耗时不计入正式结果。可显式写出默认参数或指定部分 case：

```bash
python scripts/benchmark_kda_b200_triton.py \
  --device 0 \
  --warmup 5 \
  --runs 10 \
  --cases 250,251,252,253,254,255,256,257 \
  --output-dir "$PWD/output/kda-b200-triton"
```

结果目录包含：

| 文件 | 内容 |
| --- | --- |
| `results.md` | B200 平均耗时、MFU、A5 耗时和 A5/B200 耗时比汇总表 |
| `results.csv` | 每条 case 的平均值、中位数、最小值、最大值、标准差和峰值显存 |
| `timings.csv` | 每条 case 的 10 次原始 CUDA event 耗时 |
| `results.json` | GPU、PyTorch、Triton、FLA 版本、FLA 安装位置校验状态和完整结果 |

脚本默认按单张 B200 的 BF16 dense 峰值 `2250 TFLOPS` 计算 MFU。若测试环境采用其他
功耗或频率口径，可通过 `--peak-tflops <value>` 覆盖；耗时本身不受该参数影响。

A5 PR264 一键构建、隔离安装和基础验收：

```bash
bash scripts/validate_kda_a5.sh \
  --cann-env /path/to/Ascend/ascend-toolkit/set_env.sh \
  --device 0 \
  --work-root "$PWD/outputs/kda-a5"
```

默认 `--cases smoke` 依次验证尾块同步、H96/T8K/T16K 模型 shape 和 BF16
`A_log/dt_bias` 适配；H96 放在可能触发设备超时的低占用单头诊断之前，确保模型结果能够归档。
`--cases all` 继续执行两组 msopprof。脚本默认拉取 `refs/pull/264/head`，也可通过
`--ref` 验证指定提交或分支。源码仓、Python venv、pip 下载、同提交 wheel 和 Torch 扩展均复用
`work-root/cache/`；每次运行仍使用独立结果目录。同一提交复测不会重复下载和构建。

每次运行输出 `summary.txt`、`results.json`、`results.md`、逐 case 日志和
`kda_a5_diagnostics.tar.gz`。终端自动打印短摘要，包含环境版本、模型 shape 状态及首个二进制差异；
普通精度/确定性失败后继续执行剩余 case，只有 timeout、OOM 或 device task error 才提前停止。

`probe_a5_tail.py` 默认验证 `T=64/65` 尾块与 final-state 同步；`--long-seq` 使用
BF16、BSND、`H=96`、`K=V=128`、`chunk_size=64`，依次单跑 `T=8192/16384`，
只保留必要正向输出并报告 finite、采样 fingerprint、耗时和 NPU allocator 占用。长序列探针
直接构造已完成 Q/K L2Norm 和 beta sigmoid 的输入；包含实际前处理 launch 的全链路性能使用上面的
两个 `profile.py --case-id` 命令测量。

A2/A3/A5 通过 `FLA_NPU_SOC` 选择。精度逐项比较全部公开输出并检查 NaN/Inf；性能只使用 msopprof
设备侧结果，按 JSON 的 `expect.requirement` 对比 Triton 或当前主线基线。报告记录平台、case 总数、通过数和
失败 case ID，不记录本地环境路径。

可选输出矩阵同时验证 L2 空指针契约和 `fla_npu.ops.ascendc.chunk_kda_fwd` 的 12 返回值掩码；
对标版本固定在 `tests/op_cases/chunk_kda_fwd.json` 的 `upstream_alignment`。

## 5. Triton-Ascend KDA 模型无感适配

适配目标固定为
`Ascend/triton-ascend-kernels@4cd4b506d4153ac18ac1ca8f4c770eac9fd3fcc8`
和 `triton-ascend==3.2.1`。
适配器替换该仓 `ChunkKDAFunction` 使用的低层 `chunk_kda_fwd`，并将
`chunk.py` 的前向 L2Norm 切换为仓内固定网格实现：

- 正向使用 `fla_npu.ops.ascendc.chunk_kda_fwd`；
- Q/K 前向 L2Norm 使用 `fla_npu.ops.triton.l2norm_fwd`，避免长序列按
  `T` 拆成多次 launch；
- 反向仍使用模型现有 Triton-Ascend KDA 实现；
- L2Norm 反向不替换，继续使用模型现有实现；
- 输入保持 BSND，不修改模型张量准备；
- 模型侧 `A_log` / `dt_bias` 可为 FP32 或 BF16；BF16 仅在 AscendC 正向边界
  升为 FP32，原张量继续由现有 Triton 反向使用；
- AscendC 的 BNSD 中间量在适配边界转回 BSND，供现有 Triton 反向直接消费；
- 优先使用 `cu_seqlens_cpu` 一次性构造 L2 metadata，避免逐元素 NPU 到 Host 同步；
- `disable_recompute=True` 时按 fla-org 低层接口保留 Triton 反向需要的
  `gk/h`。

### 5.1 模型启动代码增加一行

必须在第一次调用 `chunk_kda` 前安装：

```python
from fla_npu.adapters import install_triton_ascend_kda_adapter

install_triton_ascend_kda_adapter()
```

模型原调用保持不变：

```python
from triton_ascend_kernels.attention.fla.kda import chunk_kda

o, final_state = chunk_kda(
    q=q,
    k=k,
    v=v,
    g=g,
    beta=beta,
    scale=q.shape[-1] ** -0.5,
    initial_state=None,
    output_final_state=False,
    safe_gate=True,
    use_gate_in_kernel=False,
)
```

### 5.2 不修改模型源码

将仓内 startup hook 加入 `PYTHONPATH`，每个 Python/torchrun worker 会在解释器启动时安装适配器：

```bash
export FLA_NPU_ENABLE_TRITON_ASCEND_KDA_ADAPTER=1
export PYTHONPATH="$PWD/examples/adapters/triton_ascend_kda:$PYTHONPATH"

python model_entry.py ...

# 多卡命令保持相同环境变量
torchrun --nproc-per-node 8 model_entry.py ...
```

运行后可确认：

```python
from fla_npu.adapters import is_triton_ascend_kda_adapter_installed

assert is_triton_ascend_kda_adapter_installed()
```

### 5.3 模型正反向验证

```bash
# 小 shape：AscendC 正向 + Triton 反向，对比纯 Triton 正反向
python tests/operators/chunk_kda_fwd/integration/validate_triton_ascend_adapter.py \
  --case smoke --backend compare --runs 1

# 模型真实 shape：校验 finite、重复执行二进制一致和反向内部 FP32 契约
python tests/operators/chunk_kda_fwd/integration/validate_triton_ascend_adapter.py \
  --case model_bwd_h96 --backend ascendc --runs 2 \
  --output-json outputs/chunk_kda_model_bwd_h96.json
```
