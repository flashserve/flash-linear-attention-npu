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
  --work-root "$PWD/outputs/kda-main-benchmark" \
  --cases prefill_fwd_b1_s8192,prefill_fwd_b1_s16384
```

该入口的 prefill 固定为 BF16、`H=96`、`K=V=128`、`chunk_size=64`、
`initial_state=None`、`output_final_state=False`、`use_gate_in_kernel=True`、
`safe_gate=True`、`state_v_first=True`。报告使用 `msopprof` 设备侧耗时；任一 case 出现
`ERROR`、`OOM` 或 `TIMEOUT` 时，入口返回非零状态并保留逐 case 诊断。

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
