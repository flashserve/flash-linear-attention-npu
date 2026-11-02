# ChunkKdaFwd ATK 工程

本目录提供 `chunk_kda_fwd` 的 ATK 单算子工程。通用版本要求、case 范围、测试动作和
结果检查规则见 [`../README.md`](../README.md)。

**终版结果**：精度 48/48 Pass（id 250~297），Case 250 性能 809.19 μs（标杆
1323.7 μs，-38.9%）。证据固化在本目录 [`perf/`](perf/)（任务书 "pref" 路径）与
[`accuracy/`](accuracy/)，逐例精度见 [`accuracy/精度汇总_48cases.md`](accuracy/精度汇总_48cases.md)。
E2E（harness 口径）见 [`../../e2e/kda/`](../../e2e/kda/)。

## 输入限制

- `layout` 支持 `BSND/BNSD/TND/NTD`；`layout` 只描述输入，输出布局由接口固定约定。
- `BSND` 下 `q/k=[B,T,H_k,K]`，`v=[B,T,H_v,V]`，`g=[B,T,H_v,K]`，`beta=[B,T,H_v]`。
- `BNSD` 下 `q/k=[B,H_k,T,K]`，`v=[B,H_v,T,V]`，`g=[B,H_v,T,K]`，`beta=[B,H_v,T]`。
- `TND/NTD` 使用打包 token；`cu_seqlens` 从 `0` 开始、以总 token 数结束且单调不减。
- head 映射满足 `0 < H_k <= H_v <= 128` 且 `H_v % H_k == 0`。
- `q/k/v` 支持 `BFLOAT16/FLOAT16`；`g` 支持 `FLOAT/BFLOAT16`；`beta` 支持 `FLOAT/BFLOAT16`。
- `K/V` 只支持两档且必须同档：`K=V=64` 或 `K=V=128`；混合档（如 `K=64,V=128`）与其它
  取值（含 `V=256`）返回 `ACLNN_ERR_PARAM_INVALID`，负向用例覆盖同类拦截。
- `chunk_size` 只支持 `64`，其它取值（含 `128`）在参数校验阶段返回
  `ACLNN_ERR_PARAM_INVALID`，负向用例覆盖该类拦截；rank-4 变长输入要求 `B=1`，
  逻辑序列数最多 `1024`。
- `use_gate_in_kernel=true` 时必须提供 `A_log`；`safe_gate=true` 时 `lower_bound` 取 `[-5,0)`。
- `initial_state` 如提供，末两维由 `state_v_first` 解释为 `[K,V]` 或 `[V,K]`。

## 精度拓扑

精度标准统一使用 `mixed_tolerance_bm`。显式 CPU 节点调用 executor 的 FP64 路径生成
唯一 golden，显式 NPU 节点只运行 DUT。CPU golden 输出转换为 FP32，NPU 输出保持算子
原始 dtype，由 ATK 按混合容差规则比较。

```text
ATK accuracy task
|-- CPU FP64 golden (output FP32)
`-- NPU DUT
```

## 1. 环境准备

通用要求（ATK 版本 ≥26.8.8、`atk --version`/`npu-smi info` 自检、环境变量表）见
[`../README.md`](../README.md) 的"运行前准备"。本算子验证环境：

```bash
# CANN 9.1.0-beta.3（Ascend 950PR / dav-c310）
source /usr/local/Ascend/cann/set_env.sh

# ATK venv（要求 >= 26.8.8，交付机为 26.9.8）
source /workspace/venv/bin/activate    # 换成你的 ATK venv 路径
which atk && atk --version
npu-smi info

# 自定义算子包：ASCEND_CUSTOM_OPP_PATH 指向 fla_npu_transformer vendors 目录
export ASCEND_CUSTOM_OPP_PATH=<venv>/lib/python3.11/site-packages/fla_npu/opp/vendors/fla_npu_transformer
```

`ASCEND_CUSTOM_OPP_PATH` 使 ACLNN 能找到仓内编译出的 `fla_npu_transformer` 自定义
算子；atk 节点进程继承该变量。若用 `torch_custom` python 包直调（E2E 路径），同样
依赖它。

## 2. 构建与安装

```bash
# 32GB 内存机器：OPS_CPU_NUMBER 必须 <= 2（bisheng 编译单路峰值约 4.1GB，
# 并行过载会 OOM/被 cgroup SIGKILL）
OPS_CPU_NUMBER=2 FLA_NPU_OPS=chunk_kda_fwd FLA_NPU_SOC=ascend950 \
    python scripts/build_wheel.py

pip install --force-reinstall dist/fla_npu-*.whl   # 或按 build 脚本输出的 wheel 路径
```

装完后把 `ASCEND_CUSTOM_OPP_PATH` 指向**新 venv 里**的
`.../site-packages/fla_npu/opp/vendors/fla_npu_transformer`（重装后目录会重建，
旧路径失效）。快速自检：

```bash
python -c "from fla_npu.ops.ascendc import chunk_kda_fwd; print('ok')"
```

## 3. 精度测试（48 例）

在仓库根目录准备好 ATK、CANN、OPP 和 Python 包环境后执行：

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=accuracy
```

- 用例为 id **250~297** 共 **48** 例：6 个 T 档（1024/1536/2048/4096/8192/16384）
  × 8 变体（single/balanced8/mixed_tail/short64 × recompute/export，均 packed 变长）。
- **`CASE_START`/`CASE_END` 是用例位置下标（0 起），不是 case id**：`CASE_START=0
  CASE_END=24` 跑 id 250~273（T≤2048 的 24 例），`CASE_START=24 CASE_END=48` 跑
  id 274~297（T≥4096 的 24 例）；不设置则全跑 48 例。注意 id = 250 + 下标。
- 长例建议调大超时（默认 `ATK_TIMEOUT=14400` 秒通常够用，T16384 例金标耗时长）：
  `ATK_TIMEOUT=20000 bash tests/atk/run_test_cpu.sh ...`。
- **T≥8192 的用例建议逐例（或小块）串行执行**：交付容器为 32GiB cgroup 内存限制，
  ATK 的 CPU fp64 金标 worker 峰值内存大，多例并行会 OOM 被 SIGKILL
  （`WorkerLostError`）。分块示例：

```bash
CASE_START=0 CASE_END=24 bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=accuracy
CASE_START=24 CASE_END=48 bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=accuracy
```

  第二块（id 274~297，T≥4096）中，case 297（下标 47，T16384 export）在本机内存
  受限环境下需改走下节预计算金标流程，其余可正常直跑。

确定性回归会重复运行 MSS 用例并逐位比较全部可见输出。MSS 包含 #440 的 63-token 单序列
尾块场景，通过 96 个 value heads 放大并行调度覆盖，同时启用 `disable_recompute=true`、最终状态
和全部中间量：

```bash
DC_LOOP_NUMS=100 \
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=determinism
```

性能、确定性、mssanitizer 和用例生成均通过统一脚本的对应 `-scope` 执行。

### case 297（T16384 export）预计算金标

case 297 的 fp64 金标在 ATK worker 内峰值 RSS >32GiB，在 32GiB cgroup 容器内会被
SIGKILL——这是**环境内存限制，非算子问题**（NPU DUT 输出完整且数值正常）。解法是
把金标拆出来流式预计算，其余链路（DUT、混合容差比较、报告）与正式 accuracy 完全
一致。三步：

**(a) 预计算金标**（`gen_kda_golden_stream.py` 逐 head 流式复刻 executor 的 fp64
数学，峰值 RSS 仅数 GB；方法已在 case 287 上与 ATK 自算金标逐位 max|diff|=0 认证）：

```bash
venv/bin/python3 tests/atk/chunk_kda_fwd/gen_kda_golden_stream.py \
    --case-id 297 \
    --atk-json tests/atk/chunk_kda_fwd/atk_chunk_kda_fwd.json \
    --executor tests/atk/chunk_kda_fwd/executor_chunk_kda_fwd.py \
    --out-dir /tmp/golden297
```

**(b) 用预计算金标跑 case 297**。注意 `run_test_cpu.sh` 硬编码
`-p "./executor_${OP}.py"`，无法换执行器，precomputed 执行器须**直调 atk**：
用 [`scripts/run297_precomputed.sh`](scripts/run297_precomputed.sh)，或手工把
atk 命令中的 `-p` 换成 `./executor_chunk_kda_fwd_precomputed.py` 并
`export KDA_GOLDEN_PRECOMPUTED_DIR=/tmp/golden297`（该执行器仅在金标分支替换为
"加载预计算"，DUT 与比较全部走原生链路）。

**(c) 说明**：不预计算时 297 的 fp64 金标必然在 32GiB cgroup 内 OOM 被 SIGKILL
（`WorkerLostError`），属于环境限制；在内存不受限的环境内 ATK 可自算金标直接出判，
预计算只是让受限环境也能出正式判定。

## 5. 性能测试（Case 250）

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -soc=ascend950 -scope=performance
```

- `atk_chunk_kda_fwd_perf.json` 仅含 case 250（B=1, T=1024, H=96, K=V=128,
  chunk=64, BSND），与任务书模型 case 一致。
- 标杆 1323.7 μs；终版 **809.19 μs**（std 3.08，**-38.9%**）。五次终版实测
  809.19/807.92/803.89/804.66/810.72 μs，见 [`perf/perf_汇总.md`](perf/perf_汇总.md)。
- 吞吐 ≈ **1.27M tokens/s**（1024 tok / 809.19 μs；标杆口径 ≈773,600 tok/s）。
- 指标为 ATK Device Perf 口径（`device_perf(us)`，NPU AVG Profiler Time）。

## 6. 结果与报告定位

```text
atk_output/{accuracy,perf}/atk_output/<时间戳>/
|-- report/*.xlsx     # ATK 正式报告
`-- log/atk.log       # 全量日志（性能原始行 NPU AVG Profiler Time 在此）
```

脚本末尾自动运行 `common/check_atk_result.py` 汇总通过率。**本仓已把终版证据固化到
`tests/atk/chunk_kda_fwd/{perf,accuracy}/` 顶层目录**（对应任务书 pref/accuracy
路径；run 目录名 UTC+8、xlsx/日志时间戳 UTC，相差 8 小时，详见各目录 README）。

## 7. 辅助脚本

| 脚本 | 职责 |
| ---- | ---- |
| `gen_kda_golden_stream.py` | 内存受限下流式逐 head 复刻 ATK CPU fp64 金标（输出 fp32 memmap），用于 case 297 预计算金标 |
| `executor_chunk_kda_fwd_precomputed.py` | ATK 执行器钩子：`KDA_GOLDEN_PRECOMPUTED_DIR` 存在时以加载预计算金标替代 CPU 金标重算，DUT/比较链路不变 |
| `scripts/run297_precomputed.sh` | 直调 atk 跑 case 297（下标 47）+ 预计算金标的一键封装（`run_test_cpu.sh` 不能换执行器） |
| `stress_npu_determinism.py` | 固定输入 NPU-only 逐位确定性压测（如 `--case-id 250 --repeats 100`，对 run 0 逐位比较 attn_out） |
| `analyze_atk_saved_outputs.py` | 分析已落盘的 ATK NPU/Triton/fp64 金标输出（NaN/Inf/分布/逐输出对比） |

## 8. 约束速查

- `chunk_size=64`（唯一支持值）。
- `K=V∈{64,128}` 且必须同档。
- `H_k, H_v ≤ 128` 且 `H_v % H_k == 0`（`0 < H_k <= H_v`）。
- 变长最多 1024 个逻辑序列（rank-4 变长输入要求 `B=1`）。
- 注：以上与上游参考实现（main 分支参考语义）一致。
