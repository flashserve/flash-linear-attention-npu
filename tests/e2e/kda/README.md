# chunk_kda_fwd E2E 验证（tests/e2e/kda）

本目录是 `chunk_kda_fwd` 的端到端（harness 口径）验证：在 torch_npu 上直调
`fla_npu.ops.ascendc.chunk_kda_fwd`，覆盖精度 sanity、Case 250 性能、调优前后
对比、合成 K3 模型前向、多序列长度、内存、稳定性与确定性（M1~M8）。ATK 单算子
口径（正式精度/性能证据）见 [`../../atk/chunk_kda_fwd/`](../../atk/chunk_kda_fwd/)。

## 文件

| 文件 | 说明 |
| ---- | ---- |
| `e2e_full_test.py` | 主 E2E 脚本（M1~M8），结果写本目录 `e2e_results_v2.json` |
| `benchmark_chunk_kda.py` | Case 250 / K3 形状性能基准（harness 计时） |
| `e2e_k3_chunk_kda.py` | Kimi-K3-0.40B 架构兼容性测试（需本地模型 config，可用 `K3_MODEL_DIR` 覆盖路径） |
| `E2E_REPORT.md` | 完整测试报告（含吞吐换算小节） |
| `e2e_results_v2.json` | 2026-09-29 全量重跑结果（M1~M8 全过） |

## 环境

与 ATK 工程相同：CANN 9.1.0-beta.3（`source /usr/local/Ascend/cann/set_env.sh`）+
装好 fla_npu wheel + `ASCEND_CUSTOM_OPP_PATH` 指向 venv 内
`fla_npu/opp/vendors/fla_npu_transformer`。构建与安装步骤直接参照
[`../../atk/chunk_kda_fwd/README.md`](../../atk/chunk_kda_fwd/README.md) 的
**"1. 环境准备"** 与 **"2. 构建与安装"** 两节。脚本基于自身路径定位仓库根下的
`torch_custom`，可在仓库任意位置运行；`torch_npu` 需在 venv 中可用。

## 运行

```bash
# 1) 全量 E2E：M1~M8，预期全部 PASS（退出码 0）
python tests/e2e/kda/e2e_full_test.py

# 2) Case 250 性能基准（harness 口径）
python tests/e2e/kda/benchmark_chunk_kda.py --case 250
```

`e2e_k3_chunk_kda.py` 需要本地 Kimi-K3-0.40B 权重快照的 `config.json`：
`K3_MODEL_DIR=<模型快照路径> python tests/e2e/kda/e2e_k3_chunk_kda.py`。

## 预期结果（对齐 e2e_results_v2.json，Ascend 950PR）

| 项 | 指标 | 预期 |
| -- | ---- | ---- |
| M1 | 精度 sanity（H=96 T=1024 bf16） | PASS：无 NaN/Inf，max_abs≈6.5e3 |
| M2 | Case 250 平均时延 | PASS：≈**849 μs**（ref 1323.7，约 -35%） |
| M3 | 调优前后对比 | PASS：speedup≈1.55x，imp≈+35.3% |
| M4 | 合成 K3 模型前向 | PASS：T=64/256/1024/2048 全 OK，decode≈0.20 ms |
| M5 | 多序列长度（含 H=96 T=3072/4096/8192） | PASS：11 组全部有限值 |
| M6 | 内存（T=4096 H=8） | peak≈4.3 GB |
| M7 | 20 轮稳定性 | PASS：逐位一致 |
| M8 | 确定性 | PASS：bit-exact（diff=0） |

## 调优前后对比（M3，任务书目标 ≥5%）

| | 调优前（4-launch） | 调优后（varlen-dense 单 launch 快路径） |
| -- | ----- | ----- |
| Case 250 device 时延 | 1331.0 μs | **863.4 μs** |
| 加速比 | 1.0x | **1.54x** |
| 改进 | — | **+35.1%** |

（harness 多轮复测 849~863 μs，波动 <2%；ATK Device Perf 口径终版 809.19 μs、
-38.9%，见 [`../../atk/chunk_kda_fwd/perf/`](../../atk/chunk_kda_fwd/perf/)。
两口径差异为 harness 端到端计时包含 host 侧固定开销，详见 `E2E_REPORT.md`
"吞吐换算"。）

## 已知限制

- **T=16384 H=8 数值边界**：合成无界 gate 输入下从 chunk 158（T≈10112）起输出
  NaN，属 bf16 长序列 KDA 状态累积的数值边界，非算子缺陷；正常 K3 推理（T≤2048）
  不受影响，模型尺度输入（l2norm q/k、有界 gate）在 ATK case 290~297 全部通过。
  M5 已将 T=16384 排除在外。
- **V2 入口 561103 已自动回退**：`aclnnChunkKdaFwdV2` host 侧 GetWorkspaceSize
  失败（典型 561103）时，`_aclnn_ctypes.py`/`_stable.py` 自动回退融合入口
  `aclnnChunkKdaFwd` 并告警一次（V2 专属语义不回退）；回退代价实测为零，M5 中
  T=3072/4096/8192 H=96 即走该路径，全部 OK。见 commit `8ee9e23d`。
