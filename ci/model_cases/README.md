# PR 模型回归

在 PR 评论区输入 `/run-npu-ci models`，或在 Actions 的 **NPU CI → Run workflow** 勾选 **model_cases**。默认关闭。与 `quick`、`full`、`debug` 可以组合使用。`ops=...` 用于定向编译，与模型回归互斥。评论分发需要目标分支已经支持此开关。

| 每个平台的用例 | 验证内容 |
| --- | --- |
| KDA，7 条模型形状 + 32 条变长形状 | 每条验证前向输出、逐序列终态及反向的 dq、dk、dv、dbeta、dg。A2 和 A5 共用反向标杆，均关闭 beta sigmoid。 |
| GDN，3 条 Qwen Prefill | 验证分组头的输出。A5 还通过 profiler 核对 Prepare → H → O 三段调用。 |
| GDN，2 条 Qwen Decode | 验证输出和原地更新的状态。每个请求解码一个 token，使用独立状态槽。 |

选中后，CI 为 A2 和 A5 各构建一次当前 PR 合并后的完整 wheel。模型回归复用常规 CI 容器中准备好的运行时。KDA 的 `safe_gate=True`，三条用例固定 `lower_bound`，其余 36 条按这次 Actions 运行的北京时间日期分层抽样。报告记录实际取值和张量种子。

在 Actions 中打开 **KDA / GDN 模型用例（A2 + A5）** 查看各平台结果，从 `model-cases-<平台>-<run_id>` 下载逐条 JSON。缺报告、漏用例或数值超差都会使模型任务失败。修复后对新提交再次触发；同一提交需要重新实测时，也可以再发一次 `/run-npu-ci models`。

在已经安装该平台 wheel、配置好 OPP 和 ATK 的 CI 容器内，可以复现一条失败用例。日期、平台和用例 ID 取自报告，设备号使用容器内的逻辑编号。

```bash
ATEN_CPU_CAPABILITY=default OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 KDA_BWD_ATK_CPU_THREADS=1 \
python3 ci/model_cases/run.py --date 2026-11-26 --platform a5 --device 0 \
  --case glm-flash --report-dir /tmp/fla-model-results
```

省略 `--case` 可运行全部 44 条。用例清单见 [KDA](kda_cases.json) 和 [GDN](gdn_cases.json)。

## 数值标杆

- KDA 前向使用仓内 ATK 冒烟目录的 `PublicForwardApi` 同精度与 FP64 参考；反向使用 `executor_chunk_kda_bwd_A2.py`，将真实前向保存的中间张量传给两份 CPU 参考。七个输出均由 ATK 的 `cv_fused_double_benchmark` 判定，阈值为最大误差比 5、平均和均方根误差比各 1.5。[前向原件](https://gitcode.com/chen-linxin4/work_tool/blob/72d372032214b8866cd6073431b0d478e7744ba5/FLA_ATK/chunk_kda_fwd/executor_chunk_kda_fwd_v2.py) · [反向原件](https://gitcode.com/chen-linxin4/work_tool/blob/72d372032214b8866cd6073431b0d478e7744ba5/FLA_ATK/chunk_kda_bwd/executor_chunk_kda_bwd_A2.py)
- GDN Prefill 复用仓内 executor 的 `run_cpu`。Decode 的 [recurrent_reference.py](recurrent_reference.py) 摘自同一 GitCode 提交的 recurrent executor，保留输入校验和 FP64 计算。逐元素使用 `atol=rtol=0.005`，余弦相似度至少 0.999。[Prefill 原件](https://gitcode.com/chen-linxin4/work_tool/blob/72d372032214b8866cd6073431b0d478e7744ba5/FLA_ATK/chunk_gated_delta_rule_fwd/executor_chunk_gated_delta_rule_fwd.py) · [Decode 原件](https://gitcode.com/chen-linxin4/work_tool/blob/72d372032214b8866cd6073431b0d478e7744ba5/FLA_ATK/recurrent_gated_delta_rule/executor_recurrent_gated_delta_rule.py)
