# PR 模型回归

在 PR 评论区输入 `/run-npu-ci quick` 或 `/run-npu-ci full`，也可以从 Actions 的 **NPU CI** 触发。A2、A5 各自在 **05 GDR Example/ST** 先跑 Example/ST，再跑命中的算子 ATK，最后必跑模型用例。没有算子命中也会进入模型验证。单独的 `ops=...` 编译诊断在构建阶段结束。

| 每个平台的用例 | 验证内容 |
| --- | --- |
| KDA，7 条模型形状 + 32 条变长形状 | 每条验证前向输出、逐序列终态及反向的 dq、dk、dv、dbeta、dg。A2 和 A5 共用反向标杆，均关闭 beta sigmoid。 |
| GDN，3 条 Qwen Prefill | 验证分组头的输出。A5 还通过 profiler 核对 Prepare → H → O 三段调用。 |
| GDN，2 条 Qwen Decode | 验证输出和原地更新的状态。每个请求解码一个 token，使用独立状态槽。 |

模型和算子 ATK 使用同一容器内实际安装的 wheel 和 OPP。修改单个算子时也包含本次 CI 安装的 OPP overlay。KDA 的 `safe_gate=True`，三条用例固定 `lower_bound`，其余 36 条按这次 Actions 运行的北京时间日期分层抽样。报告记录实际取值和张量种子。

在 Actions 的 **05 GDR Example/ST** 查看结果，从 `model-cases-<平台>-<run_id>-<attempt>` 下载逐条 JSON 和 `summary.json`。任何一条用例失败都会使该平台冒烟失败。

## 结果缓存

GitHub Actions 在可执行源码相同的运行之间共享已通过的结果，可跨 PR 和提交，也可在同平台、同架构的匹配 runner 上恢复。记录按用例保存，身份包含实际安装的 FLA 包及生效的 OPP、容器镜像 ID、CPU/NPU 与驱动信息、测试与标杆代码，以及这条用例的参数和种子。Markdown 或提交元数据变化可以复用结果；修改某条清单参数只使输入变化的用例失效。

每次冒烟都会核对全部 44 条模型用例。命中时保留原始数值指标和来源 run/head，未命中或缓存损坏时实跑。KDA 的实际 `lower_bound` 参与匹配。算子 ATK 按本次命中的算子逐条匹配；缺少显式随机种子的用例每次实跑。`summary.json` 记录模型缓存命中数和实跑数，ATK 附件中的 `cache/report/reused.json` 保存复用的原生指标和来源。

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
