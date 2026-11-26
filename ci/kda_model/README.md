# KDA 每日模型冒烟

A2、A5 各跑 39 条，每条都执行一次前向和一次反向。GitHub Actions 在北京时间 06:30 触发，也可手动选择日期。两台机器共用同一个北京时间日期，分别安装该日 main 的对应平台 wheel，报告记录实际版本（含提交短 SHA）。

| 用例 ID | 模型或输入 | B × T | H |
| --- | --- | --- | --- |
| glm-flash | GLM Flash | 1 × 4096 | 4 |
| kimi-k3-b1 / kimi-k3-b4 | Kimi K3 | 1 × 8192 / 4 × 2165 | 96 |
| glm-next | GLM Next | 1 × 4096 | 64 |
| bailing-v3-h2 / h4 / h8 | 百灵 V3 | 1 × 131072 | 2 / 4 / 8 |
| case1 至 case32 | V01 至 V16 的原始 cu_seqlens，每组两个头数 | 1 × 各组末端 token 位置 | 4 / 8 |

Q/K/V 为 BF16，前向输入为 BNSD，K=V=128，chunk_size=64。保留变长序列边界，初态为空，输出各序列的终态，打开 Q/K 归一化和 kernel 内 gate，反向使用前向保存的中间量。beta 直接给出，损失对 attention 输出施加同一可复现的上游梯度。

**如何判定通过**　CPU FP32 标杆按序列独立计算 attention 输出、终态以及 dq、dk、dv、dbeta、dg。每个张量都检查形状、有限值、逐元素绝对/相对误差和余弦相似度，阈值复用 examples/chunk_kda.py，终态沿用前向输出阈值。任一张量超差或执行异常都会让整批返回非零。CPU 标杆每 256 个 token 保存一次状态，反向分段重算。

**如何复现**　全部用例开启 safe_gate。GLM Flash、case1、百灵 H=2 的 lower_bound 分别固定为 −5、−2.5、−0.01。其余 36 条按日分三组各 12 条，在合法负数区间内取到百分位。每条用例的输入种子跨天固定。复测时使用原报告中的 wheel、日期和用例 ID。

在已加载 CANN、安装待测 wheel 并加载其 OPP 的 CI 容器内，从仓库根目录执行：

```bash
python3 ci/kda_model/run.py --platform a2 --date 2026-11-26 --report kda-a2.json
python3 ci/kda_model/run.py --platform a5 --date 2026-11-26 --case case1 --report case1-a5.json
```

GitHub Actions 归档 kda-a2.json 和 kda-a5.json。每完成一条就更新报告，包含 case ID、形状、种子、lower_bound、通过状态和每个张量的误差指标。完整成功的报告须满足 planned=39、cases 含 39 条且 passed=true。

生成当天参数清单、独立验证标杆的命令如下。第二条需要 CPU PyTorch。

```bash
python3 ci/kda_model/daily_plan.py --date 2026-11-26 > plan.json
python3 ci/kda_model/test_reference.py
```
