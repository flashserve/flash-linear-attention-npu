# Issue #921 开发验证记录

状态：开发验证，尚未完成正式验收。修复仅改变 A5 Stage 13/14 的门控点积精度边界。

## 版本与环境

- 日期：2026-11-11；Ascend 950DT，CANN 9.2.1，driver 26.2.0.b020。
- Python 3.12.16，torch 2.15.0+cpu，torch_npu 2.15.0+gitUnknown。
- 被测实现：PR #923 合并主线后的 `38c195bf`；后续提交只增加测试和文档。
- 主线基准：`238a8ec002c45fe65ef956b17ed7c014153c1f42`。
- A/B 基线安装：`54ae09b1243cd851d986c2624a57bbb5ee63dc27` 构建的完整 A5 包。
  已核对该基线的 A5 finalize Cube、Vector、common 和 tiling struct 与上述主线一致。
  候选使用本次源码重建的 Python/Stable launcher 和 GDN OPP，独立安装。
- 安装校验：运行时 opapi MD5 为 `c2d5154cc0dfa653d38fce396f353598`，
  与本次 run 包相同；全部 25 个 kernel `.o` 与打包产物逐文件相同。

## 构建与 ABI 检查

本次同时构建完整 GDN 调用链，使用：

```bash
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu -j16 \
  --ops=chunk_gated_delta_rule_fwd,chunk_gated_delta_rule_bwd,chunk_gated_delta_rule_fwd_prepare,chunk_fwd_h,chunk_fwd_o,chunk_gdn_bwd_intra,chunk_gated_delta_rule_bwd_dhu,chunk_gated_delta_rule_bwd_finalize
```

编译、链接和 run 打包完成，但命令最终退出 1：主线 `op_abi_validate.py` 的
`adapter_calls` 在正则匹配之前未去注释，无法识别 `FLA_STABLE_EXEC(` 与接口字符串
之间带行注释的调用，报 `No ABI call sites found`。该检查器和两个受影响的
GDN adapter 均与主线相同，未在本 PR 中修改。

将 adapter 复制到临时目录，只用检查器已有的 `strip_comments` 去除注释后，
对本次打包头文件运行同一检查器：8 个调用点、0 个类型不匹配或缺失头文件。
这是独立核对结果，不能把原构建命令记为完全通过。

## 精度与回归

可直接执行仓内的两个回归脚本：

```bash
python tests/atk/chunk_gated_delta_rule_bwd_finalize/scripts/check_gate_precision.py --device cpu
python tests/atk/chunk_gated_delta_rule_bwd_finalize/scripts/check_gate_precision.py --device npu
python tests/atk/chunk_gated_delta_rule_bwd_finalize/scripts/repro_gate_zero_state.py
```

| 检查 | 基线 | 修复后 |
| --- | --- | --- |
| 最简完整前后向，单 token、零初始状态，`max(abs(dg))` | 4.9896724522e-4 | 0 |
| 独立 finalize，32 seeds × FP32/BF16 gate，64 例 | 64/64 非零，最大 8.6031854153e-5 | 64/64 精确为零；CPU 同样通过 |
| 完整前后向，8 seeds × Q/K norm 开关，16 个单 token 用例 | 16/16 非零，最大 5.7979300618e-4 | 16/16 精确为零 |
| 完整前后向诊断回放，共 36 例 | 对照同一份输入和 `do` | 36/36 的 `dg` L2 绝对误差降低 |
| 上述 64+36 例的其余输出 | 基线输出 | `dq/dk/dv/dbeta` 逐位相同，完整链的前向输出也逐位相同 |
| head/window 补充回归，8 例 | HK=3，HV/HK=1/2/3/4，T=65/2049 | 全部有限、重复确定；前向和另外四种梯度与基线逐位相同 |

36 个完整链用例包含 T=1/2/128/129/512，dense 与 packed `[0,7,72,128]`，
Q/K norm 开关，以及随机 `do` 和固定平方损失上游梯度。每个用例重复执行，结果逐位一致。
独立数学参考使用相同已量化输入及 `do`，以 FP64 逐 token 状态递推和 autograd 计算；
不使用本 PR 修改后的分阶段 CPU 实现作为完整链的数学参考。

代表性结果（B=1、HK=2、HV=4、K=V=128、T=128、seed=7、内部 Q/K norm、随机 `do`）：

| 场景 | `dg` 相对 L2：基线 → 修复 | 序列首 token `max(abs(dg))`：基线 → 修复 |
| --- | --- | --- |
| dense | 2.0888% → 0.4599% | 2.76344e-3 → 6.07688e-8 |
| packed `[0,7,72,128]` | 1.6909% → 0.4469% | 2.53301e-3 → 1.11759e-7 |

长序列首 token 仍有 FP32 级残差；本修复不承诺整个 BF16 计算链等同于 FP64。
另测单 token 的非零初始状态，以及有/无末状态梯度：非零状态的 `dg` 保留非零，
两种情况下相对 FP64 的最大绝对误差分别为 2.38956e-4、4.07805e-4。
零初始状态加末状态梯度的用例仍得到零，与数学参考一致。实现没有将首 token 梯度置零。

## 性能和内存代价

同一设备、输入、软件依赖，预热 10 次后采集 50 次，取 NPU profiler 的
`ChunkGatedDeltaRuleBwdFinalize` kernel duration 中位数。以下是算子诊断测量，
不代表完整 backward、训练 step 耗时或 ATK 性能验收。

| B / T / HK / HV（K=V=128） | 基线 µs | 修复 µs | 耗时增加 |
| --- | ---: | ---: | ---: |
| 1 / 128 / 2 / 4 | 24.918 | 28.5265 | 14.48% |
| 1 / 512 / 2 / 4 | 26.234 | 29.8640 | 13.84% |
| 1 / 2048 / 16 / 32 | 260.1635 | 315.8125 | 21.39% |

workspace 从 `7R` 增至 `13R`，每个 active AIC 增加 768 KiB；32 个 AIC 时增加 24 MiB。
UB/L1 的分配总量不变。方案、有效行处理和 ready/free 生命周期见 `design.md` Stage 13/14。

## 尚未完成

- ATK 正式 accuracy/performance/determinism/MSS：入口报找不到 `atk`，未完成；
  上述独立回归不能替代 200/64/8 正式矩阵，也不构成 sanitizer 通过。
- A2/A3 构建与实机、当前 head 的 A2+A5 NPU CI、完整 Example/ST：未完成。
  A2/A3 私有计算路径相对主线没有修改。
- 本轮未运行 GPU、短训练、训练收敛或端到端训练性能测试。
- 静态扫描仍报告主线已有的 38 处 DFX 提示、缺失算子 README，以及 `task-group`
  被 `sk-` 规则误命中的提示；不将其记为全算子静态检视通过。

PR 保持 Draft，待补齐正式验收并评审上述精度/性能取舍。
