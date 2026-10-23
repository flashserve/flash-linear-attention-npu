# ChunkKdaBwd 910B

| 项目 | 配置 / 结果 |
| --- | --- |
| 接口 | `chunk_kda_bwd(..., implementation="legacy")`，Main + Gate 完整反向 |
| 模型 shape | B=1/H=32/K=V=128/C=64，BF16 q/k/v/beta、FP32 raw gate，dense、safe、保存前向缓存 |
| 基线 | main `90176b44321ebdbf3ae2a9057243157d7085c6e1`；后续 `8c5dd635` 仅修改 GDN，KDA 源码不变 |
| 实测环境 | 910B2C / CANN 8.5.0 / torch、torch_npu 2.7.1 / ATK 26.9.8；A3/A5 未本机验证 |
| 构建兼容 | 本机旧 SDK 对两版相同替换 `ASCEND950` 查询、`Fill→InitConstValue`、`FixpipeConfig{NZ,false}→{NZ}`；这些前向编译兼容替换不属于本 PR，正常构建沿用上游推荐 CANN 8.5.2+ |
| Golden | CPU FP64 recurrent KDA + autograd，7 个梯度；另验 T=1 解析解和上游分块参考 |
| 输入 | 种子 0/20260630/7，先量化为输入 dtype；q/k/v std=0.05，beta∈[0.1,0.9]，dO std=51.2，强信号 204.8 |
| 精度规则 | 原生 `mixed_tolerance_bm`；按模型 dtype 比较，FP16/BF16 混合的 db 按 BF16 比较 |
| 校准 | 原始 dO=204.8 两版同样有 13 个失败；保留容差、统一校准值域后正式验收通过；额外强信号测试可检出 7 个输出的置零故障 |
| 正式验收 | 两版均 accuracy 72/72（66 正例 + 6 unsafe 拒绝例），accuracy_dc 12/12 × 50，memcheck 12/12 |
| 逐位对照 | 13 档长度 × 3 种子，以及 66 个短序列/回退正例：全部输入和输出 SHA256 相等；每例重复 5 次一致 |
| 构建一致性 | 安装后核验库、29 个 kernel 和 Python wrapper；最终非语义清理重新编译后 29 个 kernel SHA256 不变 |
| CI | 160/160 单元测试；shell pipeline 在隔离真实 SDK 初始化的测试环境通过，原始 SDK 初始化失败记录保留 |
| 测量边界 | 随机输入、前向缓存和 bias view 在 init；`__call__` 仅反向调用 |

| 分支 | accuracy id | MSS id |
| --- | --- | --- |
| dense，1/15/16/63/64/65/127/128/129 边界 | 0–26 | 0/12/24 |
| H=2/16/48/96、B=2 | 27–41 | 39 |
| FP16、beta FP32、无 bias | 42–50 | 42/45/48 |
| packed 对齐 / 短尾、奇数 head | 54–56、66–71 | 54/66/69 |
| 强信号 | 60–65 | — |
| unsafe 公开入口拒绝 | 51–53、57–59 | 不作为执行型 MSS 用例 |
| 多 round / 多 fold chunk | 性能 0/4 | 72/73：T=1024/4097 |

| 实际 key / BlockDim | 覆盖 | LaunchKernel 运行记录 |
| --- | --- | --- |
| Main 1/24，Gate 1/48 | dense、FP16、beta FP32 | id 0/42/45/24 |
| Main 2/24，GateVarlen 1/32 | packed 对齐 | id 54 |
| Main 1/24，Gate 1/4 | 奇数 H=3，由 wrapper 扩展为 H=4 | id 66 |
| Main 1/24，Gate 1/48 | packed 短尾拆成 dense | id 69 |
| Main 3/4/7/8 | unsafe 入口拒绝 | 6 个异常例匹配 |
| Main 5/6 | V256：当前 Kernel A/C host 限制 V128，不声明支持 | 主线直接调用返回 561103 |

## 性能

每档预热 10 次、采样 50 次。Task：按启动顺序配对 `ChunkKdaBwd` + `KdaGateBwdPost` 的 `Task Duration(us)`，求和后取 p50；13 档比值的几何平均耗时下降 **27.17%**，全部快于基线。首次 20 次采样同口径为 27.17%。

Event：完整调用，三个种子 p50 的均值，几何平均下降 **25.05%**；反序复验的单种子 13 档为 **24.31%**。数据为单算子收益。

| case | T | 基线 Task ms | 优化 Task ms | Task 耗时下降 | 完整调用耗时下降 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1024 | 1.2531 | 0.9274 | 26.00% | 25.58% |
| 1 | 2048 | 2.4569 | 1.7988 | 26.78% | 27.16% |
| 2 | 4095 | 4.7692 | 3.4538 | 27.58% | 21.52% |
| 3 | 4096 | 4.7781 | 3.4186 | 28.45% | 27.00% |
| 4 | 4097 | 4.8828 | 3.5152 | 28.01% | 22.05% |
| 5 | 8191 | 9.4977 | 6.9643 | 26.67% | 21.30% |
| 6 | 8192 | 9.4147 | 6.8183 | 27.58% | 27.47% |
| 7 | 8193 | 9.5554 | 6.9036 | 27.75% | 21.97% |
| 8 | 16384 | 18.7869 | 13.6416 | 27.39% | 26.80% |
| 9 | 16385 | 18.8286 | 13.8434 | 26.48% | 21.44% |
| 10 | 32768 | 37.2289 | 27.1293 | 27.13% | 27.12% |
| 11 | 65536 | 74.6896 | 54.4463 | 27.10% | 27.65% |
| 12 | 131072 | 148.9591 | 109.7653 | 26.31% | 27.96% |

```bash
bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu --ops=chunk_kda_fwd,chunk_kda_bwd -j16
GEN_CASES_DTYPE_NUMBERS=36 bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -soc=ascend910b -scope=gen_cases
ATK_GM_INIT_MB=512 ATK_PERFORMANCE_DATA=10,50,50 \
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -soc=ascend910b -scope=all

python3 tests/atk/chunk_kda_bwd/scripts/compare_backward.py \
  --source . --role baseline --mode performance --output baseline.json
```

切换安装优化包并核验后，执行同一输入和输出的逐位对照：

```bash
python3 tests/atk/chunk_kda_bwd/scripts/compare_backward.py \
  --source . --role candidate --mode performance --output candidate.json --reference baseline.json
```
