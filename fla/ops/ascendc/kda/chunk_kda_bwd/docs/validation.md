# A2/A3 KDA 通用反向开发验证

日期：2026-11-25。分支 `feat/fla-compat`，基线 `fc0bc08ec64e`；本记录对应本 PR 的功能验证快照。
设计见 [design.md](design.md)，接口见 [api.md](api.md)。按用户要求采用 fla-org 用例和标杆，不使用本地 ATK。

## 实现与逐阶段证据

| 阶段 | 实现 | 验证 |
| --- | --- | --- |
| Recompute | 读取 FP32 qhat/khat、有效 beta、前向 Akk，恢复 FP32 w/u/qg/kg/G | direct 五个前向用例缓存逐元素一致；独立公开接口重算缓存一致 |
| BwdPrepare | 实际 K/V、FP32 dAqk/dvLocal/dqRaw；两种状态布局 | 上游实际 dO 的阶段标杆；独立公开接口最终梯度 |
| Dhu | 每序列/head/V32 owner，跨完整 K 归约；dh 保存更新前 Dnext | 阶段矩阵公式、T65 尾块快照、两种物理布局 |
| Finalize | 完整手写反向、beta 三分支、稳定 gate 导数、norm 开关、FP32 scratch | 上游原断言与独立 recurrent/autograd 标杆 |
| 参数归并 | 每 chunk 私有 partial，独立确定顺序 reduce launch | dA/dbias、packed 空段；修复可选输出缺省导致的 launch 参数移位 |
| 公开组合 | 保存/重算都经 ACLNN 串联；原 BF16/128 热路径保留 | 8 个原始上游训练用例及扩展矩阵 |
| 独立适配层 | 按真实 K/V、缓存 dtype 分配；FP32 dh 可为 V-first | 独立阶段 4 个 FP16/BF16、K-first/V-first 用例 |

新路径为 D1.3 的纯 Vector FP32 功能实现。Dhu UB 预算 106496 B，Finalize 预算 92160 B（launch 元数据队列另加 32 B）；
全局 scratch 为 `B*H*T*(4*K+1)*4` 字节，另加私有参数 partial。此轮不宣称性能优化或 sanitizer 验收通过。

## 环境与产物

- A3：`独立开发服务器`，`Ascend910_9362`，编译目标 `ascend910_93`；物理 device 10。
- CANN 9.3.0.P009，Python 3.12，torch/torch_npu 2.10，任务独立 venv；未系统安装。
- 编译 `-j640`；完整前反向 OPP 编包成功，11 个 stable ACLNN 调用签名检查 0 mismatch。
- 上游 `fla-org/flash-linear-attention` 固定 commit `c55dc553edf44701bf703715ed052d811f8a6376`。
- 远端任务根 `<task-root>`；构建日志 `backward-public-build4.log`、增量 host 修复 `backward-public-build10.log`；最终 wheel 日志 `backward-wheel-final4.log`。
- 最终 wheel SHA256：`5926fa0135566fb0699f63d504ae03a7009fdb6d6b9e35e3d9b3c25a9f33b3d4`。
- 最终 OPP SHA256：`3c3e5a619c24896956f909ce40a7eff944693d3644574c60700ed1eb9e77623a`。
- 实际加载 overlay `30459cf0c554494e80bfe435fec0bd6c`，op_api MD5 `e7cadc9390f68057e8c7a0207adff369`；61 个 kernel .o 和 Python wrapper 与本次构建/源码一致。
  详见 `backward-provenance-delivery.log`。旧校验脚本不识别 wheel_layers，因此用 runtime_plan 的真实优先 binding 作为其校验目标，哈希比对逻辑不变。

## 最终公开入口矩阵

| 测试组 | 数量 | 参数与断言来源 |
| --- | ---: | --- |
| 原始 collinear + fused_training_contract | 8 | fla-org 原参数、原测试体、原断言，插件仅替换受测入口 |
| expanded | 20 | 相同上游函数/断言；D60/64/100/128/256 × FP16/BF16 × 保存/重算，H2 |
| nonsquare | 24 | 上游 recurrent/autograd 标杆；K64/V32、scale=.37、packed `[0,0,1,66,66,129]`、norm 二分支、beta 三分支、两 dtype/两模式 |
| dense | 8 | 上游标杆；B2/T65、K64/V32、预激活 FP32 gate、非零 beta、两 dtype/两模式/norm 二分支 |
| stages | 4 | 独立公开接口，FP16/BF16 × K-first/V-first；缓存一致和最终梯度 |
| invalid | 6 | ACLNN 161002：NaN scale、C32、beta shape、h0、sigmoid 组合、v dtype |

A3 最终统一复跑：**8 + 62 = 70 项全部通过**。

最后统一复跑日志：`public-backward-delivery-original.log`、`public-backward-delivery-extended.log`。
测试脚本及命令见 [tests/upstream/README.md](../tests/upstream/README.md)。扩展参数集与新增边界例单独计数，不冒充上游原始用例。

## 未覆盖与审查状态

- A2：在 A3 主机隔离目录 `source-a2-cross` 以 `--soc=ascend910b -j640` 交叉编译全部受影响 kernel 并生成 OPP 成功，日志 `backward-a2-cross-build2.log`。OPP SHA256 `815bbe99cabd63e934c9aee8cbbeec458be2cfe453d23f7201812a17f1d051c5`。首次隔离复制遗漏 cmake/third_party/build 源目录，补齐后正常编译；未修改算子以绕过编译。指定 `a2-server` 两次 TCP 22 超时，尚未 A2 实机验收；没有用 A3 结果替代 A2 精度结论。
- 用户明确排除组合 h0/dht/dh0；上游携带非零状态和末态梯度的 test_chunk/test_chunk_varlen 不运行，也不改断言。
- 不含 A5、chunk32、GVA、Graph 与 FP32 模型输入；未执行性能/sanitizer 验收。
- `git diff --check` 通过；测试脚本语法编译通过。
- 仓内静态 review 扫描的是受影响算子全文，初扫报 217 error/9 warning（192 为 WITHOUT_REPORT，另有 launcher/文档/测试目录等规则）。
  按当时 diff 行号交集复查，新增命中为归并 launcher 缺结构化报错和尚未落盘的 validation 链接，均已修正；全文历史报告未逐项清零，不能宣称四层审查门禁全部通过。新增 general 文件未被该脚本报告；这不替代人工审查。
- 当前保留开发期 validation 记录；A2 与审查门禁未闭合，未标记工作流正式验收完成。
