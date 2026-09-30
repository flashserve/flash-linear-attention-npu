# KDA backward 内存修复定向验证

关联 issue #760。测试日期：2026-09-28；A2 / Ascend910B3，CANN
9.1.0_910b_0605，conda wys_gdn。以 dev6b12069 wheel 的独立安装副本为基础，
应用 dense 尾块 wrapper 修复并替换从最终源码编译的目标 kernel。

## 修复范围

- state_scan 零初始化补齐 MTE3 → MTE2 → Vector 的缓冲复用依赖。
- WY 三处位掩码使用两个 uint64_t，高位显式置零，避免 API 越界读取掩码。
- WY/Intra FP32 Store 在 Adds 后等待 PIPE_V，保护调用方复用的源 UB。
- WY MMAD 阶段结束增加 M_FIX 反向同步，确保 FIX_M 消费完成后再重发。

公开接口、数学公式和内存布局不变。同步设计见 [design.md](design.md)。

## 最终版本内存与精度

使用测试提供的 atk_chunk_kda_bwd(1).json 中 case 0、1、2、3、5、12。
覆盖 T=7/15/31/63/65/257、FP16/BF16 输入、FP32/BF16 beta、gate 开关及 bias；
均为 K=V=128、chunk_size=64、safe_gate=true 的 dense 路径。
JSON 运行副本仅将 case_spec.dtype 从 non_param 改为 string 以适配 ATK 传输。

四种输入/beta dtype 组合均从最终源码编译，使用 -g -sanitizer、auto-sync=off，
只编译 tiling key 1。分别通过 mssanitizer --tool=<tool> 执行六例脚本：

| 检测 | 有效目标 kernel 次数 | ERROR | WARNING |
|---|---:|---:|---:|
| racecheck | 6 | 0 | 0 |
| memcheck | 6 | 0 | 1284 |
| initcheck | 6 | 0 | 0 |
| synccheck | 6 | 0 | 0 |

每轮核对实际加载包路径、六次 Start sanitizer / Sanitizer finished 和用例完成标记；
跳过检测或设备初始化失败的轮次不计入结果。首例额外检查 block 1、3，racecheck 无错误。
memcheck 仍报告跨核 GM ownership 的 out of bounds warning，融合阶段的 GM 区间
由不同核接续使用；本轮 racecheck 无对应竞争错误。此结果不代表所有 warning 已消除。
同一最终包的六例 ATK 双标杆精度为 6/6 Pass，未调整精度阈值。

此前仅含 wrapper 修复的版本完成过 200/200 精度及 200/200 确定性验证
（每例 50 次）；该结果不作为最终 kernel 修复的全量回归结果。

## 非插桩性能

B=1、H=64、T=4096、K=V=128，BF16/BNSD，beta/gk 为 FP32，chunk_size=64，
safe_gate=true、disable_recompute=true、use_gate_in_kernel=false、use_exp2=true。
使用同一编译选项的非插桩 before/after 包，设备 5，按 before/after/after/before
采集四轮，每轮预热 10 次、统计 50 次。仅统计 msprof op_summary 中 ChunkKdaBwd。

| 指标 | 修复前均值 (us) | 修复后均值 (us) | 变化 |
|---|---:|---:|---:|
| aicore_time | 9326.633 | 9234.147 | -0.99% |
| Task Duration | 9536.232 | 9445.199 | -0.95% |

该用例未观察到性能劣化；不将约 1% 差异解释为普遍加速。
最终非插桩目标二进制 SHA256：
fe825832f7b7af0b9b173e777d4e72213e1daa289937c139093c283e13bfb22d。

## 未覆盖

最终 kernel 版本尚未重跑全部 200 条；varlen、V256、safe_gate=false、其他 SoC、
完整 OPP/wheel 构建与整体 Example ST 未在本轮验证。A2/A5 正式 CI 和审批仍待完成。
以上为指定 wheel 副本上的定向验证，不等同于两个 PR 分支各自完整构建验收。
