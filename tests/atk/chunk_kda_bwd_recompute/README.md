# ChunkKdaBwdRecompute：A2 GVA 验证

## 范围与结论

2026-11-19，Ascend910B3（A2），CANN 9.1、PyTorch 2.9.1、torch_npu 2.9.1。
A2/A3 使用同一通用路径；本次只有 A2 实机证据，A3/A5/A6 没有实机验收结论。
公开入口 `fla_npu.ops.ascendc.chunk_kda_bwd_recompute`，返回 `(gk,w,u,qg,kg)`；gate 关闭时 gk=None。

## 用例与精度

- `atk_chunk_kda_bwd_recompute.json`：ATK 正式生成并固化的 200 条，G=1..16，104 定长、96 packed 变长；四种 g/beta dtype 组合；最大 T=1025、HV=48。
- 基础网格128条，另72条覆盖 gate、exp/exp2、bias、lower_bound、tail、跨head组与页回绕。每条固定seed。
- `reference.py` 为唯一可编辑的 CPU 数学和输入生成源码；工作流通过符号链接引用同一文件。
- 原输入值域与提供的混合容差保持一致：max_re_ratio=5，avg_re_ratio=1.5，root_mean_squared_ratio=1.5。
- 绝对误差补充检查：gk/qg/kg <0.05，w/u <0.15；实际数据无NaN/Inf。
- 最终公开入口：200/200 精度通过，每条3次结果逐位相同。ATK full：200通过、0失败；ATK确定性：20通过、0失败，每条3次。
- 补充正向：DIRECT L=1、非连续输入、NTD、G=17兼容、全空packed。非法head比例、布局、chunk size、元数据配对/逆序/末T/重复/缺失，以及gate辅助输入另有定向拒绝检查。

## TilingKey 与内存

HEAD_MODE 是独立模板轴；G=1 为 DIRECT，G>1 为 GVA。CG 仍是运行时参数，不额外扩展key。

| g | beta | DIRECT | GVA |
|---|---|---:|---:|
| bf16 | bf16 | 2570 | 133642 |
| bf16 | fp32 | 7690 | 138762 |
| fp32 | bf16 | 2590 | 133662 |
| fp32 | fp32 | 7710 | 138782 |

真实key由CANN模板编码器输出并与构建产物交叉核对；4个fat object包含8个逻辑模板。
`atk_chunk_kda_bwd_recompute_mss.json` 为20条专用用例，覆盖全部8个模板、CG1..4、packed尾块和双页回绕。
独立ACL运行器使用逐个aclrtMalloc缓冲区完成memcheck 20条，0 ERROR、0结构化WARNING。Python直调运行器曾报告GM越界WARNING，不能计为干净通过；该工具/分配器归属差异尚未定因，原日志保留。initcheck/racecheck各20条无结构化错误或警告；synccheck保留Catlass冗余通知WARNING。公开ATK内存20条报告Pass。
开发期使用独立带 `--cce-enable-sanitizer` 的直调二进制；正式性能使用普通注册算子二进制。

每AIV UB=134400 B，L1=24 KiB，L0A/B/C=8/16/32 KiB。
用户workspace=`usedAic*2*CG*32768 B`。模型实测20 AIC、40 AIV、CG4，为5 MiB；框架统计输入/必要输出之外峰值 23071232 B，F/V相同，小于200000000 B。
这个峰值结论限定于本次contiguous、gate开启的F/V；gate关闭时L2仍有历史全量gk占位，非连续大shape的复制不纳入本次模型测量结论。

## msprof：两条典型case

| case | B/HK/HV/T | 变长边界 | 中位数ms | 5轮中位数范围ms |
|---|---|---|---:|---:|
| F | 1/16/128/11264 | 定长 | 6.233915 | 6.213494–6.236794 |
| V | 1/16/128/16384 | 16×1024参考列表 | 9.079251 | 9.075251–9.099182 |

两项均K=V=128、C64；q/k/v/A/g/beta为BF16，A_log/bias为FP32；gate、exp2和bias开启，lower_bound=-5。
A使用与原测试相同的随机标杆输入，未声明来自真实模型checkpoint。V的16×1024是明确的参考实例。
普通msprof独立采样5次，每次预热20次后取100次有效调用；统计 `op_summary` 的 `Task Duration(us)` 中位数，再取5轮中位数。
每轮确认120个目标MIX_AIC任务，排除前20个；真实key=133642，Block Num=20，Mix Block Num=40。
计时窗口无CPU golden或D2H。没有native baseline；按用户明确要求豁免数值性能目标和达标结论，不报告0.4x或加速比。

## 复现

先激活安装当前wheel的独立环境并加载CANN；CPU线程建议限定4，避免ATK多worker过度订阅。

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
GEN_CASES_DTYPE_NUMBERS=100 bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_recompute -scope=gen_cases -soc=ascend910b
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_recompute -npu_device_id=0 -scope=accuracy -soc=ascend910b
DC_LOOP_NUMS=3 bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_recompute -npu_device_id=0 -scope=determinism -soc=ascend910b
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_recompute -npu_device_id=0 -scope=mssanitizer -soc=ascend910b
msprof --application="python tests/atk/chunk_kda_bwd_recompute/scripts/profile_models.py --case F --round 0 --output profile_metadata" --output=msprof_F_0 --task-time=on
```

V将`--case F`替换为`--case V`；round分别0..4并使用各自独立输出目录。取表中任务时间，排除前20次预热。
单例范围采用半开区间，例如`ACCURACY_START=0 ACCURACY_END=1`；正式full不设置任何范围。
修改头文件后必须刷新该算子的staged source和device编译标记，核对安装后的.o哈希，不能把仅重新打包的旧二进制当新实现。

## 规则版本与实现修复

采用最新CANNBot五阶段，family=linear_attention，architecture=atlas_a2_a3；数值性能目标准入与达标判断按用户明确豁免。
来源commit：cannbot e0a73261420b21cb5804bf88419d91a4380abf58；skills e3f6d21cccf39eacc6393d23d9bc18c1b03a0d0d；knowledge ea36762a12c4b534bfb2d7f884678b0a521da45c。
本次修复了V→MTE2输入缓冲复用同步、beta广播padding初始化、L1尾块初始化及其MTE2写屏障，保留双页生产/消费及同head A驻留复用。
更多公式、地址与事件见[算子设计](../../../fla/ops/ascendc/kda/chunk_kda_bwd_recompute/docs/design.md)。

最终独立ACL内存复核及ATK内存20条通过后，另补采一轮msprof（每条20预热+100有效）：F 6.215114 ms；V 9.088762 ms。此前5轮统计保留为主结果。
