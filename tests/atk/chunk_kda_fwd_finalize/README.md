# chunk_kda_fwd_finalize ATK 测试

本目录只依赖八个公开输入：

```text
qg_scaled, aqk, v_new, h, cu_seqlens, chunk_indices,
output_layout, state_v_first
```

冻结 JSON 是 ATK 原生直接输入；没有 marker tensor、序列化的
`case_spec` 或隐藏测试开关。`qg_scaled/Aqk` 使用 Prepare 的
head-major BF16 形状，packed 时 rank-3；`v_new/h` 使用 FwdH 的
rank-4/rank-5 BF16 形状，h 为 NT-first `[B,C,HV,128,128]`，packed 时首维仍为 1。`Aqk`
已经乘过 scale，CPU 和 NPU 都不再重复缩放。

## 资产

| 文件 | 用例 | 覆盖 |
| --- | ---: | --- |
| `atk_chunk_kda_fwd_finalize.json` | 200 | 保留 main 的 25 个边界/shape/变长结构 × 4 个输出 layout × 2 种 state 轴顺序，仅更新 h shape |
| `atk_chunk_kda_fwd_finalize_perf.json` | 10 | 模型大 shape、dense/varlen，供单算 profiling |
| `atk_chunk_kda_fwd_finalize_mss.json` | 12 | 4 个输出 layout × 2 种 state 轴顺序的尾块输入，以及 AIV 搬运模板的 dense/packed、KV/VK 确定性输入 |

精度包括 T 为 `1/15/16/17/31/32/33/63/64/65` 的边界、
多 chunk、严格递增变长序列、显式/自动 canonical chunk indices、
不同 HV、`B=2` dense，以及 `B=384,HV=5,T=17` 和由 382 个单
token 序列加 33/64 行序列组成的 AIV 搬运模板 dense/varlen 边界。这些
用例同时覆盖不足 4 个 head 的尾组、显式 `chunk_indices` 及
`Aqk` 的 1/2/3/4 个 16-row 数据块。
packed 组中 `qg_scaled/Aqk` 是 rank-3，
`v_new` 是 rank-4。独立 Finalize 无 HK 输入，不把这些 case
记作 GVA 映射验证；需要在三阶段组合链路另测 HK<HV。

## Tiling 模板覆盖

| `USE_AIV_INPUT_MOVER` | 选择条件 | 精度普通用例 | 精度边界用例 | `_mss.json` 用例 | 适用 SoC | 预期核类型 |
| --- | --- | --- | --- | --- | --- | --- |
| `false` | A2/A3；或 A5 的 head 被拆分；或最小单核负载不足 8 个 chunk | 64-71 | 0-63、72-87、92-191 | 0-7 | A2/A3/A5 | AIC-only |
| `true` | A5、每个 work item 包含完整 value head，且最小单核负载至少 8 个 chunk | 192-199 | 88-91、192-199 | 8-11 | A5 | MIX AIC 1:2 |

case 88-91 是 dense 的 17 行尾块；case 192-199 在同一变长输入中
包含 1/33/64 行序列并覆盖四种 layout。两组均使用非 4 倍数 HV，
因此会进入第二个 head group 并验证不足 4 个 head 的尾组。上述
模板组合同时通过 `ASCENDC_TPL_KERNEL_TYPE_SEL` 声明核类型，不在
Kernel 内按数值 TilingKey 分支。NT-first 修改前的历史 runtime profile 中，性能
case 2 命中 `_0`/cube 实例，耗时 908.545 us；性能 case 3 命中
`_1_mix_aic`/mix 实例，耗时 1794.889 us，AIC/AIV block 为
28/56。性能结论以性能 JSON 对应的大 shape profiling 为准。

CPU golden 把四个直接输入恢复为 BF16 后，使用 FP64 两项矩阵乘与求和，
不提前舍入为 BF16；executor 最后将 FP64 结果转为 FP32 比较载荷，
匹配 ATK 原生 `mixed_tolerance_bm` 支持的 BF16 DUT / FP32 golden 类型对。
`run_cpu` 自身仍返回 FP64，供独立高精度检查。
NPU 节点用本 executor 的窄 aclnn
直调适配器验证设备实现。公开
`fla_npu.ops.ascendc.chunk_kda_fwd_finalize` 稳定入口由共享 Python
wrapper 提供，其接口验证不属于本 ATK 目录。

## 生成与静态核对

```bash
python3 tests/atk/chunk_kda_fwd_finalize/gen_chunk_kda_fwd_finalize.py \
  --output-dir tests/atk/chunk_kda_fwd_finalize --summary

cd tests/atk/chunk_kda_fwd_finalize
atk case -f ./chunk_kda_fwd_finalize.yaml \
  -p ./gen_chunk_kda_fwd_finalize.py -en 0 -s 20260914
```

`dtype_numbers: 200` 与 `shape_distributions: [[0,1.0]]` 在目标 ATK
生成 200 条；生成器覆盖 YAML 临时形状，并在导出前检查八个输入
顺序。冻结 JSON 不因 `atk case` 的物理 case ID 重新排序而覆盖。

```bash
python3 -m py_compile \
  tests/atk/chunk_kda_fwd_finalize/gen_chunk_kda_fwd_finalize.py \
  tests/atk/chunk_kda_fwd_finalize/executor_chunk_kda_fwd_finalize.py
```

## 精度

加载与待测算子配套的 custom OPP、ATK 和 Python runtime 后：

```bash
cd tests/atk/chunk_kda_fwd_finalize
atk node --backend npu --devices 0 -o ./atk_output/accuracy \
  node --backend cpu task \
  -c ./atk_chunk_kda_fwd_finalize.json \
  --task accuracy -p ./executor_chunk_kda_fwd_finalize.py \
  -sp -to 60
```

单条超过 60 秒判定超时。只有最终报告确认总任务 200、执行失败
0、精度结论通过，才能声称全量通过；不能仅依据 shell 退出码。
任意 case 未通过时先使用 `--save_data output` 保留实值，再定位
shape/索引/尾块问题或执行精度复检，不能改输入 range 或阈值掩盖失败。

### GPU 双标杆

executor 除 NPU DUT 与 CPU 双标杆外，还支持把参考节点放到远端 GPU
（与 `pre_process_fwd_kernel_merged` 同范式，按 `is_benchmark_task`
区分角色，不按算子名分支）：

| 角色 | 触发条件 | 执行内容 |
| --- | --- | --- |
| DUT | `device ∈ {npu, pyaclnn}` | 本 executor 的窄 aclnn 直调适配器（不变） |
| 高精度 golden | `device = gpu` 且 `is_benchmark_task=True` | `scripts/gpu.py` 的 FP64 torch 真值（镜像 `run_cpu` 数学，设备跟随输入；输出转 FP32 载荷进 `gpu_benchmark` 目录，供 `mixed_tolerance_bm` 的 BF16 DUT / FP32 golden 对比较） |
| 同精度标杆 | `device = gpu` 且 `is_benchmark_task=False` | `scripts/gpu.py` 的真实上游 `chunk_gla_fwd_o_gk`（默认）；BF16 输入 / FP32 累加 / BF16 输出；任何执行失败直接报错，不自动切换实现 |

GPU control 默认 `upstream_strict`。GPU **性能任务强制使用真实上游 `chunk_gla_fwd_o_gk`**，覆盖服务进程中残留的 `contract` 配置；不会用 FP64 golden 或改写 kernel 替代性能标杆。混合 accuracy/performance 的独立 golden pass 仍按 ATK 调度执行 FP64 精度参考。纯精度/调试任务可在 GPU ATK 服务进程启动前设置 `KDA_FINALIZE_GPU_REFERENCE`（仅在 NPU 客户端设置不保证传给远端服务）：

- `contract`（仅显式精度/调试模式）：改写 Triton kernel，消费完整 Aqk，固定 BK64/BV128/4warps/3stages。保留完整矩阵接口；本 executor 初始化时会统一清零 Aqk 上三角，结果标记 `gpu_contract_adapted_triton`；不能称为原版 GPU 算子性能。
- `upstream_strict`（默认，性能任务固定）：直接导入 GPU 环境中 `fla.ops.gla.chunk.chunk_gla_fwd_o_gk`，解开 dispatch 装饰器调用其原始 Triton 实现，保留原版 autotune。传 `q=qg_scaled`、FP32 `g=0`、`scale=1`、`A=aqk`、`v=v_new` 和原 h；结果标记 `gpu_upstream_chunk_gla_fwd_o_gk`，记录真实源码路径和 SHA256。

上游 kernel 使用下三角。现在 `FunctionApi.init_by_input_data` 在所有 CPU/GPU/NPU 角色上统一原位处理 Aqk：按实际 `cu_seqlens` 的每段序列、每个 64 行 chunk 重置局部行号，清零 `j>r`，包括尾块无效列；保留对角线、下三角及其余输入原值。处理在 GPU control 准备前完成，设备上只在初始化同步一次，`__call__` 不执行 mask。继续使用原有 JSON 和生成器即可，不增加用例。GPU `upstream_strict` 的只读检查保留，用于防止无效输入进入上游。

ATK 的随机生成规则不变，各角色初始化时通过同一 executor 得到相同下三角数据，结果元数据记录 `aqk_structure=chunk_lower_triangular`。这改变了本 executor 的实际测试输入域；历史完整随机矩阵的精度/性能记录不能直接视作本次三角输入结果，后续对照须双方使用同一新版 executor 重新运行。

```bash
export GPU_HOST=<gpu-server-ip>  GPU_HOST_PORT=9090
cd tests/atk/chunk_kda_fwd_finalize
atk node --name npu_dut --backend npu --devices 0 --output_path ./atk_output/gpu_dual \
  node --name gpu_reference --backend gpu --host "$GPU_HOST" --port "$GPU_HOST_PORT" \
       --devices 0 --is_compare true --output_path ./atk_output/gpu_dual \
  task -c ./atk_chunk_kda_fwd_finalize.json --task accuracy --bm_device gpu \
       -p ./executor_chunk_kda_fwd_finalize.py --syc_dataset -mt 1 -to 14400
```

GPU server 需要 CUDA Torch、Triton、ATK 和完整测试资产（包括 `scripts/gpu.py`）；`upstream_strict` 还需在同一 Python 环境安装原版 FLA，核对 `export_custom_data` 中源文件 SHA。GPU 侧不依赖 CANN 或 NPU wheel。golden 仍走独立 FP64 真值，精度标准不变；GPU/NPU 不预设逐位一致。

control 的 head-major→token-major、h 连续化、metadata 和零 g 在 `init_by_input_data` 中准备一次。每次调用包含输出 zeros、目标 matmul kernel 和所需输出布局转换，**不是单 kernel 计时**。BSND/TND 通常可直接返回 token-major，BNSD/NTD 需要输出布局搬运。重复调用假设 ATK 本例输入固定；变更输入应重新初始化，不能复用旧 control。

原版参考来自 `/data/huangjunzhe/GDN/github/alvazu/flash-linear-attention`，审计 HEAD `590d9305a92d1518a030a4d26491d4814b6f4de1`：KDA `chunk_fwd.py:122` → GLA `chunk.py:990` → `chunk_gla_fwd_kernel_o:363`。真实 GPU 准备好后可先运行新增因果自测：

```bash
python scripts/gpu.py --reference upstream_strict --causal-aqk
```

该自测只打印误差，正式精度仍以 ATK 双标杆及原混合容差为准。`TRITON_INTERPRET=1 python scripts/gpu.py` 仅用 FP32 验证 contract 的寻址/mask/varlen/布局，不能作为 CUDA BF16 验收。

### GPU / NPU 性能比较口径

双标杆用于精度：FP64 golden、同精度 GPU control、NPU DUT 三路；FP64 的耗时不作性能标杆。GPU 与 NPU 之间应固定相同输入字节、shape、layout、state、有效 chunk、元数据和输出边界，再分别测量：

| 口径 | GPU | NPU | 限制 |
| --- | --- | --- | --- |
| finalize 目标 kernel | 真实 `chunk_gla_fwd_o_gk` 启动的 `chunk_gla_fwd_kernel_o` | 正常包 `ChunkKdaFwdFinalize` 目标 kernel | GPU 原版保留 g 读取与 exp2 路径；scale1 的乘法是否消除以生成代码为准，不宣称两者指令范围完全相同 |
| 准备后 control 调用 | zeros＋目标 kernel＋输出布局转换 | 同起止边界的 NPU 调用 | 输入预布局/metadata 准备均单列；不能拿 GPU 完整调用对 NPU 单 kernel |
| 完整接口或业务链路 | 从相同原始输入开始，包含全部准备、分配、layout 与输出 | 同一输入/输出边界的 NPU 链路 | 原生 GPU q/g/scale 路径的门控工作在 NPU 位于 Prepare；应统计对应上游步骤，完整 KDA 则比较完整 KDA |

性能使用 ATK 发起对应设备任务，GPU profiler 的目标 kernel duration 与 NPU msopprof 的目标 duration 对比；完整调用用双方一致的设备计时方式另报。GPU `performance_device` 的 `device_perf` 是采集范围内所有 CUDA 活动时间之和除以调用次数，并非指定 Triton kernel；必须保留 profile，从 trace 中筛选目标 kernel 的每次 duration。原版目标名含 `chunk_gla_fwd_kernel_o`；若性能 trace 出现 `_chunk_kda_fwd_finalize_kernel`，判为错误标杆并拒绝该结果。正式10次调用必须对应10个目标 kernel 事件且 duration 均大于0；工具未找到统计项时可能留下0，不能将0当有效性能。GPU 路由使用全部采样的均值，不应把 `performance_data` 的第三项解释成 GPU 尾 n 次统计。

下面直接对原有性能 JSON 使用真实 `chunk_gla_fwd_o_gk`（Aqk 在 executor 初始化时统一清零上三角）。GPU 环境须安装原版 FLA；初始化会记录实际 `fla/ops/gla/chunk.py` 路径和 SHA256，导入失败直接报错：

```bash
atk node --name gpu_reference --backend gpu --host "$GPU_HOST" --port "$GPU_HOST_PORT" \
    --devices 0 --output_path ./atk_output/gpu_upstream_perf \
  task -c ./atk_chunk_kda_fwd_finalize_perf.json \
    -p ./executor_chunk_kda_fwd_finalize.py --task performance_device \
    --performance_data 3,10,10 --save_data profile --disable_id_seed -to 2000
```

ATK 的汇总 `device_perf` 包含 wrapper 自身的输出清零和可能的布局搬运。核心两项矩阵乘由真实上游单 kernel 完成，独立 FP64 torch 仅供精度 golden。每个 case 从其本次 GPU trace 单独提取目标核：

```bash
python scripts/check_kernel_profile.py --device gpu \
  --input <case_profile.pt.trace.json> --expected-count 10 --output gpu_kernel.json
python scripts/check_kernel_profile.py --device npu \
  --input <case_op_summary.csv> --expected-count 10 --output npu_kernel.json
```

校验器只接受明确的目标核和微秒 duration，排除 zero/layout 等辅助活动；缺目标、样本数不符、非正/NaN/Inf 耗时或 GPU contract kernel 均返回非零、`passed=false`，不输出 0 μs 成绩。它不会修改 ATK 原汇总表，正式对比使用校验通过 JSON 的 `median` 与完整 `all_samples`，两端采样数/输入/统计范围须一致。

若 NPU 日志出现 `CANNTimelineParser/RelationParser/... run failed`、`op_summary is null` 和 `NPU AVG Profiler Time:0 us`，该 case 为采数失败，不能计算 GPU/NPU 比值。这是 NPU profiler 的采集/解析链，不能仅凭日志归因于 GPU 算子或精度。`AICORE_SHAPE`、Pydantic 序列化 warning 与 `Stop profiler ... RECORD` 本身不足以定位根因；已有成功采样同样包含 `Stop RECORD` 警告。保留本次 `*_ascend_pt` 原始目录，先查看其中 `logs/` 的首个异常和 CANN 导出结果，再核对实际 ATK/Torch/torch_npu/CANN 版本、模块路径、磁盘空间与目录权限。

可在失败任务的同一 NPU Python 环境重新离线解析已有目录以取得详细异常（这不重新执行算子）：

```bash
python -c 'import sys; from torch_npu.profiler.profiler import analyse; analyse(sys.argv[1])' \
  <failed_case_ascend_pt_directory>
```

离线解析仍需用校验器确认实际目标样本；Python 退出 0 不能代替文件检查。若需要先绕开 ATK 的 torch_npu profiler 排查 CANN 直接采数，可在 NPU 本机只做一条目标核 smoke：

```bash
msopprof \
  --application="atk node --backend npu --devices <ID> task -c ./atk_chunk_kda_fwd_finalize_perf.json --task run -p ./executor_chunk_kda_fwd_finalize.py --disable_id_seed -s 1 -e 2 -sp -to 2000" \
  --output=<new_empty_output_directory> --kernel-name='ChunkKdaFwdFinalize*' \
  --launch-count=1 --warm-up=0 --aic-metrics=BasicInfo --kill=off
```

此命令是独立诊断，不是正式 10 样本对比；`msopprof` 结果不冒充 ATK `performance_device` 结果。远端调度命令不要加 `-sp`。解析失败的原始环境尚未接入时，只能确定本例无有效性能，不能宣称已修复其 CANN/torch_npu 环境问题。

先通过真实 CUDA 双标杆，再完成编译/autotune 预热，正式样本排除冷启动；至少保留与既有 NPU 记录相同的10次样本，逐例报告中位数及 min/max。记录 GPU 型号、A2/A3 型号、Torch/Triton/CUDA/CANN、源码 SHA、调优配置和 cache 策略。比值统一 `GPU_us / NPU_us`。上游默认 `FLA_CACHE_MODE=disabled` 不代表禁用 Triton autotune。

2026-10-16 CPU 适配回归10项通过，覆盖四布局×两state×dense/explicit/cu-scan的24组合、真实上游 Python wrapper body 的AST参数映射、因果域检查、尾padding、异常透传、ATK仅初始化准备一次及原CPU/NPU函数AST保持。CPU模拟launcher没有执行实际Triton kernel，也没有导入上游fla。最终文件对应证据在 `/data/huangjunzhe/GDN/work/kda_finalize_gpu_baseline_20261016/cpu_adapter_validation_v3.json`；首轮测试夹具错误及修订记录保留。

检查时远程 192.168.9.234 为 A2 环境，无 CUDA GPU；尚未获得 GPU ATK host/port，故本轮没有真实 GPU BF16 精度或性能数据，也没有 GPU/NPU 加速比。此前 racecheck 的两例 FAIL 仍须单独修复验证，性能结果不能替代质量门禁。

## 性能与确定性

大 shape 只跑 NPU 单算。先核对目标 case name，再在单条 ATK
`run` 外包 `msopprof`；性能只读取目标 kernel 的 duration，不使用
Python wall time。例如性能集第 2 条（从 0 开始）：

```bash
msopprof \
  --application="atk node --backend npu --devices 0 task -c ./atk_chunk_kda_fwd_finalize_perf.json --task run -p ./executor_chunk_kda_fwd_finalize.py -s 2 -e 3 -sp -to 60" \
  --output=./atk_output/profile_case_2 \
  --aic-metrics=BasicInfo --launch-count=1 --warm-up=0 --kill=off
```

确定性使用 `_mss.json` 的十二条输入，覆盖八种输出布局/状态组合和
AIV 搬运模板的 dense/packed、KV/VK 四条分支，精确逐元素比较实际
可见结果；内存检查须先构建 sanitizer 对象，并确认运行时真正加载，
仅有 `IsFinite` 等辅助 kernel 的 sanitizer 记录不能视为 Finalize 通过。

```bash
atk node --backend npu --devices 0 task \
  -c ./atk_chunk_kda_fwd_finalize_mss.json \
  --task accuracy_dc -p ./executor_chunk_kda_fwd_finalize.py \
  -s 0 -e 12 -sp -to 60
```

## 2026-10-12 A2/A3 单 Stage 0 优化验证

基线为 `bead79ae45fca5d10a8414b215d4f762cc66538d`。只改变 A2/A3 的
chunk/head 分核：最终固定 `CG=1`，`Nwork=Nbase*HV`，grid-stride 遍历。
四项输入仍为 BF16，两项 MMAD 先初始化、再原位累加到同一 FP32 L0C，
一次 Fixpipe `F322BF16` 直接写 GM。无 Vector stage、UB 或 GM relay；
详见[设计](../../../fla/ops/ascendc/kda/chunk_kda_fwd_finalize/docs/design.md)。
本次 A2 全部精度 case 0–199 与 `_mss.json` 0–11 均使用
`USE_AIV_INPUT_MOVER=false`；前述 MIX 模板覆盖属于 A5。

### 历史单槽 CG1 消融：CG 选择

实机为 `root@192.168.9.234` 的 device 1，Ascend910B3，20 AIC/40 AIV、
192 MiB L2，CANN 9.1.0、Python 3.10、torch 2.12.0+cu130、
torch_npu 2.12.0、ATK 26.8.8。所有候选在独立
`/data/huangjunzhe/GDN/work/kda_finalize_env` 中轮换安装同一算子的正常包，
trace/sanitizer 关闭。冻结性能 JSON 不变，`--disable_id_seed` 使用各 case
的 `default_seed`，`ATK_GM_INIT_MODE=off`，warm-up 3 次、采样 10 次。
以下仅统计目标 `aclnnChunkKdaFwdFinalize` 的 profiler
`Task Duration(us)` 中位数，不计 CPU、Python、安装和编译时间。

| case | B/HV/T、场景 | 基线 µs | CG4 µs | CG3 µs | CG2 µs | CG1 µs | CG1 耗时降低 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1/16/11264 dense | 247.6750 | 251.8950 | 280.4460 | 233.9940 | 226.7845 | 8.43% |
| 1 | 1/32/11264 dense | 527.8010 | 513.1205 | 514.4210 | 513.0300 | 512.8505 | 2.83% |
| 2 | 1/96/8192 dense | 1195.8640 | 1117.6730 | 1116.8925 | 1121.5620 | 1118.2925 | 6.49% |
| 3 | 1/96/16384 varlen | 2263.3850 | 2215.4240 | 2206.4340 | 2216.3240 | 2205.9340 | 2.54% |
| 4 | 2/16/11264 dense | 529.2210 | 513.1905 | 558.8615 | 511.5900 | 513.2110 | 3.03% |
| 5 | 4/96/128 dense | 36.6005 | 37.7710 | 36.7110 | 36.9210 | 36.4005 | 0.55% |
| 6 | 1/32/160 dense | 7.6100 | 8.9300 | 7.7805 | 7.8100 | 7.3100 | 3.94% |
| 7 | 6/6/1084 dense | 34.4705 | 40.1510 | 31.4310 | 30.2410 | 30.2505 | 12.24% |
| 8 | 1/8/32768 dense | 384.0375 | 378.8675 | 374.9570 | 372.8070 | 373.5980 | 2.72% |
| 9 | 1/32/65536 varlen | 3006.9400 | 2978.8595 | 3050.1605 | 2963.1995 | 2959.8590 | 1.57% |

CG1 在六个 case 最快，其余与最快候选相差 0.03%–0.32%，低于对应采样
波动；相比 CG4，它消除了 case 6/7 的约 17%/16% 退化。因此采用统一 CG1。
全部 10 个 CG1 case 各有 10 个有效样本，任务 10/10 成功；均为
`AI_CORE`、BlockDim 20、MixBlockDim 0，四项输入/输出为 `DT_BF16`，
AIV total/vector 时间均为 0。case 5/6 的小耗时收益与采样波动相近，
表格只记录本次中位数，不据此保证小 shape 的固定加速比。

复现正常性能采样：

```bash
ATK_GM_INIT_MODE=off OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
atk node --backend npu --devices 1 task \
  -c tests/atk/chunk_kda_fwd_finalize/atk_chunk_kda_fwd_finalize_perf.json \
  --task performance_device --performance_data 3,10,10 --save_data profile \
  --disable_id_seed -p tests/atk/chunk_kda_fwd_finalize/executor_chunk_kda_fwd_finalize.py \
  -s 0 -e 10 -sp -to 60
```

### 精度、同步与架构隔离

正常 CG1 包对原 200 条精度用例逐条运行三组固定种子，实际通过
600/600，执行失败 0。每条 case 的实际 seed 为
`20260914 + case_id + offset`，offset 分别为 0、100000、200000；
三组各 200 个不同 `(case_id, seed)` 组合。使用 `--disable_id_seed`，
dev4/dev5，CPU 标杆单线程、GM init 关闭；仅临时 JSON 副本的
`default_seed` 改动，冻结 shape、数据域、golden、standard 和 executor 均未修改。
原始 XLSX 的 summary、失败表及每条 `mixed_tolerance_bm` 结果分别核验，
600 条均为 True。每组 case 192–199 的 metadata cell 达到 Excel 的
32767 字符上限；可见 seed/standard 和 inputs 前缀与冻结调用 JSON 核对，
完整输入由实际调用 JSON 保留，精度结果列本身未截断。

稳定公开入口 `fla_npu.ops.ascendc.chunk_kda_fwd_finalize` 另测两条：
dense `B2/HV3/T65, BSND, KV` 和 packed `HV5/T99, TND, VK,
cu_seqlens=[0,1,34,99]`、自动 chunk indices。seed 为 20261012/20261013，
使用原生 BF16 `mixed_tolerance_bm` 对 FP64 golden，均 matched_ratio=1、
abs/ULP 失败 0，保留输入、DUT 和 golden。

独立 trace worktree/environment 在 device6 验证原有 case 88–91、192–199、
80–87，共 20/20 精度通过。每个 case 的 runtime BlockDim=20，全部
20 个 core 实际连续处理多个 work 并回绕 L0C slot；8 个 HV48/T129 case
还验证同 core 的 head 切换及 64/64/1 行 chunk。core0 的一组实际记录为：
`work=[0,20,40,60,80,100,120,140]`、
`head=[0,20,40,12,32,4,24,44]`、`slot=[0,1,0,1,0,1,0,1]`。
trace 插桩未修改正式版本的 34 个同步语句，所有事件 credit/Wait 与退出 drain 检查通过。
trace 仅存在于独立工作区，正式源码及正常性能包无 trace、额外 workspace 或同步。

A5 的 `__CCE_AICORE__=310` 预处理 token 与基线完全一致（2787 tokens，
SHA256 `3aac387caee64ec560cb6518c9cb44e7b344c6db1e70ea02f6f7b7858b3c3637`）；
Host 默认 `headGroupSize=0` 的 60,480 个参数组合与原版五项调度结果一致，
其中 18,016 个合法、42,464 个拒绝输入。新增 HostUT 用
`g++ -std=c++17 -Wall -Wextra -Werror` 编译运行通过，覆盖 CG1 dense/varlen、
小工作域、单 head、候选尾组和 uint32 work 溢出边界。A3 `ascend910_93`
单算子构建通过；本机器只有 A2，未进行 A3 实机精度、同步或性能验证。

### 构建与原始证据

证据目录：`/data/huangjunzhe/GDN/work/kda_finalize_20261012`。
最终源文件冻结后重构建 A2，全部 6 个库/kernel 二进制与 CG1 profiling 包
及当前 normalenv OPP 的 MD5/SHA256 完全一致。对应记录为
`final_production_source_manifest.json`、`final_rebuilt_a2_binary_comparison.json`、
`final_normal_runtime_verification.json`；未用旧装包或跳过 kernel 比对。
正常 A2 包 `final_normal.run` 的 SHA256 为
`f3498c42d56df9670f3f5a555e3504d4f69506215027b9197b8d2482f8e157b6`。
A3 包 `final_a3.run` 的 SHA256 为
`50956062d8018ca6ed46ce23b8bc923b8779d0d51ef1310413e128a0e360b3b9`。

`final_cg_sweep.json/.csv/.md` 保留各 case 的 shape、seed、全部采样和
50 个原始 profiler CSV/5 个 ATK XLSX 路径。基线与 CG2 早期脚本在 ATK
完成后出现尾部 shell 127，原始退出码保留；两者原始报告仍明确各 10 个
任务成功、0 失败及每 case 10 个完整采样。最终 CG1 精度/性能退出码均 0。
600 seed 证据在 `seed_sweep_runs/cg1/seed_sweep_summary.json` 和各组
`report_validation.json`；稳定入口见 `cg1_stable_api/`；同步实测见
`trace_evidence_summary.json` 与 `trace_logs/`。

### 正式四项门禁

使用仓库原始 `run_test_cpu.sh` 的对应 scope 完成全部门禁，正常包负责
accuracy/performance/determinism，独立的无 trace sanitizer 包负责
mssanitizer。两种包来自同一份冻结 CG1 正式源码。

| 门禁 | 包与设备 | 实际结果 |
| --- | --- | --- |
| accuracy | normalenv / dev4 | 200/200 执行成功、`mixed_tolerance_bm` 全部 True，0 失败 |
| performance | normalenv / dev4 | 10/10 任务成功；冻结 standard 为 `not_key`，性能收益按前述同卡 profiler 表判定 |
| determinism | normalenv / dev4 | 12/12 全部 True；每条 50 次执行、49 次 `torch.equal` 精确比较，共 600 次执行/588 次比较，无数值容差 |
| mssanitizer memcheck | trace_env 中安装无 trace sanitizer 包 / dev6 | 12/12 memory Pass、0 执行失败；目标 kernel Start 12/Finished 12/ERROR 0 |

每项均读取真实原始报告，未依赖 shell 退出码或 runner 的“未找到报告（跳过）”
提示。确定性原始报告实际位于 workdir 的
`trace_repo/tests/atk/chunk_kda_fwd_finalize/atk_output/`，12 条
`确定性计算结果` 与结果详情逐条为 True。路径、实际统计和检查证据统一见
`final_formal_validation.json`；正常流程退出码 `normal_gates.exit=0`，
内存检测退出码 `sanitizer_gates.exit=0`。

sanitizer 包以 `--bisheng_flags=ccec_g,sanitizer` 构建并确认实际安装
kernel `.o` MD5 匹配。日志明确检测 `ChunkKdaFwdFinalize_*_0`，
不是只有 `IsFinite` 等辅助 kernel。初次将 sanitizer 包直接用于未注入的
普通 precision 流程时，case0 的 fault PC 落在
`__sanitizer_report_set_ffts_base_addr+0x1c8`，触发 scalar internal-buffer
越界；这次失败记录保留在 `sanitizer_all.log.without_injection` 与
`sanitizer_plain_run_diagnosis.json/.md`。同一包在正确注入下的 case0、case25
均 1/1 memory Pass，正式 12 条全部通过；未为此修改正式 kernel。
本机 msSanitizer 的 `--log-file` 必须与 ATK `-msl` 指向同一实际文件。

复现时正常包使用三个普通 scope；sanitizer 包仅在 msSanitizer 下启动：

```bash
# 已加载正常包及其配套 OPP/runtime；CPU 标杆线程和 GM init 设置同上。
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd_finalize -soc=A2 -npu_device_id=4 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd_finalize -soc=A2 -npu_device_id=4 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd_finalize -soc=A2 -npu_device_id=4 -scope=determinism

# 已独立安装 sanitizer 包、加载其配套 OPP/runtime；不得替换正常性能测量包。
export INJ_LOG_LEVEL=0
FINALIZE_MSS_LOG=/tmp/chunk_kda_fwd_finalize_mss.log
mssanitizer --tool=memcheck --log-level=info --log-file="$FINALIZE_MSS_LOG" -- \
  atk node --backend npu --devices 6 task \
  -c tests/atk/chunk_kda_fwd_finalize/atk_chunk_kda_fwd_finalize_mss.json \
  --task run -p tests/atk/chunk_kda_fwd_finalize/executor_chunk_kda_fwd_finalize.py \
  --mssanitizer -msl "$FINALIZE_MSS_LOG" -s 0 -e 12 -sp -to 60
```

## 2026-10-13 A2/A3 CG1 两份 L1 输入预取

两槽设计基于 `a46da76275a40e8f1236ab6808d1e482f0832a43`，即上一节已经采用的 CG1。
当前性能结论以最初 `bead79ae` 原版到当前两槽正常包的同卡对照为准；
单槽 CG1 对照作为历史消融保留。只修改 A2/A3 Cube 内部：固定 CG1，L1 input slot 0/1
分别占 `[0,72)` / `[72,144)` KiB。每槽四项 BF16 输入按 Q/H/A/V 原物理布局搬入；
下一有效 head 的 MTE2 搬运提前提交，当前 head 消费本槽；两项 L1→L0 读取完成后
发布本槽 free，下代覆盖前等待。ready/free 按槽独立使用 ID0/1，首次使用不等待 free，
尾部不多预取，最后 drain 所有真实 credit。槽随本核有效消费轮转，不用全局 work 奇偶。
两项 BF16 MMAD 仍先初始化、再累加同一 FP32 L0C；一次 Fixpipe 转 BF16 直接写 GM。
Stage0、CG1 分核、host、ABI、GM 字节数、L0 布局、四种输出 layout/state 分支保持定义。

### 当前性能：最初原版到两槽预取

原版提交为 `bead79ae45fca5d10a8414b215d4f762cc66538d`，正常包 `baseline.run`
SHA256 为 `f799fd1c8cb59c365124d7bab7bac10c7c1afeba1463da5dc17f5c2902ccefc5`；
当前两槽包为后文的 `candidate_normal.run`。本轮两版本均在同一
`/data/huangjunzhe/GDN/work/kda_finalize_prefetch_env`、device5 Ascend910B3
串行安装，每次核对全部6个库/.o，启动新进程测量。
冻结perf JSON、executor及common executor与原版提交逐字节一致；
`--disable_id_seed` 使用原default_seed，GM init off、CPU单线程、正常包无trace/sanitizer。
ATK `--performance_data 3,10,10 --save_data profile`：预热3次，采样10次，保留全部10次；
统计目标 `ChunkKdaFwdFinalize` 的 `op_summary Task Duration(us)` 中位数。

| case | B/HV/T、场景 | 最初原版 µs | 当前两槽预取 µs | 耗时降低 |
| ---: | --- | ---: | ---: | ---: |
| 0 | 1/16/11264 dense | 238.74 | 228.51 | 4.28% |
| 1 | 1/32/11264 dense | 523.78 | 463.45 | 11.52% |
| 2 | 1/96/8192 dense | 1187.96 | 1003.03 | 15.57% |
| 3 | 1/96/16384 varlen | 2250.04 | 1991.75 | 11.48% |
| 4 | 2/16/11264 dense | 523.35 | 461.28 | 11.86% |
| 5 | 4/96/128 dense | 36.93 | 22.43 | 39.26% |
| 6 | 1/32/160 dense | 7.64 | 5.93 | 22.38% |
| 7 | 6/6/1084 dense | 34.63 | 18.75 | 45.86% |
| 8 | 1/8/32768 dense | 378.44 | 341.63 | 9.73% |
| 9 | 1/32/65536 varlen | 3011.46 | 2658.08 | 11.73% |

原版与当前均为本轮fresh全10用例，每例各10个有效目标样本，10/10任务成功，
profile完整性检查通过；两个run及pair退出码均0，无缺样、补测或删除离群点。
所有当前样本均低于对应原版的最小样本；这只描述本轮采样范围。
case0当前228.51us如实记录，历史消融的160.65us保留在下一节。
目标均为AI_CORE、BlockDim20、MixBlockDim0、AIV0、四项主输入及输出BF16，
varlen辅助元数据INT64。此10case口径为BNSD/KV，A3尚无实机性能证据。

新证据根为 `/data/huangjunzhe/GDN/work/kda_finalize_original_compare_20261013`：
`freeze_provenance.json` 保存原版源码/旧build/wheel/run关联与两次测试资产核对；
`original_vs_current_perf.json`、`original_performance/perf_summary.json`、
`candidate_performance/perf_summary.json` 保存各例全样本、波动及原CSV路径；
`final_validation.json` 核对双方原始XLSX、全部6二进制和退出码。
全卡进程快照保留了其他用户任务；device5启动前空闲、结束后恢复候选正常包并释放。

### 历史单槽 CG1 消融

实机 A2 Ascend910B3 device5，20 AIC/40 AIV；CANN9.1.0、ATK26.8.8、
Python3.10.20、torch2.12.0+cu130、torch_npu2.12.0。两版本在独立
`/data/huangjunzhe/GDN/work/kda_finalize_prefetch_env` 串行安装并核对全部6库/.o。
冻结perf JSON/default_seed/input/golden/阈值均未改；GM init off、CPU单线程、
3次warm-up、每case10次采样，正常包无trace/sanitizer。
下表仅取目标 `aclnnChunkKdaFwdFinalize` 的 `op_summary Task Duration(us)` 中位数。
原始每case10个样本、min/max/标准差和CSV路径完整保留；所有候选样本均低于
对应基线的最小样本，收益超过该配对轮次的采样范围。全部均为AI_CORE、BlockDim20、
MixBlockDim0、AIV0，四项主输入/输出BF16；varlen辅助元数据INT64。
这些10个性能case使用BNSD/KV，不能把其加速比外推到全部layout/state或A3。

| case | B/HV/T、场景 | 原 CG1 µs | CG1 两槽预取 µs | 耗时降低 |
| ---: | --- | ---: | ---: | ---: |
| 0 | 1/16/11264 dense | 226.26 | 160.65 | 29.00% |
| 1 | 1/32/11264 dense | 508.94 | 463.90 | 8.85% |
| 2 | 1/96/8192 dense | 1107.14 | 996.98 | 9.95% |
| 3 | 1/96/16384 varlen | 2196.73 | 2014.57 | 8.29% |
| 4 | 2/16/11264 dense | 508.35 | 461.60 | 9.20% |
| 5 | 4/96/128 dense | 36.53 | 22.48 | 38.46% |
| 6 | 1/32/160 dense | 7.18 | 5.92 | 17.55% |
| 7 | 6/6/1084 dense | 30.14 | 18.86 | 37.43% |
| 8 | 1/8/32768 dense | 370.98 | 332.88 | 10.27% |
| 9 | 1/32/65536 varlen | 2952.98 | 2675.95 | 9.38% |

首轮候选case0包含3个毫秒级离群样本，case1缺少目标kernel记录；profiler明确
报告缺少ascend_task.db/AscendTask。该轮严格统计exit1，保留完整原始日志，未把
ATK任务SUCCESS当成有效采样。只对0/1补做baseline→candidate同条件配对，双方
各10个完整目标样本；表中0/1取该补配对，2–9取原始有效轮次。
这不删除离群点以凑中位数，也不混用不同卡或san/trace包。配对之外的运行波动仍可能存在。

### 精度、同步及正式门禁

| 检查 | 实际结果 |
| --- | --- |
| 冻结200用例 × offset0/100000/200000 | 600/600 mixed_tolerance_bm通过，0执行失败 |
| 与最初bead79原版直接逐位比较 | 相同BF16输入/属性600组；300,833,280输出元素，原始16位差异0，signed-zero差异0，非有限0 |
| 与历史单槽CG1直接逐位比较 | 相同BF16输入/属性600组；300,833,280输出元素，原始16位差异0，非有限0 |
| 正式accuracy / performance | 原200/200及10/10执行通过，逐项读取原始XLSX |
| 正式determinism | 12/12；每条50次执行、49次精确比较，共600次执行/588次比较 |
| 独立无trace sanitizer memcheck | 12/12 memory Pass；目标kernel Start12/Finished12/ERROR0 |
| 独立workspace执行trace | 36/36精度通过；L1/L0 ready/free事件账本全部退出归零 |
| Stable Python入口 | dense BSND/KV与packed TND/VK两项matched_ratio=1、abs/ULP失败0 |
| HostUT | g++17 -Wall -Wextra -Wpedantic -Werror -O2，编译/运行0，assert启用 |
| A5隔离 | Cube/Vector预处理2787 tokens与基线完全相同，SHA256 3aac387c…b3c3637 |
| A3 | ascend910_93构建通过；没有A3实机精度/性能证据 |

最初原版与当前候选的已保存BF16输出直接按case/seed比较，600组输入tensor的
dtype/shape/原始字节与全部typed属性均一致；四种layout、两种state轴顺序、
dense/varlen及尾M均覆盖。直接比较结果见新证据根
`bitwise/comparison/summary.json`，600/600通过。

历史单槽CG1逐位比较的基准为前次已保存并核过包/输入的原CG1输出；新候选600组另保存完整
input.bin和BF16输出，再按case/seed核每个输入tensor的原始字节与全部属性相同。
四种layout、两种state_v_first、dense/varlen及尾M均覆盖，不只做容差比较。
确定性XLSX保存逐caseaggregate True；次数由dc_loop_nums50及冻结ATK源码
的循环和torch.equal/NaN-mask比较核定，不声称报告含588条逐比较明细。
Excel32767字符导致长case元数据截断时，核对可见header/完整输入前缀与实际调用JSON；
精度、确定性及内存结果列未截断。没有修改输入值域或阈值。

trace在独立repo/environment插桩，正式正常源码的41条同步语句序列未变；
覆盖0–7、80–91、192–199，以及HV1/T2561×4layout×2state共36条。
HV1实例的core0实际work[0,20,40]、inputSlot[0,1,0]、末M1，其余19核各两任务；
single-task只用input0，尾部只drain实际使用的槽。trace只证明实际任务/事件
顺序和已完成等待，不是硬件pipe时间线。trace成功后torch_npu退出handler
打印ERR99999，原异常未记录，具体触发原因不确定；原plog无ERROR、HBM分配/释放
160/160、设备关闭成功、进程退出和device6空闲均已核对。原日志及调查完整保留。

公开入口额外补测第一次临时脚本将current device设5而输入仍指定npu1，触发跨设备
GM地址错误；统一为device5后两条通过。只修验证脚本，候选kernel未改，失败日志保留。

### 搬运等待与证据边界

本方案解除单L1槽复用对下一输入的串行约束，减少输入pipe空档的机会；总GM字节数
不变，不能只凭“MTE2 bound”保证收益。新证据根的 `current_pipeline_analysis.json`
仅使用本轮原版/当前fresh对照的100条当前正常包原始记录：大shape
case0/1/2/3/4/8/9的MTE2活动比例中位数为99.05%–99.9%；最短case6的
Scalar为57.8%、MTE2为55.05%，活动比例允许重叠，不能相加作为总耗时。
同一原始行/共同分母下，90/100条满足MTE2_time+MAC_time>total_time；
case6的10条均不满足，不能由此推断其完全没有重叠。
这些busy计数器支持大shape受搬运pipe占用主导的判断，不定位命名free事件的等待，
也未证明HBM带宽饱和、bank冲突被消除或带宽提升。L1并发竞争也可能增加MTE1时间。
A2工具当前无法提供950专属的真上板逐指令时间线；TimelineDetail是采输入的仿真
重放。执行trace和aggregate busy counters不定位具体跨head时刻，不据此宣称
某个slot的实机时间重叠已被直接观测。

### 原始证据与构建

两槽构建、历史CG1消融、精度及门禁证据在
`/data/huangjunzhe/GDN/work/kda_finalize_prefetch_20261013`；
最初原版到当前的fresh性能、直接逐位比较及当前pipe分析补充在
`/data/huangjunzhe/GDN/work/kda_finalize_original_compare_20261013`。
正常A2包 `candidate_normal.run` SHA256
`e07fcc2c42487f3b67d216c515d78eb7a6c4081cc91f069e2ff5a18c7189562a`；
A3包 `a3_candidate.run` SHA256
`adbd652514201fdbf0a1809a6ba8d38074f40cd1fc11bd4d2b834d09c05ae025`。
正常、trace、san包分别在独立env安装，实际全部6个库/.o与对应构建/解包一致；
san的生产kernel源文件与正常包逐文件哈希相同。

`perf_comparison.json`、`perf_selection_manifest.json`、`baseline/candidate_perf_selected.json`
保存配对数据及采样选择；`accuracy/candidate/seed_sweep_summary.json`、
`bitwise/comparison/summary.json`、`formal_normal_validation.json`、
`formal_mss_validation.json`、`prefetch_trace_evidence_summary.json`、
`a5_token_review.json`、`a3_build_summary.json`、`host_ut_validation.json`
保存各项真实检查。所有失败尝试及工具日志同时保留。

历史CG1消融的补充同口径 pipe 活动证据：`pipe_overlap_validation.json` 已逐条核对正常 dev5
性能 case2/7/9 两版本各10条原始记录，以及实际安装 CANN parser 的源码/哈希。
本机每条的 total 与 pipe 时间使用共同分母20×1800MHz；候选30/30条均满足
MTE2_time+MAC_time>total_time，支持至少部分被计数的搬运/计算活动重叠。
这不是某个core/slot的逐指令时间线，也不是命名事件等待时长；基线和不超过total
不能证明完全没有重叠。没有把分别求出的中位数相加作为逐样本证明。

独立 MemoryDetail 的尝试不计入结果：device5 后续被他人任务占用时 preflight拒绝，
另卡尝试因临时ATK loop_nums10生成重编号/变更seed被取消。已保留exit4/143及日志，
未将其称为同输入重复测量。正常 op_summary 的配对耗时和共同归一化活动计数已足够
完成本轮判断；没有有效 MemoryDetail 对比，不据此宣称bank冲突被消除或带宽提升。

最终两次独立MemoryDetail尝试均由本任务主动取消（exit143）；取消原因/实际seed保留，
没有有效baseline/candidate MemoryDetail CSV配对。只结束本任务进程，其他任务未动。
性能环境已恢复候选正常包，完整6对象一致性再次通过，device5/6本任务进程已退出。


## 2026-10-15：A2/A3 可达 tiling key 确定性及 Sanitizer 验证

本轮只补充质量验证与证据；没有修改生产算子源码或 A5 逻辑。A2/A3 的 host 合法可达 tiling key 均只有 **0**：`USE_AIV_INPUT_MOVER=1` 的 key1 被 host 限定于 ASCEND950；旧 case 名中的 key2 是历史标签。key0 是 AIC_ONLY。

|检查|实际覆盖与结果|
|---|---|
|正常包 ATK DC50＋原始 BF16 位检查|22/22 PASS；正式调用 1100 次，直接 int16 位比较 1078 次，差异元素 0；全部有限，22 例首末输入一致；另有 22 次前置 accuracy，不计入正式 50 轮|
|`mssanitizer --tool=memcheck`|22/22 完成目标 key0 检测；WARN（寄存器复位告警，未达到严格 clean）；目标 FFTS_BASE_ADDR 告警 440 条，严格 clean 0/22|
|`mssanitizer --tool=racecheck`|22/22 完成目标 key0 检测；FAIL（目标 L0C RAW/WAW hazard）；严格 clean 20/22|

racecheck 的 HV1/T2561 两例（case20：BNSD/state=false；case21：BSND/state=true）各报告一条 L0C RAW 和一条 WAW，均指向 `ConsumeHead` 内 P 初始化 MMAD 与 R 原位累加，目标结束为 `See all detected errors above.`。两例 50 轮位比较均通过，说明输出稳定不能替代竞争检测。本轮测试没有修改生产源码；错误的最终根因及修复效果尚未验证。

独立复核字段：`coverage_complete=true`、`overall_pass=false`、`no_memory_access_errors=true`、`no_race_hazards=false`。寄存器 WARNING 原文、目标 finish 状态和 strict false 均完整保留，没有以 ATK 的 memory_pass 或 exit0 替代工具判定。辅助 kernel 的寄存器告警另列。

racecheck 的目标结论限于本算子。Cast/IsFinite/ReduceAll 等 ATK 辅助核未插桩，其 NoActive/unsupported 段完整保留，不宣称辅助核也完成了竞争检测。冻结分类器把正常 `No error detected` 结束行误识别为错误的原结果亦保留；最终目标判定由独立 CPU 复核原始 Start、注入与 Finish 日志得出。

同一组 22 条冻结数学用例覆盖四布局 BNSD/BSND/NTD/TND、state_v_first 两值、dense/explicit/cu-scan、M1/17/32/33/63/64、单 head、跨 batch、双 L1 槽多次覆盖复用及退出 drain。四布局×两 state 在 explicit、cu-scan 两组均完整覆盖；本组不宣称穷举所有分支的笛卡尔积。normal/mem/race 的输入、seed、原数学 executor、FP64 golden 和精度标准相同。

硬件实测为 Ascend910B3（A2）device5，msSanitizer 26.1.0。A3 只完成源码、正常包构建产物及 CPU host 选择/排程审计，**没有 A3 硬件检测结果**。racecheck 使用默认范围：核间检查全部 block，核内流水检查 block0。生产 host 额外 24 核 CPU 排程验证不代表 A3 实测。

远程证据根：`/data/huangjunzhe/GDN/work/kda_finalize_sanitizer_keys_20261015`。正式 DC 证据为 `attempt_03/determinism`；44 条有效 sanitizer 原始记录的逐例来源、工具日志和 XLSX SHA 位于 `sanitizer_case_origins.json`（保留原 attempt 路径，不复制冒充完整运行）。共享设备占用导致的部分尝试和前期 runner 失败均单独保存，不计作通过。

完整结果与复核：`a2a3_tiling_key_quality_report_20261015.md`、`independent_cross_attempt_validation_v2.json`、`deliverables/a2a3_tiling_key_quality_cases_20261015.csv`。证据归档为 `deliverables/a2a3_tiling_key_quality_evidence_20261015.zip`；其中保留正常包原生 DC XLSX、逐轮位比较记录、两工具各 22 份原始日志/XLSX、冻结资产/runner、18 源文件及实际 normal/san 对象的前后 SHA 核验。
