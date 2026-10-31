# RecurrentKda ATK 工程

本目录提供 `recurrent_kda` 的 ATK 单算子工程，包含 `executor_recurrent_kda.py`、`gen_recurrent_kda.py`、`recurrent_kda.yaml`、`atk_recurrent_kda.json`。

## 输入约束

- 模式仅由 `cu_seqlens` 是否传入决定，忽略 `layout`：不传时为 rank4，`q/k=[B,T,H,K]`、`v/out=[B,T,HV,V]`；传入时为 rank3 packed，`q/k=[capacity,H,K]`、`v/out=[capacity,HV,V]`。`g` 对应 HV×K，`beta` 对应 HV。
- `initial_state` 是 state pool；`state_v_first=true` 时为 `[state_capacity,HV,V,K]`，否则为 `[state_capacity,HV,K,V]`。
- `cu_seqlens` 为同设备 INT32/INT64，shape `[seq_num+1]`，首项0、末项不超过 capacity、单调不减。A5 按最大相邻差分选择 X=1/2/4/8，上限8；全空取1。元数据同步与缓存限制见算子 API 文档。
- `q/k/v/out` 为 BF16；`g/beta` 支持 FP32/BF16/FP16，State 支持 FP32/BF16。传入 `A_log` 时 g 为 raw gate；不传时 g 为 step log gate，并忽略 `dt_bias/safe_gate/lower_bound`。兼容参数 `use_gate_in_kernel` 不决定数值路径。
- `HV % H == 0`，没有 H/HV<=256 或 G<=16 的人为限制，仍受地址、DMA stride 和 tiling 字段范围约束。A5 支持 K=V=128；A2/A3 保留 K=128、V=128/256 的硬件路径。
- A5 不限制每段长度为8。显式 `ssm_state_indices` 可使 `state_capacity > seq_num`。Q/K/V 支持独立的合法 batch stride。
- A2/A3 共用模式与 gate 参数语义；2026-10-31 的 A2 上板冒烟未通过，见文末记录。A3 尚未上板，不能以 A5 结果替代。

## 标杆来源

CPU recurrence 对齐 [PR #917](https://github.com/flashserve/flash-linear-attention-npu/pull/917)
提交 `082b0c704407d0b38909fd4b0f58703b8ab52873`。同一原始 dtype 量化输入提升 FP64 后
计算，最终 golden 按 ATK 规则转 FP32；L2 使用 `x/sqrt(sum(x*x)+1e-6)`。
标准沿用 `mixed_tolerance_bm`。完整 state pool 参与比较，仅排除 API 未定义的 out padding tail。

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_recurrent_kda.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。
本次新增的长序列正式包是 A5 验收矩阵，不能据此声称 A2/A3 验收通过。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 2, "T": 2, "H": 2, "HV": 4, "K": 128, "V": 128, "layout": "BSND", "state_v_first": true, "op": "recurrent_kda", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。

## Issue 680 A5 验证矩阵

公开入口默认使用安装态 Stable-ABI 后端；不强制切换 ctypes，不过滤有定义的 state 槽。
`python tests/atk/recurrent_kda/gen_recurrent_kda.py` 可重建正式包及新增对齐包；精度候选、模型性能、
人工选择的内存/确定性用例分别维护，`gen_cases` 只生成精度候选。

| 包 | 数量与来源 | 覆盖 |
| --- | --- | --- |
| `atk_recurrent_kda.json` | 44结构×3种子=132 | 原有小 shape；8融合开关组合；长序列、布局、dtype、state、slot及边界 |
| `atk_recurrent_kda_perf.json` | issue 的36组模型 shape | B=32/64/72，HK=16，HV=32/48/128，T=1/2/4/8，K=V=128 |
| `atk_recurrent_kda_mss.json` | 人工选择11组，全部 key=0 | short/window/tail/transpose/slots/packed/fallback/独立 batch stride；补充 X=1 BF16 fused/general 两遍读取 |

精度覆盖每段长度1/2/4/8/9/15/16/17/31/32/33/65/129/257/1024，HV/HK=1/2/3/16；
全零/小范数/强信号；safe/unsafe gate、大 raw gate、beta负特征值开关；融合 QKV storage
offset、head gap、feature stride=2回退；非连续 state、两种布局/dtype、原位/非原位；
无 metadata、空初始 state、空序列/padding；跨窗口重复 slot、accepted token 与未命中槽。
算子只有 key=0，host 不新增 key；11个 MSS case 覆盖当前窗口与缓冲分支，包括 X=1 BF16 fused/general。

以下原有执行次数和基线说明属于历史验收记录；本轮采样与基线见文末 2026-10-29 方案对齐记录。

性能属性固定为三融合开关开启、FP32 V-first state、原位更新、融合 storage QKV 切片。
用户确认本轮目标为不劣于库上已有实现；以变更前 `67dc99d7` 同 shape、属性为基线。
不将其他未指定属性或原 issue 的0.8倍原生目标作为本轮达标要求。

## 输入校准与公开入口回归

随机输入沿用 QKV normal(0,0.05)、state normal(0,0.02)，raw gate normal(0,0.3)、
raw beta normal(0,0.6)。小输出可能使全零结果被 BF16 atol 接受，故额外固定
Q/K单坐标=1、V=1、beta=0.5、log gate=0 的 `norm_signal`；保留原随机值域和容差。
独立解析检查覆盖 Q/K=0和2^-12、T=1/9/17，直接检查 epsilon 和每token递推。

```bash
python tests/atk/recurrent_kda/scripts/check_reference_calibration.py --artifacts /tmp/kda-calibration
python tests/atk/recurrent_kda/scripts/check_public_contract.py --device 6
ct viz /tmp/kda-calibration/correct_0.npy /tmp/kda-calibration/golden_0.npy --out_dir /tmp/kda-viz --name correct_0 --spatial
ct viz /tmp/kda-calibration/wrong_0.npy /tmp/kda-calibration/golden_0.npy --out_dir /tmp/kda-viz --name wrong_0 --spatial
# 对 output_1/state 同样执行；不能用 out 代表 state。
```

CPU 校准42结构同精度与FP64逐输出比较通过；固定PR917源码的37个非padding结构逐输出
完全一致（该参考不接受padding capacity，其余5个由本仓接口和独立recurrence验收）。
单坐标解析一致，故意置零out和错误state
均被相同标准拒绝。CT Tool v0.9.1：正确BF16 out贴近y=x，最大绝对误差约1.94e-4；
全零out明显分离，最大误差约0.0884；错误state最大误差约1。图表阈值只用于诊断。
公开入口另检查不返回final state、空初始state、输入不变性、storage、版本计数、grad契约及非法参数。

正式执行不设置 case 范围：

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -soc=ascend950 -scope=gen_cases
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -soc=ascend950 -npu_device_id=6 -scope=all
```

`all` 包含精度/NaN（GM初始化）、ATK performance_device、50轮确定性和mssanitizer memcheck。
性能另按预分配输入的脚本采修改前后 msprof Task Duration，区分目标kernel与辅助kernel：

```bash
msprof --output=/tmp/kda-prof --task-time=on \
  python tests/atk/recurrent_kda/scripts/profile_model_cases.py --device 6 --warmup 10 --samples 20 --manifest /tmp/kda-manifest.json
```

常规采集按36组各30次 `RecurrentKda` task 对齐shape，舍弃前10次，统计后20次均值；
最终交替复测统一预热30次、采样100次，逐组统计后100次均值。
增量内存为输入驻留后的allocator峰值增量，包含输出、workspace和调用内连续化。
最终验证范围、环境、通过结果及用户接受的性能例外归档如下。

## A5_ci 历史验收结果（2026-10-20）

对应基线 `67dc99d7` 上本次工作树的向量归一化+寄存器稳定softplus版本，
未采用实验性的单行Scalar归一化。功能验证通过；原始严格阈值29/36不劣、7/36未达标。
用户于2026-10-20明确确认“接受差异，先交付功能”，接受下表7个性能差异作为本轮例外；
本轮按该例外完成验收，原始逐case阈值仍未全部达到。

环境：A5_ci Ascend950PR卡6，CANN 9.2.0-beta.2，Python 3.11.17，
PyTorch 2.9.0+cpu / torch_npu 2.9.0.post6，ATK 26.9.8；独立venv与源码，
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`，安装态公开Stable-ABI入口。

被测源码SHA256：

| 文件 | SHA256 |
| --- | --- |
| `op_kernel/arch35/recurrent_kda.h` | `380e74d45a9b31fd08d35c98cb84af0b82a39d02978916ef8f051b2b78b9f46a` |
| `op_host/recurrent_kda_tiling_processor.h` | `4d6df07233b9a63fd9b450f1e1b599fc0f1ca15831a5ee5b3b4c5f71a2c774b2` |
| `torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py` | `0d79f61a80969d553a036de9411fa390b2990d7b1f8b127518ff07bd9bf66a48` |
| `executor_recurrent_kda.py` | `8852b9b04a8796d0fd09669d51e80cb8379b27c985846178cc47cabe6aa187a1` |

三份JSON语义SHA256（`sha256(json.dumps(json.load(file),sort_keys=True).encode())`）：
accuracy `b6bd764fbcb6a235fd8106e6dfe471f311ef1c99cf31dba13de249e167b2f809`；
perf `d4e9749143f1254257b86f96ab35d58e319a2716a55714b130fc3240ad25ba30`；
mss `39e16928aa594b3a8d4ba09f690714b802e4b5010c192fbc45fba4586eacbaac`。

| 动作 | 结果 | 报告或证据 |
| --- | --- | --- |
| `gen_cases`，无range | 100个精度候选生成成功；正式包42结构×3种子 | `fla-680-a5-gen-final.log` |
| FP64参考校准 | 42结构通过，37个与固定PR917完全一致；独立解析和故意错误拒绝通过 | `fla-680-a5-calibration-final.log` |
| `all`精度/NaN初始化 | 126/126，失败0 | `atk_recurrent_kda_reports_2026-10-20-19-56-33.xlsx` |
| `all` performance_device | 36/36执行成功；性能达标以独立msprof为准 | `fla-680-a5-atk-v5.log` |
| `all`确定性，50轮 | 8/8，失败0 | `atk_recurrent_kda_mss_reports_2026-10-20-20-00-43.xlsx` |
| `all` memcheck | 8/8，失败0，目标kernel无错误 | `atk_recurrent_kda_mss_reports_2026-10-20-20-01-04.xlsx` |
| 公开接口、mutation/grad | 42项通过，含无cu/空initial_state/V256 | `fla-680-a5-gatevf-public42.log` |
| 安装核验 | 72个kernel .o、opapi和Python wrapper匹配构建源码 | `fla-680-a5-gatevf-md5.log` |

交付前已恢复上述源码哈希对应的向量版本，重新安装保存的同一构建产物
`<run-root>/fla-680-gatevf.run`；未保留Scalar实验版本作为安装结果。
该run包SHA256为 `dd89ae1a7363aa744f8ad873281da5ebaafeb072d408acb54dac814e4431b03d`。
`fla-680-a5-delivery-md5.log`再次核对全部72个kernel、opapi和wrapper通过，
新Python进程重跑42项公开接口通过（`fla-680-a5-delivery-public.log`）。

新增覆盖跨窗口长序列、epsilon、稳定gate、全部融合开关、state复制与slot完整pool。
原有V256和FP16辅助输入支持保留；正式矩阵覆盖issue要求的BF16/FP32辅助输入，
不能据此新增FP16验收结论。A2/A3 kernel未改动，本次没有A2/A3设备验收。

ATK_OUTPUT_ROOT只重定向本轮accuracy/perf；确定性和memcheck实际写入算子目录的
`atk_output/`。统一脚本会对后两项显示“未找到报告”；上表已使用
`tests/atk/common/check_atk_result.py` 对实际目录分别复检，不能据“跳过”判通过。

### TilingKey与分支

| key | 条件 | 精度case id | MSS case id | SoC | 实际证据 |
| --- | --- | --- | --- | --- | --- |
| 0（唯一） | 全部合法输入，dtype由编译变体决定 | 普通0、边界1-41；每42项换种子重复 | 0-7 | A5/ascend950 | `fla-680-gatevf-ab-1-new-tiling.log`：RecurrentKda launch `tilingKey: 0,numBlocks: 56`；MSS执行 `RecurrentKda_06bba6860f70bfd6aa54df1d4ed1928e_0` |

MSS 0短序列、1跨窗、2尾窗、3 K-first/间隔state、4 speculative/repeated slot、
5 packed/空序列、6 feature-gap回退、7 BF16/V256；属性组合见生成器。
B64/HV32/T1的新旧运行日志均为UB=253952B、vStep=64、state/out队列深度2，
本次没有改变该模型的分核与队列选择。

### 36组性能逐项比较

均为Ascend950/BF16 QKV、FP32 g/beta及state，HK=16、K=V=128，属性固定如上。
基线 `67dc99d7`，目标 `new_us <= old_us`；单位us，ratio为新/旧，越小越好。
同卡交替安装新旧版本，预分配输入，预热30次后采样100次；每版本4680个目标task，
只统计RecurrentKda设备时间，辅助算子与输入生成不计入目标kernel时间。

| case id | B / HV / T | 基线us | 本次us | 比值 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 0 | 32 / 32 / 1 | 59.9578 | 57.8751 | 0.965264 | 不劣 |
| 1 | 32 / 32 / 2 | 95.2449 | 79.8444 | 0.838307 | 不劣 |
| 2 | 32 / 32 / 4 | 159.8954 | 128.1739 | 0.801611 | 不劣 |
| 3 | 32 / 32 / 8 | 288.5218 | 219.8480 | 0.761981 | 不劣 |
| 4 | 32 / 48 / 1 | 100.0874 | 95.6775 | 0.955939 | 不劣 |
| 5 | 32 / 48 / 2 | 138.9297 | 125.0125 | 0.899825 | 不劣 |
| 6 | 32 / 48 / 4 | 236.8486 | 191.5319 | 0.808668 | 不劣 |
| 7 | 32 / 48 / 8 | 430.3986 | 327.3650 | 0.760609 | 不劣 |
| 8 | 32 / 128 / 1 | 452.0251 | 451.8628 | 0.999641 | 不劣 |
| 9 | 32 / 128 / 2 | 467.1132 | 464.8114 | 0.995072 | 不劣 |
| 10 | 32 / 128 / 4 | 668.8143 | 535.2314 | 0.800269 | 不劣 |
| 11 | 32 / 128 / 8 | 1137.9879 | 865.4143 | 0.760478 | 不劣 |
| 12 | 64 / 32 / 1 | 173.2434 | 167.1979 | 0.965104 | 不劣 |
| 13 | 64 / 32 / 2 | 199.1451 | 179.8826 | 0.903274 | 不劣 |
| 14 | 64 / 32 / 4 | 317.4018 | 256.7382 | 0.808874 | 不劣 |
| 15 | 64 / 32 / 8 | 573.3985 | 435.8081 | 0.760044 | 不劣 |
| 16 | 64 / 48 / 1 | 325.8903 | 326.0334 | 1.000439 | 未达标 |
| 17 | 64 / 48 / 2 | 337.4034 | 337.4280 | 1.000073 | 未达标 |
| 18 | 64 / 48 / 4 | 495.9449 | 397.4722 | 0.801444 | 不劣 |
| 19 | 64 / 48 / 8 | 853.4129 | 647.6490 | 0.758893 | 不劣 |
| 20 | 64 / 128 / 1 | 907.1381 | 906.1128 | 0.998870 | 不劣 |
| 21 | 64 / 128 / 2 | 927.4451 | 927.4904 | 1.000049 | 未达标 |
| 22 | 64 / 128 / 4 | 1324.9615 | 1052.6453 | 0.794472 | 不劣 |
| 23 | 64 / 128 / 8 | 2249.0926 | 1700.6941 | 0.756169 | 不劣 |
| 24 | 72 / 32 / 1 | 207.0574 | 211.1663 | 1.019844 | 未达标 |
| 25 | 72 / 32 / 2 | 233.2828 | 219.5204 | 0.941005 | 不劣 |
| 26 | 72 / 32 / 4 | 364.3337 | 291.9659 | 0.801369 | 不劣 |
| 27 | 72 / 32 / 8 | 651.4040 | 494.3825 | 0.758949 | 不劣 |
| 28 | 72 / 48 / 1 | 376.6066 | 377.9932 | 1.003682 | 未达标 |
| 29 | 72 / 48 / 2 | 388.9009 | 386.1567 | 0.992944 | 不劣 |
| 30 | 72 / 48 / 4 | 561.4603 | 449.4191 | 0.800447 | 不劣 |
| 31 | 72 / 48 / 8 | 959.7260 | 727.9848 | 0.758534 | 不劣 |
| 32 | 72 / 128 / 1 | 1019.8165 | 1020.6973 | 1.000864 | 未达标 |
| 33 | 72 / 128 / 2 | 1041.6621 | 1044.9204 | 1.003128 | 未达标 |
| 34 | 72 / 128 / 4 | 1485.8383 | 1180.4448 | 0.794464 | 不劣 |
| 35 | 72 / 128 / 8 | 2521.3429 | 1907.9097 | 0.756704 | 不劣 |

T=4耗时为基线的约79.4%～80.9%，T=8为75.6%～76.2%；主要收益来自无slot时
只写最终state与批归一化。未达标case id为16/17/21/24/28/32/33；
其中24（B72/HV32/T1）慢1.9844%，其余慢0.0049%～0.3682%。
平均比值0.875758不替代逐case结论。用户接受的例外只对应上述7组、
当前属性和本次数据，不扩展为其他输入或后续变更的性能豁免。

36组原位调用在输入驻留后的allocator额外峰值最大54,526,464B（52.0005MiB），
低于issue的200MB上限，计入输出、系统workspace及调用内连续化。
该数字不能用于非原位的大state pool，后者还需复制整个初始state。

证据保留于A5_ci：源码 `<user-home>/src/fla-680-a5-port-20261020`，
日志/报告根 `<run-root>/`。精度/perf在`fla-680-a5-atk-v5/`，
确定性/内存报告在源码的 `tests/atk/recurrent_kda/atk_output/`；
性能根`fla-680-gatevf-ab-1-{old,new}-prof/`，各自manifest固定shape、属性、采样数，
`op_summary*.csv`保存逐task时间，`fla-680-gatevf-ab-results.json`保存均值/中位数/标准差。

## 方案对齐验证（2026-10-29）

本轮从已提交 PR #997 的 `d23fb9e84fb410fb857d299686b679302a32d794` 开始，
对齐用户修订的寄存器方案，未修改方案原文。开发过程记录和流水图保存在本地归档，不随 PR 提交。
方案 SHA256：`78f03d56724f4ff6ba36609f1f53dbc05148825b94ee7d253a7867c8c331d5b4`。

环境为 A5_ci / Ascend950PR，物理卡5（进程逻辑卡0），CANN 9.2.0-beta.2、Python 3.11、
torch 2.9.0+cpu / torch_npu 2.9.0.post6。独立 venv，标准 ascend950 单算子 run 包构建安装，
72个 kernel 对象；安装 opapi SHA256：
`0cc3c3d2196fe376b354ca16ffa92dc389df1646393d4f256e0c2ad97c33015b`。
CPU 标杆、值域和 mixed_tolerance_bm 精度标准不变。

| 验证 | 结果 | 证据文件 |
| --- | --- | --- |
| 标准构建安装 | 通过 | build8.log / install8.log |
| 原有正式精度，CPU 双输出对比 | 132/132 | atk_recurrent_kda_reports_2026-10-29-18-16-03.json |
| 新增对齐精度，CPU 双输出对比 | 54/54 | atk_recurrent_kda_alignment_reports_2026-10-29-18-16-19.json |
| 公开入口 | 74项及 mutation/grad 通过 | public.log |
| 确定性 | 9例各3轮，通过 | determinism-final.log |
| 原生 bank/输入/out 生命周期 trace | 8个变体通过，四个连续任务 | execution-audit.json、execution-{fp32,bf16}-x{1,2,4,8}.log |
| 真正启用 sanitizer 的原生 memcheck | FP32 V-first、BF16 K-first 快照各1例通过 | memcheck-native-fp32.log / memcheck-native-bf16.log |
| cu_seqlens 值依赖 | INT32/INT64×8种最大长度，16项通过 | metadata2.log |
| A2/A3 共用 kernel | dav-c220-vec 下 FP32/BF16 交叉编译通过 | arch22-fp32-build.log / arch22-bf16-build.log；未做设备运行验证 |

新增54例：0–23为 State dtype/layout×快照；24–35为 Lmax=1/2/4/8、packed padding及 FP16辅助输入；
36–38全空；39–44独立 batch stride；45–50忽略 inactive gate 参数；51–53定长32。
每种结构3个种子。公开入口额外覆盖同容量改变边界、非零 offset/stride3元数据、
九种 g/beta dtype组合×两种 State dtype、非法边界、mutation/grad。
trace 检查 bank 覆盖前 DMA 回收、tile 完成后批量 out、最后 tile 内下一任务输入预取及结束排空，
不是 msprof 指令流水采集。两种 State dtype/layout 都保持 R=4，不恢复逐 token 搬出 out。

### issue680 模型性能

相同设备、相同36组模型 case、安装态公开入口，对比已提交版与本轮版。
使用 msprof Task Duration；每例3次预热、5个有效样本取均值，共288次目标 kernel/版本。
不使用 ATK performance_device，不把输入生成和 host 调用时间计入 kernel 耗时。
5样本只用于本轮回归观察，不能证明每个 case 均无下降。

原始文件：`rkda-plan-alignment-baseline.csv`、`rkda-plan-alignment-final.csv`，各432行48列；
各288行为 RecurrentKda。统计文件为 `rkda-plan-alignment-performance-final.json`。

| ID | B | HV | T | 已提交版 μs | 对齐版 μs | 对齐/提交 | 耗时变化 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 32 | 32 | 1 | 46.767 | 45.594 | 0.9749 | -2.51% |
| 1 | 32 | 32 | 2 | 75.684 | 75.533 | 0.9980 | -0.20% |
| 2 | 32 | 32 | 4 | 137.317 | 136.346 | 0.9929 | -0.71% |
| 3 | 32 | 32 | 8 | 260.234 | 260.235 | 1.0000 | +0.00% |
| 4 | 32 | 48 | 1 | 65.135 | 64.480 | 0.9900 | -1.00% |
| 5 | 32 | 48 | 2 | 108.708 | 109.624 | 1.0084 | +0.84% |
| 6 | 32 | 48 | 4 | 199.057 | 198.853 | 0.9990 | -0.10% |
| 7 | 32 | 48 | 8 | 380.058 | 381.370 | 1.0035 | +0.35% |
| 8 | 32 | 128 | 1 | 382.984 | 381.637 | 0.9965 | -0.35% |
| 9 | 32 | 128 | 2 | 392.528 | 390.211 | 0.9941 | -0.59% |
| 10 | 32 | 128 | 4 | 519.259 | 517.163 | 0.9960 | -0.40% |
| 11 | 32 | 128 | 8 | 997.061 | 1000.020 | 1.0030 | +0.30% |
| 12 | 64 | 32 | 1 | 111.782 | 118.804 | 1.0628 | +6.28% |
| 13 | 64 | 32 | 2 | 144.255 | 147.153 | 1.0201 | +2.01% |
| 14 | 64 | 32 | 4 | 264.418 | 262.688 | 0.9935 | -0.65% |
| 15 | 64 | 32 | 8 | 504.087 | 505.063 | 1.0019 | +0.19% |
| 16 | 64 | 48 | 1 | 256.566 | 255.710 | 0.9967 | -0.33% |
| 17 | 64 | 48 | 2 | 269.303 | 269.058 | 0.9991 | -0.09% |
| 18 | 64 | 48 | 4 | 391.245 | 389.564 | 0.9957 | -0.43% |
| 19 | 64 | 48 | 8 | 746.706 | 748.477 | 1.0024 | +0.24% |
| 20 | 64 | 128 | 1 | 761.363 | 761.472 | 1.0001 | +0.01% |
| 21 | 64 | 128 | 2 | 773.628 | 773.502 | 0.9998 | -0.02% |
| 22 | 64 | 128 | 4 | 1021.277 | 1016.704 | 0.9955 | -0.45% |
| 23 | 64 | 128 | 8 | 1969.806 | 1975.267 | 1.0028 | +0.28% |
| 24 | 72 | 32 | 1 | 146.239 | 146.823 | 1.0040 | +0.40% |
| 25 | 72 | 32 | 2 | 168.430 | 174.046 | 1.0333 | +3.33% |
| 26 | 72 | 32 | 4 | 298.649 | 295.410 | 0.9892 | -1.08% |
| 27 | 72 | 32 | 8 | 570.931 | 571.958 | 1.0018 | +0.18% |
| 28 | 72 | 48 | 1 | 309.545 | 307.098 | 0.9921 | -0.79% |
| 29 | 72 | 48 | 2 | 320.634 | 317.692 | 0.9908 | -0.92% |
| 30 | 72 | 48 | 4 | 440.066 | 436.728 | 0.9924 | -0.76% |
| 31 | 72 | 48 | 8 | 840.571 | 842.117 | 1.0018 | +0.18% |
| 32 | 72 | 128 | 1 | 855.697 | 857.012 | 1.0015 | +0.15% |
| 33 | 72 | 128 | 2 | 870.285 | 869.388 | 0.9990 | -0.10% |
| 34 | 72 | 128 | 4 | 1146.034 | 1139.349 | 0.9942 | -0.58% |
| 35 | 72 | 128 | 8 | 2210.142 | 2216.554 | 1.0029 | +0.29% |

36例耗时比几何平均为 1.000729（+0.073%），逐例范围为
−2.51%～+6.28%。最大下降为 ID12（B64/HV32/T1）；本轮不能声明逐例均不劣于已提交版。
按 T=1/2/4/8 分组的几何平均变化分别为 +0.181%/+0.466%/−0.575%/+0.223%。

### host 调用成本和未验收项

另用 perf_counter 检查 B32/H16/HV32、FP32 V-first、原位更新、三融合开关开启的调用成本。
每例3次预热、5样本；API 时间和 API 后同步的 wall 时间均为均值（μs）。
该检查开始时无待处理工作，不能代表并发队列中的设备全局同步成本。

| 模式 / T | 提交版 API | 对齐版 API | 提交版 API+同步 | 对齐版 API+同步 |
| --- | --- | --- | --- | --- |
| packed / 1 | 81.298 | 68.533 | 141.392 | 131.197 |
| packed / 8 | 71.649 | 69.065 | 352.859 | 351.063 |
| fixed / 8 | 28.749 | 28.098 | 305.737 | 305.160 |

A5 packed 第一阶段需要全设备同步、D2H 元数据复制及 host 扫描，并禁用会复用旧边界的 executor L2 缓存。
当前不支持在 ACLGraph 捕获中调用；边界改变须重新创建 executor。固定长度不增加此同步。
A2/A3 本轮仅交叉编译，尚未通过运行验收。
静态检查仍有 DFX 报错规范问题（详见对齐记录），不能据精度通过声明全部工程门禁通过。

## K-first 与远程主线性能对比（2026-10-29）

结论：本轮72例中，当前方案对齐版全部慢于远程库实现。
远程基线为本轮开始时最新的 `main` 提交
`d77c37382aa8cf53e25005615c8c3f761fc137d2`（19:14:20 +0800）；
当前为 PR #997 的 `d23fb9e8` 加上本文件上一节记录的方案对齐改动，尚未提交。
本轮只新增性能脚本和结果记录，未修改 kernel、tiling、Python 入口或方案原文。

### 条件与版本核验

- A5_ci 同一张 Ascend950PR物理卡4、逻辑卡0，CANN 9.2.0-beta.2、Python3.11.17、torch2.9.0+cpu / torch_npu2.9.0.post6。两版使用同一独立 venv 和同一输入生成器，顺序采集；选卡前及采集后保存 npu-smi 状态，卡4为健康状态。
- issue680 全部36组 shape，B=32/64/72、H=16、HV=32/48/128、T=1/2/4/8、K=V=128；各测试 FP32/BF16 State，共72例。只把模型 case 的 State 改为 K-first，即 `state_v_first=False`，其余属性相同。
- QKV/out 为 BF16，g/beta/A_log/dt_bias 为 FP32；packed TND输入，每个请求长度等于 T，融合 QKV view；QK归一化、raw gate、beta sigmoid 全开；原位更新并返回 final State，无 indices、无逐 token 快照，safe_gate=False。
- 当前版按最大有效序列长度选择 X，因此本矩阵 T=1/2/4/8 分别取 X=1/2/4/8；不是强制全部 X=8。
- 使用 msprof Task Duration，每例3次预热、5次有效采样取均值。只计 RecurrentKda，不计 ViewCopy、输入生成、有限值检查或 host 同步；两版 manifest 中全部 spec、输入 shape/stride/dtype 和属性一致。
- 最新提交与先前构建对应的 `a7c608f2` 只在另一个算子 `chunk_kda_bwd_prepare` 上有差异。核验 recurrent_kda、公共组件、构建依赖和 Python 通路共475个源文件（忽略 CRLF 行尾差异）与最新提交一致，故复用该标准构建。两个版本均核对72个已安装 kernel 对象及 opapi 与各自构建目录逐字节一致，未替换远程版 kernel。

| 构建 | opapi SHA256 |
| --- | --- |
| 远程主线对应实现 | ff0b027a8fb73b51f4547cefa48ef36634afd1903eceb4c555671af0b6c15ec5 |
| 当前方案对齐版 | 0cc3c3d2196fe376b354ca16ffa92dc389df1646393d4f256e0c2ad97c33015b |

### 汇总

比值为当前/远程，越小越好；几何平均针对每种 State dtype 的36例耗时比。

| State dtype | 耗时比几何平均 | 耗时变化 | 逐例耗时比范围 | 当前更快 |
| --- | --- | --- | --- | --- |
| FP32 | 3.5654 | +256.54% | 2.3236～4.7164 | 0/36 |
| BF16 | 2.7137 | +171.37% | 2.0068～3.3403 | 0/36 |

按 T 分组，每组包含9个 B/HV组合；下表仍为当前/远程耗时比的几何平均。

| State dtype | T=1 | T=2 | T=4 | T=8 |
| --- | --- | --- | --- | --- |
| FP32 | 2.4433 | 3.4132 | 4.1365 | 4.6846 |
| BF16 | 2.0595 | 2.6146 | 3.0336 | 3.3200 |

随着 T 增大，本矩阵的相对回退加重。样本内最大变异系数：远程 FP32为0.936%、BF16为1.371%，
当前 FP32为0.113%、BF16为0.237%，均远小于观测到的性能差距。
这是 kernel 耗时对比，不含当前 packed 路径的 host 同步开销；不能将 V-first 的性能结论用于 K-first。
本轮未采集指令流水，不据此直接判定某条 Gather/Scatter 指令的具体耗时。

后续已在 A5_1 采集 ID7 的两版局部仿真流水。两版保留原始56个 block 和生产二进制，每版一次启动、3分钟限时；均为超时截断的局部结果，不能当作完整 kernel 耗时。当前片段存在长延迟 Gather/Scatter，尚未覆盖首 token 全部 State 行。

离散 State 读写诊断：A5_1上9次完整诊断启动均通过结果核对。512 B lane步长命中同一subbank的不同地址；独立访存实验把行距128改为129后，等量读写区间由2.494降至0.170 μs。无下一任务且无并发MTE2时仍可复现；显式MTE2对照未拉长该模型下的VF读写区间。尚未改动生产实现，局部改善不能作为原模型真机加速比。

### 逐例均值（μs）

| ID | B | HV | T | 远程 FP32 | 当前 FP32 | FP32 比值 | 远程 BF16 | 当前 BF16 | BF16 比值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 32 | 32 | 1 | 157.513 | 409.846 | 2.6020 | 103.865 | 213.713 | 2.0576 |
| 1 | 32 | 32 | 2 | 229.852 | 804.340 | 3.4994 | 157.243 | 412.381 | 2.6226 |
| 2 | 32 | 32 | 4 | 379.250 | 1592.251 | 4.1984 | 268.414 | 810.176 | 3.0184 |
| 3 | 32 | 32 | 8 | 674.403 | 3169.831 | 4.7002 | 486.573 | 1605.170 | 3.2989 |
| 4 | 32 | 48 | 1 | 231.984 | 602.711 | 2.5981 | 149.496 | 312.209 | 2.0884 |
| 5 | 32 | 48 | 2 | 339.786 | 1183.989 | 3.4845 | 228.432 | 605.368 | 2.6501 |
| 6 | 32 | 48 | 4 | 554.976 | 2346.439 | 4.2280 | 390.310 | 1190.688 | 3.0506 |
| 7 | 32 | 48 | 8 | 990.047 | 4669.504 | 4.7164 | 715.578 | 2362.515 | 3.3015 |
| 8 | 32 | 128 | 1 | 681.701 | 1584.000 | 2.3236 | 395.547 | 816.487 | 2.0642 |
| 9 | 32 | 128 | 2 | 928.828 | 3119.016 | 3.3580 | 616.452 | 1591.840 | 2.5823 |
| 10 | 32 | 128 | 4 | 1497.138 | 6190.356 | 4.1348 | 1038.529 | 3139.635 | 3.0232 |
| 11 | 32 | 128 | 8 | 2639.837 | 12332.642 | 4.6717 | 1874.365 | 6235.813 | 3.3269 |
| 12 | 64 | 32 | 1 | 318.328 | 797.841 | 2.5063 | 197.685 | 412.050 | 2.0844 |
| 13 | 64 | 32 | 2 | 456.201 | 1564.836 | 3.4301 | 302.536 | 799.160 | 2.6415 |
| 14 | 64 | 32 | 4 | 746.142 | 3099.982 | 4.1547 | 519.585 | 1573.089 | 3.0276 |
| 15 | 64 | 32 | 8 | 1319.823 | 6172.189 | 4.6765 | 939.360 | 3121.273 | 3.3228 |
| 16 | 64 | 48 | 1 | 492.219 | 1183.133 | 2.4037 | 292.676 | 609.396 | 2.0822 |
| 17 | 64 | 48 | 2 | 687.814 | 2323.555 | 3.3782 | 447.672 | 1184.945 | 2.6469 |
| 18 | 64 | 48 | 4 | 1116.327 | 4605.826 | 4.1259 | 763.894 | 2336.617 | 3.0588 |
| 19 | 64 | 48 | 8 | 1964.082 | 9170.093 | 4.6689 | 1398.378 | 4636.941 | 3.3159 |
| 20 | 64 | 128 | 1 | 1345.355 | 3137.047 | 2.3318 | 804.964 | 1617.483 | 2.0094 |
| 21 | 64 | 128 | 2 | 1824.220 | 6186.471 | 3.3913 | 1229.728 | 3155.515 | 2.5660 |
| 22 | 64 | 128 | 4 | 3123.271 | 12285.834 | 3.9336 | 2069.189 | 6230.052 | 3.0109 |
| 23 | 64 | 128 | 8 | 5224.319 | 24486.423 | 4.6870 | 3722.049 | 12381.340 | 3.3265 |
| 24 | 72 | 32 | 1 | 353.125 | 904.713 | 2.5620 | 225.402 | 466.972 | 2.0717 |
| 25 | 72 | 32 | 2 | 517.760 | 1776.208 | 3.4306 | 341.993 | 906.355 | 2.6502 |
| 26 | 72 | 32 | 4 | 844.312 | 3518.809 | 4.1677 | 583.500 | 1786.132 | 3.0611 |
| 27 | 72 | 32 | 8 | 1496.300 | 7004.659 | 4.6813 | 1060.696 | 3543.066 | 3.3403 |
| 28 | 72 | 48 | 1 | 570.628 | 1331.615 | 2.3336 | 330.688 | 685.325 | 2.0724 |
| 29 | 72 | 48 | 2 | 777.889 | 2617.408 | 3.3648 | 512.301 | 1335.408 | 2.6067 |
| 30 | 72 | 48 | 4 | 1257.310 | 5190.385 | 4.1282 | 867.438 | 2632.500 | 3.0348 |
| 31 | 72 | 48 | 8 | 2213.280 | 10336.232 | 4.6701 | 1574.187 | 5227.046 | 3.3205 |
| 32 | 72 | 128 | 1 | 1496.248 | 3519.955 | 2.3525 | 903.834 | 1813.791 | 2.0068 |
| 33 | 72 | 128 | 2 | 2051.293 | 6943.070 | 3.3847 | 1379.480 | 3541.322 | 2.5671 |
| 34 | 72 | 128 | 4 | 3311.508 | 13790.633 | 4.1645 | 2317.824 | 6994.014 | 3.0175 |
| 35 | 72 | 128 | 8 | 5861.361 | 27483.672 | 4.6890 | 4177.395 | 13897.634 | 3.3269 |

### 复现与证据

入口：[profile_kfirst_model_cases.py](scripts/profile_kfirst_model_cases.py)。
分别选择远程/当前安装包，并在进程启动前设置匹配的 OPP、opapi 和 Stable-ABI 动态库路径，
`FLA_NPU_DISABLE_PTH=1`、`ASCEND_RT_VISIBLE_DEVICES=4`。使用同一份当前输入生成器：

```bash
msprof --output="$result" --task-time=on python profile_kfirst_model_cases.py   --package "$package_root" --case-root "$case_root" --device 0   --warmup 3 --samples 5 --manifest "$manifest"
```

正式计时之前，两版分别通过4个小 shape 的 K-first FP32/BF16 调用及有限值检查；
该检查不替代上一节的正式 CPU 精度验收，本轮不额外运行 ATK 性能任务。

原始证据位于本地 `output/` 和远程任务目录 `rkda-kfirst-vs-upstream-20261029`：
`rkda-kfirst-upstream.csv`、`rkda-kfirst-current.csv` 各1152行48列，其中 RecurrentKda各576行，
去除每例前3次预热后，各保留360个有效样本。完整72例的5个原始样本及统计在
`rkda-kfirst-comparison.json`，精简表为 `rkda-kfirst-comparison.csv`。
输入清单为 `rkda-kfirst-{upstream,current}-manifest.json`；版本核验为 `rkda-kfirst-build-audit.json`，
命令、环境和正常退出记录为 `rkda-kfirst-run-final.sh`、两版 perf.log/device-before/device-after、libs.sha256 和 done。
远程版完成时间19:27:51，当前版19:28:42（UTC+8）。

## 2026-10-29：K-first UB 行距 padding

本轮仅改变 A5 K-first 的 UB State 物理排布：FP32 pitch=136、BF16 pitch=144，每行增加32B间隙；GM布局、dtype和接口保持原语义。原方案文档未修改。

A5_1 物理卡7，CANN 9.2.0-beta.2，标准 run 包构建并核对72个 kernel。正式 CPU 精度132/132及对齐精度54/54通过，公开接口74项、确定性9项通过；4个原生 sanitizer 变体覆盖 FP32/BF16 × 最终状态/逐token快照、四个连续任务和跨tile，out/State正确且无内存错误。

msprof Task Duration 每例3次预热、5次采样：issue680的36组形状×两种K-first State中，新/旧耗时几何均值 FP32为0.280507、BF16为0.364433，分别降低71.95%、63.56%；全部72组改善。ID31 BF16的旧版首轮有一个166ms异常样本，两版均整组重测后统计，原样本保留。另12个 V-first case 变化−3.86%～+0.56%。性能未使用 ATK performance_device。

正式 kernel 短请求 simulator 中 State 读写区间 FP32由20.260降至5.213μs，BF16由10.180降至3.232μs，Gather/Scatter数量不变，指令延迟显著下降。V-first短请求完整仿真另有约1.97μs入口差额，生产模型回归未见明显劣化；不以仿真数字代替真机性能。

结果位于 `output/rkda-kfirst-ub-padding-20261029` 及 A5_1 同名 run 目录，包含正式 JSON、msprof 原表、重测数据、中文流水图、二进制身份和 SHA256 清单。A2/A3未上板测试。

## 2026-10-29 当前 UB 排布优化版与最新远程 main 对比

A5_1 物理卡7，CANN 9.2.0-beta.2，msprof Task Duration；远程 main 锁定
`8c4b3691e92c6ac3685742eebe3314618c9bf53b`，475个相关源文件与已验证冻结标准构建一致。
当前为已通过精度与内存验证的 K-first 32B UB 行距优化版；本轮未改实现、未重复 ATK。
issue680 全36组形状×V-first/K-first×FP32/BF16，共144组合；每例3预热、5采样。
13个首轮异常或与同二进制历史验证不符的组合，两版均整组重测，保留全部原始数据。

| 布局 | State | 当前/远程耗时比几何均值 | 耗时变化 | 当前更快 |
| --- | --- | --- | --- | --- |
| V-first | FP32 | 0.8594 | −14.06% | 35/36 |
| V-first | BF16 | 0.9629 | −3.71% | 20/36 |
| K-first | FP32 | 1.0066 | +0.66% | 18/36 |
| K-first | BF16 | 0.9873 | −1.27% | 18/36 |

仍有实际回退：T=8的9组形状耗时比几何均值，V-first FP32为0.8908、BF16为1.1010，
K-first FP32为1.2799、BF16为1.1664。未达到全部模型 case 均不劣于远程的目标。
最终组最大CV：远程1.42%、当前1.41%。
证据位于本机 `output/rkda-upstream-layout-compare-20261029/` 与 A5_1 同名 run 目录。

## 2026-10-29 V-first 最大回退用例流水

选取上述对比中 V-first 最差 ID7：B32/H16/HV48/T8，BF16 State，真机耗时
远程401.972μs、当前448.186μs（+11.50%）。A5_1 上用相同正式 kernel 采集
Ascend950PR_957c simulator 流水：原模型56个 block 各限时3分钟，得到截断的局部图，
不比较截断记录长度判断性能；另以单 AIV、B2/H1/HV1、T8 完整执行两个任务，保留完整递推与排空诊断。
四组正常设备入口均完成，两个小规模仿真均达到 HOST_FINALIZED；原模型仿真均超时，不能当作完整 kernel 流水。

完整诊断中动态 Cast 为4736→10336次，读写次数下降，乘法与归约次数基本相同；
尚未按调用点拆分 Cast，不能据此将全部11.50%回退归因于 State 转换。
证据位于 `output/rkda-vfirst-worst-pipeline-20261029/` 及 A5_1 同名 run 目录。
本轮未修改实现或受保护的方案文档。

## 2026-10-30：按流水瓶颈顺序的四项 A5 优化

在已验证的 K-first UB 行距 padding 版本上，依次采用 v 批量转换、State Store 先于 out 收集、V-first State 加载/转换/乘法交错、当前 VF 先 Dispatch 后提交未来预取。保持 State 的 UB dtype、两个 bank 交替递推、可用第三 bank 预取、完整 tile 后批量搬出 out 和必要同步。

A5_1 物理卡7，CANN 9.2.0-beta.2；性能用 msprof `Task Duration(us)`，每例3次预热、5次采样。issue680 全36个模型形状 × V/K-first × FP32/BF16，共144组，相对同卡重测 baseline 的几何均值：V-first FP32 −3.30%（35/36更快）、V-first BF16 −3.51%（36/36）、K-first FP32 −4.25%（36/36）、K-first BF16 −3.18%（36/36），全部 −3.56%（143/144）。ID7 V-first BF16 为448.401→432.535μs（−3.54%）。

132个原有精度用例、54个对齐精度用例、74项公开契约和9项确定性测试通过。FP32/BF16 × 两种布局 × 最终 State/逐 token 快照的8组原生内存检查均通过，mssanitizer 无错误；每组覆盖4个连续任务、8个 tile，快照场景完成36次快照和3次下一任务 State 预取。输出和 State 的完整 simulator 诊断逐位一致；v 转换指令 `RV_VCVT_F2F` 10336→8320、`RV_VDUP` 4128→2080，完整局部流水38.338→36.525μs。

证据位于 `output/rkda-vfirst-ordered-optimizations-20261029/` 及 A5_1 同名目录；未修改受保护的方案文档。

## 2026-10-30：寄存器指令优化

最终保留每token公共beta×k与MulAddDst组合，仅用于V-first和K-first BF16；K-first FP32保留原乘加路径。MERGING输出收集通过精度，但叠加后无稳定额外收益，已回退。整tile完成后批量搬出out的方式保留。

标准run安装、libcust_opapi MD5及72对象审计通过；132例原有精度、54例对齐精度、74项公开接口及9项确定性通过，未修改CPU标杆或精度标准。本轮不重复内存检查，地址、UB数据类型、State生命周期及同步未变。A2/A3未上板测试。

A5_ci物理卡4，CANN 9.2.0-beta.2，msprof Task Duration，3预热+5样本；issue680全36形状×两布局×两State dtype=144组，同卡对比本轮开始的Stage16与最终reg20。V-first FP32耗时几何均值−4.15%，V-first BF16−6.81%，K-first FP32+0.009%（持平），K-first BF16−4.88%；全部−3.99%，122/144更快。最大单组增加0.83%，未超该组采样波动。ID7 V-first BF16为428.891→395.665μs。共享机器存在其他进程，保留前后快照及全部样本，最终逐组最大CV=1.63%，没有替换样本；前期明显受干扰的轮次不计入汇总。

本地证据在`output/rkda-register-optimizations-20261030`；构建/精度在A5_1的`rkda-vfirst-ordered-optimizations-20261029`，正式性能在A5_ci的`rkda-register-optimizations-20261030`。未修改受保护的方案文档。

## 2026-10-30：当前寄存器优化版与远程最新 main 对比

基线为本轮显式拉取的远程 `refs/heads/main`：`4cbcd01187c98afd0d8b187f06c3d4227f52de4c`。
该提交相对冻结标准包 `8c4b3691` 仅有两处 CI 文件修改，算子及构建源码不变；核对475文件来源、72对象和两份动态库身份后复用该包。
当前为已验收的 reg20。本轮仅比较性能，未修改实现或受保护的方案文档。

A5_ci物理卡4，CANN 9.2.0-beta.2；msprof Task Duration，每例3次预热、5次正式采样。
issue680全36形状×V/K-first×FP32/BF16 State=144组，每版1152条目标kernel记录，全部通过数量、设备和block数校验。
当前X按最大序列长度选择1/2/4/8。共享设备保留占用快照，最大组内CV为远程2.39%、当前1.15%，未替换样本或复测组。

| State布局/类型 | 当前/远程耗时比几何均值 | 耗时变化 | 当前更快 |
| --- | --- | --- | --- |
| V-first FP32 | 0.7963 | -20.37% | 36/36 |
| V-first BF16 | 0.8636 | -13.64% | 36/36 |
| K-first FP32 | 0.9621 | -3.79% | 18/36 |
| K-first BF16 | 0.9106 | -8.94% | 23/36 |
| 全部 | 0.8810 | -11.90% | 113/144 |

仍有实际回退：K-first FP32在T=4为+10.77%、T=8为+23.53%；K-first BF16在T=8为+8.35%（均为9组耗时比几何均值）。
最大回退为ID7 K-first FP32：B32/H16/HV48/T8，986.271→1230.563μs（+24.77%）。
因此不能判定全部模型用例都已优于远程实现。V-first BF16的ID7为401.745→395.555μs（-1.54%）。
证据保存在本机`output/rkda-upstream-register-compare-20261030`及A5_ci同名run目录。

## 2026-10-30：分布局寄存器优化

按V-first点积融合、K-first BF16打包、K-first点积与访存调度的顺序独立实验。最终layout26保留V-first点积融合，以及K-first BF16的AND打包、点积融合和成对加载/转换/计算交错。K-first FP32点积融合典型用例增加1.18%，访存交错无稳定收益，两者均回退。保留整tile完成后批量搬出out、State的UB类型、必要同步与预取。

layout22～26均完成标准run构建安装、libcust_opapi校验、72对象审计和132+54例精度。最终layout26另外通过74项公开接口、9项确定性（每项3次）；标杆、值域与阈值未改。当前源码与构建源18文件核对通过，A5_1与A5_ci的468项so/o SHA256一致。完整小规模BF16 simulator中，V-first每token净减256条Vector算术，K-first每token另减128条打包位操作；必要归约和同步数量不变。此轮未重复内存检查，A2/A3未上板；静态扫描仍有当轮114项host DFX-MESSAGE问题，两个修改的kernel头文件无发现。

A5_ci物理卡1，msprof Task Duration，每例3预热+5正式样本，issue680全36形状×V/K-first×FP32/BF16 State=144组。相对本轮开始的reg20，耗时比几何均值如下；此表不是与远程main的对比。

| State布局/类型 | layout26/reg20 | 耗时变化 | 均值更低组数 |
| --- | --- | --- | --- |
| V-first FP32 | 0.9927 | -0.73% | 22/36 |
| V-first BF16 | 0.9645 | -3.55% | 35/36 |
| K-first FP32 | 0.9996 | -0.04%（持平） | 26/36 |
| K-first BF16 | 0.9572 | -4.28% | 36/36 |
| 全部 | 0.9783 | -2.17% | 119/144 |

V-first FP32在T=8的9组均更快、几何均值-2.59%；T=1为+0.95%，单组最大+5.41%。针对5个FP32短序列及3个控制/波动用例反向顺序成对复测后8组均更快，连未改算术的K-first FP32也发生变化，不能用组内低CV排除跨版本时段漂移。因此短序列没有稳定逐例收益，也未判定全部模型用例无回退。完整144组和8组复测独立保留，未替换样本。完整轮次有1组CV=3.36%，复测最大CV=1.46%；卡4先前明显异常的整轮已排除。

ID7 V-first BF16为395.700→379.029μs。证据位于本机`output/rkda-layout-register-optimizations-20261030/{a5-1,a5-ci}`及两台服务器同名run目录。受保护的原方案文档未修改。

## 2026-10-30：K-first FP32连续V方向寄存器优化

最终保留kfp29：保持K-first物理布局，寄存器按连续V方向计算。第一遍累计prediction，第二遍更新State并累计out。State的UB读取量加倍，省去State Gather/Scatter、逐行归约及四行out收集；整tile计算后批量搬出out、State dtype、bank复用、预取和必要同步保留。仅out点积融合kfp27（+0.53%）和四行块内两行分组kfp28（+2.01%）均已放弃。

标准run安装、libcust_opapi核对、72对象审计、18文件源码校验和跨服务器312项so/o身份核对通过。kfp29通过132+54精度、74项公开接口、9项确定性（每项3次），CPU标杆、值域和阈值未改。使用当前Pipeline头文件编译的原生sanitizer探针在K-first FP32最终State和逐token快照两组均No error detected，每组覆盖4任务、T9、X8及跨tile/bank预取；这是探针入口的检测，不冒充正式RecurrentKda包的mssanitizer报告。

A5_ci物理卡1，msprof Task Duration，每例3预热+5正式样本。同卡对比layout26与kfp29，issue680全部36种形状的K-first FP32耗时比几何均值0.407131（**-59.29%**），36/36正式均值更低，最大组内CV1.80%，未替换样本。ID7为1230.450→343.627μs（-72.07%），ID35为7210.711→1965.797μs（-72.74%）；T1大HV组收益较小，最小约1.77%。此表口径为本轮开始的layout26，不是远程main。

另采ID7三组控制：V-first FP32 +0.61%、V-first BF16 -0.15%、K-first BF16 +0.09%；未重新采集其它布局全部模型性能，不宣称所有布局性能完全相同。其它布局/dtype的算术路径保留。完整16token K-first FP32 simulator事件61762→45336（-26.60%），区间93.012→31.187μs，仅用于诊断。必要的逐token同步保留，vFloat scratch不再使用，其专用屏障减少16条。

静态issue覆盖188项通过，仍有当轮114项host DFX-MESSAGE发现，修改的kernel头文件无发现；A2/A3未上板，原始受保护方案未修改。本地证据`output/rkda-kfirst-fp32-register-20261030/{a5-1,a5-ci}`，服务器沿用上一轮run目录，以kfp前缀区分。

## 2026-10-30：K-first BF16连续V方向验证

保留kbf30，将kfp29的连续V方向两遍计算推广到BF16。UB State只保留BF16，每次Load后在寄存器转FP32，prediction、递推及out均为FP32；先用未舍入State累计out，再Cast_RINT写另一bank。State有效UB读取由32增到64KiB/token，写入仍32KiB；不增加UB容量或GM State流量，保留必要同步、可用第三bank预取、完整tile后批量搬出out。V-first算术路径保留。

标准run构建安装、libcust_opapi MD5、72对象审计、18源文件及两服务器156项so/o SHA256一致。132原有+54对齐精度、74公开契约、9确定性（每项3次）通过；CPU标杆、值域、容差未改。当前Pipeline原生sanitizer探针的BF16 K-first最终State/逐token快照两组均No error detected，每组4任务、T9、X8，覆盖跨tile及bank复用预取；这是state_pipeline_probe入口报告，不冒充正式RecurrentKda包内存报告。K-first FP32小规模out/State与kfp29逐位一致。

A5_ci物理卡5，msprof Task Duration，3预热+5正式样本；严格串行重测kfp29与kbf30，覆盖issue680全36种形状的K-first BF16，耗时比几何均值**0.497276（-50.27%）**，36/36均更快，最大组内CV1.17%。ID7为737.222→342.829μs（-53.50%），ID35为4313.081→1989.751μs（-53.87%）。同采ID7控制：V-first FP32 -0.54%、V-first BF16基本持平、K-first FP32 -0.11%；未测其它布局全部模型性能。初次两版采集约1.076秒交叠，整对排除且原始记录保留；最终串行数据没有替换样本。此处基线是本轮kfp29，不是远程main。

完整16token BF16 simulator事件59585→57613（-3.31%），区间56.941→30.027μs，仅作诊断；连续State读写消除Gather/Scatter、逐行归约与out收集依赖，额外BF16转换使总指令降幅较小。静态issue覆盖188项通过，当轮114项host DFX-MESSAGE发现仍在，修改的kernel无发现；A2/A3未上板，受保护原方案未修改。

精度/内存在A5_1卡6。本地证据`output/rkda-kfirst-bf16-register-20261030/{a5-1,a5-ci}`，服务器沿用`rkda-layout-register-optimizations-20261030`，以kbf前缀区分。

## 2026-10-30：V-first寄存器delta候选验证（未采用）

以kbf30为基线分别验证vf31（四行prediction打包、一次Sub计算四个delta，直接消费v寄存器）和vf32（保留独立prediction/Sub，仅将vFloat scratch替换为寄存器读取）。两版均完成标准run构建安装、libcust_opapi MD5、72对象、18源文件及跨机156项so/o核对，各通过132原有+54对齐精度，小规模FP32/BF16 out/State均与kbf30逐位一致；CPU标杆、值域、容差未改。

A5_ci物理卡5，msprof Task Duration，每例3预热+5正式样本。issue680 ID0/2/3/7/20/35×V-first FP32/BF16，共12组合，每轮重新采基线并串行运行。vf31的FP32/BF16耗时比几何均值分别+22.50%/+20.80%，vf32分别+0.72%/+0.97%，均无可保留的性能收益；未扩展完整72组合，也未重复契约/确定性/内存验收。没有替换样本，最大组内CV分别1.11%/2.06%。

完整16token BF16 simulator：kbf30为50192事件/32.422μs，vf31为48403/40.311μs，vf32为50451/32.848μs，仅用于诊断。vf31静态减少指令却增加prediction打包依赖；vf32省去scratch写读和专用屏障，但寄存器Gather/轮换没有净收益。两版均未减少逐行归约；不能据此声称V-first已无其它优化空间。

当前本地和A5_1源已恢复kbf30，隔离venv通过原run包重新安装，18源文件及78个已安装so/o身份一致；保留此前已验收的K-first FP32/BF16优化。受保护原方案及正式设计未因淘汰候选改变。静态issue覆盖188项通过，当轮114项host DFX-MESSAGE发现仍在，实验kernel头文件无发现；A2/A3未上板。精度/正常启动在A5_1卡6。本地证据`output/rkda-vfirst-delta-register-20261030/{a5-1,a5-ci}`，服务器沿用前轮run目录，以vf前缀区分。

## 2026-10-30：V-first State两遍读取验证

保留v2r34：X=1使用原四行单遍，X>1第一遍计算prediction并将128个delta批量写入原512B scratch，第二遍重读State、更新并计算out。VF外选择内部模板，不增加VF循环内判断；变长调用以最大请求长度选公共X，X>1中的短请求和单token尾tile仍执行两遍。State的UB读取量加倍，写入量和UB峰值不变，BF16只在寄存器转换，out使用未舍入的FP32更新值。保留必要同步、第三bank预取及整tile完成后批量搬出out。K-first两遍算术保留。

v2r33对所有X使用两遍，典型12组合有收益，但完整模型中FP32单token最大回退7.75%、BF16最大2.00%，因此未直接采用。最终v2r34同卡重采基线，独立保留两轮完整数据，未替换样本。基线是本轮开始已验收的kbf30，不是远程main。

A5_ci物理卡7，msprof RecurrentKda Task Duration，每例3预热+5正式样本；issue680全部36种形状×V-first两State dtype，共72组目标，另含ID7 K-first两dtype控制。正式两版串行无交叠，全部原始CSV与逐例占用保留。

| State | 耗时比几何均值 | 耗时变化 | T=8耗时变化 | 均值更低组数 | 最大回退 |
| --- | --- | --- | --- | --- | --- |
| V-first FP32 | 0.949641 | -5.04% | -12.22% | 28/36 | +0.97% |
| V-first BF16 | 0.940951 | -5.90% | -8.14% | 29/36 | +1.10% |

整体72组耗时比几何均值0.945286（-5.47%），57/72均值更低，最大组内CV1.61%。ID7 FP32为333.863→292.588μs，BF16为380.511→349.592μs；ID35分别1920.490→1683.316μs、2210.913→2029.411μs。K-first ID7控制FP32 +0.24%、BF16 -0.12%，未重复其它布局全部模型性能，不能宣称所有case加速。

v2r33和v2r34各通过标准run构建安装、libcust_opapi MD5及72对象核对；各通过132原有+54对齐精度、74公开契约、9确定性（每项3次），CPU标杆、值域、容差未改。每版18源文件审计一致，基线+候选156项so/o跨机SHA256一致，78个已安装候选so/o与封装包一致。最终本地源码与v2r34构建快照一致。

精度及内存在A5_1物理卡6。两候选各4组FP32/BF16×最终State/逐token快照的当前Pipeline原生sanitizer探针均No error detected，out/State探针对比通过；每组4连续任务、T9、X8，覆盖跨tile、bank复用和第三bank预取，追踪版49/157条记录完成并HOST_FINALIZED。v2r34另通过两dtype X1/T1/逐token快照2组memcheck。以上是state_pipeline_probe入口的检测，不冒充正式RecurrentKda包的mssanitizer报告。

正式包小规模完整16token simulator中，v2r34 FP32事件42016→46328、区间30.004→27.560μs，BF16事件50192→58576、区间32.422→30.692μs。State额外Load/Cast及prediction收集使指令增加，收益来自依赖组织和生命周期改善；逐行VCADD未减少，不据此宣称总指令下降。仿真只用于诊断，模型性能采用上方真机msprof数据。

静态issue覆盖188项通过；仍有当轮114项host DFX-MESSAGE发现，两个修改kernel头文件无发现；A2/A3未上板。正式设计已同步采用的计算路径，受保护原方案未修改，开发期validation记录已收敛删除。本机证据`output/rkda-vfirst-two-pass-20261030/{a5-1,a5-ci}`及两zip逐文件SHA256索引；服务器沿用`rkda-layout-register-optimizations-20261030`，以v2r前缀区分。

## 2026-10-30 当前v2r34与远程main补充采集：共享占用，未形成可靠整体结论

远程main冻结为`6d7d32ccec99a931fe473b95779ec53cc5f76cbc`；与已核验的`8c4b3691`标准构建包相比，仅CI、文档和管理脚本有差异，算子/运行时/构建源码不变。当前为撤回BF35/36/37后的v2r34，18源文件及两版各78个so/o身份核对通过。本轮仅msprof计时，未修改、构建或安装算子，未重新运行ATK。

A5_ci物理卡3、逻辑0，两版串行覆盖issue680全部36形状×V-first/K-first×FP32/BF16共144组合；每组3预热+5正式样本。当前在第84组调用前被npu-smi查询失败中断，83组原始数据保留，仅补采剩余61组；高波动的K-first BF16 ID8另做一次完整成对复测，不在组内筛选样本。

最终逐例占用审计发现83个组合在原采集或复测时存在其他Python/ATK进程，ID8复测也有共享占用。本轮整体结论为**未获得可靠性能对比**，不将受干扰的耗时比作为144组全部加速或验收依据；已有精度与此前已验收性能记录不由本轮替代。原始CSV、重建manifest来源、占用、复测与SHA256索引见`output/rkda-upstream-v2r34-compare-20261030/`。受保护方案保持原SHA256不变。

## 2026-10-30 逐行计算与residual寄存器驻留实验（rdelta39，未采用）

按用户要求取消X>1候选四行State共同计算：两遍均逐V行执行，residual与完整out保留在寄存器，整tile完成后批量搬出out。X=1和K-first保留控制路径。标准run构建安装、72对象核对和两版各78个so/o身份通过；132原有+54对齐精度通过，小规模正式原生对象两种State dtype的out/State均与v2r34逐位一致。

A5_ci物理卡1/逻辑0，同卡串行msprof Task Duration；issue680 ID0/1/2/7/35×V-first两dtype及ID7 K-first控制共12组，每组3预热+5正式。用户明确允许共享设备，两版逐例快照均未观察到其他进程。8个X>1目标组全部变慢，候选/基线耗时比几何均值：FP32 1.8844（+88.44%）、BF16 1.6172（+61.72%）；全部12组最大组内CV为2.34%。没有组内挑选、删除或替换样本。

候选同时改变计算粒度与residual传递，不能把全部回退只归因于寄存器驻留，也不能推断所有非四行实现都更慢。停止扩展本候选，生产源和标准安装恢复v2r34，18源文件及78个已安装so/o核对通过；原方案保持不变。本机候选源码、封装包、原始CSV、正式精度与恢复证据为`output/rkda-vfirst-residual-register-20261030/`。候选在典型阶段淘汰，未扩展144组合及额外契约/确定性/内存验收。

## 2026-10-30 四条路径独立选择指令交错调度（vuf43）

按用户要求取消四行计算约束，依次测试V-first两dtype第一遍、第二遍及K-first两dtype的相邻K准备，共六候选，每次仅修改指定路径。保留V-first BF16第一遍双行Load/Cast/点积交错及FP32第二遍滚动双行；BF16第二遍、FP32第一遍、X=1和K-first保留v2r34。四行仅作为原out收集粒度；512B residual UB、必要同步、State接口dtype驻留、bank预取及整tile后批量搬出out保持。

六候选各54项对齐精度通过，最终vuf43另通过132项原有精度；74契约、9确定性（每项3次）、正式包四路径out/State共8项与v2r34逐位比较通过。首次确定性因私有脚本覆盖CANN PYTHONPATH导致acl导入失败，虽ATK返回0但9项未执行；失败日志保留，修正环境后正式报告9项SUCCESS且确定性计算结果均true。当前Pipeline sanitizer探针四路径×最终State/快照8组均No error detected；X8/T9覆盖跨tile及bank复用。该内存证据为state_pipeline_probe入口。标准run安装、72对象、18源码及78已安装so/o身份核对通过。

A5_1物理卡4/逻辑0，同卡串行msprof，issue680全36形状×四路径144组，3预热+5正式，Task Duration均值；全组前后均记录共享进程，按用户授权继续。相对v2r34耗时比几何均值：V-first BF16 0.987935（-1.21%）、FP32 0.995881（-0.41%）；T8分别-1.74%/-0.97%。K-first FP32 0.999735、BF16 0.999994，计算流程保持；最大组内CV1.87%，不宣称全部case加速。两版CSV各2304行×48列，其中目标各1152条，56 block，未挑选样本。

完整小规模BF16 V-first simulator：B2/H=HV1/T8/X8/一个AIV，完整16token。两版RV指令调用均56454，全部指令种类的调用数逐项相同；VF累计39343→38597周期（-1.90%），支持调度改善，未直接测得双发率。静态188 issue覆盖通过；当轮114项host DFX发现保持，kernel无新发现，A2/A3未上板。原方案未修改。完整144组逐例结果、命令、精度/内存范围及265文件SHA256归档证据为`output/rkda-dual-issue-20261030/`。

## 2026-10-30 四路径独立VF实验（ks54）

以vuf43为基线，分别实验V-first BF16逐行第二遍/逐行单遍、V-first FP32八结果收集、K-first两dtype双累加链和prediction系数UB广播，共七候选，各54项对齐精度通过。V-first候选耗时增加4.58%/20.65%/92.32%，撤回并保留vuf43。最终ks54只采用分别验证有收益的K-first FP32/BF16系数UB广播；复用既有512B尾部，新增两次VST及必要VST→VLD，不增加UB峰值，State dtype、Cast舍入、bank预取及整tile out批量搬出保持。

A5_1物理卡4/逻辑0，正常标准构建包，同卡串行msprof RecurrentKda Task Duration，issue680全36形状×四路径144组，每组3预热+5正式。相对vuf43耗时比几何均值：K-first FP32 0.988042（-1.20%，33/36均值更低）、BF16 0.971926（-2.81%，36/36更低）；T8分别-2.34%/-3.51%。V-first FP32/BF16为1.003262/1.001260，计算路径保留，不作为加速收益。两版CSV各2304行×48列，目标各1152条、56 block，全样本保留。

按用户授权共享设备，逐例前后均记录其他进程。原144组最大CV7.28%发生在V-first FP32 ID34（+12.72%）；单独成对复测为888.145→889.992μs（+0.21%，两版CV低于0.4%）。复测不替换原样本；K-first ID34复测仍改善1.73%/3.26%。共享条件下的均值不等于独占设备绝对性能，也不宣称所有case加速。

ks54正式186项精度（132原有+54对齐）、74契约、9确定性各3次通过；四路径小规模正式原生对象out/State共8项与vuf43逐位一致。当前Pipeline sanitizer探针10组No error detected，均bad_out=0 bad_state=0且HOST_FINALIZED：四路径×最终State/快照、X8/T9共8组，另有K-first两dtype X1/T1快照2组。该内存证据为state_pipeline_probe入口。首次辅助内存脚本CRLF失败日志保留，改LF后补跑完成；未重复已通过的精度和模型性能。

标准run安装、72对象、18源码及78已安装so/o身份核对通过。完整16token K-first BF16 simulator事件57623→55658，区间29.792→29.561μs；每token少128次VSELR及128次VADDS，增加128次VLD、2次VSTI、1次SMEM_BAR，未直接测得双发率。静态188 issue覆盖通过，仍有当轮114项host DFX发现，kernel无新发现；A2/A3未上板。原方案SHA256保持不变，正式design.md同步系数scratch生命周期，临时validation记录收敛删除。

本地证据`output/rkda-four-path-vf-20261030/`，其中a5-1-evidence.zip含577文件及逐文件SHA256索引，CRC和全部hash核验通过。

## 2026-10-30 当前ks54与远程最新main分路径性能对比

远程main冻结为`2bb7e74ecf4876acbc4527070e403bffeef06e75`，本轮在A5_1新隔离venv实际重新构建并标准安装recurrent_kda；1841个上游源文件、72对象及78个so/o核对通过。当前为已验收ks54，18源文件及78个so/o与验收快照一致。本轮不修改算子，不运行ATK任务；两版四路径正常启动与有限值检查通过。
A5_1物理卡4/逻辑0，CANN 9.2.0-beta.2，同卡串行msprof Task Duration；issue680全36形状×四路径144组，3预热+5正式，全样本保留，按用户授权共享设备并逐例记录占用。下表为当前/远程耗时比几何均值。

| 路径 | 当前/远程 | 耗时变化 | 当前均值更低 |
| --- | --- | --- | --- |
| v-bf16 | 0.779211 | -22.08% | 36/36 |
| v-fp32 | 0.723268 | -27.67% | 36/36 |
| k-bf16 | 0.421052 | -57.89% | 36/36 |
| k-fp32 | 0.350022 | -65.00% | 36/36 |

最大组内CV为2.51%，共享条件下采集，不声明独占设备绝对性能。逐例样本、占用、异常与限制均保留，不用总体均值代替逐case判断。完整原始144组及按T汇总的本机证据`output/rkda-upstream-ks54-compare-20261030/`，436文件zip的CRC及SHA256索引全部通过。首次安装缺少wheel元数据、首次smoke缺少common输入辅助目录的失败日志保留，均在正式性能采集前修正，未据此重复模型性能。

## 2026-10-30 PR #997 检视整改验收

修复非原位 State 未写槽、旧形状调用兼容、公开参数结构化错误及负向断言，整理公开报告和资产。冻结方案未修改。A5_1 上完整 wheel 构建、标准 run 安装及 libcust_opapi MD5 核对通过，22个构建源文件与78个已安装 so/o 身份核对通过。

原有132项与对齐54项正式精度全部 SUCCESS，CPU 精度字段全部为 true，未改变 CPU 标杆、值域或容差。stable、ctypes、直接 ACLNN 三入口共48项正常场景通过，覆盖两布局/两 dtype 的空请求、稀疏 pool、非零 State offset 和旧形状别名；18项负向检查断言具体返回码/消息，每次拒绝后合法调用及 CPU 对比通过。74项公开契约及 mutation/grad 通过。

本轮72个设备对象与已验收 ks54 全部 SHA256 相同，kernel、X、bank 同步和整 tile out 批量搬出保持；按用户要求不重复 msprof，上一节144组仍是历史性能记录。非原位复制、旧别名物化及新增 host 校验的端到端开销未测，不由对象一致性推断为不变。A2/A3、整体 Example/ST 和新 head 正式 CI 未执行。

静态192项 issue/21条规则覆盖通过；剩余36个 DFX 机械命中已区分内部失败与已结构化叶子的传播，10个敏感项为命令误匹配，3个断链在受保护方案中，未宣称零发现。历史114项是当轮扫描位置，其中19个位于 PR 新增行，不能统称为 PR 前已有问题。

私有证据 `output/rkda-review-fixes-20261030/evidence.zip` 共94文件，CRC 和逐文件 SHA256 全部通过；包 SHA256 为 `9bfa902ab698ac968139c03df9703ac9df348f3594c3d53306767101ff7a7caf`。包含正式精度任务行、日志、对象/源文件身份和复现脚本；不作为公共可点击工件。开发期 validation.md 已归并删除，受保护方案内的历史引用按用户要求保留。

## 2026-10-30 PR #997 二轮检视整改与三版分路径复测

修复原位独立finalState的逻辑view复制、活跃indices/accepted的host值校验、safe gate的NaN范围漏检，并将A5行为类拆为纯数据context及文件作用域Stage函数。冻结方案未修改。标准run构建安装、libcust_opapi MD5、21个实现/配置/回归源文件以及三版各78个so/o核对通过；30个原方法的可执行token经context/名称映射一致，但72个设备对象哈希全部改变，本轮重新采集性能。

132原有+54对齐正式精度全部SUCCESS且CPU精度字段为true；56项三入口正常回归、42项负向回归和74项公开契约及mutation/grad通过。未修改CPU标杆、输入值域或容差。新增正常回归覆盖原位独立输出的两布局/两dtype/连续与带间隔view；非法值拒绝后都执行合法调用并与CPU比较。

A5_1物理卡4/逻辑0，同卡串行msprof，远程main为2bb7e74ecf4876acbc4527070e403bffeef06e75，整改前为ede1185b；issue680全部36形状×四路径共144组，每组3预热+5正式，所有样本保留。各版CSV2304行×48列，目标kernel各1152条、56block。下表为每路径36组耗时比几何均值。

| 路径 | 当前/main | 耗时变化 | 当前/整改前 | 耗时变化 | 比 main 更快 |
| --- | --- | --- | --- | --- | --- |
| V-first BF16 | 0.776466 | -22.35% | 0.998342 | -0.17% | 36/36 |
| V-first FP32 | 0.718228 | -28.18% | 0.997681 | -0.23% | 36/36 |
| K-first BF16 | 0.422619 | -57.74% | 1.002541 | +0.25% | 36/36 |
| K-first FP32 | 0.348717 | -65.13% | 0.999969 | -0.00% | 36/36 |

最大组内CV2.50%，相对整改前最大单例增加1.07%；设备经授权共享，三版各288次前后查询都观察到其他进程，无查询失败，不宣称独占设备结果或所有小差异显著。小规模有/无indices端到端补测波动很大（最大CV139.66%），只保留原样本，不据此量化host成本；索引校验新增同步/D2H和快照开销，无indices模型路径不增加该步骤。

192项issue/21规则覆盖通过；静态扫描仍有已分类的内部故障/叶子传播及文档误匹配，未宣称零发现。当前对象未重跑历史确定性、sanitizer或simulator，A2/A3、整体Example/ST与正式CI未执行。私有证据zip共687文件，CRC和逐文件SHA256均通过，包SHA256为`96cc8c88fab4312b5b0d5bfd9a6f77070cd8588959df2424f3402a9a9680202e`。开发期validation记录已归并删除。

## 2026-10-31 PR #997 当前提交 A2 上板失败记录

被测提交 `be0ecad8a739de5b205d1a229058d58d2407a89d`，A2 910B3 物理卡4，CANN 9.3.0 weekly、torch/torch_npu 2.9.1、ATK 26.9.8。独立 venv 标准 wheel 构建安装及 libcust_opapi、72 个 kernel 对象核对通过，未修改算子实现和冻结方案。

`B=2,T=9,H=1,HV=2,K=128`，V=128/256 × V-first/K-first × State FP32/BF16 的8例冒烟全部触发 `507035 / vector core exception`，设备报 Scalar 地址异常，部分用例同时明确报 Vector UB 越界。普通 T=2 的原有第0例也复现。ATK 异常路径返回0不代表通过；本轮无有效 out/State 精度比较结果，未继续完整精度、公开接口、确定性、sanitizer 或性能采集，正式验收结论为未通过。

确认 A2 host 固定 UB 预算只计算一槽 packed 输入，而 generic kernel 实际分配两槽，K=V=128 时少算10 KiB。T=2 的实际 tiling 数值对应总分配204160 B，超过可用196352 B；修复后尚未重测，不能断言这是所有异常的唯一原因。私有证据79文件的CRC及SHA256全部核验通过。A3未上板；本轮未提交或推送PR。

## 2026-10-31 A2 UB 修复与 A5 X=1 两遍读取

先 rebase 到远程 main `34851b76`，原6个提交等价重放，head `c7af2aaa`。A2 host 恢复双槽 packed 预算、实际 NV 的 beta/窗口预算及 tmpBytes 容量检查。标准完整 wheel 重建安装后，原失败8例、234例正式精度和17例各3轮确定性均通过；out/State 精度字段逐项为 true。A2 无 indices 的非法 cu_seqlens 仍未被公开入口拒绝，两项扩展脚本失败于 cu_start；不宣称所有公开契约通过。

按用户选择将 A5 V-first X=1 fused/general 分派也切换为两遍 State 读取，与已验收候选逐字一致。186项原精度通过；A5_1物理卡5用 msprof 对 issue680 全36形状、四路径采集144组，每组3预热、5正式，所有样本保留。V-first T=1的9例耗时比几何均值：FP32 -4.33%、BF16 -6.66%；最差分别为 ID28 +3.51%、ID20 +1.04%。其余路径为控制组，设备有其他任务共享，不能把所有小差异解释为优化收益。

## 2026-10-31 PR #997 最终检视与双平台验收

本节取代上节“无 indices CU 尚未拒绝”的中间结论；历史失败保留。按最新 main 的 operator-review skill 修复 A2/A3 实际双槽及 scratch 预算、所有平台 CU 值校验/缓存、host opbase 报错接口，并补齐 Python 元数据生产者就绪依赖。冻结方案未修改，公开签名与已发布 tag 一致；layout/gate 的用户确认语义及旧 shape 迁移别名保留。

| 项目 | A5（A5_1，物理卡5） | A2（物理卡4） |
| --- | --- | --- |
| CPU 双输出精度 | 188/188（186+2项 X=1 BF16） | 234/234 |
| 确定性 | 11项，各50轮 | 17项，各50轮 |
| 插桩 memcheck | 11项，无错误 | 17项，无错误 |
| 公开契约与 mutation/grad | 74项 | 74项 |
| 三入口扩展 | 59正常、49负向 | 59正常、49负向 |
| 元数据就绪专项 | 20项 | 20项 |

两平台分别标准构建并核对安装产物。memcheck 使用独立 sanitizer wheel，72条设备编译命令和72个插桩对象均核验；未以正常构建替代。元数据专项 `python tests/atk/recurrent_kda/scripts/check_metadata_readiness.py --device 0` 在默认和独立 stream 中不手工预同步，覆盖 stable/ctypes 正常双输出及非法 accepted/CU；随后再次运行公开契约和三入口回归。

A5 用 msprof 对最新 main `34851b76` 与 PR 采集 issue680 全36形状×四路径，每组3预热、5正式。首轮 PR/main 耗时比几何均值：V-first BF16 0.761890、V-first FP32 0.690407、K-first BF16 0.422191、K-first FP32 0.351471。main 的7组 V-first FP32 噪声较大，反序复核后最大 CV 0.74%；29组首轮加7组复核的诊断汇总为0.708479。原样本全部保留；这不是第二轮完整144组对比，也不是端到端调用耗时。

A3、整体 Example/ST、正式 NPU CI 及内部故障注入未执行。

## 2026-10-31 新检视意见：移除元数据全设备同步

针对 PR #997 discussion_r4430769129，在调用方元数据就绪契约下移除 CU 和 indices/accepted 读取前的显式全设备同步，保留同步 D2H、活跃值校验、executor 快照和必要的值依赖缓存禁用。Python 仍等待自己的当前 stream，原始 ACLNN 回归显式保证生产者就绪；不是直接删除生产消费依赖。

A5/A2 各重新通过20项元数据就绪、74项公开契约、59正常/49负向三入口回归，以及16项具体 EZ 码和拒绝后4条 CPU 健康路径。两平台各72个设备对象与旧对照一致，本轮未重跑完整188/234项矩阵和 sanitizer，前节完整验收属于修改前版本，不能混为新一轮全量结果。

A5 同卡 msprof 对 issue680 ID0/7/35 × 四路径，每组3预热、5正式；12组耗时比几何均值0.997326，最大组内 CV 2.59%，设备计算无实质变化。额外测量8次连续 decode、两个独立 head stream 的调用链时间，并以无关 stream 的128次4096×4096 FP16 matmul 验证隔离。反序控制中，修改前5次返回时无关工作均完成、enqueue 47.207–57.724 ms；修改后5次返回时均未完成、enqueue 0.431–1.827 ms。微小负载双 head 仍有 host 调度开销和波动，未宣称所有端到端场景稳定加速。

新增复现入口为 `scripts/check_metadata_stream_perf.py`，此脚本的 host/E2E 统计不替代 msprof kernel 耗时。
