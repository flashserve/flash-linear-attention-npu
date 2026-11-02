# pre_process_bwd_kernel_merged 测试

## 本轮定向 DFX 补丁

最终补丁在A2/A5均完成14个子场景的最小错误码验收。本轮参照ops-nn PR13108和ops-transformer PR15001收敛：
输入shape缺失报EZ0004；shape存在而descriptor缺失报EZ1008，已提供的g/gk也检查元数据一致性。
二阶段失败保持原公开返回561000，native rc与首错保留；新增的C++catch不纳入本补丁。
最新测试日志在 `/home/s30080348/dfx-cp-596f808/a2-a5-validation/pr-aligned-results/`。
Host UT现为5方法24子场景，processor诊断47场景；正常tiling对照64组比较的是processor输出，
不是完整GE成功入口。最终ATK精度结果、安装核验与扩展审计状态统一见算子错误码整改报告。
下文此前29子场景、构造/执行异常UT和原始rc公开透传结论属于历史，不能作为当前行为要求。
按用户最新要求，错误码验收仅运行14个代表子场景，不再启动精度/性能矩阵：

```bash
source /data/Ascend/cann/cann-9.1.0/set_env.sh
source /home/s30080348/dfx-cp-596f808/venv/bin/activate
cd /home/s30080348/flash-linear-attention-npu
export ASCEND_RT_VISIBLE_DEVICES=1 DFX_DEVICE=0 DFX_BUILD_DIR="$PWD/build"
bash tests/operators/pre_process_bwd_kernel_merged/run_dfx_smoke.sh
```

覆盖输出指针EZ0037、缺失输入EZ0004、shape/rank/dtype/value、descriptor EZ1008、
模拟首错不重复、原生executor EZ1001及公开561000兼容返回。下文大矩阵保留为历史/可选资料，不是该命令的执行内容。

后续配套opbase源码与A2实际二进制补证已将冗余executor空检查登记REMOVED，追加原生回归4/4通过。
冻结42项当前29 FIXED/3 REMOVED/10 BLOCKED，基线和ID不变；本次未修改生产代码或启动精度/性能矩阵。
证据归档在`a2-a5-validation/opbase-source-closure/`，完整结论见上述算子错误码整改报告。

## 当前 DFX 整改验证

以下是 `179.183.2.2`、仓库 `/home/s30080348/flash-linear-attention-npu`、基线 `596f808` 的
本轮范围。后文历史精度和性能结果不代表本轮重新验收。

当前 generator 维护 123 条正常精度用例（41 个场景各 3 个固定种子）、14 条 public 负向和 14 条 raw ACLNN 契约。
新增负向覆盖 `chunk_size=(1<<32)+64`，raw 用例覆盖必选指针、输出 shape/dtype/stride 和 segment 边界。
`gen_cases` 更新精度/负向矩阵，不再同时重写独立维护的性能和 MSS 矩阵。

环境与复核命令：

```bash
source /data/Ascend/cann/cann-9.1.0/set_env.sh
source /home/s30080348/dfx-cp-596f808/venv/bin/activate
cd /home/s30080348/flash-linear-attention-npu
export ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python tests/atk/pre_process_bwd_kernel_merged/scripts/verify_dfx.py --phase all
export DFX_BUILD_DIR=/home/s30080348/flash-linear-attention-npu/build DFX_DEVICE=0
python tests/atk/pre_process_bwd_kernel_merged/scripts/test_tiling_dfx.py -v
```

该复核复用 ATK generator/executor，直接检查实际 errno、EZ、参数、原因和首错唯一性。
原始 12 条正常用例每条重复三次，检查输出有限且逐位一致；它不能替代原生混合容差精度验收。
旧 ATK 产物曾因 `KeyError: mixed_tolerance_bm` 阻塞。现已更换为独立环境内的 ATK `26.9.8`，
并按用户要求使用原生 `mixed_tolerance_bm_v2`：A2/A5 均为精度 **123/123 通过（84+39）**、负向 **14/14 通过**。
已核对 CLI 统计、XLSX 逐用例及仓库结果检查器；精度汇总为 `acc_pass_result: Pass`。
正式结论、逐候选台账及未关闭项统一见
[错误码整改报告](../../../fla/ops/ascendc/cp/pre_process_bwd_kernel_merged/docs/error_code_return_status.md)。

严格 host/API 边界 UT 的运行方式见 [CP Host Boundary UT](../../operators/pre_process_bwd_kernel_merged/README.md)：
A2/A5 各 7 个测试方法、29 个子场景通过，逐参数和调用命中均有断言；下层失败/首错为明确的测试 double。
修正后的 GE MRE 初次每平台为4通过/3跳过。A2后续已用公共IOInstanceNum验证五个必选descriptor省略，
现为9个成功场景、2项跳过；新增原生executor测试的4个子场景也通过。该后续未进入A5。
此前“缺失attrs/platformInfo已真实验证”的结论已撤回，builder仍无法用当前接口生成这两种缺失资源；
没有以进程崩溃冒充通过。最新命令和证据见上述host UT README与统一报告。

V2 使用原生默认值：BF16 `rtol=atol=0.0078`，FP16 `rtol=atol=0.001`，
点通过率至少 99%，全部元素还须满足 `abs_error<=0.1` 或 `abs_error<=32*ULP`。
`dhm` 仍通过 `output_dtype_overrides` 按模型 dtype 判定；输入、shape、seed、参考实现未修改。
JSON/YAML/generator 已切换到 V2；性能/MSS JSON 只更换比较器键，其余字节保持原样，未重新生成用例。
本轮验证范围仅为 A2/A5。两个平台均执行 12 条确定性用例，每条 50 次，通过；
8 个 TilingKey 均有真实 host 选择日志。内存工具的原始诊断需要独立审查，不能只用 ATK Pass 汇总判断。

早期 12 条用例的 V2 日志在 `/home/s30080348/dfx-cp-596f808/atk-v2/`。
本轮扩展矩阵与性能对照的证据目录：

| 平台 | 证据目录 |
| --- | --- |
| A2 / `179.183.2.2` | `/home/s30080348/dfx-cp-596f808/a2-a5-validation` |
| A5 / `179.228.2.2` | `/home/s30080348/dfx-991-xyDKNtce/cp-validation` |

重新执行 ATK：

```bash
source /data/Ascend/cann/cann-9.1.0/set_env.sh
source /home/s30080348/dfx-cp-596f808/venv/bin/activate
source /home/s30080348/dfx-cp-596f808/venv/lib/python3.11/site-packages/fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash
cd /home/s30080348/flash-linear-attention-npu
export ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT="/home/s30080348/dfx-cp-596f808/atk-rerun-$(date +%Y%m%d-%H%M%S)"

atk node --name cp_npu --backend npu --devices 0 -p 18776 --output_path "$OUT/accuracy" \
  node --name cp_cpu --backend cpu -p 18776 --output_path "$OUT/accuracy" \
  task -c tests/atk/pre_process_bwd_kernel_merged/atk_pre_process_bwd_kernel_merged.json \
  --task accuracy --bm_device cpu --save_data output \
  -p tests/atk/pre_process_bwd_kernel_merged/executor_pre_process_bwd_kernel_merged.py \
  -s 0 -e 123 -to 900

atk node --name cp_negative --backend npu --devices 0 -p 18775 --output_path "$OUT/negative" \
  task -c tests/atk/pre_process_bwd_kernel_merged/atk_pre_process_bwd_kernel_merged.json \
  --task run \
  -p tests/atk/pre_process_bwd_kernel_merged/executor_pre_process_bwd_kernel_merged.py \
  -s 123 -e 137 -to 120

python tests/atk/common/check_atk_result.py \
  --type accuracy --output-root "$OUT" --op pre_process_bwd_kernel_merged
```

运行前确认物理设备 6 和端口 18775/18776 空闲。当前环境已有绑定独立 Python 的 `atk` 可执行文件，
无需旧的 `atk_entry.py` 绕行。CLI 退出码不能替代结果报告。

ATK 预期失败标识是 `CP_DFX_VERIFIED_<case>`，只在实际诊断断言全部通过后生成；
诊断断言失败不会携带此标识，避免仅返回码匹配时的假通过。

## 扩展矩阵与模型来源

| 精度分组 | 场景数 | 固定种子数 | JSON 用例数 |
| --- | ---: | ---: | ---: |
| 原有正常分支 | 12 | 3 | 36 |
| 补齐 FP16 无门控、FP32 标量 gate、逐 K gate | 3 | 3 | 9 |
| T=1/63/65、FP16 tail、非零 segment 起点 | 6 | 3 | 18 |
| README 模型：gva/h8/kda/long 的 CP2/4/8/16/64 | 20 | 3 | 60 |
| 合计 | 41 | 3 | 123 |

原 12 个正常场景保留原 seed；新增两组分别为该 seed+1000、seed+2000。
新场景也采用相同三种子规则。输入分布、CPU reference 和 V2 原生阈值没有为通过测试而修改。

模型来自算子 README -> `docs/design.md` 第 28/31/35 节 ->
`scripts/bench_cp_chunk_delta_h_v052.py` 的 `DEFAULT_PROFILES`，不是另行选择的 shape。

| 模型档 | profile | Hk/Hv | 全局 T | gate |
| --- | --- | --- | ---: | --- |
| gva | `gdn_gva_h16_hv32` | 16/32 | 131072 | g FP32 |
| h8 | `gdn_h8_hv8` | 8/8 | 131072 | g FP32 |
| kda | `kda_h64_hv64` | 64/64 | 131072 | gk BF16 |
| long | `gdn_long_t256k` | 16/32 | 262144 | g FP32 |

四组均使用 BF16 主输入、B=1、K=V=128、chunk_size=64，CP=2/4/8/16/64，
每次调用的本地 T 为全局 T/CP。CP 仅为负载切分元数据，本算子按单卡单 segment 调用。
模型 case 的 scale 沿用 benchmark 默认值 1.0，原有四条性能 case 的 scale、seed 保持原定义。
性能矩阵为原有 4 条加模型 20 条，共 24 条。

MSS/确定性矩阵按实现人工选择 8 个 key 代表，加 varlen、非零起点、FP16 tail、96-head 连续任务，共 12 条。

| MSS case id | 模型 dtype / gate dtype / 模式 | 实际 key（A2/A5） |
| --- | --- | ---: |
| 0 | BF16 / BF16 / none | 2570 |
| 1 | BF16 / BF16 / g | 68106 |
| 2 | BF16 / FP32 / g | 73226 |
| 3 | BF16 / BF16 / gk | 133642 |
| 4 | FP16 / FP16 / none | 5140 |
| 5 | FP16 / FP16 / g | 70676 |
| 6 | FP16 / FP32 / g | 73236 |
| 7 | FP16 / FP16 / gk | 136212 |

`tiling-key-selection.log` 保存两个平台的真实 key 选择日志，不以模板声明替代实际命中证据。
手工审查模型/MSS 定义后，显式更新它们使用：

```bash
python tests/atk/pre_process_bwd_kernel_merged/gen_pre_process_bwd_kernel_merged.py \
  --summary --write-model-matrices
```

常规 `gen_cases` 只生成精度候选，不带该选项时不重写性能/MSS JSON。

## 性能对照方法

后续整改必须先冻结 commit、输入及采样配置，构建并安装原版、核验产物后采集基线，再改代码。
本次对已经存在的整改，从真实 commit `596f808` 独立构建原版，并与当前版补采同条件对照，
不把此前历史性能表或其他机器数据当成本轮基线。

两版使用相同 runtime、同卡、相同 24 条 case 和固定 seed；每条输入校验 SHA256，
每轮预热 5 次、采样 30 次，三轮交替执行原版/当前版；以 msprof `Task Duration(us)` 的样本中位数、
三轮中位数和轮间波动比较，完整 CSV/样本保留在 `results/performance-fixed/`。
设备分别为 A2 物理 1、A5 物理 4；功能测试分别使用 A2 物理 6、A5 物理 3。
性能采集不使用 sanitizer 插桩进程或调试日志进程。

`scripts/profile_pair.py` 接收 `--baseline-python`、`--current-python`、`--test-root`、`--task`、`--platform`，
使用独立 venv 的实际包和 OPP，防止 msprof 解析 Python 链接后加载错误版本。
`scripts/profile_cases.py` 会在仓库相邻的 `cp-profile-inputs/` 固定保存模型输入，
也可通过 `--input-cache` 指定位置；读取时核对完整 case spec，并在采集中记录输入 SHA256。
模型数据、profiling CSV 和 sanitizer 原始日志不入库。
最终性能数字与 sanitizer 风险统一记录到算子错误码整改报告。

统一 sanitizer 入口已使用 `--log-file="$MSS_LOG_PATH"` 与 ATK `-msl` 指向同一文件。
原始日志有诊断而 ATK 汇总 Pass 时，以原始诊断为准，不通过删除诊断或筛掉失败场景完成验收。

## 目录结构（`tests/atk` 规范）

| 文件 / 目录 | 说明 |
| --- | --- |
| `gen_pre_process_bwd_kernel_merged.py` 内联用例表 | 用例设计来源：123 条正常精度 + 14 条反向；模型/MSS 定义按上述来源独立维护 |
| `pre_process_bwd_kernel_merged.yaml` | ATK 用例 schema（`atk case -f` 用；标准 `mixed_tolerance_bm_v2`） |
| `gen_pre_process_bwd_kernel_merged.py` | 展开精度/负向矩阵；同时注册 `-scope=gen_cases` 用的 generator；性能/MSS 矩阵独立维护 |
| `executor_pre_process_bwd_kernel_merged.py` | ATK executor：按 `case_spec` 确定性构造输入，NPU 走 `fla_npu.ops.ascendc`，CPU 走 `scripts/reference.py` 标杆 |
| `atk_pre_process_bwd_kernel_merged.json` | 123 条正常精度 + 14 条负向，`output_dtype_overrides` 按模型 dtype 判 `dhm` |
| `atk_pre_process_bwd_kernel_merged_perf.json` | 24 条性能：原有 4 条加 README 模型的 20 条 |
| `atk_pre_process_bwd_kernel_merged_mss.json` | 12 条内存/确定性：8 个 key 代表 + varlen/offset/tail/多 head 复用 |
| `scripts/reference.py` | CPU 全精度参考：`preprocess_reference` 给出 `dhm = [E_r \| P_r]`；`dh_scan_direct` 用非零 `dht` 反扫得到真实 `dh0`；`check_affine` 校验 `dh0 == P_r @ dht + E_r` |
| `scripts/bench_cp_chunk_delta_h_v052.py` | GPU（H20）基线：fla-core 0.5.2 的 CP 三个 kernel device 耗时（60 个测量点），口径见下文「GPU（H20）基线脚本」 |
| `scripts/bench_cp_chunk_delta_h.py` | GPU（H20）基线：main/更新版本（含 zigzag 布局，120 个测量点） |
| `atk_pre_process_bwd_kernel_merged.json` 的 14 条反向用例 | 由 `gen_<op>.py` 生成、带 `expected_return_code`，核对实际 errno、EZ 和诊断字段后才产生 ATK 预期失败标识 |

## ATK 一键执行（规范入口）

```bash
FLA_NPU_ENV=<custom opp>/bin/set_env.bash \
ATK_OUTPUT_ROOT=<output root> \
bash tests/atk/run_test_cpu.sh \
  -op=pre_process_bwd_kernel_merged -npu_device_id=0 -soc=ascend950 -scope=accuracy
```

`-scope` 可取 `all` / `accuracy` / `performance` / `determinism` / `mssanitizer` / `gen_cases`。
本算子目录的 `atk_*.json` 与生成器都取自 `gen_<op>.py` 内联用例表，改用例只需改 `gen_<op>.py` 内联用例表 后重跑
`python tests/atk/pre_process_bwd_kernel_merged/gen_pre_process_bwd_kernel_merged.py --summary`。

### ATK 精度结果（历史记录）

| 平台 | 命令 | 结果 |
| --- | --- | --- |
| A5 / `ascend950` | `atk node --backend npu --devices <id> -o <out> node --backend cpu task -c ./atk_pre_process_bwd_kernel_merged.json --task accuracy -p ./executor_pre_process_bwd_kernel_merged.py -s 0 -e 12 -to 2000` | **12/12 通过**，`Total Task: 12, success 12, failed 0`，`acc_pass_result: Pass` |
| A2 / `ascend910b` | 同上（该机自带 ATK 版本低于 runner 要求，改用本机已有 ATK 26.9.8 运行） | **12/12 通过**，`Total Task: 12, success 12, failed 0`，`acc_pass_result: Pass` |
| A5 / `ascend950` | `... -scope=performance`（3 条：dense / varlen / 32 chunk） | **3/3 执行成功**，device 中位耗时 270.7 / 274.1 / 1066.7 µs（见下） |

上面两条精度行是把同一份 `atk_<op>.json` 用 `atk node ... --task accuracy` 直接执行的结果；同一份用例也可以走
`run_test_cpu.sh -scope=accuracy`。`fla_npu.ops.ascendc` 在本机稳定入口不可用时会自动回退 ctypes 参考通路，
两条通路共用同一份手写 aclnn，稳定入口本身另有验证（见下文"稳定入口"一节）。

`-scope=accuracy` 会走 `--bm_device cpu`，CPU 标杆的比较任务经 ATK 的 broker 队列派发；该 broker 默认占用
本机 `127.0.0.1:9090`。若 9090 已被其它服务占用（本机即是），runner 会在 compare 阶段报
`Unrecoverable error: JSONDecodeError('Expecting value: line 1 column 1 (char 0)')` 并卡住；runner 没有暴露端口参数。
此时用同一份用例显式指定空闲端口即可，实测 12/12 通过：

```bash
atk node --name npu_dut --backend npu --devices <device_id> -p <free_port> --output_path <out> \
  node --name cpu_golden --backend cpu -p <free_port> --output_path <out> \
  task -c ./atk_pre_process_bwd_kernel_merged.json --task accuracy --bm_device cpu \
  -p ./executor_pre_process_bwd_kernel_merged.py -s 0 -e 12 -to 2000
```

当前判据是 ATK 原生 `mixed_tolerance_bm_v2`；历史表格结果使用当时标准，不能视为 V2 复验。`dhm` 虽是 FP32，但链上状态按设计用模型 dtype 传递，
所以用例的 `standard.acc` 用 ATK 原生的 `output_dtype_overrides` 声明按 `bf16`/`fp16` 判，
不在 executor 里自定义指标。

### ATK 性能结果（历史记录，A5 / A2）

```bash
atk node --name npu_dut --backend npu --devices <device_id> --output_path <out> \
  task -c ./atk_pre_process_bwd_kernel_merged_perf.json --task performance_device \
  -p ./executor_pre_process_bwd_kernel_merged.py --save_data profile -sp -to 2000
```

| 用例 | shape | A5 device 耗时 | A2 device 耗时 | A2/A5 |
| --- | --- | --- | --- | --- |
| `pos_01_none_gate_dense` | `B=1, Hk=Hv=4, T=512, K=V=128`（8 chunk） | 77.1 µs（σ 0.35） | 132.5 µs（σ 0.50） | 1.72× |
| `pos_05_gk_varlen_first_segment` | 同 shape，本 launch 只算 `[0,300)`（5 chunk） | 55.4 µs（σ 0.37） | 92.8 µs（σ 0.37） | 1.67× |
| `pos_13_long_nt_chain_accumulation` | 同 shape，`T=2048`（32 chunk） | 283.2 µs（σ 0.39） | 481.0 µs（σ 1.50） | 1.70× |
| `pos_15_head_contiguous_partition` | `B=1, Hk=Hv=96, T=256, K=V=128`（4 chunk，Hv > 核数 ⇒ 每核串多个 head） | 135.1 µs（σ 0.53） | 360.8 µs（σ 1.00） | **2.67×** |

两次都是 `Total Task: 4, success 4, failed 0`。数值口径是 ATK `performance_device` 的 device 耗时
（xlsx `npu_*_Device性能（us）`），不是 Python wall time。dense/32-chunk 两档单 chunk ≈ 9~10 µs
（随 chunk 数近似线性），A2/A5 ≈ 1.7×；**`Hv=96` 档 A2/A5 拉到 2.67×**，是 arch22 上"每核串多个 head
时逐 head 同步开销"的直接体现，也是 A2 侧后续要压的点（见 `docs/design.md` §31.2 的 `kda`（Hv=64）档）。
（此前记录过的 270.7 / 274.1 / 1066.7 µs 是更早的实现版本，且当时 `cu_seqlens` 没送到 executor、
varlen 那一条实际按 dense 跑的，已作废。）
`-scope=determinism` / `-scope=mssanitizer` 使用同一份 `atk_<op>_mss.json`（5 条，覆盖 4 种模板参数组合），
历史版本未执行；本轮已执行 12 条，实际工具诊断与未通过项见统一整改报告。

### ATK 反向拦截结果（历史 13 条，`--task run`）

命令（NPU 单节点，不带 CPU 标杆节点）：

```bash
atk node --backend npu --devices <device_id> task \
  -c ./atk_pre_process_bwd_kernel_merged.json --task run \
  -p ./executor_pre_process_bwd_kernel_merged.py -s 12 -e 25
```

| 平台 | 结果 |
| --- | --- |
| A5 / `ascend950` | `Total Task: 13, success 13, failed 0`，13 条命中的 `aclnnStatus` 均为 161001 |
| A2 / `ascend910b` | `Total Task: 13, success 13, failed 0`，同上 |

判据：executor 先从算子异常文本里取 `aclnnStatus=<code>` 与用例声明的 `expected_return_code` 比对，
一致后才抛出与用例 `expected_error_msg` 完全一致的文本，交由 ATK 的"预期失败"判定（与
`tests/atk/chunk_kda_fwd` 同一口径）。返回码不符或算子没有拦截时，用例记为失败。

接入时踩到的两个 ATK 行为（记下来避免误判）：

- `BaseBackend.before_call` 会**丢掉所有 `dtype=non_param` 的输入**，`case_spec` 正是这一类；executor
  实际读到的是同一份元数据的**标量副本**。因此 `cu_seqlens` 按逗号分隔的 string 下发——用 `case_spec`
  或张量通道传都到不了 executor，会让 varlen 用例静默退化成 dense（dense 与 dense 互相比较仍会"通过"）。
- `compare_error_msg` 在字符串包含匹配未命中后会把 `expected_error_msg` 当**正则**编译；编译失败时会
  `raise` 一个非异常对象，报 `TypeError: exceptions must derive from BaseException`，掩盖真实比对结论。
  生成器因此保证下发的 `expected_error_msg` 本身是合法正则。

## GPU（H20）基线脚本

算子「×H20」对比的分母来自上游 [`fla/ops/cp/chunk_delta_h.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/chunk_delta_h.py)
里的三个 kernel，脚本放在本目录 `scripts/` 下（**需要 NVIDIA GPU** + `torch`/`triton` + 对应版本的
`flash-linear-attention`）：

| 脚本 | 适用版本 | 场景矩阵 |
| --- | --- | --- |
| `scripts/bench_cp_chunk_delta_h_v052.py` | fla-core **0.5.2**（按该版本的真实签名逐字对齐） | cp ∈ {2,4,8,16,64} × contiguous × 4 个模型档 × 3 kernel = **60 个测量点** |
| `scripts/bench_cp_chunk_delta_h.py` | main / 更新版本（含 `AFFINE_CHAIN_PRECISION`、zigzag CP 布局） | 再加 zigzag = **120 个测量点** |

被测的三个 kernel：`pre_process_fwd_kernel_merged`（每 rank 本地前向摘要）、
`pre_process_bwd_kernel_merged`（本算子对应的反向摘要）、`merge_fwd_bwd_kernel`（rank 间仿射链合并）。

```bash
cd tests/atk/pre_process_bwd_kernel_merged
# 先空跑看场景矩阵（不需要 GPU）
python scripts/bench_cp_chunk_delta_h_v052.py --dry-run
# fla-core 0.5.2 全量：60 个测量点，每点 warmup 5 次 + 计时 20 次
python scripts/bench_cp_chunk_delta_h_v052.py --cp-sizes 2 4 8 16 64 --iters 20 --warmup 5 --save-dir ./out
# main / 更新版本：120 个测量点
python scripts/bench_cp_chunk_delta_h.py --cp-sizes 2 4 8 16 64 --iters 20 --warmup 5 --save-dir ./out
```

口径（脚本 docstring 是同一份）：

- **device 侧耗时**用 `torch.cuda.Event` 打在 kernel launch 前后（同一 stream），warmup 后循环
  `--iters` 次（默认 20），输出 mean / median / min / max，结果落 `--save-dir` 下的 CSV + JSON。
- 默认只占 1 张卡，仿真每个 rank 的本地负载（cp ∈ {2,4,8,16,64} 都能测）；`--dist` 才走上游真实的
  分布式路径（需要 `cp_size` 张卡，含 all-gather + merge）。
- `pre_process_*` 报的是**每 rank**本地耗时（rank 间并行 ⇒ 端到端里这一步 ≈ 该值）；
  `merge_fwd_bwd_kernel` 取**最坏 rank**（fwd 链与 bwd 链各合并 `cp-1` 个），一次计时含两次 launch。
- 脚本按"已安装版本的真实签名"过滤实参，启动时打印签名并提示"本版本不支持的参数已自动忽略"，
  版本差异不会退化成看不懂的 `KeyError`（0.5.2 上被过滤的如 `AFFINE_CHAIN_PRECISION`）。

A5/A2 的「×H20」比值用的 H20 分母就是这两个脚本产出的；逐档对照见算子
[`docs/design.md`](../../../fla/ops/ascendc/cp/pre_process_bwd_kernel_merged/docs/design.md)
§11.7 / §28 / §31.2 / §35。

> 这两个脚本是 GPU 侧基线工具：**本仓 NPU 侧 CI 不执行**，ATK 的 `-scope=*` 流程也不会调用它们，
> 放在 `scripts/` 只为让「×H20」的口径可复现。

## 本地自检（无需 NPU）

```bash
python tests/atk/pre_process_bwd_kernel_merged/scripts/reference.py
```

该自检对无门控 / `USE_G` / `USE_GK` 三种模式分别构造随机输入，用**非零** `dht` 验证仿射恒等式。
只测 `dht = 0` 无法验证 `P_r`，因此自检固定使用非零 `dht`。

## 稳定入口（`fla_npu.ops.ascendc`）验证

`from fla_npu.ops.ascendc import pre_process_bwd_kernel_merged`（等价 `npu_pre_process_bwd_kernel_merged`）默认走 Stable-ABI 适配层（`csrc/src/stable_pre_process_bwd_kernel_merged.cpp` + `torch.ops.fla_npu_stable`），`libfla_npu_stable.so` 不可用时回退 ctypes 参考实现；两条通路与上文 ATK 用例共用同一份输入与 CPU 标杆。已执行用例与结论（A5 / ascend950）：

| 用例 | E rel_norm | P rel_norm | 结论 |
| --- | --- | --- | --- |
| `pos_01_none_gate_dense`（无门控） | 5.562e-03 | 7.098e-03 | PASS |
| `pos_03_g_fp32_dense`（`g` 为 FP32） | 5.076e-03 | 5.466e-03 | PASS |
| `pos_04_gk_bf16_dense`（逐 K gate） | 6.280e-03 | 5.498e-03 | PASS |
| `pos_05_gk_varlen_first_segment`（gk + varlen） | 3.787e-03 | 4.515e-03 | PASS |
| `pos_14_fp16_inputs`（fp16） | 6.506e-04 | 6.721e-04 | PASS |
| `pos_15_head_contiguous_partition`（96 head 多 task） | 4.321e-03 | 4.356e-03 | PASS |

数值与 aclnn 直调口径逐项一致（同一份源码）。A2 / ascend910b 同口径复核也是 **6/6 PASS**
（`pos_01` 5.521e-03 / 7.169e-03、`pos_03` 5.068e-03 / 5.442e-03、`pos_04` 6.301e-03 / 6.572e-03、
`pos_05` 3.724e-03 / 4.519e-03、`pos_14` 7.124e-04 / 7.043e-04、`pos_15` 4.321e-03 / 4.357e-03）。

环境要点（复现该入口时容易踩）：稳定入口要求 custom OPP 指向**本算子 run 包的安装目录**
（`<install>/vendors/<vendor>`，同时把 `FLA_NPU_ENV` 指向该目录的 `set_env.bash`）。把 OPP
**复制**到别处再用会在 A2 上解析不到 tiling compile-info（`InitTilingParseCtx failed ...
compile info not contain [_pattern]`），从而在 `GetWorkspaceSize` 报 `aclnnStatus=161001`；
指回原安装目录即恢复正常。

## 精度判据

以下 rel_norm/2% 为早期辅助观察口径；当前正式比较仅使用文首原生 `mixed_tolerance_bm_v2`，不以这些历史指标替代 V2 判定。

- `dhm` 为 FP32，但**链上状态按设计用模型 dtype 传递**（`Pc`/`PBf`/`dHBf` 为 bf16 或 fp16），因此绝对误差随
  序列长度放大（例如 32 chunk 用例 `max_abs` 可达 5.7e28）。判据采用**相对参考幅值**：
  `rel_norm = max_abs / max|参考|`，阈值 2%。
- 同时打印逐元素 `within2%`（相对误差小于 2% 的元素占比）作为辅助观察项；`max_rel` 会被参考中的极小值
  放大，不作为判据。
- 不允许通过收窄输入 range、跳过失败用例或放宽阈值来制造通过结论。

GE attrs/platform null-resource fixtures are excluded from this CP error-code remediation by user instruction. Their historical skip evidence is retained for any future GE-specific work.
