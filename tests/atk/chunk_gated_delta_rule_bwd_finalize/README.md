# ChunkGatedDeltaRuleBwdFinalize ATK 测试说明

本目录提供 `chunk_gated_delta_rule_bwd_finalize` 的 ATK 精度、性能、确定性和
内存检查资产。CPU 标杆完整实现 Stage 0--15 公式，不使用零输出或
NPU 结果作为标杆。

## CPU golden 接口

`scripts/chunk_gated_delta_rule_bwd_finalize_cpu.py` 提供完整 CPU 标杆：

```python
from scripts.chunk_gated_delta_rule_bwd_finalize_cpu import (
    chunk_gated_delta_rule_bwd_finalize_golden,
)

dq, dk, dv, dbeta, dg = chunk_gated_delta_rule_bwd_finalize_golden(
    q, k, v, v_new, do, du, g, beta, h, dh, a,
    q_rstd=q_rstd,
    k_rstd=k_rstd,
    beta_raw=beta_raw,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    scale=scale,
    chunk_size=64,
    use_qk_l2_norm_in_kernel=use_qk_l2_norm_in_kernel,
    use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
    use_gate_in_kernel=False,
    state_v_first=state_v_first,
    use_exp2=True,
)
```

该函数与 `fla_npu.ops.ascendc.npu_chunk_gated_delta_rule_bwd_finalize`
使用相同的参数顺序、默认值和五输出语义；作为独立 CPU 标杆，输入
tensor 必须位于 CPU。

## 支持范围

- Ascend 950，`q/k/v/v_new/do/du/h/dh/A` 为 BF16。
- `g/beta` 支持 BF16 或 FP32 且 dtype 必须相同，`beta_raw` 跟随 `beta.dtype`。
- `state_v_first=false/true` 分别验证 `h/dh` 末两维的 `[K,V]` 和 `[V,K]` 存储语义。
- `K=V=128`，`chunk_size=64`，支持定长和变长序列。
- `use_qk_l2_norm_in_kernel` 和 `use_beta_sigmoid_in_kernel` 为两个独立
  TilingKey 模板参数，默认值均为 `false`，支持 `false/true`。
- `use_gate_in_kernel` 只支持 `false`，`use_exp2` 支持 `true` 和 `false`；ATK executor
  显式传入这两个固定属性。
- `q/k` 由 ATK 生成，数据范围为 `[-0.2, 0.2]`，`v` 数据范围为
  `[-0.5, 0.5]`；`v_new/do/du/h/dh/a`
  数据范围为 `[-0.03, 0.03]`，`g/beta/beta_raw` 数据范围为
  `[-0.09, 0.09]`；CPU 节点在
  算子 ABI 边界将 ATK 升精度输入还原为实际输入 dtype，并按 NPU 各 Stage
  的 BF16 落盘和 FP32 Vector 驻留边界执行标杆。
- 输出顺序为 `dq, dk, dv, dbeta, dg`；`dbeta` 跟随 `beta.dtype`，
  `dg` 跟随 `g.dtype`。executor 会对所有输出的 shape 和 dtype 做强校验。

## 用例矩阵

| 文件 | 用例数 | 覆盖 |
|---|---:|---|
| `atk_chunk_gated_delta_rule_bwd_finalize.json` | 200 | 4 类边界 shape + 8 个模型派生 shape + 13 个泛化 shape x 8 个模板 key |
| `atk_chunk_gated_delta_rule_bwd_finalize_perf.json` | 64 | 8 个模型派生 case x 8 个模板 key |
| `atk_chunk_gated_delta_rule_bwd_finalize_mss.json` | 8 | 每个模板 key 一条尾块 GVA 用例 |

精度矩阵覆盖完整 chunk、尾 chunk、多 chunk、GVA、varlen、泛化 shape
和全部 8 个模型派生 shape。为控制 ATK CPU 标杆耗时，各派生 shape 的 `T`
约为原模型长度的四分之一；`T=64/65` 的最小完整块和尾块边界保留。

## 执行

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_finalize -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_finalize -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_finalize -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_finalize -npu_device_id=0 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_finalize -scope=gen_cases
```

默认不保存正常输入、输出或 CPU golden；保留 ATK 原始 Excel、日志和数值摘要。
本套泛化用例按上述分组值域生成，不得通过修改精度阈值或跳过失败用例制造通过结论。


## A2/A3 新增回归（原 A5 路径保留）

上方 200/64/8 条为原 A5 资产，ID、seed、payload 和原 CPU 标杆未改动。
原三个 JSON 保持原始文件内容；`formal_payloads` 与默认 `Generator` 只生成这些 A5 用例，
旧 `run_test_cpu.sh` 的 accuracy/MSS/DC 入口仍读取原文件，不会混入 arch22 矩阵。
新增 `executor_chunk_gated_delta_rule_bwd_finalize_arch22` 独立注册，使用真实公开输入与属性，
不新增 `case_spec` 或用来替代真实 shape 的控制字段。`cu_seqlens/chunk_indices` 是公开的
int64 列表参数，以 `non_param` JSON 列表字面量存储后解码成列表；不是随机整数张量。
ATK 在 `before_call` 过滤该通道之前，先调用 executor 的 `init_by_input_data`；此时已将
真实元数据量化/解析并缓存到 `actual/case/data`，后续调用使用该缓存，不重新读取被过滤的字段。

| 正式文件 | 用途 | 用例数与源 ID |
| --- | --- | --- |
| `atk_chunk_gated_delta_rule_bwd_finalize.json` | 原 A5 精度 | 200，ID 0–199 |
| `atk_chunk_gated_delta_rule_bwd_finalize_perf.json` | 原 A5 性能 | 64，ID 0–63 |
| `atk_chunk_gated_delta_rule_bwd_finalize_mss.json` | 原 A5 MSS/DC | 8，ID 0–7 |
| `atk_chunk_gated_delta_rule_bwd_finalize_arch22.json` | 独立 A2/A3 精度 | 68 个逻辑代表 ×3 seed = 204，源 ID 200–403 |
| `atk_chunk_gated_delta_rule_bwd_finalize_arch22_mss.json` | 独立 A2/A3 MSS/DC/keys | 32 key 代表 +8 条关键分支 =40，源 ID 8–47 |

`formal_arch22_payloads` 单独生成 204/40；生成器 CLI 默认同时更新这五个文件。
`scripts/arch22_cases.py` 只读取两个独立 arch22 文件，保留源 ID 并为 ATK 从零编号，
不会改变旧 A5 文件、默认 Generator 或旧测试入口的用例数量。

三个精度 seed 固定为 `20261030/20261031/20261032`，每个逻辑用例分别保留。
32 条普通代表覆盖 GT BF16/FP32、norm、sigmoid、exp2、owner 五个独立模板维度。
32 条边界代表保留同一 key：owner=0 覆盖 BT128/V256、varlen 和空序列；owner=1
覆盖奇数 head、31/33/1 尾块及空序列，执行 owner key 内的 fallback。
额外 4 条覆盖 BT64/V256+ratio3、BT128/V128+B2、偶数 MHA dense 尾块、合法元数据重排。
`state_v_first` 两种布局在真实 V=128/256 状态张量中覆盖，不以 K=V 掩盖转置。

### CPU 两路与输入边界

原 A5 CPU 函数和调用路径保持原样。A2/A3 的 `scripts/arch22_cpu.py` 只依赖 NumPy，
抽取已验证外部标杆的纯函数，原件 SHA256 为
`891b42da51150735eb4306ebfd6b573d8fa58f21950a8febab33a5cc1386a340`；不导入其 ATK
执行器或 fixture 脚手架。调整仅为公开 `scale` 入参、合法零长度序列和公开状态 bank
布局/重排适配；`REFERENCE_ARTIFACT_SHA256` 可追溯原件。

- `cpu_lowp`：公开输入先按真实 BF16/FP32 量化，模拟 BF16 GEMM 落盘、FP32 Vector、
  GVA 累加、BT64/128 DG2 规约、gate 尾处理和最终输出 dtype。BLAS MMA、Vector 指令
  超越函数、FTZ/NaN 尚未证明逐位等价，不能声称 bit exact。
- CPU FP64 golden：共享同一份量化输入，独立执行 FP64 数学式；不调用 lowp 函数，
  不把中间量或输出降成 BF16/FP32。ATK 原生 `mixed_tolerance_bm` 保持不变，
  lowp 成功不能替代或遮盖 NPU 对 FP64 的失败。
- 三个角色均记录同一公开输入的 SHA，只有摘要进入回执，不保存张量。
- ATK 属性在输入边界按公开 ABI 规范成 Python 标量：scale 为 `float`、chunk_size 为
  `int`、五个开关为 `bool`。NumPy FP32 scale 精确提升后供三个角色共享，不替换成常数。
- DUT 先严格校验原生输出 dtype；lowp 的 `dq/dk/dv` 按 BF16、`dbeta/dg` 按公开 GT
  保留 BF16 或 FP32 数值。三个角色交 ATK 的比较载体统一无损升为 FP64，真 FP64
  golden 的计算、中间量与结果始终保持 FP64，不通过降精度绕开框架检查。
- ATK 26.9.8 的前置 dtype 检查不接受 BF16/FP32 与 FP64 配对。每条新 case 使用官方
  `standard.acc={"mixed_tolerance_bm":{"output_dtype_overrides":...}}`，输出 0/1/2 固定
  `bf16`，输出 3/4 按该 case 的真实 GT 声明 `bf16` 或 `fp32`。只选择原有 ABI
  dtype 对应标准，不修改任何阈值，也不让 FP64 载体落入其他比较方法。
- 输出回执的 `dtype/sha256` 保留原生或 CPU 生产值；`comparison_dtype/comparison_sha256`
  单列实际交给 ATK 的 FP64 载体。BF16 原生 SHA 仅用独立 FP32 临时副本计算。

新增矩阵保留原 q/k `[-0.2,0.2]`、v `[-0.5,0.5]`、gradient/state/A
`[-0.03,0.03]`、g/beta/beta_raw `[-0.09,0.09]`，未收窄范围或提高阈值。
q/k rstd 为新增真实可选输入，范围 `[0.8,1.2]`。A 和 g 使用 ATK 实际生成值，
没有复制外部标杆的三角逆矩阵或累计 g 构造器。随机 g 可能非单调，而源码精度链的
`exp(min(delta,0))` 与独立 FP64 数学式 `exp(delta)` 在这种输入上可能有差异；
此矩阵用于暴露该公共输入合同边界，不能预先声明全部通过，也不能改 FP64 截断来通过。
真实 GDN 累计 g 校准可另作标明的附加实验，不能替换这套随机范围矩阵。

本地 `scripts/check_cpu_reference.py` 检查 68 个代表的输出 shape/有限值、独立 FP64 dtype、
零梯度、合法空序列、状态 bank 与 pair 同步重排以及 BF16 RNE tie；它不代表 NPU 精度通过。

### TilingKey 映射与实际选择

预期 key 来自当前 arch22 模板字段和 Host owner 位：
`10 | (gKey << 8) | (norm << 16) | (sigmoid << 17) | (exp2 << 18) | (owner << 19)`，
其中 gKey 为 10(BF16)/30(FP32)。owner 仅在 K128/BT64/V128/HK==HV 时置位；
owner=1 不代表每个运行时块都使用 fast owner。

下表 ID 是正式源 JSON ID；运行入口筛选后从 0 编号，并把源 ID 保存于 `selection.json`。
A2 与 A3 使用相同源逻辑矩阵，必须分别取得设备实际证据；任何一端的静态预期或实测不能代替另一端。

| 预期 key | GT | norm/sigmoid/exp2 | 精度普通 ID | 精度边界 ID | MSS ID | A2/A3 实际选择 |
| ---: | --- | --- | --- | --- | ---: | --- |
| 2570 | bf16 | 0/0/0 | 200–202 | 296–298 | 8 | 待实际运行记录 |
| 264714 | bf16 | 0/0/1 | 203–205 | 299–301 | 9 | 待实际运行记录 |
| 133642 | bf16 | 0/1/0 | 206–208 | 302–304 | 10 | 待实际运行记录 |
| 395786 | bf16 | 0/1/1 | 209–211 | 305–307 | 11 | 待实际运行记录 |
| 68106 | bf16 | 1/0/0 | 212–214 | 308–310 | 12 | 待实际运行记录 |
| 330250 | bf16 | 1/0/1 | 215–217 | 311–313 | 13 | 待实际运行记录 |
| 199178 | bf16 | 1/1/0 | 218–220 | 314–316 | 14 | 待实际运行记录 |
| 461322 | bf16 | 1/1/1 | 221–223 | 317–319 | 15 | 待实际运行记录 |
| 7690 | fp32 | 0/0/0 | 224–226 | 320–322 | 16 | 待实际运行记录 |
| 269834 | fp32 | 0/0/1 | 227–229 | 323–325 | 17 | 待实际运行记录 |
| 138762 | fp32 | 0/1/0 | 230–232 | 326–328 | 18 | 待实际运行记录 |
| 400906 | fp32 | 0/1/1 | 233–235 | 329–331 | 19 | 待实际运行记录 |
| 73226 | fp32 | 1/0/0 | 236–238 | 332–334 | 20 | 待实际运行记录 |
| 335370 | fp32 | 1/0/1 | 239–241 | 335–337 | 21 | 待实际运行记录 |
| 204298 | fp32 | 1/1/0 | 242–244 | 338–340 | 22 | 待实际运行记录 |
| 466442 | fp32 | 1/1/1 | 245–247 | 341–343 | 23 | 待实际运行记录 |
| 526858 | bf16 | 0/0/0 | 248–250 | 344–346 | 24 | 待实际运行记录 |
| 789002 | bf16 | 0/0/1 | 251–253 | 347–349 | 25 | 待实际运行记录 |
| 657930 | bf16 | 0/1/0 | 254–256 | 350–352 | 26 | 待实际运行记录 |
| 920074 | bf16 | 0/1/1 | 257–259 | 353–355 | 27 | 待实际运行记录 |
| 592394 | bf16 | 1/0/0 | 260–262 | 356–358 | 28 | 待实际运行记录 |
| 854538 | bf16 | 1/0/1 | 263–265 | 359–361 | 29 | 待实际运行记录 |
| 723466 | bf16 | 1/1/0 | 266–268 | 362–364 | 30 | 待实际运行记录 |
| 985610 | bf16 | 1/1/1 | 269–271 | 365–367 | 31 | 待实际运行记录 |
| 531978 | fp32 | 0/0/0 | 272–274 | 368–370 | 32 | 待实际运行记录 |
| 794122 | fp32 | 0/0/1 | 275–277 | 371–373 | 33 | 待实际运行记录 |
| 663050 | fp32 | 0/1/0 | 278–280 | 374–376 | 34 | 待实际运行记录 |
| 925194 | fp32 | 0/1/1 | 281–283 | 377–379 | 35 | 待实际运行记录 |
| 597514 | fp32 | 1/0/0 | 284–286 | 380–382 | 36 | 待实际运行记录 |
| 859658 | fp32 | 1/0/1 | 287–289 | 383–385 | 37 | 待实际运行记录 |
| 728586 | fp32 | 1/1/0 | 290–292 | 386–388 | 38 | 待实际运行记录 |
| 990730 | fp32 | 1/1/1 | 293–295 | 389–391 | 39 | 待实际运行记录 |

MSS ID 40–47 依次选择逻辑代表 32、39、48、63、64、65、66、67，覆盖 BT128/V256
varlen/空序列、奇数 MHA 尾块 fallback、两种混合 BT/V、B2、ratio3、偶数 MHA 尾块和重排。

`keys` 专项只开启现有 SDK debug/stdout，不增加 kernel/Host 功能开关，不作性能结论。
worker 每次真实 ACLNN 调用前后打印 case 边界，Host `SetTilingKey` 后输出
`GDN_FINALIZE_ARCH22_TILING actualKey=... owner=... gKey=... norm=... sigmoid=... exp2=...`。
`scripts/capture_key_evidence.py` 仅提取调用边界内的实际 Host 行，
`scripts/check_key_evidence.py` 对照全部 case，保留原日志路径、行号和 SHA。
没有 Host 行、关联不全、key 不一致或证据文件缺失均不能通过；输入参数推导只写入 expected_key。

### 运行入口、身份与 no-save

准备 ATK >=26.8.8（已验证环境可使用 26.9.8）、CANN/OPP、当前构建和冻结 binding。
入口不设置 PYTHONPATH、不安装依赖、不构建、不替换设备环境。

```bash
export ATK_PYTHON=<python>
export ATK_BIN=<ATK bin/atk>
export FINALIZE_SOURCE_COMMIT=<当前源码提交>
export FINALIZE_BINDING_PATH=<runtime-binding.json绝对路径>
export FINALIZE_BINDING_SHA256=<binding原始文件字节SHA256>
export REVIEWFIX_RESULT_DIR=<全新的阶段结果目录>
export FINALIZE_CPU_EVIDENCE="$REVIEWFIX_RESULT_DIR/role-evidence"
bash tests/atk/chunk_gated_delta_rule_bwd_finalize/scripts/run_arch22.sh accuracy 0 A2
# A3 使用独立输出目录并把末参数改为 A3。
# 每个动作使用自己的全新 REVIEWFIX_RESULT_DIR：
bash tests/atk/chunk_gated_delta_rule_bwd_finalize/scripts/run_arch22.sh mssanitizer 0 A2
bash tests/atk/chunk_gated_delta_rule_bwd_finalize/scripts/run_arch22.sh determinism 0 A2
bash tests/atk/chunk_gated_delta_rule_bwd_finalize/scripts/run_arch22.sh keys 0 A2
```

可选小批量使用 `ARCH22_SOURCE_CASE_IDS=<逗号分隔源ID>`，入口传给选择器的
`--source-case-ids`。省略时仍运行该 scope 的全部 204/40 条；筛选只匹配源 ID，
保留原输入、seed 与源 ID，运行 ID 重新从零编号。空值、重复或不存在的 ID 会报错。

worker 首次调用核 binding 原始 SHA、frozen、源码 HEAD/clean、完整 installed_files 成员/字节/SHA，
并核实际 `fla_npu.__file__`、目标双符号解析到的 cust_opapi 及该安装树内库的实测 SHA；
允许的 CANN 默认 registry/proto 库按精确路径单列 `external_dependencies`。
DUT 成功调用后刷新身份，要求新包 tiling 库已映射且 SHA 匹配，独立保存新的 maps 快照；
库映射不能证明回调被选中，还须与该 case 的真实 `actualKey` Host 标记联合核验。
每个 DUT 回执记录实测身份、库 SHA 与 `/proc/self/maps` 原件；CPU 角色不加载 NPU 库。
仅请求中的 `SOURCE_COMMIT/BINDING_SHA` 环境值不能充当实测证据。

所有动作均不传 `--save_data` 或 `--print_data`。结果目录保留原始 Excel、日志、
`selection.json`、`role-evidence/`、`summary.json`，keys 另有 `actual-keys.json/key-evidence.json`。
默认后验盘点拒绝 tensor 归档和符号链接；不通过删除数据后改判 no-save。
`summary.json` 只有原生结果、角色完整性、实测身份、同输入 SHA 与 no-save 均满足时才写 pass。
204 个新精度 case 是 68 种小 shape ×3 seed；MSS 仅 40 条，不扩展成全形状开关笛卡尔积。
MSS 外层 `--log-file` 与 ATK 的 `-msl` 使用同一个新日志文件，保留 info 级工具记录；
不得通过创建空日志或只检查 ATK 退出码代替实际 sanitizer 覆盖和错误检查。
MSS 运行时间尚未实测；建议预留 10–20 分钟，实际总时长按原日志统计，不能把估计当验收耗时。

### 真实 Host 负向

```bash
# 使用上面的冻结身份环境；只申请 workspace，不调用 launch。
python -B tests/atk/chunk_gated_delta_rule_bwd_finalize/scripts/check_host_contract.py \
  --device 0 --output "$REVIEWFIX_RESULT_DIR/host-contract.json"
```

固定 18 条：9 条非法元数据（首非0、尾非T、下降、seq越界、localChunk越界、缺漏、重复、
空序列误配、奇数 pair 数）断言真实 `ACLNN_ERR_PARAM_INVALID=161002`、参数路径、
actual 数字与 expected 关键词，且拒绝返回 executor；另 3 条合法空序列、合法 pair 重排、dense
断言 SUCCESS。真实消息来自 `aclGetRecentErrMsg`，兼容 EZ0002 模板的 incorrect value/should be，
不硬编码日志前缀为 actual/expected。不会调用 NPU DUT，也不会拿旧版危险 kernel 试跑非法输入。
另 6 条代表验证 K/V、HV%HK、state NT、启用的 q_rstd/beta_raw shape 前置校验，
同样断言 161002 和真实参数级消息。地址乘积/workspace 溢出仍可能在 tiling 层发现并映射到其他返回码，
不把未覆盖的超大描述符场景伪装为已测试。

## 本次验证状态

本地 CPU 自检通过；三份 JSON 和 Python/YAML 的静态检查见本次交付回执。
A2/A3 原生 ATK、32 key 实际选择、真实 Host 返回码/消息和 MSS/确定性尚待固定构建运行。
上述清单表示已提供可执行资产与预期映射，不表示硬件验收已经通过。
