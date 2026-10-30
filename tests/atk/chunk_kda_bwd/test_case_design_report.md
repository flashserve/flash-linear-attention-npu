# ChunkKdaBwd public V2 用例设计报告

## 1. 范围与公开契约

本目录只覆盖 Ascend950 上 `fla_npu.ops.ascendc.chunk_kda_bwd(...,
implementation="optimized")` 的 public V2 正例：BF16 token、K=V=128、chunk=64、
`use_gate_in_kernel=True`、`use_exp2=True`、`state_v_first=False`、safe gate。
dense 使用连续 `[B,H,T,D]`，packed 使用连续 `[H,T,D]`，不引入其他布局或维度。

公开反向调用为：

```python
chunk_kda_bwd(
    q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h, d_o, scale,
    raw_g=raw_g, A_log=A_log, dt_bias=dt_bias, cu_seqlens=cu_seqlens,
    chunk_size=64, safe_gate=True, lower_bound=-5.0,
    use_gate_in_kernel=True, disable_recompute=disable_recompute,
    use_exp2=True, state_v_first=False, implementation="optimized",
    q_rstd=q_rstd, k_rstd=k_rstd,
)
```

ATK 用例提供 `q/k/v/raw_g/beta/d_o/A_log/dt_bias/cu_seqlens` 以及独立属性。
executor 先调用公开 `npu_chunk_kda_fwd_prepare`、`npu_chunk_fwd_h` 和
`npu_chunk_kda_fwd_finalize`，由这些真实输入构造 Aqk/Akk 和反向缓存，再调用上述公开反向接口；
这些准备输入不是对完整反向位置参数的替换。`scale=1/sqrt(128)`。

公开返回固定为八槽，executor 过滤为 ATK 可见输出的规则如下：

| 槽位 | 名称 | 当前契约与可见性 |
| --- | --- | --- |
| 0 | dq | BF16，保留 |
| 1 | dk | BF16，保留 |
| 2 | dv | BF16，保留 |
| 3 | db | 与 beta 同为 BF16 或 FP32，保留 |
| 4 | dg | FP32，保留 |
| 5 | dh0 | 当前不支持 initial state/dht，返回 None，过滤 |
| 6 | dA | FP32，保留 |
| 7 | dbias | 有 dt_bias 时为 FP32 并保留；无 dt_bias 时为 None 并过滤 |

因此有 bias 用例可见七项 `(dq,dk,dv,db,dg,dA,dbias)`，无 bias 用例可见六项。
dense 的 token 梯度为 `[B,H,T,*]`，packed 为 `[H,T,*]`。A5/V2 的公开 db/dg 均为
head-major，不在 executor 中增加布局恢复。

## 2. ATK 输入、动态 shape 与数值生成

三份 JSON 和 generator `after_case_config` 共用 `_inputs()` 构造输入；YAML 的名称和顺序
必须与其完全一致。`after_case_config` 按逻辑 profile 同时改写每个张量的 dtype、shape、
range，以及每个独立属性的值，不再通过聚合配置字符串传递用例。

| 输入 | dense shape | packed shape | dtype / 当前值域 |
| --- | --- | --- | --- |
| q, k, v | `[B,H,T,128]` | `[H,T,128]` | BF16；正态分布 mean=0、std=0.05 |
| raw_g | `[B,H,T,128]` | `[H,T,128]` | FP32；正态分布 mean=0、std=1.25 |
| beta | `[B,H,T]` | `[H,T]` | BF16或FP32；均匀分布 `[0.1,0.9]` |
| d_o | `[B,H,T,128]` | `[H,T,128]` | BF16；正态分布 mean=0、std=51.2 |
| A_log | `[H]` | `[H]` | FP32；正态分布 mean=0、std=0.12 |
| dt_bias | `[H,128]` | `[H,128]` | 可选 FP32；正态分布 mean=-3、std=1.65 |

正态分布使用 ATK `range_values={"name":"nd","mean":[...],"std":[...]}`，
张量先保持声明的输入 dtype，再由 CPU golden 转为 FP64；NPU 保持声明 dtype。
beta 的 BF16/FP32 分支与既有 profile 一一对应。

`cu_seqlens` 是精确的字符串属性：dense 为空字符串并解析为 None，packed 为
`0,64,128` 或 `0,97,130` 并解析为 INT 列表。这样不会把累计长度元数据随机化。
`disable_recompute/norm/safe_gate/lower_bound` 是执行属性；`soc/dtype/beta_dtype/seed`
描述目标环境、张量类型和固定种子。`norm` 只控制准备阶段是否导出并使用
q_hat/k_hat/q_rstd/k_rstd，不是新增生产参数。`has_dt_bias` 控制可选输入是否存在；
为 false 时 JSON 将 dt_bias 标为 default，executor 还必须从 `input_data.kwargs`
删除 dt_bias，公开调用不接收该输入。

CPU golden 与 NPU DUT 都消费 `input_data.kwargs` 中的真实张量。packed 张量在内部
CPU recurrent 前补回大小为 1 的 batch 维，黄金输出再删除该维；NPU 准备阶段按
packed TND/NTD 契约转换，最终反向仍接收 `[H,T,*]`。`build_inputs(..., provided=None)`
保留原确定性随机构造回退，只供 `scripts/check_cpu_reference.py` 等独立自校验使用；
ATK 路径始终传入 provided，不再凭 shape/seed 重建另一套随机输入。

CPU golden 使用 FP64 recurrent + autograd，最终返回 FP32。norm-on 对原始 q/k 做
`epsilon=1e-6` 的 L2Norm，先量化到 BF16 q_hat/k_hat，再转回 FP64 求导。
saved 模式传入完整 `gk/w/qg/kg/v_new/h`；recompute 模式这六项均为 None，
Aqk/Akk 始终保留。前向缓存准备和有限值检查不属于反向性能计时。

## 3. 精度矩阵与 case id

固定种子为 `0 / 20260630 / 7`。12 个逻辑 profile 各生成三个 `(profile,seed)`，
accuracy 共 36 条，id 0–35；逻辑 profile 内的种子顺序不变。下表同时给出 mss 与
预期 Finalize TilingKey 映射，不表示已经取得新的设备选择日志或通过结果。

| 逻辑 id / profile | accuracy id | mss id | 路径 | layout / norm | B/H/T 或 packed cu | beta / dt_bias | Key |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 saved_dense | 0–2 | 0 | saved | dense / off | B1/H8/T64 | BF16 / 有 | 1 |
| 1 saved_packed | 3–5 | 1 | saved | packed / off | H8/cu=[0,64,128] | BF16 / 有 | 2 |
| 2 saved_dense_norm_beta32 | 6–8 | 2 | saved | dense / on | B1/H8/T100 | FP32 / 有 | 3 |
| 3 saved_packed_norm | 9–11 | 3 | saved | packed / on | H8/cu=[0,97,130] | BF16 / 有 | 4 |
| 4 recompute_dense | 12–14 | 4 | recompute | dense / off | B1/H8/T100 | BF16 / 有 | 1 |
| 5 recompute_packed | 15–17 | 5 | recompute | packed / off | H8/cu=[0,64,128] | BF16 / 有 | 2 |
| 6 recompute_dense_norm_beta32 | 18–20 | 6 | recompute | dense / on | B1/H8/T64 | FP32 / 有 | 3 |
| 7 recompute_packed_norm | 21–23 | 7 | recompute | packed / on | H8/cu=[0,97,130] | BF16 / 有 | 4 |
| 8 saved_dense_odd_norm | 24–26 | 8 | saved | dense / on | B1/H3/T100 | BF16 / 有 | 3 |
| 9 saved_packed_odd | 27–29 | 9 | saved | packed / off | H3/cu=[0,97,130] | BF16 / 有 | 2 |
| 10 saved_dense_b2_nobias_tail | 30–32 | 10 | saved | dense / off | B2/H8/T100 | BF16 / 无 | 1 |
| 11 recompute_dense_b2_nobias_tail | 33–35 | 11 | recompute | dense / off | B2/H8/T100 | BF16 / 无 | 1 |

覆盖关系保持在现有范围内：saved 与 recompute 均覆盖 dense/packed 和 norm on/off；
H3 只用于 saved odd-head，recompute 仍满足 H 为 8 的倍数。T100 覆盖 dense 尾块，
packed `[0,97,130]` 覆盖序列内尾块，T64 保留整块对齐对照。逻辑 10/11 同时覆盖
B2、无 bias 和尾块，不额外扩展组合。

## 4. TilingKey 与 mss 映射

Finalize key 为 `(variable ? 2 : 1) + (hasRstd ? 2 : 0)`：dense 无 rstd 为 1，
packed 无 rstd 为 2，dense 有 rstd 为 3，packed 有 rstd 为 4。内部 gate layout
属性不增加 key。

`atk_chunk_kda_bwd_mss.json` 对 12 个逻辑 profile 各保留 seed=0 一条，共 12 条，
mss id 与逻辑 id 相同。Key 1 对应 mss `0,4,10,11`，Key 2 对应 `1,5,9`，
Key 3 对应 `2,6,8`，Key 4 对应 `3,7`。这些是用例到预期 key 的覆盖设计，
实际选择和内存检测结论必须由相应设备运行证据另行确认。

## 5. 性能模型

`atk_chunk_kda_bwd_perf.json` 仅保留两条模型 case：id 0 为 saved，id 1 为 recompute。
二者均为 B1/H96/T8192、K=V=128、C64、dense、norm off、BF16 token、FP32 raw_g、
BF16 beta、有 dt_bias、safe gate、`lower_bound=-5`、Ascend950、seed=20260630，
预期 Finalize Key 1。前向缓存构造不计入反向性能；本报告不填写未执行的耗时、
增速或达标结论。

## 6. 排除项与布局看护

本包不把 legacy、auto 路由或拒绝路径计入 public V2 正例，也不覆盖 unsafe gate、
FP16、initial state/dht。扩展 saved 集合中的 `42,43,44,51,52,53,57,58,59`
仅用于 legacy/拒绝路径看护，不计入 V2 正例，不属于本文件所述 36 条 accuracy、
2 条 perf 或 12 条 mss 正例。

独立 Finalize 默认使用 token-major db/dg 输出 shape；A5/V2 组合路径通过现有
`output_layout` 的 `BNSD/NTD_GATE_HEAD_MAJOR` 显式选择并直接产生公开 head-major db/dg，
不新增 attr、L0 入参或 TilingData 字段。输出 shape 只用于校验，`T == H` 时不猜测布局。
看护时必须保留独立 Finalize 默认布局用例、方形用例，以及 A2/A3 对显式 head-major 请求的
拒绝检查，不得用 public V2 的 head-major 结果替代或放宽这些契约。本设计不据此宣称 A2/A3
已完成设备验收。
