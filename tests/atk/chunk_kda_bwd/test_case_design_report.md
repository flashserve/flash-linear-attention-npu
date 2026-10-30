# ChunkKdaBwd public V2 用例设计报告

## 1. 范围与公开契约

本目录的当前资产覆盖 Ascend950 上 `fla_npu.ops.ascendc.chunk_kda_bwd(...,
implementation="optimized")` 的 public V2 正例：BF16 token、K=V=128、chunk=64、
`use_gate_in_kernel=True`、`use_exp2=True`、`state_v_first=False`、safe gate。
公开接口只有一个 `layout`，合法值为 `BSND/BNSD/TND/NTD`；当前 ATK 执行入口只使用
dense `BSND` 和 packed `TND`，不把 legacy、auto 或拒绝路径计入本包正例。

PR993 的布局与 dtype 契约如下：

- q/k/v 及 dq/dk/dv 随 `layout`；成对提供 q_rstd/k_rstd 时，q/k 为前向保存的
  head-major q_hat/k_hat。
- do、raw_g、beta 固定为 token-major 公开输入；dg/dbeta 也固定为 token-major，
  不随 q/k/v 的 head/token 顺序改变。
- dq/dk/dv 为 BF16；dg 跟随 raw_g，可为 BF16/FP32；db 跟随 beta，可为 BF16/FP32；
  A_log/dt_bias 及其梯度固定 FP32。
- ACLNN 将 do 转置一次供 Prepare/Dhu 共用；普通 beta 转置一次供 Recompute/Finalize
  共用。Finalize 内部的 beta 仍为 head-major。

公开反向调用为：

```python
chunk_kda_bwd(
    q, k, v, beta, gk, Aqk, Akk, w, qg, kg, v_new, h, d_o, scale,
    raw_g=raw_g, A_log=A_log, dt_bias=dt_bias, cu_seqlens=cu_seqlens,
    chunk_size=64, safe_gate=True, lower_bound=-5.0,
    use_gate_in_kernel=True, disable_recompute=disable_recompute,
    use_exp2=True, state_v_first=False, implementation="optimized",
    q_rstd=q_rstd, k_rstd=k_rstd, layout=layout,
    beta_eff=None, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
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
| 3 | db | 跟随 beta，BF16 或 FP32，保留 |
| 4 | dg | 跟随 raw_g，BF16 或 FP32，保留 |
| 5 | dh0 | 当前不支持 initial state/dht，返回 None，过滤 |
| 6 | dA | FP32，保留 |
| 7 | dbias | 有 dt_bias 时为 FP32 并保留；无 dt_bias 时为 None 并过滤 |

因此有 bias 用例可见七项 `(dq,dk,dv,db,dg,dA,dbias)`，无 bias 用例可见六项。
当前执行入口下，dense 的五个公开 token 梯度为 BSND，packed 为 TND；
executor 在比较前将 dq/dk/dv/db/dg 转回 CPU golden 的 head-major 排列，
dA/dbias 不做布局转换。这只是测试侧比较适配，不改变公开输出契约。

## 2. ATK 输入、动态 shape 与数值生成

三份 JSON 和 generator `after_case_config` 共用 `_inputs()` 构造输入；YAML 的名称和顺序
必须与其完全一致。`after_case_config` 按逻辑 profile 同时改写每个张量的 dtype、shape、
range，以及每个独立属性的值，不通过聚合配置字符串传递用例。

下表描述 generator 和三份 JSON 中的**真实 ATK 输入**。它们沿用 CPU golden 的
head-major 规范排列；executor 不重新生成或替换这些张量，而是在 NPU 准备和公开反向调用前
按当前执行入口转换为 BSND/TND。

| 输入 | dense ATK shape | packed ATK shape | dtype / 当前值域 |
| --- | --- | --- | --- |
| q, k, v | `[B,H,T,128]` | `[H,T,128]` | BF16；正态分布 mean=0、std=0.05 |
| raw_g | `[B,H,T,128]` | `[H,T,128]` | BF16或FP32；正态分布 mean=0、std=1.25 |
| beta | `[B,H,T]` | `[H,T]` | BF16或FP32；均匀分布 `[0.1,0.9]` |
| d_o | `[B,H,T,128]` | `[H,T,128]` | BF16；正态分布 mean=0、std=51.2 |
| A_log | `[H]` | `[H]` | FP32；正态分布 mean=0、std=0.12 |
| dt_bias | `[H,128]` | `[H,128]` | 可选 FP32；正态分布 mean=-3、std=1.65 |

正态分布使用 ATK `range_values={"name":"nd","mean":[...],"std":[...]}`。
张量保持声明的输入 dtype，CPU golden 转为 FP64，NPU 保持声明 dtype。
raw_g 的 BF16/FP32 分支由 `raw_g_dtype` 控制，beta 由 `beta_dtype` 控制。
generator 的 `output_dtype_overrides` 是**逐输出比较判据**，不是公开输出 dtype 声明：
dq/dk/dv/db/dA/dbias 使用已验证的 BF16 mixed-tolerance 校准，dg 按 raw_g 真实 dtype
选择 FP32 或 BF16；executor 另行断言公开存储 dtype，db 跟随 beta、dg 跟随 raw_g。

这些比较配置属于自定义泛化回归，不是上游官方精度门禁。严格 FP32 匹配阈值只作本地诊断，
不单独作为 PR993 或本 PR 的缺陷、通过或阻塞结论；也不把它替换成 BF16 来制造通过结果。
回归判定使用同一输入、缓存和 golden 的跨版本输出等价性，原始日志仅留作本地诊断。

`cu_seqlens` 是精确的字符串属性：dense 为空字符串并解析为 None，packed 为
`0,64,128`、`0,97,130` 或 `0,8` 并解析为 INT 列表，不随机化累计长度元数据。
`disable_recompute/norm/safe_gate/lower_bound` 是执行属性；`soc/dtype/beta_dtype/raw_g_dtype/seed`
描述目标环境、张量类型和固定种子。`norm` 只控制准备阶段是否导出并使用
q_hat/k_hat/q_rstd/k_rstd，不是新增生产参数。`has_dt_bias` 控制可选输入是否存在；
为 false 时 JSON 将 dt_bias 标为 default，executor 还必须从 `input_data.kwargs`
删除 dt_bias，公开调用不接收该输入。

CPU golden 与 NPU DUT 都消费 `input_data.kwargs` 中的真实张量。packed 张量在内部
CPU recurrent 前补回大小为 1 的 batch 维，黄金输出再删除该维。NPU 侧通过
`_token_major_input` 将 q/k/v/raw_g/beta/d_o 转为公开 BSND/TND；norm-on 时 q/k 改用
前向保存的 head-major q_hat/k_hat。`build_inputs(..., provided=None)` 只保留独立
CPU 自校验的确定性随机构造回退，ATK 路径始终传入 provided。

CPU golden 使用 FP64 recurrent + autograd，最终返回 FP32，并始终保持 head-major。
norm-on 对原始 q/k 做 `epsilon=1e-6` 的 L2Norm，先量化到 BF16 q_hat/k_hat，
再转回 FP64 求导。saved 模式传入完整 `gk/w/qg/kg/v_new/h`；recompute 模式这六项均为
None，Aqk/Akk 始终保留。前向缓存准备和有限值检查不属于反向性能计时。

## 3. 精度矩阵与 case id

固定种子为 `0 / 20260630 / 7`。当前 generator 有 **14 个逻辑 profile**，每个生成三个
`(profile,seed)`，所以 `atk_chunk_kda_bwd.json` 共 **42 条 accuracy 资产**，id 0–41；
逻辑 profile 内的种子顺序不变。`atk_chunk_kda_bwd_mss.json` 对每个 profile 保留
seed=0 一条，共 **14 条**。下表同时给出 mss 与预期 Finalize TilingKey 映射；
映射本身不证明实际选 key 或 msSanitizer 通过，accuracy 实测结果见 README。

表中 dtype 顺序为 raw_g/beta；执行 layout 指公开反向入口，不是 JSON 的规范输入排列。

| 逻辑 id / profile | accuracy id | mss id | 路径 | 执行 layout / norm | B/H/T 或 packed cu | raw_g/beta | dt_bias | Key |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 saved_dense | 0–2 | 0 | saved | BSND / off | B1/H8/T64 | FP32/BF16 | 有 | 1 |
| 1 saved_packed | 3–5 | 1 | saved | TND / off | H8/cu=[0,64,128] | FP32/BF16 | 有 | 2 |
| 2 saved_dense_norm_beta32 | 6–8 | 2 | saved | BSND / on | B1/H8/T100 | FP32/FP32 | 有 | 3 |
| 3 saved_packed_norm | 9–11 | 3 | saved | TND / on | H8/cu=[0,97,130] | FP32/BF16 | 有 | 4 |
| 4 recompute_dense | 12–14 | 4 | recompute | BSND / off | B1/H8/T100 | FP32/BF16 | 有 | 1 |
| 5 recompute_packed | 15–17 | 5 | recompute | TND / off | H8/cu=[0,64,128] | FP32/BF16 | 有 | 2 |
| 6 recompute_dense_norm_beta32 | 18–20 | 6 | recompute | BSND / on | B1/H8/T64 | FP32/FP32 | 有 | 3 |
| 7 recompute_packed_norm | 21–23 | 7 | recompute | TND / on | H8/cu=[0,97,130] | FP32/BF16 | 有 | 4 |
| 8 saved_dense_odd_norm | 24–26 | 8 | saved | BSND / on | B1/H3/T100 | FP32/BF16 | 有 | 3 |
| 9 saved_packed_odd | 27–29 | 9 | saved | TND / off | H3/cu=[0,97,130] | FP32/BF16 | 有 | 2 |
| 10 saved_dense_b2_nobias_tail | 30–32 | 10 | saved | BSND / off | B2/H8/T100 | FP32/BF16 | 无 | 1 |
| 11 recompute_dense_b2_nobias_tail | 33–35 | 11 | recompute | BSND / off | B2/H8/T100 | FP32/BF16 | 无 | 1 |
| 12 saved_dense_t_eq_h_raw_g_bf16 | 36–38 | 12 | saved | BSND / off | B1/H2/T2 | BF16/BF16 | 有 | 1 |
| 13 recompute_packed_t_eq_h_norm_raw_g_bf16 | 39–41 | 13 | recompute | TND / on | H8/cu=[0,8] | BF16/BF16 | 有 | 4 |

覆盖关系保持在当前 public V2 范围内：saved 与 recompute 均覆盖 dense/packed 和
norm on/off；H3 只用于 saved odd-head，所有 recompute profile 均满足 H≤256 且 H 为
8 的倍数。T100 覆盖 dense 尾块，packed `[0,97,130]` 覆盖序列内尾块，T64 保留整块对齐对照。
逻辑 10/11 同时覆盖 B2、无 bias 和尾块。

逻辑 12 以 **H=T=2**、norm off、BF16 raw_g 覆盖 dense 方形消歧；逻辑 13 以
**H=T=8**、`cu_seqlens=[0,8]`、norm on、BF16 raw_g 覆盖 packed 方形和 rstd 路径。
后者 H=8 满足 recompute 的 8 倍数约束。两者同时把 dg 的 BF16 写回纳入资产设计。

## 4. TilingKey 与 mss 映射

Finalize key 为 `(variable ? 2 : 1) + (hasRstd ? 2 : 0)`：dense 无 rstd 为 1，
packed 无 rstd 为 2，dense 有 rstd 为 3，packed 有 rstd 为 4。内部 `model_layout`
描述混合布局，不增加 key。

mss id 与逻辑 profile id 相同，当前完整映射为：

| Key | 条件 | 逻辑 profile id | mss id |
| --- | --- | --- | --- |
| 1 | dense，无 rstd | 0,4,10,11,12 | 0,4,10,11,12 |
| 2 | packed，无 rstd | 1,5,9 | 1,5,9 |
| 3 | dense，有 rstd | 2,6,8 | 2,6,8 |
| 4 | packed，有 rstd | 3,7,13 | 3,7,13 |

这些是用例到预期 key 的覆盖设计。实际选择和内存检测结论必须由相应设备运行证据另行确认，
不能由 JSON 条数或映射表推断。

## 5. 性能模型资产

`atk_chunk_kda_bwd_perf.json` 保留 **2 条**模型 case：id 0 为 saved，id 1 为 recompute。
二者均为 B1/H96/T8192、K=V=128、C64、dense BSND、norm off、BF16 token、FP32 raw_g、
BF16 beta、有 dt_bias、safe gate、`lower_bound=-5`、Ascend950、seed=20260630，
预期 Finalize Key 1。前向缓存构造不计入反向性能。四组六轮配对性能及 2 条
`performance_device` 的真实执行结果见 README；资产数量或执行完成均不等于性能达标。

## 6. 排除项与布局看护

本包不覆盖 unsafe gate、FP16、initial state/dht，也不使用 legacy 扩展结果作为当前结论。
42 条 accuracy、14 条 mss、2 条 perf 是自定义资产规模，不直接代表官方验收通过数量。
严格 FP32 匹配阈值只作本地诊断；回归以跨版本输出等价性判定，determinism 的 14×50 和
`performance_device` 的 2/2 执行结果分别按 README 中的真实报告解释。A2/A3 实机、官方
精度 CI 和整链验收不由本报告推断。

独立 Finalize 的 `output_layout` 控制全部五个梯度。总接口内部通过可选字符串
`model_layout` 传递 BSND/BNSD/TND/NTD，默认空字符串表示独立调用；输出 shape 只用于校验，
`T == H` 时不猜测布局。看护时必须保留独立 Finalize 两种输出顺序、总接口混合布局、
raw_g shape 负向检查及上述两个方形 profile，不得以测试侧的布局恢复替代生产契约检查。
