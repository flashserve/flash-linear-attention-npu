# RecurrentKda API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.recurrent_kda` | 支持 |
| aclnn | `aclnnRecurrentKdaGetWorkspaceSize` / `aclnnRecurrentKda` | 支持 |
| Ascend C `<<<>>>` | `recurrent_kda<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `torch.ops.npu.npu_recurrent_kda` | 不支持 |

各入口表达同一个 fused recurrent KDA 前向语义。算子计算本身在一个 AICore kernel 内完成，不把
`KdaGateCumsum + recurrent` 暴露为两段式公共实现。

## 2. 公共参数与约束

Shape 符号统一引用 [KDA 模型符号表](../../README.md#model-shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | 必选 | `BSND=[B,T,H_k,K]` 或 `TND=[T,H_k,K]` | BF16 | BSND/TND | query |
| `k` | 必选 | 与 `q` 相同 | BF16 | BSND/TND | key |
| `v` | 必选 | `BSND=[B,T,H_v,V]` 或 `TND=[T,H_v,V]` | BF16 | BSND/TND | value |
| `g` | 必选 | `BSND=[B,T,H_v,K]` 或 `TND=[T,H_v,K]` | FP32/BF16/FP16 | BSND/TND | 预计算 step log gate 或 raw gate |
| `beta` | 必选 | `BSND=[B,T,H_v]` 或 `TND=[T,H_v]` | FP32/BF16/FP16 | BSND/TND | delta 更新系数 |
| `initial_state` | Python 可选，aclnn 必选 | V-first `[state_capacity,H_v,V,K]` 或 K-first `[state_capacity,H_v,K,V]` | FP32/BF16 | ND | 可变 state pool；Python 仅在非原位模式下允许为空，并创建 `[seq_num,...]` 全零 FP32 状态 |
| `cu_seqlens` | 可选 | `[seq_num+1]` | INT32/INT64 | ND | 提供时使用 fla-org 累积 offset；标准表示为空时为 rank4 BSND，按 batch 行划分；提供时为 rank3 packed TND；旧 rank 组合的迁移见第 4.1 节 |
| `ssm_state_indices` | 可选 | packed `[>=T]` 或 speculative `[seq_num,max_step]` | INT32/INT64 | ND | 每个 token 对应的 state pool 槽索引 |
| `A_log` | 可选 | `[H_v]` | FP32 | ND | 传入时 g 为 raw gate，否则为 step log gate |
| `dt_bias` | 可选 | `[H_v*K]` 或 `[H_v,K]` | FP32 | ND | raw gate 偏置；无 A_log 时忽略，不检查其专用 shape/dtype |
| `num_accepted_tokens` | 可选 | `[seq_num]` | INT32/INT64 | ND | 必须与 `ssm_state_indices` 一起传 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `out` | 与 `v` 相同 | BF16 | recurrent 输出 |
| `final_state` | 与 `initial_state` 同 shape/dtype | FP32/BF16 | 原位模式与输入 alias；非原位模式为独立输出；`output_final_state=False` 时 Python 主入口不返回 state |

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `layout` | str | `BSND` | 兼容参数 | 忽略；仅由 cu_seqlens 是否传入判定模式 |
| `scale` | float? | Python 为 `K ** -0.5`；aclnn 由调用方传入 | 浮点数 | 乘到 query 上 |
| `output_final_state` | bool | `false` | `{false, true}` | Python 主入口是否返回最终状态 |
| `inplace_final_state` | bool | `true` | `{false, true}` | 是否把最终状态写回 `initial_state` |
| `use_qk_l2norm_in_kernel` | bool | `false` | `{false, true}` | 是否在 kernel 内对 q/k 做 L2 normalize；A5 使用 `x/sqrt(sum(x*x)+1e-6)` |
| `use_gate_in_kernel` | bool | `false` | `{false, true}` | 兼容参数；数值路径忽略，由 A_log 是否传入决定 |
| `use_beta_sigmoid_in_kernel` | bool | `false` | `{false, true}` | 是否在 kernel 内计算 `sigmoid(beta)` |
| `allow_neg_eigval` | bool | `false` | `{false, true}` | beta sigmoid 后是否乘 2 |
| `safe_gate` | bool | `false` | `{false, true}` | raw gate 的 safe 分支；无 A_log 时忽略 |
| `lower_bound` | float? | `-5.0` | 仅 A_log 存在且 safe_gate=True 时为 `[-5,0)` | safe gate 下界，其余情况不使用或校验 |
| `state_v_first` | bool | `false` | `{false, true}` | true 为 `[state_capacity,H_v,V,K]`，false 为 `[state_capacity,H_v,K,V]` |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnRecurrentKdaGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *gate,
    const aclTensor *beta,
    aclTensor *initialStateRef,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *ssmStateIndicesOptional,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *numAcceptedTokensOptional,
    const char *layout,
    double scale,
    bool outputFinalState,
    bool inplaceFinalState,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool stateVFirst,
    const aclTensor *attnOut,
    const aclTensor *finalState,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnRecurrentKda(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

`GetWorkspaceSize` 完成参数校验、Q/K/V stride 判定、其余非 state tensor 连续化预处理和 executor
创建；第二段在传入 stream 上异步执行。满足约束的非连续 Q/K/V 与 state 均通过 `CreateView` 保留
shape、storage、stride 和 offset，kernel 按 tiling
中的真实 stride 直接访问。原位模式直接写回输入 view；非原位模式直接写入 `finalState` view。原位模式若调用者
另外传入独立的 `finalState` 输出，仅为该输出保留一次必要的 `ViewCopy`。`cuSeqlensOptional` 为空时，BSND
标准表示使用 rank4 BSND；提供时标准表示使用 rank3 TND；旧 shape 别名见第 4.1 节。A5 host 通过 cu_seqlens 值依赖计算最大有效序列长度，选择不超过8的 X；元数据复制或同步开销计入调用链。
A2/A3/A5 的设备 cu_seqlens 均在 `GetWorkspaceSize` 中通过同步 D2H 取得；算子不再调用全设备同步。调用方须先保证元数据就绪，不能依赖本算子等待无关 stream。校验起点、单调性及容量后，由 executor 持有 host 常量并向 kernel 提供该次值；不论是否传入 indices，均禁用依赖旧元数据的 executor 缓存。A5 还使用该快照选择 X。固定长度且无 indices 不增加元数据等待。host placement 元数据直接复制，隔离实测及限制见[测试记录](../../../../../../tests/atk/recurrent_kda/README.md)。
stable/ctypes Python 入口在存在 CU 或 indices 时，先等待 PyTorch 当前 NPU stream，保证框架队列中的元数据生产者已提交并完成，再进入 ACLNN host 快照阶段。其他 stream 产生的输入须先按 PyTorch 规则向当前 stream 建立依赖。直接调用 ACLNN 时，调用方须保证设备元数据生产者已提交并完成；单独的 ACL runtime 同步不能替代框架提交队列的排空。这些同步成本属于调用链，不包含在目标 kernel 计时中。
当前 packed 路径不能在 ACLGraph 捕获中调用；改变 cu_seqlens 后须重新创建 executor，不能依赖旧 executor/ACLGraph 自动更新边界或 X。device 仍核验元数据。首项必须为 0，offset 必须单调不减，末项不得超过输入 token capacity，
A5 使用固定容量窗口递推，不限制每段长度为8；A2/A3仍按各自实现的支持范围执行。
输入、输出、workspace 和 executor 必须保持有效，直到 stream 完成。

### 3.2 调用示例

```cpp
// 按 2.1/2.2 的 shape、dtype 和 layout 创建输入/输出 aclTensor。
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
ACLNN_CHECK(aclnnRecurrentKdaGetWorkspaceSize(
    q, k, v, g, beta, state, cuSeqlens, ssmStateIndices,
    aLog, dtBias, numAcceptedTokens, "BSND", scale, true, true, true,
    true, true, false, false, -5.0, true, out, finalState,
    &workspaceSize, &executor));
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnRecurrentKda(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
```

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
recurrent_kda(q, k, v, g, beta, initial_state=None, *,
              cu_seqlens=None, ssm_state_indices=None, A_log=None,
              dt_bias=None, num_accepted_tokens=None, layout="BSND",
              scale=None, output_final_state=False,
              inplace_final_state=True,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
              use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
              safe_gate=False, lower_bound=None, state_v_first=False)
```

稳定入口经 Stable-ABI 适配层调用 aclnn（纯 Python `ctypes` 直调保留为参考实现与回退），不依赖 legacy `torch.ops.npu` 注册。`initial_state=None` 仅在
`inplace_final_state=False` 时有效，此时 wrapper 创建与 `state_v_first` 对应布局的全零 FP32 状态；原位模式必须
显式传入 state。显式传入的非连续 state view 由 kernel 按真实 stride 直接访问；满足第 7 节约束的 Q/K/V view
同样保留 storage offset 并直接读取，不满足时自动连续化。原位模式返回与输入相同 storage/stride
的 state，非原位模式保持输入不变并返回独立 state。

`output_final_state=True` 的非原位输出保留全部未更新槽的初值，包括空请求和稀疏索引未命中的 pool 槽。
ctypes 和直接 ACLNN 同样遵循此约定；ACLNN 在更新前复制初态的逻辑 view，可能额外产生复制任务。
该复制仅出现在非原位且要求输出 State 的调用中。

为兼容已发布的 `v26.9.1` 调用，Python 两后端及 ACLNN 保留两种已弃用的 shape 别名：

- rank3 且不传 `cu_seqlens`：在入口补一个大小为 1 的 batch 维，按单请求定长处理。
- rank4 且传 `cu_seqlens`：在入口将前两维展平为 packed token 维，offset 作用于展平后的输入。

这两种别名保留至 `v26.12`，最早在 `v27.1` 移除。Python 发出 `DeprecationWarning`，ACLNN 记录弃用警告。
请迁移到 rank4 无 `cu_seqlens` 或 rank3 有 `cu_seqlens` 的标准表示。`layout` 的值始终忽略；
归一化后的内核仍只根据 `cu_seqlens` 是否传入选择模式，输出保持调用方 `v` 的原 shape。
旧 rank4 packed 的输入 view 必要时连续化；直接 ACLNN 的该别名要求 `attnOut` 连续。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import recurrent_kda

B, T, H, H_v, K, V = 2, 2, 2, 4, 128, 128
q = torch.randn(B, T, H, K, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(B, T, H_v, V, device="npu", dtype=torch.bfloat16)
g = torch.randn(B, T, H_v, K, device="npu", dtype=torch.float32)
beta = torch.randn(B, T, H_v, device="npu", dtype=torch.float32)
A_log = torch.randn(H_v, device="npu", dtype=torch.float32)
state = torch.zeros(B, H_v, V, K, device="npu", dtype=torch.float32)
cu_seqlens = torch.arange(B + 1, device="npu", dtype=torch.int32) * T

out, final_state = recurrent_kda(
    q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1),
    g.flatten(0, 1), beta.flatten(0, 1), state, cu_seqlens=cu_seqlens,
    A_log=A_log, layout="BSND",
    output_final_state=True, use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True, state_v_first=True)
torch.npu.synchronize()
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果。参数顺序与 kernel 定义保持一致：

```cpp
recurrent_kda<<<blockDim, nullptr, stream>>>(
    q, k, v, g, beta, initialState, cuSeqlens, ssmStateIndices,
    aLog, dtBias, numAcceptedTokens, out, initialStateOut, finalState, workspace, tiling);
```

直调通路只作为 route/诊断入口；公开 Python 和 aclnn API 负责完整参数校验。直调通路按连续物理地址或 tiling data 中的 Q/K/V/state stride 解释 GM 地址；非连续直调必须使用
与实际 view shape、stride 和首地址匹配的 host tiling 结果。

## 6. legacy `torch.ops.npu` 通路

`torch.ops.npu.npu_recurrent_kda` 不属于当前支持范围，也不提供兼容性保证。请使用
`fla_npu.ops.ascendc` 的稳定入口或 aclnn API。

## 7. 已知限制

- `q/k/v/out` 当前仅支持 BF16。
- A5 仅支持 `K=V=128`；A2/A3 支持 `K=128,V=128` 或 `K=128,V=256`。head 数无256静态上限，受 tiling/DMA 表示范围和内存容量约束。
- Python/aclnn 入口支持符合 stride 约束的非连续 Q/K/V 与 `initial_state`。
- Q/K/V 直接访问要求 feature stride=1、head/token 区间不重叠、stride 为正；BSND `B>1` 时
  batch stride 可包含不重叠间隔，按各输入的独立 batch 基址搬运。feature-gap、zero-stride view 由 ACLNN
  回退连续化；token/head 置换或正 stride 重叠 descriptor 返回参数错误。
- 未传 `ssm_state_indices` 时 `state_capacity=seq_num`；传入后容量可大于序列数，所有有效 slot 必须位于 `[0,state_capacity)`。
- `ssm_state_indices` 支持 packed `[T]` 和 speculative `[seq_num,max_step]`；活跃序列不得共享正在写入的 state slot。
- `num_accepted_tokens[b]` 对非空请求须在 `[1, length[b]]` 内，只选择初态槽，不截短本次计算。
- 空序列不读取 `ssm_state_indices/num_accepted_tokens`，speculative padding 不参与值校验，也不读写空序列的 state pool。
- 传入 indices 时，ACLNN 第一阶段在 host 校验活跃 slot/accepted 及 speculative 容量，保存校验后的元数据快照供本次 executor 消费。非法值返回 `ACLNN_ERR_PARAM_INVALID`，消息包含请求、token、实际值及边界。设备元数据读取需要同步及 D2H，与已有 CU 快照共享必要同步；无 indices 的调用不增加 slot/accepted 快照开销，但 packed 调用仍执行上述 CU 校验。使用 indices 的调用不支持 ACLGraph 捕获，修改元数据后须重建 executor。
- `cu_seqlens` 可选；标准表示为空时为 rank4 BSND，按 batch 行划分；提供时为 rank3 packed TND；旧 shape 别名见第 4.1 节。提供时首项必须为 0，offset
  必须单调不减，末项为有效 token 数且不得超过输入
  token capacity；相邻差值为序列长度。A5 host 读取实际值选择 X，device kernel 仍检查边界。
- 末项小于 capacity 时，kernel 仅处理有效前缀并逐行跳过零长度序列；padding tail 输出不作保证。
- state 支持 V-first 与 K-first，支持 FP32/BF16，以及原位/非原位 final state。
- 非连续 state 仅允许 slot/head 外层维存在间隔，最后两维必须为行主序稠密矩阵，且外层地址区间不得重叠。
- A5 dense/varlen 每段有效序列长度不再受8限制；内部按最大有效序列长度及 UB 选择 X=1/2/4/8 分窗递推，支持范围受uint32 token/state容量和设备内存约束。A2/A3当前工作树仍限制为 `<=8`，对应扩展见参考 PR #917。
- A2/A3/A5 共用 cu_seqlens/A_log 判定，忽略 layout/use_gate_in_kernel 的模式选择作用。
- 未传入 A_log 时忽略 dt_bias、safe_gate 和 lower_bound，不校验、连续化、搬运或读取 dt_bias。

## 8. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| 必选 tensor 为空 | `ACLNN_ERR_PARAM_INVALID`（参数校验分支） |
| workspaceSize 或 executor 为空 | `ACLNN_ERR_PARAM_NULLPTR` |
| rank/shape/dtype/layout、活跃 slot/accepted 或属性组合非法（含启用 safe gate 时的 NaN lower_bound） | `ACLNN_ERR_PARAM_INVALID` |
| 内部 tensor 创建或 L0 调用失败 | `ACLNN_ERR_INNER_NULLPTR` |
| Python 输入不是 NPU tensor 或 runtime/op_api 未加载 | `RuntimeError` |

参数错误由 Python wrapper 与 aclnn `GetWorkspaceSize` 按各自接口约束返回。
