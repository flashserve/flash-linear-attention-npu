# GdnRecoverssmVerify 接口与支持范围

## 参考来源

- 固定参考提交：`0935884b3bc1e24ed36194539a9a420438587b39`。
- 参考实现：`vllm_ascend/ops/triton/gdn/gdn_recoverssm_verify.py`。
- 参考测试：`test_gdn_recoverssm_verify_npu.py` 与 `gdn_recoverssm_test_utils.py`。
- 用户请求将参考接口和数学语义冻结为本算子的接口依据。

本算子验证一个 GDN speculative window。每个请求从分页 recurrent checkpoint
读取初态，逐 token 推进临时状态，写出每个 token 的输出及 commit 所需记录；
checkpoint 全程只读，replay cache 按有效请求和有效 token 原地写入。

## 接口签名

GE/Ascend C 算子名为 `GdnRecoverssmVerify`。公开入口固定为：

```text
GdnRecoverssmVerifyInnerGetWorkspaceSize   # 内部接口，不对外发布
GdnRecoverssmVerifyInner                  # 内部接口，不对外发布
fla_npu.ops.ascendc.gdn_recoverssm_verify # 唯一公开入口
```

Python 接口：

```python
def gdn_recoverssm_verify(
    A_log: Tensor,
    a: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    checkpoint_state: Tensor,
    replay_cache: Tensor,
    query_start_loc: Tensor,
    state_indices: Tensor,
    scale: float | None,
    softplus_beta: float,
    softplus_threshold: float,
    use_qk_l2norm_in_kernel: bool,
    out: Tensor | None,
) -> Tensor
```

固定参数顺序如下。Stable-ABI schema 与内部 launcher 参数表必须保持同一顺序和名称。

| 位置 | 参数 | 方向 | Shape | dtype | 说明 |
| ---: | --- | --- | --- | --- | --- |
| 0 | `A_log` | 输入 | `[HV]` | FP32 | 每 value head 的对数衰减尺度 |
| 1 | `a` | 输入 | `[T,HV]` 或 `[1,T,HV]` | BF16 | 原始 gate 输入 |
| 2 | `b` | 输入 | `[T,HV]` 或 `[1,T,HV]` | BF16 | 原始 beta 输入 |
| 3 | `dt_bias` | 输入 | `[HV]` | FP32 | 每 value head 的 gate bias |
| 4 | `q` | 输入 | `[1,T,HK,K]` | BF16 | Query |
| 5 | `k` | 输入 | `[1,T,HK,K]` | BF16 | Key |
| 6 | `v` | 输入 | `[1,T,HV,V]` | BF16 | Value |
| 7 | `checkpoint_state` | 输入、只读 | `[P,HV,V,K]` | BF16 或 FP32 | 分页 recurrent checkpoint |
| 8 | `replay_cache` | 输出、原地写 | `[P,HV,L,V+K+1]` | FP32 | commit 所需紧凑记录 |
| 9 | `query_start_loc` | 输入 | `[B+1]` | INT32 或 INT64 | 请求 token 起止位置 |
| 10 | `state_indices` | 输入 | `[B]` | INT32 或 INT64 | 每个请求的 checkpoint 页号 |
| 11 | `scale` | 属性 | 标量 | FLOAT | Query 缩放，`None` 使用 `K**-0.5` |
| 12 | `softplus_beta` | 属性 | 标量 | FLOAT | Softplus beta，默认 `1.0` |
| 13 | `softplus_threshold` | 属性 | 标量 | FLOAT | Softplus 线性分支阈值，默认 `20.0` |
| 14 | `use_qk_l2norm_in_kernel` | 属性 | 标量 | BOOL | 是否在 kernel 内归一化 Q/K，默认 `True` |
| 15 | `out` | 可选输出 | `[1,T,HV,V]` | BF16 | 未提供时由接口分配 |

Python wrapper 返回 `out`。`replay_cache` 作为 mutation 输出保留调用者视角中的原对象，
不与 `out`、`q`、`k`、`v` 或 `checkpoint_state` 发生部分重叠。
每请求记录容量 `L` 由 `replay_cache.shape[2]` 推导。

## Python 调用示例

```python
import torch

from fla_npu.ops.ascendc import gdn_recoverssm_verify


batch = 2
window = 4
hk, hv = 2, 6
k_dim = v_dim = 128
offsets = [0, 3, 7]
tokens = offsets[-1]
device = "npu"

q = torch.randn((1, tokens, hk, k_dim), dtype=torch.bfloat16, device=device)
k = torch.randn_like(q)
v = torch.randn((1, tokens, hv, v_dim), dtype=torch.bfloat16, device=device)
a = torch.randn((tokens, hv), dtype=torch.bfloat16, device=device)
b = torch.randn((1, tokens, hv), dtype=torch.bfloat16, device=device)
A_log = torch.linspace(-0.4, 0.3, hv, dtype=torch.float32, device=device)
dt_bias = torch.linspace(-0.2, 0.2, hv, dtype=torch.float32, device=device)

checkpoint = torch.randn(
    (batch + 1, hv, v_dim, k_dim), dtype=torch.float32, device=device
) * 0.02
replay = torch.empty(
    (batch + 1, hv, window, v_dim + k_dim + 1),
    dtype=torch.float32,
    device=device,
)

out = gdn_recoverssm_verify(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    checkpoint,
    replay,
    torch.tensor(offsets, dtype=torch.int32, device=device),
    torch.tensor([1, 2], dtype=torch.int32, device=device),
    scale=k_dim**-0.5,
    softplus_beta=0.5,
    softplus_threshold=2.0,
    use_qk_l2norm_in_kernel=True,
)

torch.npu.synchronize()
assert out.shape == (1, tokens, hv, v_dim)
assert out.dtype == torch.bfloat16
assert replay.dtype == torch.float32
```

`gdn_recoverssm_verify` 是唯一公开 Python 入口。`GdnRecoverssmVerifyInnerGetWorkspaceSize`
和 `GdnRecoverssmVerifyInner` 仅作为内部 launcher ABI，不安装公开头文件。

## 数学语义

对请求 `r`、value head `hv` 和 token `t`，令：

```text
ratio = HV / HK
hk = hv // ratio
S = checkpoint_state[state_indices[r], hv]
x = a[t, hv] + dt_bias[hv]
sp = log(1 + exp(softplus_beta * x)) / softplus_beta
     if softplus_beta * x <= softplus_threshold
sp = x otherwise
g = -exp(A_log[hv]) * sp
d = exp(g)
beta = sigmoid(b[t, hv])
qh = q[0, t, hk]
kh = k[0, t, hk]
qh = qh / sqrt(sum(qh * qh) + 1e-6)   # 归一化开启时
kh = kh / sqrt(sum(kh * kh) + 1e-6)   # 归一化开启时
qh = qh * scale
```

按请求内 token 顺序执行，当前 token 的结果使用更新后的状态：

```text
D = S * d
u = beta * (v[0,t,hv] - D @ kh)
S_next = D + u[:, None] * kh[None, :]
out[0,t,hv] = S_next @ qh

replay_cache[page,hv,t,0:V] = u
replay_cache[page,hv,t,V:V+K] = kh
replay_cache[page,hv,t,V+K] = d
```

计算语义：

- State、gate、归一化、点积、Beta 和 replay 使用 FP32 中间量。
- `out` 仅在最终写回时转换为 BF16。
- `checkpoint_state` 的 BF16/FP32 初态只读入 FP32 临时状态，不写回 checkpoint；
  BF16 量化不回灌后续 token。
- `replay_cache` 写入 FP32 correction、有效 normalized key 和标量 decay。
- `query_start_loc` 定义的有效 token 区间内逐 token 递推；区间外不写输出或 replay。
- 零长度请求不读取 checkpoint、不写 output/replay，也不改变同一核后续请求的执行状态。
- 调用方必须为非空请求提供正整数 page；kernel 不执行设备端 page 防御检查。

## 布局与步进

`q`、`k`、`v`、`out` 的最后一维 stride 必须为 1；head、token 维允许正 stride，
但同一逻辑元素不得被多个索引重复引用。`A_log`、`dt_bias`、`replay_cache` 的最后一维
stride 必须为 1。

`a`、`b` 的总元素数必须等于 `T * HV`，token/head 维允许 view；`a.stride(-1)` 和
`b.stride(-1)` 必须为 1。`checkpoint_state` 支持 page、head、value 三轮 stride，
其中 key 维 stride 必须为 1。`replay_cache` 支持 page、head、position、payload 四轮
stride，其中 payload 维 stride 必须为 1。所有 stride 必须为正，且不允许内部重叠。

`out` 提供时，其 shape 必须为 `[1,T,HV,V]`，value 维 stride 为 1，并满足相同的
非重叠约束。输入、checkpoint、replay、metadata 和输出必须位于同一 NPU device。

## 支持范围

- 目标 SoC：`ascend950`；本版本不声明 A2/A3 支持。
- Q/K/V 与 `a`/`b`：BF16；`A_log`/`dt_bias`：FP32。
- checkpoint：BF16 或 FP32；replay 与输出：分别为 FP32、BF16。
- `K`、`V`：固定为 128。首版仅验证 Qwen3.6-35B-A3B 与 Qwen3.8-27B 的模型宽度。
- `HK >= 1`、`HV % HK == 0`、`1 <= HV/HK <= 16`。
- `replay_cache.shape[2]` 必须位于 `[1,32]`；每个请求有效长度不大于该值。
- `query_start_loc[0] == 0`、末项等于 `T`，且序列非递减。
- 每个非空请求的 page 必须位于 `[1,P)`；页号随请求在调用内唯一，保证 replay 写区间不冲突。
- 零长度请求可与普通请求混排，作为 padding 请求支持。
- 支持 BF16/FP32 checkpoint padding、输出 view、INT32/INT64 metadata 和空 batch。

## 边界行为

- replay window 小于 1、`softplus_beta <= 0`、非有限 `scale` 或维度超限由 host 拒绝。
- Q 与 K 的 shape、dtype、device 必须一致；V、checkpoint、replay、metadata 的 shape
  和 dtype 必须满足上表。
- 空 batch 或 `T == 0` 时不启动 kernel，返回形状正确的空/零输出，checkpoint 和 replay
  不变。
- 空请求不执行读取或写回；非空请求仍要求正整数 page。
- 对任意输入路径，checkpoint 字节内容在调用前后完全一致。
- 支持范围内不允许 NaN/Inf gate 属性；输入中的非有限值遵循 IEEE 浮点运算，不另行
  静默替换。
- 调用为确定性前向推理，不建立 autograd 图。

## 错误行为

Stable-ABI C++ 适配层在创建 descriptor 前检查 NPU device、shape、dtype、stride、
storage offset、view 边界和内存重叠。host tiling 检查 SoC、维度上限、属性组合、
metadata 容量和 workspace。参数级错误返回具体参数名和原因；空指针返回参数错误；
不支持的 SoC 或超限 shape 在启动 kernel 前拒绝。
