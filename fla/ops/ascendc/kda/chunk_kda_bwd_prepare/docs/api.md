# ChunkKdaBwdPrepare 接口

## A2/A3 通用分支（本次新增）

FP32 Aqk/v_new/h 缓存配 FP16/BF16 dO；dAqk、dv、dq_raw 均为 FP32。K 与 V 分别从状态和 value 取值，K-first 使用 dO@h.T，V-first 使用 dO@h。

C64、等 head，六组 K/V 和 FP32 缓存契约见 [组合接口](../../chunk_kda_bwd/docs/api.md)。其余旧路径约束仍按下文。


```python
from fla_npu.ops.ascendc import chunk_kda_bwd_prepare
d_aqk, dv, dq_raw = chunk_kda_bwd_prepare(
    aqk, v_new, d_o, h, scale=scale, layout="BNSD",
    chunk_size=64, state_v_first=False,
    cu_seqlens=None, chunk_indices=None,
)
```

输入均为连续 BF16，固定 C=64、K=V=128。`scale` 必须可表示为有限 FP32。

| layout | aqk | v_new / d_o | h | 输出布局 |
| --- | --- | --- | --- | --- |
| BNSD | [B,H,T,64] | [B,H,T,128] | [B,ceil(T/64),H,128,128] | BNSD |
| BSND | [B,T,H,64] | [B,T,H,128] | 同上 | BNSD |
| NTD | [H,T,64] | [H,T,128] | [total_chunks,H,128,128] | NTD |
| TND | [T,H,64] | [T,H,128] | 同上 | NTD |

B、H、T 必须为正。layout 只决定 rank 和轴顺序，`cu_seqlens` 决定是否变长：

- 不传 `cu_seqlens`：四维按 B 个等长序列处理，三维按单序列处理。
- 传入 `cu_seqlens`：四种 layout 均可用，四维要求 B=1。元数据至少有 2 项，无固定序列数上限，
  非递减、起点为 0、终点为 T，允许空序列。
- 变长 h 为 `[1,total_chunks,H,128,128]`（四维 Token）或
  `[total_chunks,H,128,128]`（三维 Token）；`total_chunks = sum(ceil(seq_len/64))`。
- `chunk_indices` 为规范序列顺序的扁平 `(sequence_id, local_chunk_id)` 列表；
  省略时 Host 自动生成。不能只传 `chunk_indices`。

每个 chunk 的有效行数为 m，状态为对应 chunk 的 h：

- `d_aqk[:, :m] = tril(d_o @ v_new.T) * scale`，FP32；上三角及尾列严格为零。
- `dv = aqk[:, :m].T @ d_o`，最终一次 RN 舍入为 BF16，不乘 scale。
- `dq_raw = d_o @ h.T`，FP32；`state_v_first=True` 时改为 `d_o @ h`。

三路输出均为连续 head-major；维度分别为 64、128、128。输入不会做 NPU 布局转换。
不支持的 dtype、维度、布局、元数据和非连续输入由调用层及 ACLNN 校验拒绝。

ACLNN 两段式入口中，`const char *layout` 位于两项可选元数据之后、`double scale` 之前。
保留 A2/A3/950 的算子注册；arch22 使用 GM 中转，arch35 保留 L0C→UB 路径。
A5 接口一致，目前只开放定长 BNSD、变长 NTD，其余组合由 Host 拒绝，功能后续适配。
A6（ascend960dt / DAV_9201）同样只开放定长 BNSD、变长 NTD，两种
`state_v_first` 均支持。变长允许空 sequence，但总 T 必须为正。
A6 三路结果经 UB 交给 AIV 写出，用户 workspace 为 0；ACLNN workspace
仍包含 SDK 系统 workspace。A6 按 AIC/AIV 核数相同的约定，仅查询 AIC 核数；
AIC 核数为 0 时 tiling 拒绝发射，不以固定默认核数替代平台查询。
原提交的 A2 设备验证、A3 编译记录不代表当前整合版本已经通过回归。
A5 的 BSND/TND 要求 H*128 不超过 uint32_t 上限（CATLASS 的行距字段限制），
不满足时 host tiling 拒绝；常规模型 head 数不受影响。
实现详见 [design.md](design.md)。

兼容性：原完整 backward 的两处 L0 调用显式传 BNSD/NTD，输出消费方式不变。
旧 standalone prepare 和 FP16 分支已删除；A2/A3 总 backward 的 FP16 输入明确报错。
总 backward 的公开签名不变，BF16 模型调用进入新 prepare。
新增 `layout` 改变 ACLNN C ABI，旧二进制调用者不能与新 OPP 混用，须同步重新编译。

参数校验通过 Error Message 上报错误码和具体原因：缺失必需参数为 EZ0004，
shape 不匹配为 EZ0008，dtype 不匹配为 EZ0019，属性值非法为 EZ0024，
非连续输入或元数据约束非法为 EZ0024；同时保留对应 ACLNN 状态返回值。
Stable-ABI 与 ctypes 入口均在 ACLNN 失败时显示 Error Message 的详情。
对于缺少 `OP_LOGE_FOR_INVALID_*` 的旧 SDK（如 CANN 9.0.0），shape/属性值/元数据
使用其已有的 EZ0002，dtype 使用 EZ0007；缺失参数仍为 EZ0004。
两种路径都保留参数名、实际值、期望值或原因，以及原有 ACLNN 状态返回值。

调用层及错误信息可用以下脚本回归（`--device` 为进程可见设备编号）：

```bash
python tests/stable_abi/test_chunk_kda_bwd_prepare.py --device 0
python tests/stable_abi/test_chunk_kda_bwd_prepare.py --device 0 --extended
python tests/stable_abi/test_chunk_kda_bwd_prepare_errors.py --device 0
```

正常输入脚本覆盖四种布局、两种状态方向、奇数 head、密集多 batch、packed 空序列及尾块，
并断言使用 Stable 后端且未发生回退。错误脚本覆盖参数、元数据、输出 buffer 和空指针校验，
同时检查 ACLNN 返回码、Error Message 错误码及参数名，并验证公开入口的异常详情。

`--extended` 增加 16 行分形边界、H=1/2/5、H=513 的 token 行距、长序列及
多任务 slot 复用输入；另补充全部 layout × 定变长 × 状态方向 × 自动元数据组合。
实际 slot 回绕仍需设备 trace 证明。
另外执行四布局的非有限上三角/尾列掩码检查。
A6 功能 smoke（整数 BF16 输入、独立 CPU 标杆、逐位比较；不是 ATK 精度验收）：

```bash
python tests/atk/chunk_kda_bwd_prepare/scripts/prepare_a6_smoke.py --tokens 65 --heads 3
python tests/atk/chunk_kda_bwd_prepare/scripts/prepare_a6_smoke.py --varlen --state-v-first --tokens 129 --heads 5
```

上面的四布局 Stable-ABI 扩展脚本用于 A2/A3 回归；A5/A6 设备验证只选择对应平台支持的组合。
