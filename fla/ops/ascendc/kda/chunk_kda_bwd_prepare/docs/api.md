# ChunkKdaBwdPrepare 接口

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
- 传入 `cu_seqlens`：四种 layout 均可用，四维要求 B=1。元数据有 2..1025 项，
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
纯 CPU 标杆自检：`python tests/test_chunk_kda_bwd_prepare_reference.py`。
