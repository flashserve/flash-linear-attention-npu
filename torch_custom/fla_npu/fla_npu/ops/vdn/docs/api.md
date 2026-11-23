# 前向接口

```python
vdn_forward(q, k, v, beta, alpha, bounds, text_state=None, *,
            delta_rule="vdn_solve", fuse_silu=False, fuse_qk_l2norm=False,
            short_conv=None, frame_size=None, frame_tile=None,
            pad_tail=True, value_first=False)
```

| 参数 | 规格与语义 |
| --- | --- |
| `q, k, v` | 同一 NPU 上同形状、同 dtype 的 FP32 或 BF16 张量 `[F,H,S,128]`。F、H、S 均为正数。H 是当前进程可见头数。支持连续、token-major 转置视图及其他合法步长。 |
| `beta` | `[F,H,S]`，逐 token 权重，FP32 或特征 dtype，与 Q 同设备。模型取值位于 [0,1]。 |
| `alpha` | `[F,H,128]`，与 Q 同设备的 FP32 逐列衰减。模型取值位于 (0,1]。 |
| `bounds` | CPU 或 Q 所在设备上的 int64 `[F,2]`。每行给出含端点的帧区间，满足 `0 ≤ lo ≤ t ≤ hi < F`。CPU 输入会检查每行范围；设备输入的数值由调用方保证。 |
| `text_state` | 可选 FP32 `[H,128,128]`，与 Q 同设备，按 value × key 排列。传入上游已缩放的状态，两端使用同一初态；省略时两端从零开始。 |
| 返回值 | 与 Q 同 dtype、同形状的读出结果，每次分配新输出。 |
| `delta_rule` | `vdn_solve` 或 `newton_schulz`。Newton 路径支持 S ≤ 405，要求归一化的 K 或调用方已保证加权 Gram 的谱界 ≤ 405。 |
| `fuse_silu, fuse_qk_l2norm` | 独立开关。先对指定投影应用可选卷积，再对 Q/K/V 做可选 SiLU，最后对 Q/K 按末维做可选 L2 归一化，分母下限 1e-6。 |
| `short_conv, frame_size` | 可选字典，以 `q`、`k` 或 `v` 为键。每个值为一对 depthwise 权重，空间核 `[H*128,1,5,5]`，时间核 `[H*128,5]` 或 `[H*128,1,5]`；dtype 和设备与 Q 一致。`frame_size=(height,width)`，两者乘积为 S；空间和时间端点零填充。 |
| `frame_tile, pad_tail, value_first` | 帧块上限可设正的 2 的整幂或由实现选择。`pad_tail=True` 将尾块补至最近的上方 2 次幂；关闭时将余数拆为多个 2 次幂块。`value_first=True` 先算 value Gram 以改善缓存复用。 |

令每帧每头的 token 沿矩阵的行排列。对预处理输出，构造
`A = Kᵀ diag(β) K`、`M = I + (A+Aᵀ)/2`、`B = Vᵀ diag(β) K`。
两方向使用相同更新 `step(X,t) = (X diag(α_t) + B_t) M_t⁻¹`。
令 `P[0] = R[F] = text_state`，则 `P[t+1] = step(P[t],t)`，
`R[t] = step(R[t+1],t)`。对窗口 [lo,hi]，左状态取 `P[lo]`，右状态取
`R[hi+1]`。分别对状态的 key 列乘上 `α[lo:t+1]` 和 `α[t:hi+1]` 的逐列乘积，再用
`Q_t @ (left + right).T` 读出。计算窗口乘积时将衰减下限取 1e-12。

模型参考固定在 [OpenVDN 视频线性注意力分支](https://github.com/OpenVDN/vdn-minimax-h3/blob/e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b/src/models/linear_attention/branch.py)。精度验证在 HF32 关闭时运行。Gram A、求逆、状态递推和衰减使用 FP32。BF16 路径保留加权 V、value Gram 及状态读出等模型舍入点。不同求解器、分块和重复运行之间只承诺经测试的数值精度。

