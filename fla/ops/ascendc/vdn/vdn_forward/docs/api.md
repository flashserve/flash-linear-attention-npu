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

发布模型对 K、V 分别使用各自的空间和时间卷积权重。Q 直接进入后续特征处理。调用方式为

```python
out = vdn_forward(q, k, v, beta, alpha, bounds, text_state,
                  short_conv={"k": (k_spatial, k_temporal), "v": (v_spatial, v_temporal)},
                  frame_size=(height, width), fuse_silu=True, fuse_qk_l2norm=True,
                  delta_rule="newton_schulz", value_first=True)
```

空间阶段为逐通道 5×5 卷积。时间阶段对空间输出做逐通道五点卷积，分块边界从相邻帧取值，仅片段的两端补零。BF16 时间卷积的每次乘法和加法均按特征 dtype 舍入。时间核同时计算 SiLU 和可选的 Q/K 归一化。V 有卷积且当前块的 β 连续并与 V 同 dtype 时，时间核还会在写回前完成 β 加权。已在上游完成卷积的调用方应省略对应的权重。

令每帧每头的 token 沿矩阵的行排列。对预处理输出，构造
`A = Kᵀ diag(β) K`、`M = I + (A+Aᵀ)/2`、`B = Vᵀ diag(β) K`。
两方向使用相同更新 `step(X,t) = (X diag(α_t) + B_t) M_t⁻¹`。
令 `P[0] = R[F] = text_state`，则 `P[t+1] = step(P[t],t)`，
`R[t] = step(R[t+1],t)`。对窗口 [lo,hi]，左状态取 `P[lo]`，右状态取
`R[hi+1]`。分别对状态的 key 列乘上 `α[lo:t+1]` 和 `α[t:hi+1]` 的逐列乘积，再用
`Q_t @ (left + right).T` 读出。计算窗口乘积时将衰减下限取 1e-12。

模型参考固定在 [OpenVDN 视频线性注意力分支](https://github.com/OpenVDN/vdn-minimax-h3/blob/e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b/src/models/linear_attention/branch.py)。精度验证在 HF32 关闭时运行。Gram A、求逆、状态递推和衰减使用 FP32。BF16 路径保留加权 V、value Gram 及状态读出等模型舍入点。不同求解器、分块和重复运行之间只承诺经测试的数值精度。


## Ascend C 接口

完整前向由 Python 编排。A2 OPP 提供下列三个子算子的两段式 aclnn 接口。第一段 `GetWorkspaceSize` 返回本次调用所需的 workspace 字节数和 executor，第二段使用 workspace、executor 和目标 stream 执行。

| Stable PyTorch 接口（`fla_npu.ops.ascendc`） | aclnn 名称 | 写回语义 |
| --- | --- | --- |
| `npu_vdn_scan(transition, prefix, suffix)` | `aclnnVdnScanGetWorkspaceSize`、`aclnnVdnScan` | 就地修改两份 FP32 状态库。调用前 prefix[0]、suffix[F] 为初态，其余目标槽为各步的 injection。 |
| `npu_vdn_bridge(prefix, suffix, bounds, logs, start, out)` | `aclnnVdnBridgeGetWorkspaceSize`、`aclnnVdnBridge` | 从 start 帧起写入 out.shape[0] 帧，out 可为 FP32 或 BF16。 |
| `npu_vdn_add_diagonal(correction, two_eye)` | `aclnnVdnAddDiagonalGetWorkspaceSize`、`aclnnVdnAddDiagonal` | 对 FP32 [N,128,128] 校正缓冲就地加 2I，two_eye 为同设备的 FP32 [128,128] 二倍单位阵。 |

三个子算子使用同一 NPU 上的连续 ND 张量。令 H 为头数，D=128，C 为本次写回的有效帧数：

- Scan 的 transition 为 [F,H,D,D]，prefix/suffix 为 [F+1,H,D,D]。两个可写状态库和 transition 的有效字节区间互不重叠。
- Bridge 的 prefix/suffix 为 [F+1,H,D,D]，bounds 为 int64 [F,2]，logs 为 FP32 [F+1,H,D]，out 为 [C,H,D,D]。start 满足 0 ≤ start ≤ F−C，out 的有效区间与输入分离。
- AddDiagonal 的 N 为正，correction 与 two_eye 的有效区间分离。切片共享底层存储且数据区间不相交时可传入。

调用 C 接口前分配相应输出与工作区，并检查两个阶段的返回值。`ACLNN_SUCCESS` 为成功，参数不合法时按对应头文件与 L2 实现返回错误。完整张量参数顺序见 [Scan](../../vdn_scan/op_host/op_api/aclnn_vdn_scan.h)、[Bridge](../../vdn_bridge/op_host/op_api/aclnn_vdn_bridge.h)、[AddDiagonal](../../vdn_add_diagonal/op_host/op_api/aclnn_vdn_add_diagonal.h)。
