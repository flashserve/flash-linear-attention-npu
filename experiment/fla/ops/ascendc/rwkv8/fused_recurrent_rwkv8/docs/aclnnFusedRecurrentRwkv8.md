# aclnnFusedRecurrentRwkv8

源码路径：`fla/ops/ascendc/rwkv8/fused_recurrent_rwkv8/`

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| Atlas A2 训练系列产品/Atlas A2 推理系列产品（Ascend 910B） | √ |
| 其他 | 未验证 |

## 功能说明

- 接口功能：RWKV-v8（WKV7）fused recurrent 前向递推。逐 token 更新每个 (b, h) 的 (N, N) 状态并产生输出 `o`，可选携带初始状态、可选产出 chunk 快照 `s` 与每 token 的 `sa`（训练预埋）。

- 计算公式（z/b 参数化，对齐 RWKV-LM `wkv7_cuda.cu` forward）：

  $$
  sa_t = S_{t-1}\, z_t
  $$

  $$
  S_t = S_{t-1} \odot \mathrm{diag}(decay_t) + sa_t\, b_t^T + v_t\, k_t^T,\quad decay_t = \exp(-\exp(w_t))
  $$

  $$
  o_t = S_t\, (q_t \cdot scale)
  $$

  其中 $S \in R^{N \times N}$（行 = v/q 侧，列 = k/z 侧），$q_t, w_t, k_t, v_t, z_t, b_t \in R^N$。

## 函数原型

两段式接口，必须先调用 GetWorkspaceSize 再调用执行接口。

```cpp
aclnnStatus aclnnFusedRecurrentRwkv8GetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *w,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *z,
    const aclTensor *b,
    const aclTensor *initialState,
    float           scale,
    bool            reverse,
    bool            outputChunkState,
    bool            outputSa,
    int64_t         chunkLen,
    aclTensor       *out,
    aclTensor       *sOut,
    aclTensor       *saOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnFusedRecurrentRwkv8(
    void          *workspace,
    uint64_t      workspaceSize,
    aclOpExecutor *executor,
    aclrtStream   stream)
```

## aclnnFusedRecurrentRwkv8GetWorkspaceSize

- 参数说明

  | 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) | 可空 |
  |:---|:---|:---|:---|:---|:---|:---:|
  | q | 输入 | 公式中的 q | FLOAT16/BF16/FLOAT32 | ND | (B, H, T, K) | × |
  | w | 输入 | log 域衰减参数，decay = exp(-exp(w)) | FLOAT16/BF16/FLOAT32 | ND | (B, H, T, K) | × |
  | k | 输入 | 公式中的 k | FLOAT16/BF16/FLOAT32 | ND | (B, H, T, K) | × |
  | v | 输入 | 公式中的 v | FLOAT16/BF16/FLOAT32 | ND | (B, H, T, V) | × |
  | z | 输入 | 公式中的 z（状态读出侧） | FLOAT16/BF16/FLOAT32 | ND | (B, H, T, K) | × |
  | b | 输入 | 公式中的 b（sa 写回系数） | FLOAT16/BF16/FLOAT32 | ND | (B, H, T, K) | × |
  | initialState | 输入 | 初始状态，接口朝向 = 内核账本 Sᵀ 原样（与 sOut 一致） | FLOAT32 | ND | (B, H, K, V) | √（空 = 零初态） |
  | scale | 属性 | q 的缩放系数 | FLOAT32 | - | 标量 | - |
  | reverse | 属性 | true = T 维倒序递推（对齐 fla reverse） | BOOL | - | 标量 | - |
  | outputChunkState | 属性 | true = 产出 chunk 快照 s（训练预埋，默认 false） | BOOL | - | 标量 | - |
  | outputSa | 属性 | true = 产出每 token 的 sa（训练预埋，默认 false） | BOOL | - | 标量 | - |
  | chunkLen | 属性 | s 快照间隔（默认 16，对齐官方 CUDA backward 的 chunk 重建粒度；非 16 值与官方 backward 不兼容） | INT64 | - | 标量，>= 1 | - |
  | out | 输出 | 公式中的 o | 同 q | ND | (B, H, T, V) | × |
  | sOut | 输出 | chunk state 快照，**官方 CUDA 转置布局**（快照 [j][i]=S[i][j]，UB 中 S^T 朝向即官方布局） | FLOAT32 | ND | (B, H, T//chunkLen, K, V) | outputChunkState=true 时必填 |
  | saOut | 输出 | 每 token 移除读出 state@z（更新前） | FLOAT32 | ND | (B, H, T, V) | outputSa=true 时必填 |
  | workspaceSize | 输出 | 需在 device 侧申请的 workspace 大小 | - | - | - | - |
  | executor | 输出 | op 执行器 | - | - | - | - |

- 约束说明
  - q/w/k/z/b 支持 float16/bfloat16/float32，dtype 必须一致，ND 格式，shape 必须全等 (B,H,T,K)；v/out/saOut 的末维 V 独立（(B,H,T,V)，K≠V 合法，K=V 为特例），dtype 同样必须与 q 一致。
  - initialState/sOut/saOut 恒为 float32（递推累加全程 fp32，state 不降精度）。
  - sOut：每满 chunkLen 个 token 拍一次快照（chunkLen 默认 16，对齐官方 CUDA backward 的 chunk 重建粒度）；T 非 chunkLen 倍数时 floor(T/chunkLen) 个快照，尾部不满的一段无快照；T<chunkLen 时为零尺寸。
  - sOut/saOut 默认关闭：关闭时传 nullptr 即可，kernel 跳过写出（推理零带宽开销）。
  - K 需为 8 的倍数且 ≤ 128（K 维不切分，K×V 的 state 须整体放入单核 UB）。
  - V 需为 8 的倍数且 ≤ 128（V 维不切分，与 K 同上由 UB 预算决定）。
  - B、T、H 为任意正整数（T 无需对齐）。
  - 当前仅前向；反向由 chunk 算子另行承担（s/sa 输出即为其预埋）。

## aclnnFusedRecurrentRwkv8

- 参数说明

  | 参数名 | 描述 |
  |:---|:---|
  | workspace | device 侧 workspace 内存起址（workspaceSize 为 0 时传 nullptr） |
  | workspaceSize | 第一段接口返回的 workspace 大小 |
  | executor | 第一段接口返回的执行器 |
  | stream | acl stream |

## 调用示例

参见 `examples/test_aclnn_fused_recurrent_rwkv8.cpp`（自包含：内置输入生成 + CPU golden + rel-RMSE 对拍），通过以下命令构建运行：

```bash
bash build.sh --run_example fused_recurrent_rwkv8 eager cust --vendor_name=fla_npu --soc=ascend910b
```
