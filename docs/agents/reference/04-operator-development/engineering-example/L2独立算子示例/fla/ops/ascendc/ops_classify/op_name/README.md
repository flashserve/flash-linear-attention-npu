# OpName 示例算子说明

<!--
示例文件：fla/ops/ascendc/ops_classify/op_name/README.md

注意事项：
  1. 本文件是输入限制与输出布局的**唯一定义来源**：tiling 校验、JSON 用例、executor 输入构造、
     docs/api.md 都必须与本文件一致。三者不一致时以本文件为准并修正其它文件。
  2. 「已知限制」要写死具体取值或档位（例如 D 只支持 128），不要写"视情况而定"。
  3. 只有本文件维护 Shape 变量附录；docs/design.md 与 docs/api.md 链接到这里，不重复定义。
  4. 不写内部环境信息（机器、路径、日志、账号）。
-->

## 功能

对每个 chunk 内的门控 `g` 做局部扫描，并按 `x` 的归一化结果更新输出：

```text
norm  = rsqrt(sum(x^2, dim=D) / D + epsilon)     # NORM_MODE=L2
scan  = 前缀和(g * scale)                        # chunk 内
y     = x * scan * norm
state = 每个 chunk 末尾的 scan                   # 可选导出
```

## 输入

<!-- 注意事项：逐个参数写 shape、dtype、layout、约束；可选输入的"缺席语义"必须写清。 -->

| 参数 | 必选/可选 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `x` | 必选 | BSND `[B,T,H,D]`、BNSD `[B,H,T,D]`、TND/NTD 打包 token | BF16 / FP16 | 主输入；varlen 时为打包 token |
| `g` | 必选 | 与 `x` 的 token/head 轴一致，无 D 维 | FP32 / BF16 | 门控 |
| `a_log` | 可选 | `[H]` | FP32 | 非空时参与门控缩放；为空时按 `scale` 直接使用 |
| `initial_state` | 可选 | `[B,H,D]` | 与 `x` 同 dtype | 非空时从状态续算 |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | 变长元数据，必须从 0 开始、单调不减、末元素等于总 token 数 |
| `chunk_indices` | 可选 | `[N_c,2]` | INT64 | 必须与 `cu_seqlens` 同时提供 |

## 属性

| 属性 | 取值 | 默认 | 说明 |
| --- | --- | --- | --- |
| `layout` | `BSND` / `BNSD` / `TND` / `NTD` | `BSND` | 只解释输入布局 |
| `scale` | 任意正数 | `1.0` | 门控缩放 |
| `chunk_size` | `64` / `128` | `64` | chunk 内扫描长度 |
| `epsilon` | `> 0` | `1e-6` | 仅 L2 归一化使用 |
| `output_mode` | `0` / `1` | `0` | L2 内部档位，不是用户参数 |

## 输出

| 参数 | 必选/可选 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| `y` | 必选 | BSND/TND，与输入 token 轴一致 | 与 `x` 相同 | 扫描结果 |
| `state` | 可选导出 | `[B,H,chunk_count,D]` | 与 `x` 相同 | 仅在 `output_mode=1` 时导出 |
| `x_norm` | 可选导出 | 与 `x` 相同 | FP32 | 反向中间量，仅在 `output_mode=1` 时导出 |

## 已知限制

1. `D` 只支持 `128`；其它取值在参数校验阶段返回 `ACLNN_ERR_PARAM_INVALID`，报错打印实际 `Ddim`。
2. `chunk_size` 只支持 `64`/`128`；两者命中不同的 tiling key 模板参数。
3. 变长输入要求 `B=1`，且 `cu_seqlens`/`chunk_indices` 必须成对出现。
4. `a_log` 为空时 `scale` 直接作为门控系数；非空时先算 `exp(a_log)`。
5. SoC 支持：A2（`ascend910b`）、A3（`ascend910_93`）、A5（`ascend950`），三个平台共用同一公开原型。
