# aclnnGdnCoreFwdPhase6

`aclnnGdnCoreFwdPhase6` 将 GDN 系数生成、状态更新和输出计算收敛到同一个
`ChunkGdnCoreFwd` 底层入口。算子保持四个固定输出槽：`oOut`、
`finalStateOutOptional`、`gCumsumOut` 和 `aOut`。

## 辅助输出语义

- `oOut` 始终必选。
- `finalStateOutOptional` 在 `outputFinalState=true` 时必选，否则可以为 null。
- `gCumsumOut` 和 `aOut` 可以分别为 null。
- null 不会改变底层输出数量、顺序或公开 C ABI。L2 使用 rank-1 `[1]`
  placeholder 保持 REQUIRED L0 固定槽，并从公开 null 指针显式生成私有
  `output_mask` attr；tiling 不从输出 storage shape 反推 mask。
- `gCumsumOut=null` 时，kernel 仍生成 H/O 所需的内部 BHT cumsum，只跳过公开
  BTH 搬出。
- `aOut=null` 时，kernel 仍生成 Solve/Recompute 所需的 A，并将其保存在内部
  `a_storage`，不写公开 A 输出。

两个辅助输出可以独立省略。`output_mask` 的 bit 0 表示写出
`gCumsumOut`，bit 1 表示写出 `aOut`；其他 bit 无效。

## Python 调用

默认行为保持不变，返回全部四个槽：

```python
o, final_state, g_cumsum, A = gdn_core_fwd_phase6(
    q, k, v, g, beta, return_aux=True
)
```

推理可以显式关闭两个公共辅助输出：

```python
o, final_state, g_cumsum, A = gdn_core_fwd_phase6(
    q, k, v, g, beta, return_aux=False
)
assert g_cumsum is None and A is None
```

`return_aux` 是 keyword-only 参数，默认值为 `True`。接口不会依据
`requires_grad`、全局 grad mode 或 `model.training` 隐式切换，因此训练和推理行为
由调用方明确决定。

## 兼容性

- `aclnnGdnCoreFwdPhase6GetWorkspaceSize` 的参数数量、顺序和 C 类型不变。
- `ChunkGdnCoreFwd` 的 L0 输入和四个 REQUIRED 输出不变；私有 L0 新增
  `output_mask` attr，不暴露给公开 ACLNN C ABI。
- 旧调用方继续提供 `gCumsumOut/aOut` 时，shape、dtype 和数值语义不变。
- Python 返回值始终是四元组；关闭辅助输出时对应位置为 `None`。
