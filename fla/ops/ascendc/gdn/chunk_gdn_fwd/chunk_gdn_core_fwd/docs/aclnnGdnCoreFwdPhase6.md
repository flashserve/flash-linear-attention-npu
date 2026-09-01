# aclnnGdnCoreFwdPhase6

`aclnnGdnCoreFwdPhase6` 将 GDN 系数生成、状态更新和输出计算收敛到同一个
`ChunkGdnCoreFwd` 底层入口。算子保持四个固定输出槽：`oOut`、
`finalStateOutOptional`、`gCumsumOut` 和 `aOut`。

## 辅助输出语义

- `oOut` 始终必选。
- `finalStateOutOptional` 在 `outputFinalState=true` 时必选，否则可以为 null。
- `gCumsumOut` 和 `aOut` 可以分别为 null。
- null 不会改变底层输出数量、顺序或 C ABI。L2 使用 rank-1 `[1]` placeholder
  保持 REQUIRED L0 固定槽，tiling 再生成内部 output mask。
- `gCumsumOut=null` 时，kernel 仍生成 H/O 所需的内部 BHT cumsum，只跳过公开
  BTH 搬出。
- `aOut=null` 时，kernel 仍生成 Solve/Recompute 所需的 A，并将其保存在内部
  `a_storage`，不写公开 A 输出。

两个辅助输出可以独立省略。公开输出是否存在按 rank 判断：正常 cumsum 为 rank 3、
正常 A 为 rank 4、placeholder 为精确的 rank-1 `[1]`；不得使用元素数量判断，以免
把合法的 `[1,1,1]` 或 `[1,1,1,1]` 输出误认为 placeholder。

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
- `ChunkGdnCoreFwd` 的 L0 输入、输出和属性数量不变。
- 旧调用方继续提供 `gCumsumOut/aOut` 时，shape、dtype 和数值语义不变。
- Python 返回值始终是四元组；关闭辅助输出时对应位置为 `None`。
