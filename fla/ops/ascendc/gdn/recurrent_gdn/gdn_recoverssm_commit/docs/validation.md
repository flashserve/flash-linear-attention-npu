# 验证记录

## 02 标杆运行

| 范围 | 运行结果 | 计算语义 | 精度策略 | 证据 | 结论 |
| --- | --- | --- | --- | --- | --- |
| CPU FP64，固定 seed，常规 final commit、boundary prefix、零长度、null page、同页 final/boundary、BF16 state | PASS | PASS | PASS | `test/test_reference.py` 6 项全部通过 | 通过 |

## 04 算子开发

| 检查项 | 结果 |
| --- | --- |
| Python 语法检查 | PASS |
| Stable-ABI 源码审计 | PASS |
| Stable-ABI 覆盖检查（严格模式） | PASS，`npu_gdn_recoverssm_commit` 的 wrapper/schema/run/def/impl 均完整 |
| Ascend C host/API 编译 | PASS |
| Ascend C kernel 编译 | PASS，BF16/FP32 与 `align_mode` 模板组合均已实际生成 |
| run 包封装 | 待测试机连接恢复后使用现有构建缓存完成 |

## 待完成验收

- 安装本轮 run 包并校验 `libcust_opapi.so` 与轮子源一致。
- 在可用 A5 卡上运行 `torch_custom/fla_npu/test/test_npu_gdn_recoverssm_commit.py`。
- 对 BF16/FP32 state、`align_mode` 两种组合和执行 graph replay 做单算子精度验收。
- 使用 `msprof` 采集主场景 `Task Duration(us)`。
