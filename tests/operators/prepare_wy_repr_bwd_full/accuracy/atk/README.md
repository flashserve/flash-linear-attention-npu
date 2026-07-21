# prepare_wy_repr_bwd_full ATK 回归资产

## 用例来源

唯一用例来源是 tests/op_cases/prepare_wy_repr_bwd_full.json 中 legacy.suite=atk_legacy_mislabeled 的 2 条记录。
本目录只保留 ATK schema、generator 和 executor，不再提交第二份 all_*.json。

> 原文件的算子名、输入 schema 与当前算子不一致，这 2 条 case 已原样归档并标记为 disabled；修正 schema 前不得作为通过结论。

## 物化与执行

    python3 -m tests.operators._shared.legacy_cases materialize \
      --op prepare_wy_repr_bwd_full \
      --suite atk_legacy_mislabeled \
      --output /tmp/prepare_wy_repr_bwd_full_atk.json

将生成文件作为 atk task -c 的输入；生成文件是临时产物，不得提交。执行前仍需按本目录 yaml 注册 generator/executor，并以 yaml 中的精度标准判定结果。
