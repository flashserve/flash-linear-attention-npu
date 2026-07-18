# chunk_gated_delta_rule_bwd_dhu ATK 回归资产

## 用例来源

唯一用例来源是 tests/op_cases/chunk_gated_delta_rule_bwd_dhu.json 中 legacy.suite=atk_regression 的 200 条记录。
本目录只保留 ATK schema、generator 和 executor，不再提交第二份 all_*.json。

物化结果与迁移前 JSON 已做逐字段一致性校验；case 本身只在 manifest 中维护。

## 物化与执行

    python3 -m tests.operators._shared.legacy_cases materialize \
      --op chunk_gated_delta_rule_bwd_dhu \
      --suite atk_regression \
      --output /tmp/chunk_gated_delta_rule_bwd_dhu_atk.json

将生成文件作为 atk task -c 的输入；生成文件是临时产物，不得提交。执行前仍需按本目录 yaml 注册 generator/executor，并以 yaml 中的精度标准判定结果。
