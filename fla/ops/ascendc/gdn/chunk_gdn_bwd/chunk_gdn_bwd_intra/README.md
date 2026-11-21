# ChunkGdnBwdIntra

[设计文档](docs/design.md) | [API 文档](docs/api.md) |
[ATK 测试](../../../../../../tests/atk/chunk_gdn_bwd_intra/README.md)

`ChunkGdnBwdIntra` 融合 `recompute_w_u_fwd` 与 `chunk_bwd_dv_local`，按顺序返回
`w`、`u` 和 `dv_local`。

稳定入口、参数、属性、输出和支持范围以 [API 文档](docs/api.md) 为准；Stage 划分、
TilingKey、L1/UB 空间和同步见[设计文档](docs/design.md)；最终测试范围与验收
结果见 [ATK 测试](../../../../../../tests/atk/chunk_gdn_bwd_intra/README.md)。

## 支持范围

- SoC：A2 / Ascend 910B、A3 / Ascend 910C（`ascend910_93`）、A5 / Ascend 950、A6 / Ascend 960DT（`ascend960dt`）。
- `G = HV/HK`：A2/A3（DAV_2201）为 `1~16` 中的整数（`5~16` 走运行时 G 桶，
  TilingKey `G_KEY=0`）；A5 为 `1~16`，其中 G>4 仅支持 BF16 主输入。A6 为 `1~4`。
- 完整参数、dtype 与 layout 约束见 [API 文档](docs/api.md) 的“输入与输出”与
  “支持范围”章节。

