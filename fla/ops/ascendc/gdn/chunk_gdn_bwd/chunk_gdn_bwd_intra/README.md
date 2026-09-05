# ChunkGdnBwdIntra

[设计文档](docs/design.md) | [API 文档](docs/api.md) |
[ATK 测试](../../../../../../tests/atk/chunk_gdn_bwd_intra/README.md)

`ChunkGdnBwdIntra` 融合 `recompute_w_u_fwd` 与 `chunk_bwd_dv_local`，按顺序返回
`w`、`u` 和 `dv_local`。

稳定入口、参数、属性、输出和支持范围以 [API 文档](docs/api.md) 为准；Stage 划分、
TilingKey、L1/UB/GM workspace 和同步见[设计文档](docs/design.md)；最终测试范围与验收
结果见 [ATK 测试](../../../../../../tests/atk/chunk_gdn_bwd_intra/README.md)。
