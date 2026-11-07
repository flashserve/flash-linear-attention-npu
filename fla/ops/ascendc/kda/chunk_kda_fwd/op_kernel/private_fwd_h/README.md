# ChunkKdaFwd private FwdH

本目录保存融合入口 `aclnnChunkKdaFwd` 使用的最小 FwdH 头文件集合，
用于把 V1 的编译和实现边界收敛到 `chunk_kda_fwd` 内。

- 初始来源：`34e36de0a35609042d5007b8444037543cf2ee0e` 的
  `chunk_gated_delta_rule_fwd_h/op_kernel` 头文件。
- 不包含 GDN 独立算子的入口 `.cpp`、op_host、tiling 或 aclnn 代码。
- `KdaFwdHPrivateTilingData` 只描述 `ChunkKdaFwd` 融合入口传给私有
  FwdH 的字段，不与独立 GDN 算子共享 ABI。
- 后续修改属于 `ChunkKdaFwd` 私有实现，不得默默同步到独立
  `ChunkGatedDeltaRuleFwdH`；若需要同步，必须分别验证两个算子。
