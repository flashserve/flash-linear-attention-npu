# VdnAddDiagonal

在 Ascend 910B 上为一批 FP32 方阵原地加上 `2I`，用于 VDN 的 Newton–Schulz 更新。每个方阵规格为 128 × 128。

- [ACLNN 接口及输入要求](docs/api.md)
- [向量任务与 tiling](docs/design.md)
- [完整 VDN 前向](../vdn_forward/README.md)
