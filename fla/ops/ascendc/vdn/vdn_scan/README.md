# VdnScan

在 Ascend 910B 上执行 VDN 的双向帧状态递推。调用方先把每帧的注入项写入两个状态库，本算子将前向和反向结果写回原处。状态与转移矩阵均为 FP32，head dim 为 128。

- [ACLNN 接口及输入要求](docs/api.md)
- [分核、tiling 与计算顺序](docs/design.md)
- [完整 VDN 前向](../vdn_forward/README.md)
