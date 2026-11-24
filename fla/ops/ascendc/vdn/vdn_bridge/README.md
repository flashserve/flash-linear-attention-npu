# VdnBridge

VDN 双向状态的 alpha bridge。每个输出行使用当前帧局部窗口之前的 prefix 和窗口之后的 suffix，分别按 FP32 对数前缀和计算逐通道衰减，最后相加。

- [接口和张量约束](docs/api.md)
- [核内实现与 tiling](docs/design.md)
