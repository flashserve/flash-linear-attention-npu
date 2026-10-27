# ChunkKdaBwdPrepare

BF16、C=64、K=V=128 的 KDA backward prepare 算子。
输入支持 BNSD、BSND、NTD、TND，三路输出固定为 head-major：FP32 dAqk、BF16 dv、FP32 dq_raw。

- [公开接口及数学定义](docs/api.md)
- [当前实现及资源](docs/design.md)
- [ATK 用例及验收结果](../../../../../tests/atk/chunk_kda_bwd_prepare_a3/README.md)

arch22 实现采用 AIC Cube 与 AIV Vector 的 GM 中转；arch35 使用原 A5 L0C→UB
路径。统一 ACLNN/L0 接口使用显式 layout 参数；A5 暂只开放定长 BNSD 和变长 NTD。
A2/A3/950 均保留注册；当前整合版本已通过 A2 prepare 单算子验证；总 backward 与 A5 设备验证尚未完成，详见 [开发记录](docs/validation.md)。
