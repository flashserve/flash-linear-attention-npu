# ChunkKdaBwdPrepare

BF16、C=64、K=V=128 的 KDA backward prepare 算子。
输入支持 BNSD、BSND、NTD、TND，三路输出固定为 head-major：FP32 dAqk、BF16 dv、FP32 dq_raw。

- [公开接口及数学定义](docs/api.md)
- [当前实现及资源](docs/design.md)
- [ATK 用例及验收状态](../../../../../tests/atk/chunk_kda_bwd_prepare/README.md)

## 支持范围

arch22 实现采用 AIC Cube 与 AIV Vector 的 GM 中转；arch35 使用原 A5 L0C→UB
路径。统一 ACLNN/L0 接口使用显式 layout 参数；A5 暂只开放定长 BNSD 和变长 NTD。
A6（ascend960dt / DAV_9201）新增独立 arch92 路径：1 AIC + 1 AIV、一个
`(chunk, head)` 一个任务，L1 四份输入、UB 四组结果，三路结果均经 L0C→UB→GM。
A6 支持定长 BNSD、变长 NTD 和两种状态方向，用户 workspace 为 0。
A2/A3/950 注册保留；A6 编译及设备验收尚未完成，详见 [开发记录](docs/validation.md)。
