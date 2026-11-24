# MegaRecurrentKda

Atlas A2/A3 和 Ascend 950 的融合 KDA 推理算子。从 hidden_states 开始，一次调用完成输入投影、四抽头因果卷积、KDA 递推、门控归一化和输出投影。

- [接口与输入约束](docs/api.md)
- [Atlas A2/A3 模型的 INT8 QKV 与设备前缀](docs/api.md#atlas-a2a3-int8-qkv-投影)
- [模型接入指南（A2/A3 INT8 / A5 MXFP8）](docs/model_integration.md)
- [模板划分、计算与 workspace](docs/design.md)
- [CPU 标杆](../../../../../tests/reference/mega_recurrent_kda_reference.py)
- [ATK 验收](../../../../../tests/atk/mega_recurrent_kda/)
