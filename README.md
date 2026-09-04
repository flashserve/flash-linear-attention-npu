# flash-linear-attention-npu

> 面向昇腾（Ascend）NPU 的高性能线性注意力算子库，对标 Flash-Linear-Attention。本文档只给最小上手路径，完整说明见[开发者指南](docs/开发者指南.md)。

## 最新动态

- [2026/09] **torch_npu 解耦**：默认 wheel 通过 `fla_npu.ops.ascendc` ctypes 直调 aclnn/opapi，不再依赖 torch_npu 注册与代码生成；新增 **KDA 正反向**（`recurrent_kda` / `chunk_kda_fwd` / `chunk_kda_bwd_intra`）与 **GDN 融合算子**（fused forward / backward finalize）。
- [2026/06] 发布 v26.6.0 预编译 wheel，覆盖 A2 / A3 / A5：[Release v26.6.0](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.6.0)。
- [2026/03] 项目首次上线。

## 简介

本仓提供 AscendC / Triton 双后端线性注意力算子，算子按芯片在运行时自动选择，Python 稳定入口为 `fla_npu.ops.ascendc`（`torch.ops.npu.*` 仅支持到 v26.6.0，迁移见[兼容与迁移指南](docs/兼容与迁移指南.md)）。

> 本仓不自动安装 `torch` / `torch_npu` / `triton-ascend`，需与 CANN、Python 版本匹配后自行安装。

## 快速开始

先确认机器芯片：`npu-smi info`。A2→`ascend910b`，A3→`ascend910_93`，A5→`ascend950`。

### 1. 安装 CANN（≥ 8.5.2）

安装 **toolkit** 与对应机型 **ops** 两个安装包：

```text
Ascend-cann-toolkit_<version>_linux-<arch>.run
Ascend-cann-<chip>-ops_<version>_linux-<arch>.run
```

下载与安装方式见 [CANN 社区下载页](https://www.hiascend.com/zh/cann/download?versionId=752&ids=d803%2Ch0501%2Ch0601%2Ch0703)。每个新 shell（含 Docker/Conda/venv）先执行：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2. 编译并安装 wheel

先安装与 CANN/Python 匹配的 `torch` / `torch_npu` / `triton-ascend`（版本组合与安装指引见[开发者指南](docs/开发者指南.md)），然后在仓库根目录：

```sh
python -m pip install -r requirements.txt
python scripts/check_npu_env.py          # 无 NPU 的纯构建环境可加 --build-only
FLA_NPU_SOC=ascend910b python scripts/build_wheel.py
python -m pip install --force-reinstall --no-cache-dir --no-deps dist/<wheel文件名>.whl
```

常用构建参数（其余环境变量见[开发者指南](docs/开发者指南.md)）：

- `FLA_NPU_SOC`：目标芯片，如 `ascend910b` / `ascend910_93` / `ascend950`
- `FLA_NPU_OPS`：只构建指定算子，如 `chunk_fwd_o,chunk_bwd_dv_local`

也可直接使用 [Release v26.6.0](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.6.0) 官方 wheel，安装命令同上（替换 wheel 路径）。

### 3. 验证与测试

```sh
python -c "import fla_npu; print('ok')"
python -c "from fla_npu.ops import ascendc; print(hasattr(ascendc, 'chunk_fwd_o'))"
```

端到端示例：

```sh
python examples/flash_gated_delta_rule.py
```

单算子测试使用 [ATK](tests/atk/README.md)：

```sh
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=0
```

## 文档

- [开发者指南](docs/开发者指南.md)：单独编译算子、新增算子、环境变量、工具链依赖等
- [离线编译与使用指南](docs/离线编译与使用指南.md)：离线安装与离线二次编译
- [兼容与迁移指南](docs/兼容与迁移指南.md)：v26.6.0 及更早版本迁移
- 维护：NPU CI 部署见 [Fla-npu仓CI部署教程](docs/Fla-npu仓CI部署教程.md)
- [贡献指南](CONTRIBUTING.md) · [安全声明](SECURITY.md) · [许可证](LICENSE) · [NOTICE](NOTICE)

## 致谢

部分实现参考 [ops-transformer](https://gitcode.com/cann/ops-transformer)，感谢华为 CANN 社区及相关团队。