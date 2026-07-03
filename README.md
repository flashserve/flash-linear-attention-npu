# flash-linear-attention-npu

## 🔥Latest News

- [2026/03] flash-linear-attention-npu 项目首次上线。

## 🚀概述

flash-linear-attention-npu 算子库由天津大学主导开发，是一个面向昇腾架构的高性能线性注意力算子库，对标 Flash-Linear-Attention 项目，旨在为昇腾平台提供高效的线性注意力计算实现。

## ⚡️快速上手

### ​CANN 开发环境部署

首先需安装 CANN 开发包，提供 NPU 算子运行所需的底层驱动与工具链。
推荐使用是社区版8.5.2，总共要下2个run包，这里以A3机器为例（即需要下载A3-ops、toolkit）
下载地址为
[https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.2](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.2)
需要找到与你当前机器对应的包

```
#设置需要安装的路径
export INSTALL_PATH=/usr/local/Ascend

./Ascend-cann-toolkit*run --install-path=$INSTALL_PATH --full  --quiet
./Ascend-cann-A3*run --install-path=$INSTALL_PATH --install --quiet
source $INSTALL_PATH/ascend-toolkit/set_env.sh
```

### 编译自定义算子包

编译GDN算子run包并安装

```
# 编译命令，注意 --soc=${soc_version} 需指定为当前机器芯片类型 {ascend910b/ascend910_93/ascend950}
bash build.sh --soc=ascend910b --pkg --vendor_name=fla_npu

# 安装 run 包（custom 包名：fla-npu-<vendor>_linux-<arch>.run）
./build_out/fla-npu-*.run
```

### ​torch_custom 框架编译构建

下载并安装对应python和torch版本的最新发行版[Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)


编译torch适配whl包并安装
```sh
cd torch_custom/fla_npu
bash build.sh  # 一键编译安装脚本，先调用torchnpugen自动接入算子，再运行setup编whl包，最后安装whl包
```

如果仍采用“先编译并安装 run 包，再单独编译 torch_custom”的老流程，请在运行 Python 前 source custom OPP 的 `set_env.bash`，或设置 `FLA_NPU_OPP_PATH` 指向 OPP root / vendor 目录。`import fla_npu` 会优先使用 wheel 内嵌 OPP，找不到时会继续从 `FLA_NPU_OPP_PATH`、`ASCEND_CUSTOM_OPP_PATH` 和 `ASCEND_OPP_PATH` 查找已安装 OPP。

### 源码一键编译并生成 wheel

在已完成 CANN、PyTorch、torch-npu、triton-ascend 环境准备后，可以在仓库根目录执行源码态一键安装。默认目标芯片为 `ascend910b`，A3/A5 机器需要显式指定 `FLA_NPU_SOC`：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -U setuptools wheel packaging psutil
FLA_NPU_SOC=ascend910b FLA_NPU_VENDOR_NAME=fla_npu pip install --no-build-isolation .
```

该命令会自动完成：

```text
bash build.sh --soc=${FLA_NPU_SOC} --pkg --vendor_name=${FLA_NPU_VENDOR_NAME}
./build_out/fla-npu-*.run --quiet --install-path=<wheel-staging>/fla_npu/opp
cd torch_custom/fla_npu && bash gen.sh npu_custom.yaml
cd torch_custom/fla_npu && python setup.py build_ext --inplace
将 <wheel-staging>/fla_npu/opp/vendors/<vendor>_transformer 打进 wheel
```

安装后的 wheel 内同时包含 torch 适配 `.so` 和 AscendC OPP 运行产物。用户侧只需要安装 wheel，并在 Python 中 `import fla_npu`；`fla_npu` 会自动设置内嵌 OPP 路径、预加载 `libcust_opapi.so` 并注册 `torch.ops.npu.*`，不需要手动执行 run 包或 source vendor 目录。wheel 构建时会检查 `op_api`、`op_host` 和 `binary_info_config.json` 等关键 OPP 产物，避免打出缺少运行文件的 wheel。

可用环境变量：

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `FLA_NPU_SOC` | 目标芯片 | `ascend910b` |
| `FLA_NPU_VENDOR_NAME` | custom vendor 名 | `fla_npu` |
| `FLA_NPU_OPS` | 单算子过滤，空表示全量 | 空 |
| `FLA_NPU_SKIP_RUN_BUILD` | 跳过 run 包编译 | `FALSE` |
| `FLA_NPU_SKIP_RUN_INSTALL` | 跳过将 run 包安装产物内嵌到 wheel | `FALSE` |
| `FLA_NPU_SKIP_TORCH_GEN` | 跳过 torchnpugen 代码生成 | `FALSE` |
| `FLA_NPU_OPP_PATH` | 运行时指定外部 OPP root 或 vendor 目录 | wheel 内嵌 OPP |
| `FLA_NPU_DISABLE_LOCAL_VERSION` | wheel 版本号不追加 SOC/torch/ABI 本地版本 | `FALSE` |

安装后可验证：

```sh
python -c "import fla_npu; import torch; print(hasattr(torch.ops.npu, 'npu_chunk_fwd_o'))"
python scripts/check_packaged_wheel_api.py
```

推荐调用路径：

```python
from fla_npu.ops.triton import chunk_local_cumsum, causal_conv1d_triton
from fla_npu.ops.ascendc import chunk_fwd_o, causal_conv1d
```

兼容调用路径：

```python
import torch
import torch_npu
import fla_npu

torch.ops.npu.npu_chunk_fwd_o(...)
torch_npu.ops.chunk_fwd_o(...)
torch_npu.ops.npu_chunk_fwd_o(...)
```

`fla_npu.ops.ascendc` 会为已有明确一对一关系的算子提供自动绑定入口，例如 `fast_gelu_custom -> fast_gelu_custom_backward`，以及 `causal_conv1d -> causal_conv1d_bwd` 的 prefill / `activation_mode=0` 场景；其他复杂 GDN 组合算子仍显式暴露 forward/backward 原语，便于上层 `torch.autograd.Function` 按完整计算图组合。

### 单 wheel 离线交付

如果需要给客户离线交付，只需要提供构建出来的 wheel：

```sh
python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
```

`--force-reinstall` 会整体替换 Python 包目录，因此 wheel 内的 `.so` 和内嵌 OPP 产物会同步覆盖。

高级调试场景下，可以把 wheel 内嵌 OPP 复制到外部目录：

```sh
python -m fla_npu.install_opp --install-path /path/to/custom_opp --force
export FLA_NPU_OPP_PATH=/path/to/custom_opp
```

普通客户不需要执行这一步。

本地环境检查可先执行：

```sh
python scripts/check_npu_env.py
```

### 测试单算子

```sh
# 运行测试
cd torch_custom/fla_npu/test
bash test.sh --device 0                 # 全量测试
bash test.sh --device 0 --op causal_conv1d  # 单算子测试
```


### 算子调用方式参考

使用torch.ops.npu.npu_{算子名称}()调用对应算子，具体可参考torch_custom/fla_npu/test下面的对应算子测试脚本

例如：

```python
import torch
import torch_npu
import fla_npu

torch.ops.npu.npu_chunk_bwd_dv_local(...)
```

### 接入实践

环境准备：除本仓根目录 `requirements.txt` 外，Example ST 还依赖包含 GDN `aclnn_extension` stream 修复的 Ascend PyTorch release 对应的 PyTorch / torch-npu / torchnpugen、[triton-ascend](https://gitcode.com/Ascend/triton-ascend) 和 `pybind11`。构建脚本会拒绝未包含该修复的 `torch_npu`，尤其会拒绝 `v26.0.0-pytorch2.x` 包。当前按 PyTorch release family 限制最低版本：`2.7.1.post5`、`2.8.0.post5`、`2.9.0.post3`、`2.10.0.post2`、`2.11.0rc3`、`2.12.0rc1`，或后续 `2.13.0+` release。注意 `release v26.0.0-pytorch2.7.1/2.8.0/2.9.0/2.10.0` 附带的 `torch_npu-2.7.1.post4`、`2.8.0.post4`、`2.9.0.post2`、裸 `2.10.0` 都不包含该 GDN 修复；PyTorch 2.10 请使用包含修复的 `v26.0.1-pytorch2.10.0` (`torch_npu-2.10.0.post2`) 或后续版本：[Ascend/pytorch releases](https://gitcode.com/Ascend/pytorch/releases?presetConfig={%22tags%22:229,%22release%22:122})。对应 wheel 已包含 `torchnpugen`；不要再拉取 `op-plugin` 仓库重新编译。`triton-ascend` 会提供 `triton` Python 模块；3.2.0 及以前不要和社区版 `triton` 共存，否则可能触发 `torch_npu` 的 `triton` namespace 重复注册。

```sh
# 以下示例使用 Python 3.10/aarch64 + PyTorch 2.7.1；其他 PyTorch 小版本请切换到同属 v26.1.0-beta.1 的配套 tag 和 wheel。
pip install -r requirements.txt
pip install torch==2.7.1
curl -fL --retry 3 --retry-delay 2 -o /tmp/torch_npu-2.7.1.post5-cp310-cp310-manylinux_2_28_aarch64.whl \
  https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-beta.1-pytorch2.7.1/torch_npu-2.7.1.post5-cp310-cp310-manylinux_2_28_aarch64.whl
pip install /tmp/torch_npu-2.7.1.post5-cp310-cp310-manylinux_2_28_aarch64.whl
pip uninstall -y triton
pip install triton-ascend==3.2.0
pip uninstall -y triton
pip install pybind11
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTORCH_VERSION=2.7.1
```

一键运行GDN模块，组装了所有GDN相关算子，包括前向和反向，包括AscendC和Triton算子
```sh
python examples/flash_gated_delta_rule.py
```

NPU CI 的 Example/ST 用例由 [`ci/example_st_cases.json`](ci/example_st_cases.json) 管理。当前默认启用 `case1_current_default`，shape 与上面的直接运行默认值一致；后续 GVA、`Vdim=256` 等泛化场景可以在该文件中新增用例，显式填写 `B`、`T`、`chunk_size`、`query_head`、`value_head`、`Kdim`、`Vdim` 等 shape 字段，以及 `gate_source`、`gate_function`、`initial_state`、`output_final_state`、`qk_l2norm` 等行为字段。

当前端到端 Example/ST 已支持 `gate_source=g`；`gk` / `g+gk` 先作为用例 schema 预留，待 NPU fwd_h 路径支持后再启用。

## 维护文档

NPU CI 维护说明见 [`docs/Fla-npu仓CI部署教程.md`](docs/Fla-npu仓CI部署教程.md)。

## 🔍目录结构
关键目录如下：
```
├── cmake                              # 项目工程编译目录
├── common                             # 项目公共头文件和公共源码
├── fla                                # 算子库核心包
│   └── ops
│       ├── ascendc                    # AscendC 算子实现
│       │   ├── common                 # 公共模块（GroupedMatMul 等）
│       │   └── gdn                    # GDN 算子
│       │       ├── chunk_gdn_fwd      # 前向传播算子
│       │       │   ├── chunk_fwd_o
│       │       │   ├── chunk_gated_delta_rule_fwd_h
│       │       │   └── recompute_wu_fwd
│       │       ├── chunk_gdn_bwd      # 反向传播算子
│       │       │   ├── chunk_bwd_dqkwg
│       │       │   ├── chunk_bwd_dv_local
│       │       │   ├── chunk_gated_delta_rule_bwd_dhu
│       │       │   ├── prepare_wy_repr_bwd_da
│       │       │   └── prepare_wy_repr_bwd_full
│       │       ├── gdn_preprocess     # 预处理算子
│       │       │   └── causal_conv1d
│       │       └── recurrent_gdn      # 推理算子
│       │           └── recurrent_gated_delta_rule
│       └── triton                     # Triton 算子实现
├── torch_custom                       # 自定义PyTorch算子适配
├── examples                           # 端到端算子开发和调用示例
│   └── flash_gated_delta_rule.py      # 完整GDN接入调用示例
├── scripts                            # 脚本目录，包含算子构建相关配置文件
├── tests                              # 测试工程目录
├── gdn-verify.sh                      # GDN 一键验证脚本
├── CMakeLists.txt
├── README.md
├── build.sh                           # 项目工程编译脚本
├── install_deps.sh                    # 安装依赖包脚本
├── QUICKSTART.md                      # 快速入门文档
├── CONTRIBUTING.md                    # 贡献指南
├── SECURITY.md                        # 安全声明
├── LICENSE                            # 许可证
└── requirements.txt                   # 本项目需要的第三方依赖包
```

## 📝相关信息

- [安全声明](SECURITY.md)
- [许可证](LICENSE)

## 🙏致谢

本项目的部分实现参考了 [ops-transformer](https://gitcode.com/cann/ops-transformer) 仓库，感谢华为 CANN 社区及相关开发团队的开源贡献。
