# flash-linear-attention-npu

## 🔥Latest News

- [2026/09] torch_npu 解耦；新增算子：KDA 正反向（`recurrent_kda` / `chunk_kda_fwd` / `chunk_kda_bwd_intra`）、GDN 大融合（fused forward / backward finalize）。
- [2026/06] 发布 v26.6.0 预编译 wheel，覆盖 A2 / A3 / A5 目标，可在 [Release v26.6.0](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.6.0) 下载。
- [2026/03] flash-linear-attention-npu 项目首次上线。

## 🚀概述

flash-linear-attention-npu 算子库由天津大学主导开发，是一个面向昇腾架构的高性能线性注意力算子库，对标 Flash-Linear-Attention 项目，旨在为昇腾平台提供高效的线性注意力计算实现。

本仓不自动安装 `torch`、`torch_npu`、`torchnpugen`、`triton-ascend`，这些包必须与 CANN 与 Python 版本匹配，需要使用者按环境自行安装；版本不匹配时，构建或运行会报错。依赖匹配关系与检查方式见下文 Step 1 / Step 2。

## ⚡️快速上手

### Step 0. 确认硬件与目标芯片

在开始前，先确认机器上可用的 NPU 类型：

```sh
npu-smi info
```

确认机器类型后，按目标芯片选择后续构建参数（`--soc` / `FLA_NPU_SOC`）：

| 产品 | `--soc` / `FLA_NPU_SOC` |
| ---- | --------------------------- |
| A2   | `ascend910b`              |
| A3   | `ascend910_93`            |
| A5   | `ascend950`               |

### Step 1. 部署 CANN 开发环境

安装 toolkit 与对应机型 ops 两个包（A2/A3：CANN ≥ 8.5.2；A5：CANN ≥ 9.0.0），下载页：[CANN 社区下载页](https://www.hiascend.com/developer/download/community/result?module=cann)

- `Ascend-cann-toolkit_<version>_linux-<arch>.run`
- `Ascend-cann-<chip>-ops_<version>_linux-<arch>.run`

### Step 2. 编译并安装 wheel

以下命令在已激活的 Python 环境（conda/venv）的仓库根目录执行：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 每次进入新 shell / Docker / venv 都要重新执行

# 本仓不自动安装 torch / torch_npu / triton-ascend，需按 CANN 与 Python 版本自行安装
python -m pip install torch==2.7.1
# torch_npu 与 torch 严格配对：gitcode.com/Ascend/pytorch/releases 下载同版本 wheel
# triton-ascend>=3.2.1：--extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
# 须在 torch / torch_npu 之后安装，避免 torch 覆盖 Ascend triton 栈；
# 若 import fla_npu.ops.triton 报 triton._C.libtriton.ascend 缺失，按此顺序重装即可

python -m pip install -r requirements.txt
python scripts/check_npu_env.py            # 无 NPU 的纯构建环境可加 --build-only
FLA_NPU_SOC=ascend910b python scripts/build_wheel.py            # A2；A3→ascend910_93，A5→ascend950
python -m pip install --force-reinstall --no-cache-dir --no-deps dist/<wheel文件名>.whl

# 可选：只构建指定算子；其余环境变量见开发者指南
FLA_NPU_OPS=chunk_fwd_o,chunk_bwd_dv_local FLA_NPU_SOC=ascend910b python scripts/build_wheel.py
```

不想编译时，可直接安装 [Release v26.6.0](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.6.0) 官方 wheel，把安装命令中的 wheel 路径换成下载文件的路径。

需要单独编译一个或多个算子 run 包的开发者场景见[开发者指南](docs/开发者指南.md) 场景 1。

### Step 3. 验证与测试

```sh
python -c "import fla_npu; print('ok')"
python -c "from fla_npu.ops import ascendc; print(hasattr(ascendc, 'chunk_fwd_o'))"
python scripts/check_packaged_wheel_api.py
```

单算子测试（更多算子见 [ATK 说明](tests/atk/README.md)，ATK 安装见 [Ascend/ATK](https://gitcode.com/Ascend/ATK)）：

```sh
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=0
```

## 开发者指引

开发者相关操作（单独编译单算子、一键编包、增加新算子、确认 wheel 来自最新源码）按场景拆分为独立文档；测试单算子和端到端验证见上文 Step 3：

- [开发者指南](docs/开发者指南.md)
- [在线 / 离线使用与编译指南](docs/离线编译与使用指南.md)（直接使用 wheel、在线编译后离线二次编译、全离线编译）

旧版本（v26.6.0 及更早）用户升级与兼容迁移见[兼容与迁移指南](docs/兼容与迁移指南.md)。

## 维护文档

- NPU CI 维护说明见 [`docs/Fla-npu仓CI部署教程.md`](docs/Fla-npu仓CI部署教程.md)。
- 旧版本用户升级与兼容迁移见 [`docs/兼容与迁移指南.md`](docs/兼容与迁移指南.md)。
- 开发者分场景指南见 [`docs/开发者指南.md`](docs/开发者指南.md)。

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
│       │       │   ├── chunk_fwd_h
│       │       │   ├── chunk_fwd_o
│       │       │   ├── chunk_gated_delta_rule_fwd_h
│       │       │   └── recompute_w_u_fwd
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
├── docs                               # 文档目录（兼容迁移指南、开发者指南等）
├── tests                              # 测试工程目录
├── gdn-verify.sh                      # GDN 一键验证脚本
├── CMakeLists.txt
├── README.md
├── build.sh                           # 项目工程编译脚本
├── install_deps.sh                    # 安装依赖包脚本
├── CONTRIBUTING.md                    # 贡献指南
├── SECURITY.md                        # 安全声明
├── LICENSE                            # 仓库级许可证说明
├── LICENSES                           # 许可证全文
├── NOTICE                             # 来源与再分发说明
└── requirements.txt                   # 本项目需要的第三方依赖包
```

## 📝相关信息

- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [NOTICE](NOTICE)

## ⚖️许可证说明

本仓库包含多种许可证文件：未在文件头或更具体说明中另行标识的原创代码使用 BSD 3-Clause License；从 CANN ops-transformer 改编的代码，以及文件头标识为 CANN Open Software License Agreement Version 2.0 的代码，使用 CANN Open Software License Agreement Version 2.0。该 CANN 许可证全文见 [LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt](LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt)，来源和再分发说明见 [NOTICE](NOTICE)。若文件级许可证说明与仓库级说明不一致，以文件级说明为准。

## 🙏致谢

本项目的部分实现参考了 [ops-transformer](https://gitcode.com/cann/ops-transformer) 仓库，感谢华为 CANN 社区及相关开发团队的开源贡献。
