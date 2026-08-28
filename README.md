# flash-linear-attention-npu

## 🔥Latest News

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

首先需安装 CANN 开发包，提供 NPU 算子运行所需的底层驱动与工具链。
推荐使用最新的社区稳定版本（不低于 8.5.2，如需使用更新版本请参考 `check_npu_env.py` 支持的 CANN / torch_npu 版本组合），总共需要下载 2 个 run 包。
下载地址为社区 CANN 下载页（最新稳定版）
[https://www.hiascend.com/zh/cann/download?versionId=752&ids=d803%2Ch0501%2Ch0601%2Ch0703](https://www.hiascend.com/zh/cann/download?versionId=752&ids=d803%2Ch0501%2Ch0601%2Ch0703)
在其中找到与你当前机器对应的包

```
# 设置需要安装的路径（请替换为实际安装路径）
export INSTALL_PATH=/usr/local/Ascend

# toolkit 与机型对应的 ops 包都必须安装；ops 包命名格式为
# Ascend-cann-<chip_type>-ops_<version>_linux-<arch>.run，
# 例如 A3 机器对应 Ascend-cann-A3-ops*.run，A2 机器对应 Ascend-cann-910b-ops*.run，A5 机器对应 Ascend-cann-950-ops*.run
./Ascend-cann-toolkit*run --install-path=$INSTALL_PATH --full  --quiet
./Ascend-cann-*-ops*.run --install-path=$INSTALL_PATH --install --quiet
source $INSTALL_PATH/ascend-toolkit/set_env.sh
```

> 若 CANN 安装在自定义路径，请将 `INSTALL_PATH` 设置为实际安装路径，并 source 实际路径下对应的 `set_env.sh`（上述 `/usr/local/Ascend` 仅为默认安装路径）。每次进入新的 shell（含 Docker / Conda / venv）后，都需要重新 source `set_env.sh` 才能正常编译与运行。

### Step 2. 编译

#### 【推荐】源码一键编译并生成 wheel

在已完成 CANN、PyTorch、torch-npu、torchnpugen、triton-ascend 环境准备后，推荐直接在仓库根目录生成单 wheel。默认目标芯片为 `ascend910b`，A3/A5 机器需要显式指定 `FLA_NPU_SOC`。本仓不会自动安装 `torch`、`torch_npu`、`torchnpugen` 或 `triton-ascend`，因为这些包必须和 CANN、Python、`torch_npu` 可用版本匹配；在新的 conda 环境中请先安装匹配依赖，再执行预检：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install -r requirements.txt
python scripts/check_npu_env.py
```

完整预检会同时检测运行与编译环境，检查 `torch` / `torch_npu` / `triton-ascend` 是否可导入、版本下限与 NPU 可用性（`torch.npu.is_available()`）；torch 系依赖缺失或版本不匹配时，`pip wheel` 会在构建或打包阶段报错，请按 CANN 与 Python 版本匹配的列表先行安装。在无 NPU 卡的纯构建环境里，`is_available()` 为 `False` 且该检查会报 `FAIL`，属预期现象——此时若只需产出 wheel 构建，可用 `--build-only` 跳过 torch 系检查：

```sh
python scripts/check_npu_env.py --build-only
```

预检覆盖编译链上的 `cmake`、`gcc`/`g++`、`setuptools` 版本要求，`make` / `patch` / `bisheng` 存在性检查，以及 `wheel` / `packaging` / `psutil`（`--no-build-isolation` 构建时需本机已装）的导入检查。其余组件未纳入预检，缺失时会在 `pip wheel` 阶段才报错。各组件的最低版本要求与详细说明见[开发者指南](docs/开发者指南.md) 场景 2 的工具链依赖表。

> `triton-ascend` 与 CANN 版本需要匹配：CANN 8.x 使用 `>=3.2.0` 即可；CANN 9.x（9.0.0+）因 Ascend Triton 后端 JIT 编译 `npu_utils.cpp` 依赖更新的 `rt.h` 头文件，需要 **`>=3.2.1`**（3.2.0 在 CANN 9.1.0 上会编译失败）。预检会按检测到的 CANN 版本自动校验 `triton-ascend` 是否满足对应下限。

预检通过后再生成 wheel：

```sh
FLA_NPU_SOC=ascend910b python scripts/build_wheel.py
```

脚本内部仍使用 `pip wheel --no-build-isolation --no-deps` 完成构建，并在成功后打印本轮
wheel 的绝对路径和可直接复制执行的强制覆盖安装命令。可通过 `--wheel-dir` 修改默认的
`dist/` 输出目录。

修改任何源码或适配后，重新执行同一条命令做全量构建。构建流程会清理上一轮
`build/`、`build_out/` 和 `output/` 中间产物，不依赖 Git diff 或旧 CMake 状态决定
编译范围：

```sh
FLA_NPU_SOC=ascend910b python scripts/build_wheel.py
```

构建完成后，wheel 仍统一输出到 `dist/`。该目录可能同时存在不同版本或构建标签
的 wheel，因此安装时必须传入本轮构建生成的准确文件名，并使用 Step 3 的强制覆盖
命令，避免通配符选中旧产物。

编译可用环境变量：

| 环境变量                          | 可选范围                                          | 作用 / 建议                                                                                                                        | 默认           |
| --------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------- |
| `FLA_NPU_SOC`                   | `ascend910b` / `ascend910_93` / `ascend950` | 目标芯片；按实际运行机器选择                                                                                                       | `ascend910b` |
| `FLA_NPU_OPS`                   | 算子名，逗号分隔（如 `chunk_fwd_o,chunk_bwd_dv_local`） | 只构建指定算子的 wheel；适合已安装完整 wheel 后快速替换少量算子的 Ascend C 产物，未设置则全量构建 | 空（全量） |
| `FLA_NPU_DISABLE_LOCAL_VERSION` | `TRUE` / `FALSE`                              | wheel 版本号不追加 SOC/torch/ABI 本地版本；内部统一发版需要固定版本号时可设`TRUE`，日常构建建议保持 `FALSE` 以区分产物兼容范围 | `FALSE`      |

布尔变量设为 `TRUE` 时也接受 `1`、`YES`、`ON`；未设置或其他值按 `FALSE` 处理。

> 需要单独编译一个或多个算子 run 包的开发者场景（如已安装完整 wheel 后快速替换少量算子的
> Ascend C 产物），见[开发者指南](docs/开发者指南.md) 场景 1。

### Step 3. 安装

产物可以来自本地源码一键编译，也可以直接使用 [Release v26.6.0](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.6.0) 提供的官方验证 wheel。下载或构建完成后执行：

```sh
# 将 WHEEL_PATH 设置为实际 wheel 文件路径：
# 本地构建产物位于 dist/ 目录，请使用构建日志输出的准确文件名（勿用通配符，避免匹配多个产物）；
# Release 下载的 wheel 则填实际下载路径。
WHEEL_PATH="dist/<准确wheel文件名>.whl"
python -m pip install --force-reinstall --no-cache-dir --no-deps "$WHEEL_PATH"
```

> 重新构建的 wheel 版本号与已安装的旧 wheel 可能相同。版本号相同时，不带 `--force-reinstall` 的 `pip install` 会认为"已是最新版本"而跳过，导致实际仍是旧代码。上面的命令已带 `--force-reinstall` 强制覆盖；若想先清理再装，可先执行 `python -m pip uninstall -y flash-linear-attention-npu`。

wheel 不安装或执行 shell 环境钩子。无论使用系统 Python、Conda、venv
还是 Docker，每次进入新的 shell 后都需要先按 Step 1 手工 source CANN 的
`set_env.sh`。调用 `fla_npu.ops.ascendc` 算子时会在当前 Python 进程内定位并加载
wheel 内嵌 OPP；wheel 通过绝对路径加载 `libcust_opapi.so`，不会再生成或加载
可能覆盖 CANN 运行库的自定义 `libopapi.so`。如果旧版 run 包曾在 wheel 中遗留该别名，
新 runtime 会在首次加载 OPP 时删除它；目录不可写时会给出明确的手工清理提示。

`import fla_npu` 会定位 OPP 并加载 `libcust_opapi.so`。执行前必须先 source CANN
的 `set_env.sh`；CANN 环境未初始化、OPP 不完整或动态库加载失败时，import 会直接
报错。该过程不会自动导入 `torch` / `torch_npu`，也不会注册 `torch.ops.npu`。
默认 wheel 通过 Python ctypes 直调 aclnn/opapi，推荐使用 `fla_npu.ops.ascendc`；
只有用 `FLA_NPU_BUILD_LEGACY_EXTENSION=1` 额外编出 legacy 扩展时，才可显式调用
`fla_npu.load_legacy_torch_ops()` 兼容旧 `torch.ops.npu.*`。

`fla_npu.ops.ascendc` 只使用当前 wheel 内嵌的 custom OPP，不从 `FLA_NPU_OPP_PATH`、`ASCEND_CUSTOM_OPP_PATH`、`ASCEND_OPP_PATH` 或 CANN 的 `vendors` 目录回退查找其他 `libcust_opapi.so`。外部 OPP 仅用于 CANN 侧的 host、tiling 与 kernel 发现，不再作为 Python runtime 加载 `libcust_opapi.so` 的来源。单独构建的 run 包应使用默认 `--install` / `--full` 流程覆盖当前 wheel 内的 OPP（见[开发者指南](docs/开发者指南.md) 场景 1）。

> 已安装完整 wheel 后，如需用单算子 run 包快速替换部分算子的 Ascend C 产物（含安装器
> 的算子状态说明），见[开发者指南](docs/开发者指南.md) 场景 1。

### Step 4. 测试安装成功

安装后可用以下命令验证：

```sh
python -c "import fla_npu; print('ok')"
python -c "from fla_npu.ops import ascendc; print(hasattr(ascendc, 'chunk_fwd_o'))"
python scripts/check_packaged_wheel_api.py
```

`import fla_npu` 成功即表示 wheel 与内嵌 OPP 加载正常。推荐使用 `fla_npu.ops.ascendc` 稳定 Python 入口调用算子。

`torch.ops.npu.*` / `torch_npu.ops.*` 是旧版本（v26.6.0 及更早）的调用方式，**v26.6.0 之后不再维护旧版本兼容接口**，新代码请使用 `fla_npu.ops.ascendc` 下的稳定 Python 入口。迁移期如需临时兼容（`install_torch_npu_ops_compat()` / `load_legacy_torch_ops()`）及其注意事项（如 `hasattr(torch_npu.ops, ...)` 的版本差异），见[兼容与迁移指南](docs/兼容与迁移指南.md)。

不再使用时，按 distribution 名卸载：

```sh
python -m pip uninstall -y flash-linear-attention-npu
```

### 测试单算子

单算子看护统一使用 `tests/atk` 下的 ATK 工程。每个算子目录包含
`atk_<op>.json`、`<op>.yaml`、`gen_<op>.py`、`executor_<op>.py` 和本算子
`README.md`。各算子的输入 shape、dtype、可选输入和 tiling 限制以对应算子 README 为准。

运行前先加载 ATK、CANN 和当前安装的 OPP/Python 环境：

```sh
source <cann_install_path>/set_env.sh
atk --version
npu-smi info
```

一键执行某个算子的完整 ATK 动作：

```sh
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=0
```

当前 `-op` 可选值为：

- `causal_conv1d`
- `causal_conv1d_bwd`
- `chunk_bwd_dqkwg`
- `chunk_bwd_dv_local`
- `chunk_fwd_o`
- `chunk_gated_delta_rule_bwd_dhu`
- `chunk_gated_delta_rule_fwd_h`
- `chunk_kda_fwd`
- `chunk_local_cumsum`
- `chunk_scaled_dot_kkt`
- `prepare_wy_repr_bwd`
- `prepare_wy_repr_bwd_da`
- `prepare_wy_repr_bwd_full`
- `recompute_w_u_fwd`
- `recurrent_gated_delta_rule`
- `recurrent_kda`
- `solve_tri`

`run_test_cpu.sh` 支持以下 scope：

```sh
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=0 -scope=mssanitizer
```

默认 `-scope=all` 会执行混合容差精度、确定性和 mssanitizer；性能测试需显式指定
`-scope=performance`。精度任务以 CPU 高精度结果作为唯一 golden，以 NPU 输出作为 DUT。
未设置 `CASE_START/CASE_END` 时不向 ATK 传入 `-s/-e`，会执行 JSON 中的全部用例；
需要只跑指定顺序范围时使用：

```sh
CASE_START=0 CASE_END=1 \
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=0
```

ATK 工程结构、支持算子索引、环境变量和新增算子规范见
[`tests/atk/README.md`](tests/atk/README.md)。

### 算子调用方式参考

推荐通过 `fla_npu.ops.ascendc` 或 `fla_npu.ops.triton` 导入对应算子；具体入参可参考 `torch_custom/fla_npu/test` 下的对应算子测试脚本。

例如：

```python
import torch
import fla_npu
from fla_npu.ops.ascendc import chunk_bwd_dv_local

dv = chunk_bwd_dv_local(...)
```

### 端到端 Example/ST 验证

完成安装后，可以一键运行 GDN 模块。该示例会组装 GDN 相关前向/反向算子，覆盖 AscendC 和 Triton 调用链：

```sh
python examples/flash_gated_delta_rule.py
```

NPU CI 的 Example/ST 用例由 [`ci/example_st_cases.json`](ci/example_st_cases.json) 管理。当前默认启用 `case1_current_default`，shape 与上面的直接运行默认值一致；后续 GVA、`Vdim=256` 等泛化场景可以在该文件中新增用例，显式填写 `B`、`T`、`chunk_size`、`query_head`、`value_head`、`Kdim`、`Vdim` 等 shape 字段，以及 `gate_source`、`gate_function`、`initial_state`、`output_final_state`、`qk_l2norm` 等行为字段。

当前端到端 Example/ST 已支持 `gate_source=g`；`gk` / `g+gk` 先作为用例 schema 预留，待 NPU fwd_h 路径支持后再启用。

## 开发者指引

开发者相关操作（单独编译单算子、一键编包、增加新算子、确认 wheel 来自最新源码）按场景拆分为独立文档；测试单算子和端到端验证见上文 Step 4：

- [开发者指南](docs/开发者指南.md)

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
