# flash-linear-attention-npu

## 🔥Latest News

- [2026/06] 发布 v26.6.0 预编译 wheel，覆盖 A2 / A3 / A5 目标，可在 [Release v26.6.0](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.6.0) 下载。
- [2026/03] flash-linear-attention-npu 项目首次上线。

## 🚀概述

flash-linear-attention-npu 算子库由天津大学主导开发，是一个面向昇腾架构的高性能线性注意力算子库，对标 Flash-Linear-Attention 项目，旨在为昇腾平台提供高效的线性注意力计算实现。

## ⚡️快速上手

### Step 1. 部署 CANN 开发环境

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

### Step 2. 编译

#### 方式 A：【推荐】源码一键编译并生成 wheel

在已完成 CANN、PyTorch、torch-npu、torchnpugen、triton-ascend 环境准备后，推荐直接在仓库根目录生成单 wheel。默认目标芯片为 `ascend910b`，A3/A5 机器需要显式指定 `FLA_NPU_SOC`。本仓不会自动安装 `torch`、`torch_npu`、`torchnpugen` 或 `triton-ascend`，因为这些包必须和 CANN、Python、`torch_npu` 可用版本匹配；在新的 conda 环境中请先安装匹配依赖，再执行预检：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install -r requirements.txt
python scripts/check_npu_env.py --build-only
```

如果依赖缺失，预检和一键编包都会在真正编译前失败，并列出缺失项，例如 `torch`、`torch_npu`、`torchnpugen.*`、`triton` 或 `triton-ascend distribution was not found`。依赖通过后再生成 wheel：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
FLA_NPU_SOC=ascend910b python -m pip wheel --no-build-isolation --no-deps . -w dist
```

如果已经做过一次完整编译，之后只修改少量算子源码，可以复用上一次 CMake build 目录做完整 wheel 的真增量构建：

```sh
FLA_NPU_SOC=ascend910b FLA_NPU_INCREMENTAL_BUILD=1 python -m pip wheel --no-build-isolation --no-deps . -w dist
```

增量构建仅建议用于本地反复调试。构建完成后，wheel 会输出到 `dist/` 目录，按 Step 3 安装即可。

方式 A 编译可用环境变量：

| 环境变量 | 可选范围 | 作用 / 建议 | 默认 |
|---|---|---|---|
| `FLA_NPU_SOC` | `ascend910b` / `ascend910_93` / `ascend950` | 目标芯片；按实际运行机器选择 | `ascend910b` |
| `FLA_NPU_INCREMENTAL_BUILD` | `TRUE` / `FALSE` | 复用 `build/` 做完整 wheel 的真增量构建；本地反复调试可设 `TRUE`，release wheel 或干净验证建议保持 `FALSE` | `FALSE` |
| `FLA_NPU_OPS` | 逗号分隔的算子名，如 `chunk_fwd_o,recompute_wu_fwd` | 仅构建指定算子；用于单算子定位，不要用于 release wheel | 空 |
| `FLA_NPU_SKIP_RUN_BUILD` | `TRUE` / `FALSE` | 跳过 run 包编译；仅在已准备好匹配的 `build_out/fla-npu-*.run` 且只重打 wheel 时可设 `TRUE`，常规构建建议保持 `FALSE` | `FALSE` |
| `FLA_NPU_SKIP_RUN_INSTALL` | `TRUE` / `FALSE` | 跳过将 run 包安装产物内嵌到 wheel；会得到不含内嵌 OPP 的 wheel，除非使用外部 OPP 调试，否则建议保持 `FALSE` | `FALSE` |
| `FLA_NPU_DISABLE_LOCAL_VERSION` | `TRUE` / `FALSE` | wheel 版本号不追加 SOC/torch/ABI 本地版本；内部统一发版需要固定版本号时可设 `TRUE`，日常构建建议保持 `FALSE` 以区分产物兼容范围 | `FALSE` |

布尔变量设为 `TRUE` 时也接受 `1`、`YES`、`ON`；未设置或其他值按 `FALSE` 处理。

#### 方式 B：【备选】单独编译算子 run 包和 Python wheel

只有在已经安装方式 A 的完整 wheel、但需要快速替换少量算子的 Ascend C 产物时，才建议使用该方式。`--ops=op1,op2,...` 只会生成指定算子的 run 包；run 包安装时会把当前 run 包里的 `packages/vendors/fla_npu_transformer` 合并覆盖到当前 Python 环境已安装的 `site-packages/fla_npu/opp/vendors/fla_npu_transformer`，从而更新 `aclnn`、tiling、kernel 和相关配置。

```sh
# 编译一个或多个算子 run 包，--soc 需指定为当前机器芯片类型 {ascend910b/ascend910_93/ascend950}
bash build.sh --soc=ascend910b --pkg --vendor_name=fla_npu --ops=chunk_fwd_o

# 如果 Python wrapper 也有修改，再单独编译 Python runtime wheel
cd torch_custom/fla_npu
python3 setup.py bdist_wheel

# 如需继续验证旧 torch.ops.npu 路径，可显式编译 legacy PyTorch C++ extension
FLA_NPU_BUILD_LEGACY_EXTENSION=1 bash gen.sh npu_custom.yaml
FLA_NPU_BUILD_LEGACY_EXTENSION=1 python3 setup.py bdist_wheel
```

### Step 3. 安装

#### 方式 A 产物安装

方式 A 产物可以来自本地源码一键编译，也可以直接使用 [Release v26.6.0](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.6.0) 提供的官方验证 wheel。下载或构建完成后执行：

```sh
python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
```

如果使用 Release 下载的 wheel，将命令中的 `dist/flash_linear_attention_npu-*.whl` 替换为实际下载路径。

#### 方式 B 产物安装

先确认方式 A 的完整 wheel 已经安装到当前 Python 环境，然后安装 run 包。安装器会在覆盖前列出当前 run 包携带的算子，并标出安装后的算子状态：`WARNING` 表示安装后不可用，包括不在当前 run 包范围内但会受局部 `libcust_opapi.so`、tiling so、proto so 整体替换影响的算子，以及当前 run 包内但 aclnn ABI 修改或删除的算子；`NOTICE` 表示新增或无法完整确认的 ABI，需要确认当前 Python wheel 是否已有对应 wrapper；`OK` 表示当前 run 包内且 aclnn ABI 一致的算子。`op_api/include/aclnnop` 中新增、删除、修改的 aclnn ABI 头文件会合并显示到对应算子的状态原因里；删除只按当前 run 包携带的算子范围判断，非 `--quiet` 模式只在状态表后确认一次。

```sh
# 覆盖当前 Python 环境中 flash-linear-attention-npu wheel 内嵌的 OPP
./build_out/fla-npu-*.run --install
# 或等价写法
./build_out/fla-npu-*.run --full

# 如果 Python wrapper 也有修改，再安装单独编译出的 wheel
python -m pip install --force-reinstall --no-deps torch_custom/fla_npu/dist/flash_linear_attention_npu-*.whl
```

安装 run 包后需要重启 Python 进程，已经 `dlopen` 的 `libcust_opapi.so` 不会在同一进程内热替换。

`import fla_npu` 是轻量导入，不会自动导入 `torch` / `torch_npu`，也不会自动注册 `torch.ops.npu`。默认 wheel 通过 Python ctypes 直调 aclnn/opapi，推荐使用 `fla_npu.ops.ascendc`；`torch_npu.ops.*` 会在导入 `fla_npu.ops.ascendc` 后挂到同一套 Python wrapper。只有用 `FLA_NPU_BUILD_LEGACY_EXTENSION=1` 额外编出 legacy 扩展时，才可显式调用 `fla_npu.load_legacy_torch_ops()` 兼容旧 `torch.ops.npu.*`。

### Step 4. 测试安装成功

安装后两种方式均可用以下命令验证：

```sh
python -c "import fla_npu; print(fla_npu.is_legacy_torch_ops_loaded())"
python -c "from fla_npu.ops import ascendc; import torch_npu; print(hasattr(torch_npu.ops, 'chunk_fwd_o'))"
python scripts/check_packaged_wheel_api.py
```

`torch.ops.npu.*` 是 legacy extension 的过渡用法，后续版本不再支持。新代码优先使用 `fla_npu.ops.ascendc` 下的稳定 Python 入口。

### 测试单算子

```sh
# 运行测试
cd torch_custom/fla_npu/test
bash test.sh --device 0                      # 全量测试
bash test.sh --device 0 --op causal_conv1d   # 单个 AscendC 测试任务
```

`--op` 当前仅覆盖 `test.sh` 已接入的 AscendC 测试任务，可选值：

- `prepare_wy_repr_bwd_full`
- `chunk_gated_delta_rule_bwd_dhu`
- `chunk_bwd_dv_local`
- `causal_conv1d`
- `prepare_wy_repr_bwd_da`
- `chunk_bwd_dqkwg`
- `gdn_fwd_o`
- `gdn_fwd_h`
- `recompute_wu_fwd`


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
