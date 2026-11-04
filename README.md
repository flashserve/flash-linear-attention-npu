# flash-linear-attention-npu

## 🔥Latest News

- [2026/11] 新增 [Overlay 增量安装](docs/Overlay增量安装指南.md)：保留完整基础包，只更新指定算子，移除增量后恢复基础实现。
- [2026/09] torch_npu 解耦；新增算子：KDA 正反向（`recurrent_kda` / `chunk_kda_fwd`）、GDN 大融合（fused forward / backward finalize）。
- [2026/06] 发布 v26.6.0 预编译 wheel，覆盖 A2 / A3 / A5 目标，可在 [Release v26.6.0](https://github.com/flashserve/flash-linear-attention-npu/releases/tag/v26.6.0) 下载。
- [2026/03] flash-linear-attention-npu 项目首次上线。

## 🚀概述

flash-linear-attention-npu 算子库由天津大学主导开发，是一个面向昇腾架构的高性能线性注意力算子库，对标 Flash-Linear-Attention 项目，旨在为昇腾平台提供高效的线性注意力计算实现。

所有 AscendC 算子统一通过 Stable ABI 调用。wheel 声明 `torch>=2.7.1`、`torch_npu>=2.7.1` 运行时依赖；源码构建不自动安装框架，使用 `--no-deps` 安装 wheel 时也需自行准备与 CANN/Python 配套的 torch、torch_npu 和 triton-ascend。Stable launcher 构建仍需要 torch 的链接库；不需要 torchnpugen 或 legacy 扩展。依赖匹配关系与检查方式见下文 Step 1 / Step 2。

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

### Step 2. 安装 wheel

#### 2.1 环境检查

以下命令在已激活的 Python 环境（conda/venv）的仓库根目录执行：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 每次进入新 shell / Docker / venv 都要重新执行

# 本仓不自动安装 torch / torch_npu / triton-ascend，需按 CANN 与 Python 版本自行安装

python -m pip install -r requirements.txt
python scripts/check_npu_env.py            # 无 NPU 的纯构建环境可加 --build-only
```

#### 2.2 安装

##### 2.2.1 预编译包安装（推荐）

完成环境检查后，推荐从[华为云内源镜像站](https://mirrors.huaweicloud.com/ascend/repos/pypi/)直接安装正式 Release，无需本地编译。以下以 **A2** 为例，A3 / A5 将包名中的 `a2` 替换为 `a3` / `a5`；pip 自动匹配 ARM / x86。以下命令使用 `--no-deps`，运行时依赖需在 2.1 中准备好。

```sh
# 以正式版 26.9.1 为例
python -m pip install --upgrade --no-deps \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/ \
  "flash-linear-attention-npu-a2==26.9.1"
```

Daily、`latest` 安装及版本命名等说明见 [Daily 和 Release 包发布与安装说明](docs/Daily和Release包发布与安装说明.md)。安装完成后进入 **Step 3 验证与测试**。

##### 2.2.2 手工编译安装

如需修改源码或定制编译算子，可按以下步骤编译并安装。

先编译 wheel：

```sh
FLA_NPU_SOC=ascend910b python scripts/build_wheel.py            # A2；A3→ascend910_93，A5→ascend950

# 可选：只构建指定算子；其余环境变量见开发者指南
FLA_NPU_OPS=chunk_fwd_o,chunk_bwd_dv_local FLA_NPU_SOC=ascend910b python scripts/build_wheel.py
```

编译完成后安装本地产物：

```sh
python -m pip install --force-reinstall --no-cache-dir --no-deps dist/<wheel文件名>.whl
```

需要单独编译一个或多个算子 run 包的开发者场景见[开发者指南](docs/开发者指南.md) 场景 1。

### 可选：只更新部分算子（Overlay）

**已经装好完整包，只想更新其中一个算子？** Overlay（增量叠加层）会把更新单独安装：例如完整包有 A、B 两个算子，增量包只更新 A，运行时 A 使用新实现，B 继续使用完整包，原有 Python 调用方式不变。

下面以 **A2 上更新 `chunk_fwd_o`** 为例，演示从准备基础包到撤销增量的完整流程。先完成 Step 0 / 1 和 2.1 的环境准备，在包含 overlay 功能的当前源码根目录、同一个 Bash 终端和 Python 环境中依次执行。每一步成功后再继续；安装或移除前退出相关 Python 程序、Notebook 内核和测试 worker。

**① 安装完整基础包（base）**

已经安装支持 overlay 的完整 wheel，可跳过编译安装，直接执行最后两条检查命令。首次体验可从当前源码构建；不要将上面示例中的旧 `26.9.1` 正式包直接当作支持 overlay 的 base。

```sh
SOC=ascend910b  # A3 改为 ascend910_93，A5 改为 ascend950；后续使用相同值
unset FLA_NPU_OPS  # 基础包需要全部算子
BASE_WHEEL_DIR="$(mktemp -d /tmp/fla-npu-base.XXXXXX)"
FLA_NPU_SOC="$SOC" FLA_NPU_BUILD_ARGS=-j32 \
  python scripts/build_wheel.py --wheel-dir "$BASE_WHEEL_DIR"
python -m pip install --force-reinstall --no-cache-dir --no-deps "$BASE_WHEEL_DIR"/*.whl
python -m fla_npu_opp_manage --help
python -m fla_npu_opp_manage --status
```

应能看到 `base` 条目。基础包只需安装一次，后续修改算子时从第 ② 步开始。

**② 修改算子，编译增量包**

修改 `fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/` 下的实现，保持参数和返回值等公开接口不变，再只编译该算子：

```sh
SOC=ascend910b  # 与基础包一致；A3 / A5 按第 ① 步替换
bash build.sh --pkg --soc="$SOC" --ops=chunk_fwd_o -j32
OVERLAY_RUN="$PWD/build_out/fla_npu_linux-$(uname -m).run"
test -f "$OVERLAY_RUN"
```

如果已从开发者处拿到与 base 匹配的增量 `.run` 包，可跳过源码修改和编译，改为设置 `OVERLAY_RUN=/实际路径/增量包.run`。未修改源码也可演练安装流程，但算子结果不会因此改变。

**③ 安装为 overlay，保留 base**

```sh
STAGING_DIR="$(mktemp -d /tmp/fla-npu-overlay.XXXXXX)"
bash "$OVERLAY_RUN" --install-path="$STAGING_DIR" --quiet
unset FLA_NPU_OVERLAY_VENDOR FLA_NPU_DISABLE_PTH  # 使用默认发现和启动配置
python -m fla_npu_opp_manage --overlay --source "$STAGING_DIR/vendors"
python -m fla_npu_opp_manage --status
```

应看到 base、overlay 两类条目，overlay 的 `ops=` 包含 `chunk_fwd_o`，并显示 `ABI: OK`。不要用单算子 `.run --install` 代替上述命令，那会进入 base 覆盖流程。已有同名 overlay 时，确认替换后在 `--overlay --source ...` 命令末尾追加 `--force`。

**④ 启动新进程，验证更新**

```sh
python -c "import os; print(os.environ.get('ASCEND_CUSTOM_OPP_PATH', ''))"
python -c "import fla_npu; print('fla_npu import ok')"
```

路径中应先出现 `fla_npu_overlay_transformer`，再出现 `fla_npu_transformer`。随后重新启动原有程序，或按下面的 Step 3 运行算子测试：`chunk_fwd_o` 使用增量实现，未覆盖的算子继续使用 base。状态和导入检查不替代数值测试，已运行的进程也不会自动切换到新实现。

**⑤ 不再需要增量时，移除并恢复 base**

先退出相关进程，再执行：

```sh
python -m fla_npu_opp_manage --remove-overlay
python -m fla_npu_opp_manage --status
```

默认 overlay 条目消失，base 保留；再次启动程序即可使用基础实现，无需重装完整包。

上述流程适用于已有算子且公开接口不变的更新。替换、临时禁用、兼容边界与故障处理见 **[Overlay 增量安装指南](docs/Overlay增量安装指南.md)**。

### Step 3. 验证与测试

```sh
python -c "import fla_npu; print('ok')"
python -c "from fla_npu.ops import ascendc; print(hasattr(ascendc, 'chunk_fwd_o'))"
python scripts/check_packaged_wheel_api.py
```

单算子测试（ATK 安装见 [Ascend/ATK](https://gitcode.com/Ascend/ATK)，用法见 [ATK 说明](tests/atk/README.md)）：

```sh
bash tests/atk/run_test_cpu.sh -op=<算子名> -npu_device_id=0
```

`-op` 使用 `tests/atk` 下的测试工程目录名；算子工程名及 ATK 使用说明见 [算子概览](docs/op_list.md#atk-op-options)。

## 开发者指引

开发者相关操作（单独编译单算子、一键编包、增加新算子、确认 wheel 来自最新源码）按场景拆分为独立文档；测试单算子和端到端验证见上文 Step 3：

- [开发者指南](docs/开发者指南.md)
- [Overlay 增量安装指南](docs/Overlay增量安装指南.md)（准备基础包、安装 / 替换 / 移除增量、确认生效）
- [在线 / 离线使用与编译指南](docs/离线编译与使用指南.md)（直接使用 wheel、在线编译后离线二次编译、全离线编译）

旧版本（v26.6.0 及更早）用户升级与兼容迁移见[兼容与迁移指南](docs/兼容与迁移指南.md)。

## 维护文档

- 算子介绍、API 索引及弃用备注见 [算子概览](docs/op_list.md)。
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
