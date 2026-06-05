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
bash build.sh --soc=ascend910_93 --pkg --vendor_name=fla_npu

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

环境准备：除本仓根目录 `requirements.txt` 外，Example ST 还依赖 Ascend PyTorch `v26.1.0-beta.1` release family 对应的 PyTorch / torch-npu / torchnpugen、[triton-ascend](https://gitcode.com/Ascend/triton-ascend) 和 `pybind11`。`v26.1.0-beta.1` 是必须的 Ascend PyTorch release 版本；PyTorch 小版本可以按环境选择，但必须安装同一 release family 下匹配的 `torch_npu` wheel。对应 wheel 已包含 `torchnpugen`，并修复了 GDN 算子自定义适配中 `aclnn_extension` 未传 stream 导致算子间数据同步不生效的问题；不要再拉取 `op-plugin` 仓库重新编译。`triton-ascend` 会提供 `triton` Python 模块；3.2.0 及以前不要和社区版 `triton` 共存，否则可能触发 `torch_npu` 的 `triton` namespace 重复注册。

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

## 触发 NPU CI

NPU CI 支持两种手动触发方式。

方式一：在 GitHub `Actions` 页面点击 `NPU CI`，再点击 `Run workflow`，填写：

- `pr_number`: 需要验证的 PR 编号
- `ci_mode`: `quick` 或 `full`
- `ops`: 可选，逗号分隔的算子列表
- `example_args`: 可选，传给 `examples/flash_gated_delta_rule.py` 的参数；不填时保持用例原始 shape，仅由 CI 覆盖容器内逻辑设备号 `--device`

方式二：在 PR 评论区发送命令。只有维护账号可以触发：

```text
/run-npu-ci
/run-npu-ci quick
/run-npu-ci full
/run-npu-ci quick ops=causal_conv1d,chunk_bwd_dv_local
```

无论通过按钮还是评论触发，NPU CI 都会执行 `examples/flash_gated_delta_rule.py`，默认保持该用例原始 shape。后续增加已批准的泛化场景时，可在 self-hosted runner 环境中通过 `CI_EXAMPLE_CASES` 扩展多组 Example ST。

如果当前 PR head commit 已经有成功的 `NPU CI / manual` 状态，重复点击按钮或重复评论不会再次启动 NPU CI。

### 注册 self-hosted runner

完整部署教程见 [`docs/npu-ci-deployment-guide.md`](docs/npu-ci-deployment-guide.md)。

仓库管理员需要先在 GitHub 仓库设置里生成 self-hosted runner 注册 token，然后在目标服务器执行：

```sh
bash ci/setup_self_hosted_runner.sh \
  --url https://github.com/flashserve/flash-linear-attention-npu \
  --token <registration-token>
```

runner 默认标签为 `linux,arm64,npu,flash-linear-attention-npu`，与 `.github/workflows/ci.yml` 中的 `runs-on` 保持一致。

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
