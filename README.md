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
# 编译命令，注意--soc=${soc_version}需要指定为当前机器的芯片类型{ascend910b/ascend910_93/ascend950}
bash build.sh --soc=ascend910_93 --pkg --ops=chunk_bwd_dv_local,chunk_bwd_dqkwg,chunk_gated_delta_rule_bwd_dhu,prepare_wy_repr_bwd_da,prepare_wy_repr_bwd_full,chunk_fwd_o,chunk_gated_delta_rule_fwd_h,recurrent_gated_delta_rule,recompute_wu_fwd

# 安装run包
./build_out/cann-*.run
```

### ​torch_custom 框架编译构建

下载并安装对应python和torch版本的最新发行版[Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)


编译torch适配whl包并安装
```sh
cd torch_custom/aclnn_extension
bash build.sh  # 一键编译安装脚本，先调用torchnpugen自动接入算子，再运行setup编whl包，最后安装whl包
```

### 测试单算子

```sh
# 运行测试
cd torch_custom/aclnn_extension/test
bash test.sh  # 包含所有GDN单算子的一键测试脚本
```


### 算子调用方式参考

使用torch.ops.npu.npu_{算子名称}()调用对应算子，具体可参考torch_custom/aclnn_extension/test下面的对应算子测试脚本

例如：

```python
import torch
import torch_npu
import aclnn_extension

torch.ops.npu.npu_chunk_bwd_dv_local(...)
```

### 接入实践

环境准备：[triton-ascend](https://gitcode.com/Ascend/triton-ascend)包安装，可以使用以下命令安装
```sh
pip install triton-ascend
```

一键运行GDN模块，组装了所有GDN相关算子，包括前向和反向，包括AscendC和Triton算子
```sh
python examples/flash_gated_delta_rule.py
```

## 🔍目录结构
关键目录如下：
```
├── cmake                              # 项目工程编译目录
├── common                             # 项目公共头文件和公共源码
├── chunk_gated_delta_rule             # Chunk Gated Delta Rule训练算子
│   ├── chunk_bwd_dqkwg                # 反向传播dq/dk/dw/dg算子
│   ├── chunk_bwd_dv_local             # 反向传播dv局部计算算子
│   ├── chunk_fwd_o                    # 前向传播输出o算子
│   ├── chunk_gated_delta_rule_bwd_dhu # 反向传播dh/du算子
│   ├── chunk_gated_delta_rule_fwd_h   # 前向传播隐藏状态h算子
│   ├── common                         # chunk算子公共模块
│   ├── prepare_wy_repr_bwd_da         # WY表示反向传播da算子
│   ├── prepare_wy_repr_bwd_full       # WY表示完整反向传播算子
│   ├── recompute_wu_fwd               # 重新计算wu前向算子
│   ├── recurrent_gated_delta_rule     # Recurrent Gated Delta Rule推理算子
│   └── CMakeLists.txt
├── torch_custom                       # 自定义PyTorch算子适配
├── examples                           # 端到端算子开发和调用示例
│   └── flash_gated_delta_rule.py      # 完整GDN接入调用示例
├── scripts                            # 脚本目录，包含算子构建相关配置文件
├── tests                              # 测试工程目录
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
