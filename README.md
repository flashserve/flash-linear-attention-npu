# flash-linear-attention-npu

## 🔥Latest News

- [2026/03] flash-linear-attention-npu 项目首次上线。

## 🚀概述

flash-linear-attention-npu 算子库由天津大学主导开发，是一个面向昇腾架构的高性能线性注意力算子库，对标 Flash-Linear-Attention 项目，旨在为昇腾平台提供高效的线性注意力计算实现。

## ⚡️快速入门

若您希望快速体验算子的调用和开发过程，请访问如下文档获取简易教程。

- [算子列表](docs/zh/op_list.md)：介绍项目提供的全量算子信息，方便快速查询。
- [环境部署](docs/zh/context/quick_install.md)：介绍项目基础环境的搭建，包括软件包和第三方依赖的获取和安装。
- [算子调用](docs/zh/invocation/quick_op_invocation.md)：环境部署后，介绍如何快速调用算子，包括编译执行算子包和UT等。

## 📖学习教程

若您希望深入体验项目功能并修改算子源码，请访问如下文档获取详细教程。
- [算子调用方式](docs/zh/invocation/op_invocation.md)：介绍不同的调用算子方式，方便快速应用于不同的AI业务场景。

## 🔍目录结构
关键目录如下，详细目录介绍参见[项目目录](./docs/zh/context/dir_structure.md)。
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
│   └── CMakeLists.txt
├── recurrent_gated_delta_rule         # Recurrent Gated Delta Rule推理算子
│   ├── op_host                        # 算子信息库、Tiling相关实现
│   ├── op_kernel                      # 算子Kernel目录
│   ├── docs                           # 算子说明文档
│   ├── examples                       # 算子使用示例
│   ├── tests                          # 算子测试目录
│   └── README.md
├── op-plugin                          # PyTorch算子插件
├── torch_custom                       # 自定义PyTorch算子
├── docs                               # 项目文档介绍
├── examples                           # 端到端算子开发和调用示例
├── scripts                            # 脚本目录，包含算子构建相关配置文件
├── tests                              # 测试工程目录
├── CMakeLists.txt
├── README.md
├── build.sh                           # 项目工程编译脚本
├── install_deps.sh                    # 安装依赖包脚本
└── requirements.txt                   # 本项目需要的第三方依赖包
```

## 📝相关信息

- [安全声明](SECURITY.md)
- [许可证](LICENSE)

## 🙏致谢

本项目的部分实现参考了 [ops-transformer](https://gitcode.com/cann/ops-transformer) 仓库，感谢华为 CANN 社区及相关开发团队的开源贡献。
