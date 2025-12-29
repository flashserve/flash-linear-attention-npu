# 离线编译
离线编译是指在没有连接互联网的环境下，将软件源代码编译成可执行程序，并安装或配置到目标服务器上的过程。
本项目[算子调用](../invocation/quick_op_invocation.md)或[算子开发](../develop/aicore_develop_guide.md)过程中均需编译算子包，编译过程中会依赖一些开源第三方软件，这些软件联网时会自动下载，离线状态下无法直接下载。

本章提供了离线编译安装指导，在此之前请确保已按[环境部署](quick_install.md)完成基础环境搭建。
## 获取依赖
离线编译时，需手动安装如下依赖，否则无法正常编译和执行算子，其中`${cann_3rd_lib_path}`表示第三方软件存放的目录。

- 依赖json

下载[json](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip)，并解压到`${cann_3rd_lib_path}/json/include`，若无该目录请自行创建。

- 依赖makeself

下载[makeself](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz)，并解压到`${cann_3rd_lib_path}/makeself`，若无该目录请自行创建。

## 离线编译（自定义算子包）
自定义算子包编译时，需增加--cann_3rd_lib_path配置选项并指定路径，命令如下：

```bash
bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}] --cann-3rd_lib_path=${cann_3rd_lib_path}
# 以FlashAttentionScore算子编译为例，假设第三方软件存放的目录为/path/cann_3rd_lib_path
# bash build.sh --pkg --soc=ascend910b --ops=flash_attention_score --cann-3rd_lib_path=/path/cann_3rd_lib_path
```

## 离线编译（ops-transformer包）
整包编译时，需增加--cann_3rd_lib_path配置选项并指定路径，命令如下：

```bash
bash build.sh --pkg [--jit] --soc=${soc_version} --cann-3rd_lib_path=${cann_3rd_lib_path}
```