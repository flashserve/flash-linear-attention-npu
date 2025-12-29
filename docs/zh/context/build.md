# build参数说明

## 简介
build.sh是本项目的构建脚本，默认在项目根目录下，作用是将源代码自动编译、链接和配置，最终生成可执行文件、库文件或其它可供安装或直接运行的目标文件。具体来说，脚本中通过配置不同参数实现多种功能，包含构建多种目标库（如：libophost_transformer.so）、编译算子包、执行单元测试等。


## 使用方法 
1. **配置环境变量**
   
   参考[环境准备](../invocation/quick_op_invocation.md#环境准备)完成环境变量配置。
   ```bash
   # 默认路径安装，以root用户为例
   source /usr/local/Ascend/cann/set_env.sh
   ```
2. **构建命令格式**

   以编译算子包命令为例，样式如下，其中`--vendor_name`与`--ops`在该场景为可选项。
   ```bash
   bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
   ```
   全量参数含义参见下方参数说明章节，请按实际情况选择合适的参数。

## 参数说明
build.sh支持多种功能，可通过如下命令查看所有功能参数。
```bash
bash build.sh --help
```

| 参数名              | 可选/必选  | 参数说明                                                                         |
|------------------|--------|------------------------------------------------------------------------------|
| -j${n}           | 可选     | 指定编译线程数，${n}为具体线程数，默认值为8（如：-j8）；若线程数超过CPU核心数，会自动调整为CPU核心数。                   |
| -v               | 可选     | 查看CMake编译配置信息。                                                               |
| -O${n}           | 可选     | 指定编译优化级别，支持O0/O1/O2/O3（如：-O3），${n}为优化级别标识。                                   |
| -u               | 可选     | 启用单元测试（UT）编译模式，编译所有UT目标。                                                     |
| --help，-h        | 可选     | 打印脚本使用帮助信息。                                                                  |
| --ops            | 可选     | 指定待编译的算子，如：apply_rotary_pos_emb,rope_quant_kvcache，多个算子用英文逗号“,”分隔，不可与--ophost、--opapi、--opgraph同时使用。 |
| --soc            | 可选     | 指定NPU型号，多个soc用英文逗号“,”分隔。                                               |
| --jit            | 可选     | 配置后，表示不编译算子的二进制文件。                                                                 |
| --vendor_name    | 可选     | 指定自定义算子包的名称，默认值为custom。                                                      |
| --debug          | 可选     | 启用调试模式。                                                                      |
| --cov            | 可选     | 预留参数，开发者暂不需要关注。                                                              |
| --noexec         | 可选     | 仅编译单元测试二进制文件，不自动执行编译后的UT可执行文件。                                               |
| --opkernel       | 可选     | 编译二进制内核。                                                                     |
| --pkg        | 可选     | 生成安装包，不可与-u（UT模式）或--ophost、--opapi、--opgraph同时使用。                            |
| --make_clean     | 可选     | 执行基础清理操作（清理编译产物），执行后脚本退出。                                                    |
| --ophost         | 可选     | 编译libophost_transformer.so库，不可与--pkg、--ops同时使用。                                     |
| --opapi          | 可选     | 编译libopapi_transformer.so库，不可与--pkg、--ops同时使用。                                      |
| --opgraph        | 可选     | 编译libopgraph_transformer.so库，不可与--pkg、--ops同时使用。                                    |
| --ophost_test   | 可选     | 编译ophost相关单元测试，与-u --ophost组合等效。                                          |
| --opapi_test    | 可选     | 编译opapi相关单元测试，与-u --opapi组合等效。                                            |
| --opgraph_test  | 可选     | 预留参数，开发者暂不需要关注。                                                           |
| --opkernel_test | 可选     | 编译opkernel相关单元测试，与-u --opkernel组合等效。                                      |
| --run_example    | 可选     | 编译指定算子及模式的样例并执行编译后的可执行文件。                                                    |
| --genop         | 可选     | 创建AI Core自定义算子初始目录。                                                       |
| --experimental  | 可选     | 编译experimental目录下的用户算子。                                                   |
| --oom           | 可选     | 开启kernel侧oom内存检测功能。                                                   |
| --cann_3rd_lib_path           | 可选     | 离线编译场景下第三方库存放的目录。                                                   |