# 通算融合算子Experimental使用说明

## 前提条件

- 环境部署：调用算子之前，请先参考[环境部署](../../docs/zh/context/quick_install.md)完成基础环境搭建。
- 算子源码：可参考通算融合示例算子[all_gather_add](../../examples/mc2/all_gather_add)完成代码编写，本说明后续将以all_gather_add算子为例。

## 编译执行

1. **编译自定义算子包**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg --experimental --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
    # 执行命令示例：（在910b环境下编译all_gather_add算子）
    # bash build.sh --pkg --experimental --soc=ascend910b --ops=all_gather_add
    # 执行命令示例：（在910b环境下编译experimental全量包）
    # bash build.sh --pkg --experimental --soc=ascend910b
    ```

    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件使用"ascend910b"（默认），Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。同一环境中如需要安装多个自定义算子包，则必须使用不同的\$\{vendor\_name\}加以区分。
    - --ops：自定义算子名称。

    > 注意：编译experimental全量包时，需要将ops-transformer/gmm/grouped_matmul_swiglu_quant算子拷贝至ops-transformer/experimental/gmm目录下。

    若提示如下信息，说明编译成功。

    ```bash
    # experimental自定义单算子包
    Self-extractable archive "cann-ops-transformer-${vendor_name}_linux-${arch}.run" successfully created.
    # experimental全量包
    # Self-extractable archive "cann-${soc}-ops-transformer_${version}_linux-${arch}.run" successfully created.
    ```

    编译成功后，run包存放于项目根目录的build_out目录下。

2. **安装自定义算子包**

    ```bash
    ./cann-ops-transformer-${vendor_name}_linux-${arch}.run
    ```

    自定义单算子包安装时可指定参数`--install-path=${ASCEND_HOME_PATH}/opp`，并存放于`${ASCEND_HOME_PATH}/opp/vendors`目录下，全量包安装路径为`--install-path=${ASCEND_HOME_PATH}`，并存放于`${ASCEND_HOME_PATH}/opp`目录下，\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/cann。注意自定义算子包不支持卸载。安装完成后，需要激活自定义算子包环境变量，激活方式如下：

    ```bash
    # experimental自定义单算子包
    source ${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_transformer/bin/set_env.bash
    # experimental全量包
    # source ${ASCEND_HOME_PATH}/cann/set_env.bash
    ```

3. **本地验证**

    通过项目根目录build.sh脚本，可快速调用算子和UT用例，验证项目功能是否正常，build参数介绍参见[build参数说明](../../docs/zh/context/build.md)。

    - **执行算子样例**

        以[all_gather_add](../../examples/mc2/all_gather_add)算子为例，样例执行步骤如下：

        1. 将自定义算子包安装在默认路径下，参考**安装自定义算子包**章节；

        2. 编写算子样例代码，通过调用[aclnn_all_gather_add.h](../../examples/mc2/all_gather_add/op_host/op_api/aclnn_all_gather_add.h)文件中的两段式接口`aclnnAllGatherAddGetWorkspaceSize`和`aclnnAllGatherAdd`来完成算子调用。

        > 注意：算子样例代码需放置在算子名目录下的examples内，且文件命名需遵循"test_[调用方式]_[算子名称].cpp"格式。其中，调用方式可选择aclnn（单算子直调）或geir（图模式），对应mode为eager和graph。

        完成自定义算子包的安装和算子执行代码的编写后，执行如下命令：

        ```bash
        bash build.sh --run_example ${op} ${mode} ${pkg_mode} [--vendor_name=${vendor_name}]
        # 执行命令示例:
        # experimental自定义单算子包
        # bash build.sh --run_example all_gather_add eager cust
        # experimental全量包
        # bash build.sh --run_example all_gather_add eager
        ```

        - \$\{op\}：表示待执行算子，算子名采用小写下划线命名方式，如all_gather_add。
        - \$\{mode\}：表示执行模式，目前支持eager（aclnn调用）。
        - \$\{pkg_mode\}：表示包模式，目前仅支持cust，即自定义算子包。
        - \$\{vendor\_name\}：与构建的自定义算子包设置一致，默认名为custom。

        执行算子样例后会打印执行结果，以all_gather_add算子为例，结果如下：

        ```bash
        device_0 aclnnAllGatherAdd execute successfully.
        device_1 aclnnAllGatherAdd execute successfully.
        device_0 aclnnAllGatherAdd golden compare successfully.
        device_1 aclnnAllGatherAdd golden compare successfully.
        ```
