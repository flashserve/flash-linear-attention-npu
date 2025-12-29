# 算子调用
## 前提条件

- 环境部署：调用算子之前，请先参考[环境部署](../context/quick_install.md)完成基础环境搭建。
- 调用算子列表：项目可调用的算子参见[算子列表](../op_list.md)，算子对应的aclnn接口参见[aclnn列表](../op_api_list.md)。

## 编译执行

基于社区版CANN包对算子源码修改时，可采用如下方式进行源码编译：

- [自定义算子包](#自定义算子包)：选择部分算子编译生成的包称为自定义算子包，以**挂载**形式作用于CANN包，不改变原始包内容。生成的自定义算子包优先级高于原始CANN包。该包支持aclnn和图模式调用AI Core算子。

- [ops-transformer包](#ops-transformer包)：选择整个项目编译生成的包称为ops-transformer包，可**完整替换**CANN包对应部分。该包支持aclnn和图模式调用AI Core算子。

### 自定义算子包

1. **编译自定义算子包**

    进入项目根目录，执行如下编译命令：

    > 说明：编译过程依赖第三方开源软件，联网场景会自动下载，离线编译场景需要自行安装，具体参考[离线编译](../context/build_offline.md)。
    
    ```bash
    bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
    # 以FlashAttentionScore算子编译为例
    # bash build.sh --pkg --soc=ascend910b --ops=flash_attention_score
    # 编译experimental贡献目录下的算子
    # bash build.sh --pkg --experimental --soc=ascend910b --ops=${experimental_op}
    ```
    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"，Ascend 950PR/Ascend 950DT产品使用"ascend950"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。
    - --ops（可选）：\$\{op\_list\}表示待编译算子，不指定时默认编译所有算子。格式形如"apply_rotary_pos_emb,rope_quant_kvcache,..."，多算子之间用英文逗号","分隔。
    - --experimental（可选）：表示编译experimental贡献目录下的算子，${experimental_op}为新贡献算子目录名。

    若\$\{vendor\_name\}和\$\{op\_list\}都不传入编译的是ops-transformer包；若编译所有算子的自定义算子包，需传入\$\{vendor\_name\}。当提示如下信息，说明编译成功。
    ```bash
    Self-extractable archive "cann-ops-transformer-${vendor_name}_linux-${arch}.run" successfully created.
    ```
    编译成功后，run包存放于项目根目录的build_out目录下。
    
2. **安装自定义算子包**
   
    ```bash
    ./cann-ops-transformer-${vendor_name}_linux-${arch}.run
    ```
    
    自定义算子包安装路径为`${ASCEND_HOME_PATH}/opp/vendors`，\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/cann。

3. **（可选）删除自定义算子包**

    注意自定义算子包不支持卸载，如需卸载，请删除vendors\/\$\{vendor\_name}目录，并删除vendors/config.ini中load_priority对应\$\{vendor\_name\}的配置项。

### ops-transformer包

1. **编译ops-transformer包**

    进入项目根目录，执行如下编译命令：

    > 说明：编译过程依赖第三方开源软件，联网场景会自动下载，离线编译场景需要自行安装，具体参考[离线编译](../context/build_offline.md)。

    ```bash
    # 编译除experimental贡献目录外的所有算子
    bash build.sh --pkg [--jit] --soc=${soc_version}
    # 编译experimental贡献目录下的所有算子
    # bash build.sh --pkg --experimental [--jit] --soc=${soc_version}
    ```
    - --jit（可选）：推荐设置，表示不编译算子的二进制文件。
    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"，Ascend 950PR/Ascend 950DT产品使用"ascend950"。
    - --experimental（可选）：表示编译experimental贡献目录下的算子。

    若提示如下信息，说明编译成功。

    ```bash
    Self-extractable archive "cann-${soc_name}-ops-transformer_${cann_version}_linux-${arch}.run" successfully created.
    ```

   \$\{soc\_name\}表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。编译成功后，run包存放于build_out目录下。

2. **安装ops-transformer包**

    ```bash
    # 安装命令
    ./cann-${soc_name}-ops-transformer_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
    ```

    \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

3. **（可选）卸载ops-transformer包**

    ```bash
    # 卸载命令
    ./${install_path}/cann/share/info/ops_transformer/script/uninstall.sh
    ```

## 本地验证 

通过项目根目录build.sh执行算子和UT用例，验证项目功能是否正常，build参数参见[build参数说明](../context/build.md)。目前算子支持API方式（aclnn接口）和图模式调用，**推荐aclnn调用**。

- **执行算子样例**
  
    - 完成自定义算子包安装后，执行如下命令：
        ```bash
        bash build.sh --run_example ${op} ${mode} ${pkg_mode} [--vendor_name=${vendor_name}] [--soc=${soc_version}]
        # 以FlashAttentionScore算子example执行为例
        # bash build.sh --run_example flash_attention_score eager cust --vendor_name=custom
        ```
        
        - \$\{op\}：表示待执行算子，算子名小写下划线形式，如flash_attention_score。            
        - \$\{mode\}：表示执行模式，目前支持eager（aclnn调用）、graph（图模式调用）。
        - \$\{pkg_mode\}：表示包模式，目前仅支持cust，即自定义算子包。         
        - \$\{vendor\_name\}（可选）：与构建的自定义算子包设置一致，默认名为custom。
        - \$\{soc_version\}（可选）：表示NPU型号。当设置为"ascend950"时会额外运行"arch35"目录下的示例文件。        
        
        说明：\$\{mode\}为graph时，不指定\$\{pkg_mode\}和\$\{vendor\_name\}

    - 完成ops-transformer包安装后，执行命令如下：
        ```bash
        bash build.sh --run_example ${op} ${mode} [--soc=${soc_version}]
        # 以FlashAttentionScore算子example执行为例
        # bash build.sh --run_example flash_attention_score eager
        ```
        
        - \$\{op\}：表示待执行算子，算子名小写下划线形式，如flash_attention_score。       
        - \$\{mode\}：表示算子执行模式，目前支持eager（aclnn调用）、graph（图模式调用）。
        - \$\{soc_version\}（可选）：表示NPU型号。当设置为"ascend950"时会额外运行"arch35"目录下的示例文件。

        执行算子样例后会打印结果，以FlashAttentionScore算子执行为例：
    
        ```
        mean result[0] is: 256.000000
        mean result[1] is: 256.000000
        mean result[2] is: 256.000000
        mean result[3] is: 256.000000
        mean result[4] is: 256.000000
        mean result[4] is: 256.000000
        ...
        mean result[65532] is: 256.000000
        mean result[65533] is: 256.000000
        mean result[65534] is: 256.000000
        mean result[65535] is: 256.000000
        ```
- **执行算子UT**

	> 说明：执行UT用例依赖googletest单元测试框架，详细介绍参见[googletest官网](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

    ```bash
  # 安装根目录下test相关requirements.txt依赖
  pip3 install -r tests/requirements.txt
  # 方式1: 编译并执行指定算子和对应功能的UT测试用例（选其一）
  bash build.sh -u --[opapi|ophost|opkernel] --ops=flash_attention_score
  # 方式2: 编译并执行所有的UT测试用例
  # bash build.sh -u
  # 方式3: 编译所有的UT测试用例但不执行
  # bash build.sh -u --noexec
  # 方式4: 编译并执行对应功能的UT测试用例（选其一）
  # bash build.sh -u --[opapi|ophost|opkernel]
  # 方式5: 编译对应功能的UT测试用例但不执行（选其一）
  # bash build.sh -u --noexec --[opapi|ophost|opkernel]
  # 方式6: 编译并执行除公共用例外指定soc的UT测试用例
  # bash build.sh -u --[opapi|ophost|opkernel] --soc=${soc_version}
    ```
  
    如需验证ophost功能是否正常，执行如下命令：
    ```bash
  bash build.sh -u --ophost
    ```

    执行完成后出现如下内容，表示执行成功。
    ```bash
  Global Environment TearDown
  [==========] ${n} tests from ${m} test suites ran. (${x} ms total)
  [  PASSED  ] ${n} tests.
  [100%] Built target transformer_op_host_ut
    ```
    \$\{n\}表示执行了n个用例，\$\{m\}表示m项测试，\$\{x\}表示执行用例消耗的时间，单位为毫秒。