

# 算子调试调优

## 调试定位（AI Core算子）

算子运行过程中，如果出现算子执行失败、精度异常等问题，可以打印各阶段信息，如Kernel中间结果，进行问题分析和定位。

以`AddExample`算子为例，常见调试方法如下：

* **printf**

  该接口支持打印Scalar类型数据，如整数、字符、布尔型等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > printf”。
  
  ```c++
  blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;
  // 打印当前核计算Block长度
  AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
  ```
* **DumpTensor**

  该接口支持Dump指定Tensor的内容，同时支持打印自定义附加信息，比如当前行号等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > DumpTensor”。
  
  ```c++
  AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
  // 打印zLocal Tensor信息
  DumpTensor(zLocal, 0, 128);
  AscendC::DataCopy(outputGMZ[progress * tileLength_], zLocal, tileLength_);
  ```

对于复杂场景的问题定位，比如算子卡死、GM/UB访问越界等场景，可以采取**单步调试**的方式，具体操作请参见[msDebug](https://www.hiascend.com/document/redirect/CannCommunityToolMsdebug)算子调试工具。

## 性能调优

### 方式一（针对Atlas A2/A3系列产品）

算子运行过程中，如果出现执行精度下降、内存占用异常等问题，可通过[msProf](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof)性能分析工具分析算子各运行阶段指标数据（如吞吐率、内存占用、耗时等），从而确定问题根源，并针对性地优化。

本章以`AddExample`自定义算子为例，主要介绍算子调优中常用的算子上板性能采集和流水图仿真两种方式。通过采集算子上板运行时各项流水指标分析算子Bound场景，了解仿真流水图便于优化算子内部流水。

1. 前提条件。

   完成算子开发和编译后，假设采用aclnn接口方式调用，生成的算子可执行文件（test_aclnn_add_example）所在目录为本项目`examples/add_example/examples/build/bin/`。

2. 采集性能数据。

   当需要采集算子上板运行各项流水指标时可以进入算子可执行文件所在目录，执行如下命令：

   ```bash
   msprof op ./test_aclnn_add_example
   ```
   采集结果在本项目`examples/add_example/examples/build/bin/OPPROF_*`目录，采集完成后打印如下信息：
   
    ``` text
    Op Name: AddExample_a1532827238e1555db7b997c7bce2928_high_performance_1
    Op Type: vector             
    Task Duration(us): 97.861954 
    Block Dim: 8
    Mix Block Dim:
    Device Id: 0
    Pid: 2776181
    Current Freq: 1800
    Rated Freq: 1800
    ```
   其中Task Duration是当前算子Kernel耗时，Block Dim是当前算子执行核数。

   算子各项流水详细指标可关注`OPPROF_*`下`ArithmeticUtilization`文件，包含了当前各项流水的占比，具体介绍参见[msProf](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof)中”性能数据文件 > msprof op > ArithmeticUtilization（cube及vector类型指令耗时和占比）“章节。

3. 采集仿真流水图。
   
   msProf工具进行算子仿真调优之前，需执行如下命令配置环境变量。
   ```bash
   export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH 
   ```
   请根据CANN软件包实际安装路径和AI处理器型号对以上环境变量进行修改。
   
   之后进入算子可执行文件所在目录，执行如下命令：
   ```bash
   msprof op simulator --output=$PWD/pipline_auto --kernel-name"AddExample" ./test_aclnn_add_example
   ```
   采集结果在本项目`$PWD/pipeline_auto/OPPROF_**`目录中。
   其中流水相关文件路径为`OPPROF**/simulator/visualize_data.bin`，可以借助[MindStudio Insight](https://www.hiascend.com/document/redirect/MindStudioInsight)工具查看。
   
### 方式二（针对Ascend 950PR）

算子开发过程中，如果出现执行精度下降、内存占用异常等问题，可以通过[CANN Simulator](./cann_sim.md)仿真工具分析算子的指令流水情况，从而确定问题根源，并针对性地优化。

本章以`AddExample`自定义算子为例，主要介绍仿真工具的使用。如何通过仿真工具进行精度和性能调优。

1. 前提条件。

   完成算子开发和编译后，假设采用aclnn接口方式调用，生成的算子可执行文件（test_aclnn_add_example）所在目录为本项目`examples/add_example/examples/build/bin/`。

2. 执行仿真命令，生成仿真数据

   ```
   cannsim record ./test_aclnn_add_example -s Ascend950 --gen-report
   ```

   仿真结果在本项目`examples/add_example/examples/build/bin/cannsim_*`目录，流水相关文件为：

   ```
   trace_core0.json
   ``` 

3. 在Chrome浏览器中输入“chrome://tracing”地址，并将生成的指令流水图文件（trace_core0.json）拖到空白处打开，具体参数介绍参考CANN Simulator中[“仿真结果解析”](./cann_sim.md#仿真结果解析)章节。

