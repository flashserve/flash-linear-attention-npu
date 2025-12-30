# aclnnGroupedMatmulV4

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT AI处理器</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明

- 接口功能：实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。基本功能为矩阵乘，如$y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中g为分组个数，$m_i/k_i/n_i$为对应的维度。输入输出参数类型均为aclTensorList，对应的功能为：

  - k轴分组：$k_i$各不相同，但$m_i/n_i$每组相同，此时$x_i/weight_i$可以在$k_i$上拼接。
  - m轴分组：$k_i$各组相同，$weight_i/y_i$可以在$n_i$上拼接。

    相较于[GroupedMatmulV3](aclnnGroupedMatmulV3.md)接口，**此接口新增：**
  - 支持groupListOptional中数值为分组轴上每组大小。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 支持静态量化（pertensor+perchannel）（量化方式请参见[量化介绍](../../../docs/zh/context/量化介绍.md)，下同）BFLOAT16和FLOAT16输出，带激活及不带激活场景
    - 支持动态量化（pertoken+perchannel）BFLOAT16和FLOAT16输出，带激活及不带激活场景。
    - 支持伪量化weight是INT4的输入，不带激活场景，支持perchannel和pergroup两种模式。
  - <term>Ascend 950PR/Ascend 950DT AI处理器</term>：
    - 支持静态量化（1.pertensor-perchannel(T-C)；2.pertensor-pertensor(T-T)）BFLOAT16，FLOAT16和FLOAT32输出，带bias，不带激活场景。
    - 支持动态量化（1.pertoken-perchannel(K-C)；2.pertoken-pertensor(K-T)；3.pertensor-pertensor(T-T)；4.pertensor-perchannel(T-C)；4.mx量化；5.pergroup-perblock(G-B)）BFLOAT16，FLOAT16和FLOAT32输出，带bias，不带激活场景。
    - 支持伪量化weight是INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8的输入，不带激活场景，仅支持perchannel模式。

    **说明：**
  - 单tensor指一个tensor list中所有分组的tensor在groupType指定的分组轴上合并为1个；否则为多tensor。
  - tensor转置：指若tensor shape为[M,K]时，则stride为[1,M],数据排布为[K,M]的场景，即非连续tensor。

- 计算公式：
  - **非量化场景：**

    $$
     y_i=x_i\times weight_i + bias_i
    $$

  - **量化场景（静态量化，T-C && T-T量化，无perTokenScaleOptional）：**
    $$
      y_i=(x_i\times weight_i) * scale_i + offset_i
    $$
  - x为INT8，bias为INT32
      $$
        y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
      $$
  - x为INT8，bias为BFLOAT16/FLOAT16/FLOAT32，无offset
      $$
        y_i=(x_i\times weight_i) * scale_i + bias_i
      $$
  - **量化场景（动态量化，T-T && T-C && K-T && K-C量化）：**
    $$
     y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i
    $$
  - x为INT8，bias为INT32
      $$
        y_i=(x_i\times weight_i + bias_i) * scale_i * per\_token\_scale_i
      $$
  - x为INT8，bias为BFLOAT16/FLOAT16/FLOAT32
      $$
        y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i  + bias_i
      $$
  - **量化场景（动态量化，MX && G-B量化）：**
    $$
    y_i[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (xSlice_i * weightSlice_i)) * (per\_token\_scale_i[m/gsM, j] * scale_i[j, n/gsN])) + bias_i[n]
    $$
    其中，gsM,gsN和gsK分别代表M/N/K轴的量化的block size，$xSlice_i$代表$x_i$第m行长度为gsK的向量，$weightSlice_i$代表$weight_i$第n列长度为gsK的向量，K轴均从j * gsK起始切片，j的取值范围[0, kLoops), kLoops=ceil($K_i$ / gsK)，支持最后的切片长度不足gsK。

  - **伪量化场景：**

    $$
     y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupedMatmulV4GetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnGroupedMatmulV4”接口执行计算。

```cpp
aclnnStatus aclnnGroupedMatmulV4GetWorkspaceSize(
  const aclTensorList *x,
  const aclTensorList *weight,
  const aclTensorList *biasOptional,
  const aclTensorList *scaleOptional,
  const aclTensorList *offsetOptional,
  const aclTensorList *antiquantScaleOptional,
  const aclTensorList *antiquantOffsetOptional, 
  const aclTensorList *perTokenScaleOptional,
  const aclTensor     *groupListOptional,
  const aclTensorList *activationInputOptional,
  const aclTensorList *activationQuantScaleOptional,
  const aclTensorList *activationQuantOffsetOptional,
  int64_t              splitItem, int64_t groupType,
  int64_t              groupListType,
  int64_t              actType,
  aclTensorList       *out,
  aclTensorList       *activationFeatureOutOptional,
  aclTensorList       *dynQuantScaleOutOptional,
  uint64_t            *workspaceSize,
  aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulV4(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupedMatmulV4GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1483px"><colgroup>
  <col style="width: 210px">
  <col style="width: 90px">
  <col style="width: 370px">
  <col style="width: 232px">
  <col style="width: 339px">
  <col style="width: 86px">
  <col style="width: 92px">
  <col style="width: 64px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>Device侧的aclTensorList，公式中的输入x</td>
      <td>支持的最大长度为128个</td>
      <td>FLOAT16、BFLOAT16、FLOAT32、INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E1M2、FLOAT4_E2M1</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>Device侧的aclTensorList，公式中的weight</td>
      <td>支持的最大长度为128个</td>
      <td>FLOAT16、BFLOAT16、FLOAT32、INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E1M2、FLOAT4_E2M1</td>
      <td>ND、NZ</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>输入</td>
      <td>Device侧的aclTensorList，公式中的bias</td>
      <td>长度与weight相同</td>
      <td>INT32、BFLOAT16、FLOAT16、FLOAT32、INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleOptional</td>
      <td>输入</td>
      <td>Device侧的aclTensorList，代表量化参数中的缩放因子</td>
      <td>一般情况下，长度与weight相同</td>
      <td>UINT64、INT64、BFLOAT16、FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>输入</td>
      <td>Device侧的aclTensorList，代表量化参数中的偏移量</td>
      <td>长度与weight相同</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional</td>
      <td>输入</td>
      <td>Device侧的aclTensorList，代表伪量化参数中的缩放因子</td>
      <td>长度与weight相同</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional</td>
      <td>输入</td>
      <td>Device侧的aclTensorList，代表伪量化参数中的偏移量</td>
      <td>长度与weight相同</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>perTokenScaleOptional</td>
      <td>输入</td>
      <td>Device侧的aclTensorList，代表量化参数中的由x量化引入的缩放因子</td>
      <td>仅支持x、weight、out均为单tensor</td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListOptional</td>
      <td>输入</td>
      <td>Device侧的aclTensor类型，代表输入和输出分组轴方向的matmul大小分布</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activationInputOptional</td>
      <td>输入</td>
      <td>Device侧的aclTensorList类型，代表激活函数的反向输入</td>
      <td>当前只支持传入nullptr</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activationQuantScaleOptional</td>
      <td>输入</td>
      <td>Device侧的aclTensorList类型</td>
      <td>当前只支持传入nullptr</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activationQuantOffsetOptional</td>
      <td>输入</td>
      <td>Device侧的aclTensorList类型</td>
      <td>当前只支持传入nullptr</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>splitItem</td>
      <td>输入</td>
      <td>整数型参数，代表输出是否要做tensor切分</td>
      <td>枚举值0、1、2、3</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupType</td>
      <td>输入</td>
      <td>整数型参数，代表需要分组的轴</td>
      <td>枚举值-1、0、1、2</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>输入</td>
      <td>-</td>
      <td>枚举值0、1、2</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actType</td>
      <td>输入</td>
      <td>代表激活函数类型</td>
      <td>枚举值1、2、3、4、5</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>Device侧的aclTensorList，公式中的输出y</td>
      <td>支持的最大长度为128个</td>
      <td>FLOAT16、BFLOAT16、INT8、FLOAT32、INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activationFeatureOutOptional</td>
      <td>输出</td>
      <td>Device侧的aclTensorList，激活函数的输入数据</td>
      <td>当前只支持传入nullptr</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dynQuantScaleOutOptional</td>
      <td>输出</td>
      <td>Device侧的aclTensorList，当前只支持传入nullptr</td>
      <td>当前只支持传入nullptr</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - x支持FLOAT16、BFLOAT16、FLOAT32、INT8、INT4
    - weight支持FLOAT16、BFLOAT16、FLOAT32、INT8、INT4，格式支持ND、FRACTAL_NZ
    - biasOptional支持FLOAT16、FLOAT32、INT32
    - scaleOptional支持UINT64、BFLOAT16、FLOAT32
    - perTokenScaleOptional支持FLOAT32
    - out支持FLOAT16、BFLOAT16、INT8、FLOAT32、INT32
    - groupType不支持n轴分组
  - <term>Ascend 950PR/Ascend 950DT AI处理器</term>：
    - x支持FLOAT8_E4M3FN、FLOAT8_E5M2、INT8、HIFLOAT8、FLOAT16、BFLOAT16、FLOAT4_E1M2、FLOAT4_E2M1
    - weight支持FLOAT8_E4M3FN、FLOAT8_E5M2、INT8、INT4、HIFLOAT8、FLOAT16、BFLOAT16、FLOAT4_E1M2、FLOAT4_E2M1，当x与weight都为int8时支持ND和FRACTAL_NZ格式，其余场景只支持ND格式。使用weightNz特性时可使用aclnnNpuFormatCast接口完成输入Format从ND到AI处理器亲和数据排布格式（NZ）的转换。如原始weight为转置状态且想使用性能更高的非转置通路计算，可使用aclnnPermute接口转为非转置后再调用aclnnNpuFormatCast接口。
    - biasOptional支持INT32、BFLOAT16、FLOAT16、FLOAT32，在输入x为INT8、FLOAT16、BFLOAT16时支持INT32、BFLOAT16、FLOAT16、FLOAT32，在输入x为FLOAT4_E1M2、FLOAT4_E2M1时仅支持FLOAT32，其它类型输入需传空指针
    - scaleOptional支持UINT64、INT64、BFLOAT16、FLOAT32、FLOAT8_E8M0
    - perTokenScaleOptional支持FLOAT32、FLOAT8_E8M0
    - groupListType不支持取2
    - actType只支持传入0
    - out支持BFLOAT16、FLOAT16、FLOAT32
    - 不支持offsetOptional
    - groupType支持m轴分组，仅非量化和量化支持k轴分组，仅非量化和伪量化支持不分组

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一阶段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed; width: 1055px"><colgroup>
  <col style="width: 242px">
  <col style="width: 78px">
  <col style="width: 735px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERRPARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入参数是必选输入、输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>x、weight、biasOptional、scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、groupListOptional、out的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>weight的长度大于128；若bias不为空，bias的长度不等于weight的长度。</td>
    </tr>
    <tr>
      <td>groupListOptional维度为1。</td>
    </tr>
    <tr>
      <td>splitItem为2、3的场景，out长度不等于1。</td>
    </tr>
    <tr>
      <td>splitItem为0、1的场景，out长度不等于weight的长度，groupListOptional长度不等于weight的长度。</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedMatmulV4

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 834px"><colgroup>
    <col style="width: 118px">
    <col style="width: 87px">
    <col style="width: 629px">
    </colgroup>
    <thead>
      <tr>
        <th>参数说明</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulV4GetWorkspaceSize获取。</ td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
      </tr>
    </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnGroupedMatmulV4默认确定性实现。
- 如果传入groupListOptional，当groupListType为0时，groupListOptional必须为非负单调非递减数列；当groupListType为1时，groupListOptional必须为非负数列；groupListType为2时，groupListOptional的第二列数据必须为非负数列，且长度不能为1。
- x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32的最大值2147483647。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 非量化场景支持的输入类型为：
    - x为FLOAT16、weight为FLOAT16、biasOptional为FLOAT16、scaleOptional为空、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空、activationInputOptional为空、out为FLOAT16。
    - x为BFLOAT16、weight为BFLOAT16、biasOptional为FLOAT32、scaleOptional为空、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空、activationInputOptional为空、out为BFLOAT16。
    - x为FLOAT32、weight为FLOAT32、biasOptional为FLOAT32、scaleOptional为空、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空、activationInputOptional为空、out为FLOAT32（仅x、weight、y都为单tensor场景支持）。
  - 量化场景支持的输入类型为：
    - x为INT8、weight为INT8、biasOptional为INT32、scaleOptional为UINT64、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空、activationInputOptional为空、out为INT8。
    - x为INT8、weight为INT8、biasOptional为INT32、scaleOptional为BFLOAT16、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空或FLOAT32、activationInputOptional为空、out为BFLOAT16。
    - x为INT8、weight为INT8、biasOptional为INT32、scaleOptional为FLOAT32、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空或为FLOAT32、activationInputOptional为空、out为FLOAT16。
    - x为INT8、weight为INT8、biasOptional为INT32、scaleOptional为空、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空、activationInputOptional为空、out为INT32。
    - x为INT4、weight为INT4、biasOptional为空、scaleOptional为UINT64、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空或FLOAT32、activationInputOptional为空、out为FLOAT16。
    - x为INT4、weight为INT4、biasOptional为空、scaleOptional为UINT64、offsetOptional为空、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为空或FLOAT32、activationInputOptional为空、out为BFLOAT16。
  - 伪量化场景支持的输入类型为：
    - x为FLOAT16、weight为INT8或INT4、biasOptional为FLOAT16、scaleOptional为空，offsetOptional为空，antiquantScaleOptional为FLOAT16、antiquantOffsetOptional为FLOAT16、perTokenScaleOptional为空、activationInputOptional为空、out为FLOAT16。
    - 伪量化参数antiquantScaleOptional和antiquantOffsetOptional的shape要满足下表（其中g为matmul组数，G为pergroup数，$G_i$为第i个tensor的pergroup数）：

        | 使用场景 | 子场景 | shape限制 |
        |:---------:|:-------:| :-------|
        | 伪量化perchannel | weight单 | $[g, n]$|
        | 伪量化perchannel | weight多 | $[n_i]$|
        | 伪量化pergroup | weight单 | $[g, G, n]$|
        | 伪量化pergroup | weight多 | $[G_i, n_i]$|

    - x为BFLOAT16、weight为INT8或INT4、biasOptional为FLOAT32、scaleOptional为空，offsetOptional为空，antiquantScaleOptional为BFLOAT16、antiquantOffsetOptional为BFLOAT16、perTokenScaleOptional为空、activationInputOptional为空、out为BFLOAT16。
    - x为INT8、weight为INT4、biasOptional为FLOAT32、scaleOptional为UINT64、antiquantScaleOptional为空、antiquantOffsetOptional为空、perTokenScaleOptional为FLOAT32、activationInputOptional为空。此场景支持对称量化和非对称量化：
      - 对称量化场景：
        - 该场景下输出out的dtype为BFLOAT16或FLOAT16
        - 该场景下offsetOptional为空
        - 该场景下仅支持count模式（算子不会检查groupListType的值），k要求为quantGroupSize的整数倍，且要求k <= 18432。其中quantGroupSize为k方向上pergroup量化长度，当前支持quantGroupSize=256。
        - 该场景下scale为pergroup与perchannel离线融合后的结果，shape要求为$[e, quantGroupNum, n]$，其中$quantGroupNum=k \div quantGroupSize$。
        - Bias为计算过程中离线计算的辅助结果，值要求为$8\times weight \times scale$，并在第1维累加，shape要求为$[e, n]$。
        - 该场景下要求n为8的整数倍。
      - 非对称量化场景：
        - 该场景下输出out的dtype为FLOAT16
        - 该场景下仅支持count模式（算子不会检查groupListType的值）。
        - 该场景下{k, n}要求为{7168, 4096}或者{2048, 7168}。
        - scale为pergroup与perchannel离线融合后的结果，shape要求为$[e, 1, n]$。
        - 该场景下offsetOptional不为空。非对称量化offsetOptional为计算过程中离线计算辅助结果，即$antiquantOffset \times scale$，shape要求为$[e, 1, n]$，dtype为FLOAT32。
        - Bias为计算过程中离线计算的辅助结果，值要求为$8\times weight \times scale$，并在第1维累加，shape要求为$[e, n]$。
        - 该场景下要求n为8的整数倍。
  - 量化场景下，若weight的类型为INT4，需满足以下约束（其中g为matmul组数，G为k轴被pergroup划分后的组数）：
    - weight的数据格式为ND时，要求n为8的整数倍。
    - 支持perchannel和pergroup量化。perchannel场景的scale的shape需为$[g, n]$，pergroup场景需为$[g, G, n]$。
    - pergroup场景下，$G$必须要能整除$k$，且$k/G$需为偶数。
    - 该场景仅支持groupType=0(x,weight,y均为单tensor)，actType=0，groupListType=0/1。
    - 该场景不支持weight转置。
  - 伪量化场景下，若weight的类型为INT8，仅支持perchannel模式；若weight的类型为INT4，对称量化支持perchannel和pergroup两种模式。若为pergroup，pergroup数G或$G_i$必须要能整除对应的$k_i$。若weight为多tensor，定义pergroup长度$s_i = k_i / G_i$，要求所有$s_i(i=1,2,...g)$都相等。非对称量化支持perchannel模式。
  - 伪量化场景下若weight的类型为INT4，则weight中每一组tensor的最后一维大小都应是偶数。$weight_i$的最后一维指weight不转置时$weight_i$的N轴或当weight转置时$weight_i$的K轴。并且在pergroup场景下，当weight转置时，要求pergroup长度$s_i$是偶数。

  - 不同groupType支持场景：
    - 量化、伪量化仅支持groupType为-1和0场景。
    - 支持场景中单表示单tensor，多表示多tensor，表示顺序为x，weight，y，例如单多单表示支持x为单tensor，weight多tensor，y单tensor的场景。

        | groupType | 支持场景 | 场景限制 |
        |:---------:|:-------:| :-------|
        | -1 | 多多多 |1）仅支持splitItem为0/1<br>2）x中tensor要求维度一致，支持2-6维，weight中tensor需为2维，y中tensor维度和x保持一致<br>3）groupListOptional必须传空<br>4）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>5）x不支持转置 |
        | 0 | 单单单 |1）仅支持splitItem为2/3<br>2）weight中tensor需为3维，x，y中tensor需为2维<br>3）必须传groupListOptional，且当groupListType为0时，最后一个值与x中tensor的第一维相等，当groupListType为1时，数值的总和与x中tensor的第一维相等，当groupListType为2时，第二列数值的总和与x中tensor的第一维相等<br>4）groupListOptional第1维最大支持1024，即最多支持1024个group<br>5）支持weight转置<br>6）x不支持转置 |
        | 0 | 单多单 |1）仅支持splitItem为2/3<br>2）必须传groupListOptional，且当groupListType为0时，最后一个值与x中tensor的第一维相等，当groupListType为1时，数值的总和与x中tensor的第一维相等且长度最大为128，当groupListType为2时，第二列数值的总和与x中tensor的第一维相等且长度最大为128<br>3）x,weight,y中tensor需为2维<br>4）weight中每个tensor的N轴必须相等<br>5）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>6）x不支持转置 |
        | 0 | 多多单 |1）仅支持splitItem为2/3<br>2）x,weight,y中tensor需为2维<br>3）weight中每个tensor的N轴必须相等<br>4）若传入groupListOptional，当groupListType为0时，groupListOptional的差值需与x中tensor的第一维一一对应，当groupListType为1时，groupListOptional的数值需与x中tensor的第一维一一对应，且长度最大为128，当groupListType为2时，groupListOptional第二列的数值需与x中tensor的第一维一一对应，且长度最大为128<br>5）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>6）x不支持转置 |
        | 2 | 单单单 |1）仅支持splitItem为2/3<br>2）x，weight中tensor需为2维，y中tensor需为3维<br>3）必须传groupListOptional，且当groupListType为0时，最后一个值与x中tensor的第二维相等，当groupListType为1时，数值的总和与x中tensor的第二维相等，当groupListType为2时，第二列数值的总和与x中tensor的第二维相等<br>4）groupListOptional第1维最大支持1024， 即最多支持1024个group<br>5）x必须转置，weight不能转置<br>6）bias必须传空|
        | 2 | 单多多 |1）仅支持splitItem为0/1<br>2）x，weight，y中tensor需为2维<br>3）groupListOptional必须传空<br>4）weight长度最大支持128，即最多支持128个group<br>5）x必须转置，weight不能转置<br>6）原始shape中weight每个tensor的第一维之和不应超过x第一维<br>7）bias必须传空 |

  - x和weight中每一组tensor的最后一维大小都应小于65536。$x_i$的最后一维指当x不转置时$x_i$的K轴或当x转置时$x_i$的M轴。$weight_i$的最后一维指当weight不转置时$weight_i$的N轴或当weight转置时$weight_i$的K轴。
  - 仅量化场景 (per-token)、反量化场景支持激活函数计算。

- <term>Ascend 950PR/Ascend 950DT AI处理器</term>：
  <details>
    <summary><term>静态量化场景约束</term></summary>
      <a id="静态量化场景约束"></a>

  - 以下入参为空：offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、 perTokenScaleOptional、 activationInputOptional
  - 不为空的参数支持的数据类型组合要满足下表：

    |groupType| x       | weight  | biasOptional | scaleOptional | out     |
    |:-------:|:-------:|:-------:| :------      |:-------       | :------ |
    |0|INT8     |INT8     |INT32/null    | UINT64/INT64  |BFLOAT16/FLOAT16|
    |0|INT8     |INT8     |INT32/BFLOAT16/FLOAT32/null    | BFLOAT16/FLOAT32  | BFLOAT16|
    |0|INT8     |INT8     |INT32/FLOAT16/FLOAT32/null    | FLOAT32  |FLOAT16|
    |0|HIFLOAT8     |HIFLOAT8    |null    | UINT64/INT64  |BFLOAT16/FLOAT16/  FLOAT32|
    |0/2|HIFLOAT8     |HIFLOAT8    |null    | FLOAT32  |BFLOAT16/FLOAT16/FLOAT32|
    |0|FLOAT8_E5M2/FLOAT8_E4M3FN   |FLOAT8_E5M2/FLOAT8_E4M3FN   |null    |  UINT64/INT64  |BFLOAT16/FLOAT16/FLOAT32|
    |0/2|FLOAT8_E5M2/FLOAT8_E4M3FN   |FLOAT8_E5M2/FLOAT8_E4M3FN   |null    |  FLOAT32  |BFLOAT16/FLOAT16/FLOAT32|

  - scaleOptional要满足下表（其中g为matmul组数即分组数）：

    |groupType| 使用场景 | shape限制 |
    |:---------:|:---------:| :------ |
    |0/2|weight单tensor|perchannel场景：每个tensor 2维， shape为（g, N）；  pertensor场景：每个tensor 2维或1维，shape为 （g, 1）或（g,）|

  </details>

  <details>
    <summary><term>动态量化（T-T && T-C && K-T && K-C量化）场景约束</term></summary>
      <a id="动态量化（T-T && T-C && K-T && K-C量化）场景约束"></a>

  - 动态量化（T-T && T-C && K-T && K-C量化）场景支持的输入类型为：
    - 以下入参为空：offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、 activationInputOptional
    - 不为空的参数支持的数据类型组合要满足下表：

        |groupType| x       | weight  | biasOptional | scaleOptional |  perTokenScaleOptional |out     |
        |:-------:|:-------:|:-------:| :------      |:-------    | :------   |   :------ |
        |0|INT8  |INT8| INT32/BFLOAT16/FLOAT32/null     |BFLOAT16/FLOAT32    |  FLOAT32   | BFLOAT16 |
        |0|INT8  |INT8| INT32/FLOAT16/FLOAT32/null     |FLOAT32    | FLOAT32   |  FLOAT16 |
        |0/2|HIFLOAT8  |HIFLOAT8| null     |FLOAT32    | FLOAT32   | BFLOAT16/  FLOAT16/FLOAT32 |
        |0/2|FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| null     |  FLOAT32    | FLOAT32   | BFLOAT16/  FLOAT16/FLOAT32 |

    - scaleOptional要满足下表（其中g为matmul组数即分组数），推荐在pertensor场景scaleOptional的shape使用（g,），防止与G-B量化模式混淆：

        | groupType | 使用场景 | shape限制 |
        |:---------:|:---------:| :------ |
        |0/2|weight单tensor|perchannel场景：每个tensor 2维，shape为（g, N）； pertensor场景：每个tensor 2维或1维，shape为（g, 1）或（g,）|

    - perTokenScaleOptional要满足下表：

        | groupType | 使用场景 | shape限制 |
        |:---------:|:---------:| :------ |
        |0|x单tensor|pertoken场景：每个tensor 1维，shape为（M,）；pertensor场景：每个tensor 2维或1维，shape为（g, 1）或  （g,），输入为INT8时不支持pertensor场景|
        |2|x单tensor|pertoken场景：每个tensor 2维，shape为（g, M）；pertensor场景：每个tensor 2维或1维，shape为（g, 1）  或（g,）|

  </details>

  <details>
    <summary><term>动态量化（mx量化）场景约束</term></summary>
      <a id="动态量化（mx量化）场景约束"></a>

  - 以下入参为空：offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、 activationInputOptional
  - 计算公式中量化block size为：gsM = gsN = 1，gsK = 32。mx量化是特殊的pergroup量 化。
  - 不为空的参数支持的数据类型组合要满足下表：

      |groupType| x       | weight  | biasOptional | scaleOptional |  perTokenScaleOptional |out     |
      |:-------:|:-------:|:-------:|:-------:| :-------    | :------   | :------ |
      |0/2|FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| null|   FLOAT8_E8M0    | FLOAT8_E8M0    | BFLOAT16/FLOAT16/FLOAT32 |
      |0|FLOAT4_E2M1/FLOAT4_E1M2 |FLOAT4_E2M1/FLOAT4_E1M2| FLOAT32/null |   FLOAT8_E8M0    | FLOAT8_E8M0    |   BFLOAT16/FLOAT16/FLOAT32 |

  - scaleOptional要满足下表（其中g为matmul组数即分组数，g\_i为第i个分组（下标从0开  始））：

      |groupType| 使用场景 | shape限制 |
      |:---------:|:---------:| :------ |
      |0|weight单tensor|每个tensor 4维，当weight转置时，shape为(g, N, ceil(K / 64), 2)；当weight不转置时，shape为(g, ceil(K / 64), N, 2)|
      |2|weight单tensor|每个tensor 3维，shape为((K / 64) + g, N, 2)，scale\_i起始地 址偏移为((K\_0 + K\_1 + ...+ K\_ {i-1})/ 64 + g\_i)*N* 2，即scale_0的起始地 址偏移为0，scale_1的起始地址偏移为（K\_0 / 64 + 1）*N* 2， scale_2的起始地址偏移为((K\_0 + K\_1) / 64 + 2) *N* 2, 依此类推|

  - perTokenScaleOptional要满足下表：

      |groupType| 使用场景 | shape限制 |
      |:---------:|:---------:| :------ |
      |0|x单tensor|每个tensor 3维，shape为（M, ceil(K / 64), 2）|
      |2|x单tensor|每个tensor 3维，shape为((K / 64) + g, M, 2), 起始地址偏移与scale 同理|

  - 对于mx量化中输入x为FLOAT4_E2M1/FLOAT4_E1M2时，需要满足K为偶数且K不为2。当weight 非转置时还需满足N为偶数。
  </details>

  <details>
    <summary><term>动态量化（G-B量化）场景约束</term></summary>
      <a id="动态量化（G-B量化）场景约束"></a>

  - 动态量化（G-B量化）场景支持的数据类型为：
  - 以下入参为空：biasOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、activationInputOptional
  - 计算公式量化block size为：当前仅支持gsM = 1， gsN = gsK = 128。
  - 不为空的参数支持的数据类型组合要满足下表：

      |groupType| x       | weight  |  scaleOptional | perTokenScaleOptional |  out     |
      |:-------:|:-------:|:-------:| :-------    | :------   | :------ |
      |0/2|HIFLOAT8  |HIFLOAT8| FLOAT32    | FLOAT32    | BFLOAT16/FLOAT16/ FLOAT32 |
      |0/2|FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| FLOAT32    |  FLOAT32    | BFLOAT16/FLOAT16/FLOAT32 |

  - scaleOptional要满足下表（其中g为matmul组数即分组数，g\_i为第i个分组（下标从0开  始））：

      |groupType| 使用场景 | shape限制 |
      |:---------:|:---------:| :------ |
      |0|weight单tensor|每个tensor 3维，weight转置时shape为（g, ceil(N / gsN), ceil (K / gsK)），weight非转置时shape为（g, ceil(K / gsK), ceil(N / gsN)）|
      |2|weight单tensor|每个tensor 2维，shape为（K / gsK + g, ceil(N / gsN)），scale\_i地址偏移为（(K\_0 + K\_1 + ...+   K\_{i-1})/ gsK + g\_i）*ceil(N /  gsN)，即scale\_0的起始地址偏移为0，scale\_1的起始地址偏移为（K\_0 / gsK + 1）* ceil(N / gsN)， scale_2的起始地址偏移为((K\_0 + K\_1) / gsK + 2) * ceil(N / gsN), 依此类推|

  - perTokenScaleOptional要满足下表：

      |groupType| 使用场景 | shape限制 |
      |:---------:|:---------:| :------ |
      |0|x单tensor|每个tensor 2维，shape为（M, ceil(K / gsK)）|
      |2|x单tensor|每个tensor 2维，shape为（K / gsK + g, M），per\_token\_scale\_i地址偏移为（(K\_0 + K\_1 + ...+ K\_{i-1}) / gsK + g\_i）\* M，即  per\_token\_scale\_0的起始地址偏移为0，per\_token\_scale\_1的起始地址偏移为（K\_0 / gsK + 1）\* M， per\_token\_scale\_2的起始地址偏移为((K\_0 + K\_1) / gsK + 2) * M, 依此类推|

  - 动态量化特殊场景处理：
    - 在动态量化场景M分组或K分组情况下，当N等于1且scaleOptional的shape为（g, 1）时，weight既可以pertensor量化也可以perchannel量化时, 优先选择pertensor量化模式。
    - 在动态量化场景M分组情况下，当g = M且perTokenScaleOptional的shape为（g,）时，x选择pertoken量化模式；当g = M，K <= 128且perTokenScaleOptional的shape 为（g, 1）时，根据weight的量化模式选择x的量化模式（weight如果是perchannel或者pertensor量化，x选择pertensor量化；weight如果是perblock量化，x选择pergroup量化）。
    - 在动态量化场景K分组情况下，K小于128，N小于等于128且scaleOptional的shape为（g, 1）时，按照现有量化模式区分规则，既可以为非pergroup量化，又可以为G-B量化，此种场景现一律按照G-B量化处理。
    - 在动态量化场景K分组情况下，当M等于1且perTokenScaleOptional的shape为（g, 1）时，x既可以pertoken量化也可以pertensor量化时, 优先选择pertensor量化模式。
    - 在动态量化场景K分组情况下，K小于128, M等于1且perTokenScaleOptional的shape为（g, 1）时，如果N小于等于128，x则选择pergroup量化；如果N大于128，根据weight的量化模式选择x的量化模式（weight如果是perchannel或者pertensor量化，x选择 pertensor量化；weight  如果是perblock量化，x选择pergroup量化）。
    - 在动态量化场景K分组情况下，K小于128, M不等于1时，如果N小于等于128，x则选择pergroup量化；如果N大于128，根据weight的量化模式选择x的量化模式（weight如果是 perchannel或者pertensor量化，x选择pertoken量化；weight如果是perblock量化，x选择pergroup量化）。
  </details>

  <details>
    <summary><term>非量化场景约束</term></summary>
      <a id="非量化场景约束"></a>

  - 非量化场景支持的数据类型为：
    - 以下入参为空：scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、perTokenScaleOptional、activationInputOptional、activationQuantScaleOptional、activationQuantOffsetOptional、actType、activationFeatureOutOptional
    - 不为空的参数支持的数据类型组合要满足下表

        |groupType| x       | weight  | biasOptional | out     |
        |:-------:|:-------:|:-------:| :------      |:------ |
        |-1/0/2   |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
        |-1/0/2   |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|

  </details>

  <details>
    <summary><term>伪量化场景约束</term></summary>
      <a id="伪量化场景约束"></a>

  - 伪量化场景支持的数据类型为：
    - 以下入参为空：scaleOptional、offsetOptional、perTokenScaleOptional、activationInputOptional、activationQuantScaleOptional、activationQuantOffsetOptional
    - 不为空的参数支持的数据类型组合要满足下表

        |groupType| x       | weight  | biasOptional |antiquantScaleOptional| antiquantOffsetOptional| out     |
        |:-------:|:-------:|:-------:| :------      |:------|:------|:------|
        |-1/0   |BFLOAT16     |INT8/INT4 |BFLOAT16/FLOAT32/null| BFLOAT16 | BFLOAT16/  null | BFLOAT16 |
        |-1/0   |FLOAT16     |INT8/INT4     |FLOAT16/null    | FLOAT16 | FLOAT16/null |  FLOAT16 |
        |0   |BFLOAT16     |FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 |BFLOAT16/FLOAT32/ null| BFLOAT16 | null | BFLOAT16 |
        |0   |FLOAT16     |FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 |FLOAT16/null    |   FLOAT16 | null | FLOAT16 |

    - 当weight的数据类型为FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8时，antiquantOffsetOptional仅支持传入空指针或空tensorList，weight仅支持转置。
    - 若weight的类型为INT4，则weight中每一组tensor的最后一维大小都应是偶数。$weight_i$的最后一维指weight不转置时$weight_i$的N轴或当weight转置时$weight_i$的K轴。
    - antiquantScaleOptional和非空的biasOptional、antiquantOffsetOptional要满足下表（其中g为matmul组数即分组数）：

        |groupType| 使用场景 | shape限制 |
        |:---------:|:---------:| :------ |
        |-1|weight多tensor|每个tensor 1维，shape为（$n_i$），不允许存在一个tensorList中部分tensor的shape为（$n_i$）部分tensor为空的情况 |
        |0|weight单tensor|每个tensor 2维，shape为（g, N）|

  </details>

  <details>
    <summary><term>不同groupType约束</term></summary>
      <a id="不同groupType约束"></a>

  - 不同groupType支持场景:
    - 支持场景中单表示单tensor，多表示多tensor，表示顺序为x，weight，out，例如单多单表示支持x为单tensor，weight多 tensor，out单tensor的场景。

        | groupType | 支持场景 | 场景限制 |
        |:---------:|:-------:| :------ |
        | -1 | 多多多 |1）仅支持splitItem为0/1<br>2）非量化x，out中tensor需为2维，shape分别为（$m_i$, $k_i$）和（$m_i$, $n_i$）；伪量化场景x中tensor要求维度一致，支持2-6维，y中tensor维度和x保持一致；weight中tensor需为2维，shape为（$n_i$, $k_i$）或（$k_i$, $n_i$）；bias中tensor需为1维，shape为（$n_i$）<br>3） groupListOptional必须传空<br>4）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>5）x不支持转置<br>6）仅支持非量化和伪量化  <br>7）仅支持ND进ND出 <br>|
        | 0 | 单单单 |1）仅支持splitItem为2/3<br>2）weight中tensor需为3维，shape为（g, N, K）或（g, K, N）；x，out中tensor需为2维，shape分别为（M, K）和（M, N）；bias中tensor需为2维，shape为（g, N）<br>3）必须传groupListOptional，且当groupListType为0时，最后一个值不大于x中tensor的第一维，当groupListType为1时，数值的总和不大于x中tensor的第一维<br>4）groupListOptional第1维最大支持1024，即最多支持1024个group<br>5）支持x不转置，weight转置、不转置均支持<br>6）x与weight为int8时支持weight为FRACTAL_NZ数据格式，其余场景仅支持ND进；仅支持ND出<br>|
        | 0 | 单多单 |1）仅支持splitItem为2/3<br>2）必须传groupListOptional，且当groupListType为0时，最后一个值与x中tensor的第一维相等，当groupListType为1时，数值的总与x中tensor的第一维相等，长度最大128<br>3）x，out中tensor需为2维，shape分别为（M, K）和（M, N）；weight中tensor需为2维，shape为（N, K）或（K, N）；bias中tensor需为1维，shape为（N）<br>4）weight中每个tensor的N轴必须相等<br>5）支持weight转置，但weight的tensorList中每tensor是否转置需保持统一<br>6）x不支持转置<br>7）仅支持非量化<br>8）仅支持ND进ND出<br> |
        | 0 | 多多单 |1）仅支持splitItem为2/3<br>2）x，out中tensor需为2维， shape分别为（M, K）和（M, N）；weight中tensor需为2维，shape为（N, K）或（K, N）；bias中tensor需为1维，shape为（N）<br>3）weight中每个tensor的N轴必须相等<br>4）若传入groupListOptional，当groupListType为0时，groupListOptional的差值需与x中tensor的第一维一一对应，当groupListType为1时，groupListOptional的数值需与x中tensor的第一维一一对应，且长度最大为128<br>5）支weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>6）x不支持转置<br>7）仅支持非量化<br>8）仅支持ND进ND出<br> |
        | 2 | 单单单 |1）仅支持splitItem为2/3<br>2）x，weight中tensor需为2维，shape分别为（K, M）和（K, N）；out中tensor需为3维, shape为（g, M, N）<br>3）必须传groupListOptional，且当groupListType为0时，最后一个值不大于x中tensor的第一维，当groupListType为1时，数值的总和不大于x中tensor的第一维<br>4）groupListOptional第1维最大支持1024，即最多支持1024个group<br>5）仅支持x转置且weight不转置<br>6）仅支持非量化和量化<br>7）仅支持ND进ND出|

  </details>

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

weight为ND时调用示例
  ```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_v4.h"

#define CHECK_RET(cond, return_expr) \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
      printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor_New(const std::vector<int64_t>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    std::vector<T> hostData(size, 0);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}


int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr,
                        aclDataType dataType, aclTensorList** tensor) {
    int size = shapes.size();
    aclTensor* tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor<uint16_t>(shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}


int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<std::vector<int64_t>> xShape = {{512, 256}};
    std::vector<std::vector<int64_t>> weightShape= {{2, 256, 256}};
    std::vector<std::vector<int64_t>> biasShape = {{2, 256}};
    std::vector<std::vector<int64_t>> yShape = {{512, 256}};
    std::vector<int64_t> groupListShape = {{2}};
    std::vector<int64_t> groupListData = {256, 512};
    void* xDeviceAddr[1];
    void* weightDeviceAddr[1];
    void* biasDeviceAddr[1];
    void* yDeviceAddr[1];
    void* groupListDeviceAddr;
    aclTensorList* x = nullptr;
    aclTensorList* weight = nullptr;
    aclTensorList* bias = nullptr;
    aclTensor* groupedList = nullptr;
    aclTensorList* scale = nullptr;
    aclTensorList* offset = nullptr;
    aclTensorList* antiquantScale = nullptr;
    aclTensorList* antiquantOffset = nullptr;
    aclTensorList* perTokenScale = nullptr;
    aclTensorList* activationInput = nullptr;
    aclTensorList* activationQuantScale = nullptr;
    aclTensorList* activationQuantOffset = nullptr;
    aclTensorList* out = nullptr;
    aclTensorList* activationFeatureOut = nullptr;
    aclTensorList* dynQuantScaleOut = nullptr;
    int64_t splitItem = 3;
    int64_t groupType = 0;
    int64_t groupListType = 0;
    int64_t actType = 0;

    // 创建x aclTensorList
    ret = CreateAclTensorList(xShape, xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensorList
    ret = CreateAclTensorList(weightShape, weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensorList
    ret = CreateAclTensorList(biasShape, biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y aclTensorList
    ret = CreateAclTensorList(yShape, yDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建group_list aclTensor
    ret = CreateAclTensor_New<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupedList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. 调用CANN算子库API
    // 调用aclnnGroupedMatmulV4第一段接口
    ret = aclnnGroupedMatmulV4GetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupedList, activationInput, activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, out, activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGroupedMatmulV4第二段接口
    ret = aclnnGroupedMatmulV4(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmul failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    for (int i = 0; i < 1; i++) {
        auto size = GetShapeSize(yShape[i]);
        std::vector<uint16_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr[i],
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
        }
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensorList(x);
    aclDestroyTensorList(weight);
    aclDestroyTensorList(bias);
    aclDestroyTensorList(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    for (int i = 0; i < 1; i++) {
        aclrtFree(xDeviceAddr[i]);
        aclrtFree(weightDeviceAddr[i]);
        aclrtFree(biasDeviceAddr[i]);
        aclrtFree(yDeviceAddr[i]);
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
  ```
weightNz特性调用示例
```c++
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_v4.h"
#include "aclnnop/aclnn_npu_format_cast.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1L;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1L);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
template <typename T>
int CreateAclTensorList(const std::vector<T> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                        void **deviceAddr, aclDataType dataType, aclTensorList **tensor)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; i++) {
        int ret = CreateAclTensor(hostData, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}
template <typename T>
int CreateAclTensorWithFormat(const std::vector<T> &hostData, const std::vector<int64_t> &shape, int64_t **storageShape,
                              uint64_t *storageShapeSize, void **deviceAddr, aclDataType dataType, aclTensor **tensor,
                              aclFormat format)
{
    auto size = hostData.size() * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, *storageShape,
                              *storageShapeSize, *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorNz(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor, aclrtStream &stream)
{
    void *srcDeviceAddr = nullptr;
    aclTensor *srcTensor = nullptr;
    auto size = hostData.size() * sizeof(T);

    auto ret = aclrtMalloc(&srcDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(srcDeviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1L);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    srcTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), srcDeviceAddr);

    int64_t *dstShape = nullptr;
    uint64_t dstShapeSize = 0;
    int actualFormat;
    ret = aclnnNpuFormatCastCalculateSizeAndFormat(srcTensor, 29, aclFormat::ACL_FORMAT_ND, &dstShape, &dstShapeSize,
                                                   &actualFormat);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret);
              return ret);

    aclTensor *dstTensor = nullptr;
    void *dstDeviceAddr = nullptr;

    uint64_t tensorSize = 1;
    for (int64_t i = 0; i < dstShape[i]; i++) {
        tensorSize *= dstShape[i];
    }
    ret = aclrtMalloc(&dstDeviceAddr, tensorSize * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    int64_t weightLen = shape.size();
    for (int64_t i = 0; i < weightLen + 2; i++) {
        tensorSize = tensorSize * dstShape[i];
    }
    std::vector<uint16_t> dstTensorHostData(tensorSize, 0);

    ret = CreateAclTensorWithFormat(dstTensorHostData, shape, &dstShape, &dstShapeSize, &dstDeviceAddr, dataType,
                                    &dstTensor, static_cast<aclFormat>(actualFormat));
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(workspaceAddr, aclrtFree);

    // 调用aclnnNpuFormatCastGetWorkspaceSize第一段接口
    ret = aclnnNpuFormatCastGetWorkspaceSize(srcTensor, dstTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnNpuFormatCastGetWorkspaceSize第二段接口
    ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    *tensor = dstTensor;
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensorListNz(const std::vector<T> &hostData, const std::vector<std::vector<int64_t>> &shapes,
                          void **deviceAddr, aclDataType dataType, aclTensorList **tensor, aclrtStream &stream)
{
    int size = shapes.size();
    aclTensor *tensors[size];
    for (int i = 0; i < size; ++i) {
        int ret = CreateAclTensorNz<T>(hostData, shapes[i], deviceAddr + i, dataType, tensors + i, stream);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnGourpedMatmulTest(int32_t deviceId, aclrtStream &stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t m = 512L;
    int64_t k = 256L;
    int64_t n = 4L;
    int64_t groupnum = 2L;
    std::vector<std::vector<int64_t>> xShape = {{m, k}};
    std::vector<std::vector<int64_t>> weightShape = {{groupnum, k, n}};
    std::vector<std::vector<int64_t>> biasShape = {{groupnum, n}};
    std::vector<std::vector<int64_t>> scaleShape = {{groupnum, n}};
    std::vector<std::vector<int64_t>> pertokenShape = {{
        m,
    }};
    std::vector<std::vector<int64_t>> yShape = {{m, n}};
    std::vector<int64_t> groupListShape = {{groupnum}};
    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *biasDeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *pertokenDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr;
    void *groupListDeviceAddr = nullptr;
    aclTensorList *x = nullptr;
    aclTensorList *weight = nullptr;
    aclTensorList *bias = nullptr;
    aclTensor *groupedList = nullptr;
    aclTensorList *scale = nullptr;
    aclTensorList *offset = nullptr;
    aclTensorList *antiquantScale = nullptr;
    aclTensorList *antiquantOffset = nullptr;
    aclTensorList *perTokenScale = nullptr;
    aclTensorList *activationInput = nullptr;
    aclTensorList *activationQuantScale = nullptr;
    aclTensorList *activationQuantOffset = nullptr;
    aclTensorList *out = nullptr;
    aclTensorList *activationFeatureOut = nullptr;
    aclTensorList *dynQuantScaleOut = nullptr;
    int64_t splitItem = 3L;
    int64_t groupType = 0L;
    int64_t groupListType = 0L;
    int64_t actType = 0L;
    std::vector<int8_t> xHostData(m * k, 10);
    std::vector<int8_t> weightHostData(groupnum * k * n, 10);
    std::vector<uint16_t> yHostData(m * n, 0);
    std::vector<int64_t> groupListData = {256, 512};
    std::vector<int8_t> scaleHostData(groupnum * n, 1);
    std::vector<int8_t> biasHostData(groupnum * n, 1);
    std::vector<int8_t> pertokenHostData(m, 1);

    // 创建x aclTensorList
    ret = CreateAclTensorList(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> xTensorPtr(x, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    // 创建weight aclTensorList
    ret = CreateAclTensorListNz(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> weightTensorPtr(weight,
                                                                                           aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    // 创建scale aclTensorList
    ret = CreateAclTensorList(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_BF16, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> scaleTensorPtr(scale, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    // 创建pertoken aclTensorList
    ret = CreateAclTensorList(pertokenHostData, pertokenShape, &pertokenDeviceAddr, aclDataType::ACL_FLOAT,
                              &perTokenScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> pertokenTensorPtr(perTokenScale,
                                                                                             aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> pertokenDeviceAddrPtr(pertokenDeviceAddr, aclrtFree);
    // 创建y aclTensorList
    ret = CreateAclTensorList(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_BF16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensorList, aclnnStatus (*)(const aclTensorList *)> yTensorPtr(out, aclDestroyTensorList);
    std::unique_ptr<void, aclError (*)(void *)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);

    // 创建group_list aclTensorList
    ret = CreateAclTensor<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                                   &groupedList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupedList, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(workspaceAddr, aclrtFree);

    // 3. 调用CANN算子库API
    // 调用aclnnGroupedMatmulV4第一段接口
    ret = aclnnGroupedMatmulV4GetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset,
                                               perTokenScale, groupedList, activationInput, activationQuantScale,
                                               activationQuantOffset, splitItem, groupType, groupListType, actType, out,
                                               activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV4GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnGroupedMatmulV4第二段接口
    ret = aclnnGroupedMatmulV4(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV4 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    for (int i = 0; i < 1; i++) {
        auto size = GetShapeSize(yShape[i]);
        std::vector<uint16_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t j = 0; j < size; j++) {
            LOG_PRINT("result[%ld] is: %d\n", j, resultData[j]);
        }
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnGourpedMatmulTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```