# aclnnGroupedMatmulV5

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT AI处理器</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明

- 接口功能：实现分组矩阵乘计算。如$y_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i], i=1...g$，其中g为分组个数。当前支持m轴和k轴分组，对应的功能为：

  - m轴分组：$k_i$、$n_i$各组相同，$m_i$可以不相同。
  - k轴分组：$m_i$、$n_i$各组相同，$k_i$可以不相同。

- 基础计算公式如下（详细公式请参见[计算公式](#计算公式)）：

  $$
  y_i=x_i\times weight_i + bias_i
  $$

- 版本演进：

  |版本变化      | Atlas A2 训练系列产品/Atlas A2 推理系列产品<br>Atlas A3 训练系列产品/Atlas A3 推理系列产品 |Ascend 950PR/Ascend 950DT AI处理器|
  |---------|---------|----------------|
  |V4 -> V5|  增加可选参数tuningConfigOptional，调优参数。数组中第一个值表示各个专家处理的token数的预期值，算子tiling时会按照该预期值进行最优tiling。   |  /  |
  |V1 -> V4|     支持不同分组轴，由groupType表示。<br />非量化场景，支持x，weight转置（转置指若shape为[M,K]时，则stride为[1, M],数据排布为[K,M]的场景）。<br />量化、伪量化场景，支持weight转置，支持weight为单tensor。<br />x、weight、y都为单tensor非量化场景，支持x，weight输入都为float32类型。<br />支持静态量化（pertensor+perchannel）（量化方式请参见[量化介绍](../../../docs/zh/context/量化介绍.md)，下同）BFLOAT16和FLOAT16输出，带激活及不带激活场景。<br />支持动态量化（pertoken+perchannel）BFLOAT16和FLOAT16输出，带激活及不带激活场景。<br />支持伪量化weight是INT4的输入，不带激活场景，支持perchannel和pergroup两种模式。     |支持不同分组轴，由groupType表示。<br />非量化场景，支持x，weight转置（转置指若shape为[M,K]时，则stride为[1, M],数据排布为[K,M]的场景）。<br />支持静态量化（1.pertensor-perchannel；2.pertensor-pertensor）BFLOAT16，FLOAT16和FLOAT32输出，带bias，不带激活场景。<br />支持动态量化（1.pertoken-perchannel；2.pertoken-pertensor；3.pertensor-pertensor；4.mx量化；5.pergroup-perblock）BFLOAT16，FLOAT16和FLOAT32输出，带bias，不带激活场景。<br />支持伪量化weight是INT4、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8的输入，不带激活场景，仅支持perchannel模式。|

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnGroupedMatmulV5GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupedMatmulV5”接口执行计算。

```c++
aclnnStatus aclnnGroupedMatmulV5GetWorkspaceSize(
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
    int64_t              splitItem,
    int64_t              groupType,
    int64_t              groupListType,
    int64_t              actType,
    aclIntArray         *tuningConfigOptional,
    aclTensorList       *out,
    aclTensorList       *activationFeatureOutOptional,
    aclTensorList       *dynQuantScaleOutOptional,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```

```c++
aclnnStatus aclnnGroupedMatmulV5(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulV5GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px;">
  <colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
      <col style="width: 212px">
      <col style="width: 100px">
      <col style="width: 190px">
      <col style="width: 145px">
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
          <th>非连续Tensor</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td>x</td>
          <td>输入</td>
          <td>公式中的输入<code>x</code>。</td>
          <td>最多支持128个tensor。</td>
          <td>FLOAT<sup>1</sup>、FLOAT16、INT16<sup>1</sup>、INT8、INT4<sup>1</sup>、BFLOAT16、FLOAT8_E5M2<sup>2</sup>、FLOAT8_E4M3FN<sup>2</sup>、HIFLOAT8<sup>2</sup></td>
          <td>ND</td>
          <td>2-6</td>
          <td>√</td>
      </tr>
      <tr>
          <td>weight</td>
          <td>输入</td>
          <td>公式中的<code>weight</code>。</td>
          <td>最多支持128个tensor。</td>
          <td>FLOAT<sup>1</sup>、FLOAT16、INT16<sup>1</sup>、INT8、INT4、BFLOAT16、FLOAT8_E5M2<sup>2</sup>、FLOAT8_E4M3FN<sup>2</sup>、HIFLOAT8<sup>2</sup></td>
          <td>ND/NZ</td>
          <td>2-3</td>
          <td>√</td>
      </tr>
      <tr>
          <td>biasOptional</td>
          <td>可选输入</td>
          <td>公式中的<code>bias</code>。</td>
          <td>长度与weight相同。</td>
          <td>FLOAT、FLOAT16、INT32、BFLOAT16<sup>2</sup></td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
      </tr>
      <tr>
          <td>scaleOptional</td>
          <td>可选输入</td>
          <td>公式中的<code>scale</code>，代表量化参数中的缩放因子。</td>
          <td>一般情况下，长度与weight相同。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
          <td>FLOAT、UINT64、BFLOAT16、FLOAT8_E8M0<sup>2</sup>、INT64<sup>2</sup></td>
          <td>ND</td>
          <td>1-3</td>
          <td>√</td>
      </tr>
      <tr>
          <td>offsetOptional</td>
          <td>可选输入</td>
          <td>公式中的<code>offset</code>，代表量化参数中的偏移量。</td>
          <td>长度与weight相同。</td>
          <td>FLOAT</td>
          <td>ND</td>
          <td>3</td>
          <td>√</td>
      </tr>
      <tr>
          <td>antiquantScaleOptional</td>
          <td>可选输入</td>
          <td>公式中的<code>antiquant_scale</code>，代表伪量化参数中的缩放因子。</td>
          <td>长度与weight相同。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>1-3</td>
          <td>√</td>
      </tr>
      <tr>
          <td>antiquantOffsetOptional</td>
          <td>可选输入</td>
          <td>公式中的<code>antiquant_offset</code>，代表伪量化参数中的缩放因子。</td>
          <td>长度与weight相同。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>1-3</td>
          <td>√</td>
      </tr>
      <tr>
          <td>perTokenScaleOptional</td>
          <td>可选输入</td>
          <td>公式中的<code>per_token_scale</code>，代表量化参数中的由x量化引入的缩放因子。</td>
          <td>一般情况下，只支持1维且长度与x的M相同。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
          <td>FLOAT、FLOAT8_E8M0<sup>2</sup></td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
      </tr>
      <tr>
          <td>groupListOptional</td>
          <td>可选输入</td>
          <td>代表输入和输出分组轴方向的matmul大小分布。</td>
          <td>根据groupListType输入不同格式数据。</td>
          <td>INT64</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
      </tr>
      <tr>
          <td>activationInputOptional</td>
          <td>可选输入</td>
          <td>代表激活函数的反向输入，当前只支持传入nullptr。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>activationQuantScaleOptional</td>
          <td>可选输入</td>
          <td>当前只支持传入nullptr。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>activationQuantOffsetOptional</td>
          <td>可选输入</td>
          <td>当前只支持传入nullptr。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>splitItem</td>
          <td>输入</td>
          <td>代表输出是否要做tensor切分。</td>
          <td>0/1代表输出为多tensor；2/3代表输出为单tensor。aclnn接口不感知0/1的差异，2/3同理。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>groupType</td>
          <td>输入</td>
          <td>代表需要分组的轴。</td>
          <td>取值范围-1、0、2。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>groupListType</td>
          <td>输入</td>
          <td>代表groupList输入的分组方式。</td>
          <td>取值范围0-2。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>actType</td>
          <td>输入</td>
          <td>代表激活函数类型。</td>
          <td>取值范围为0-5。综合约束请参见<a href="#约束说明">约束说明</a>。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>tuningConfigOptional</td>
          <td>可选输入</td>
          <td>第一个数代表各个专家处理的token数的预期值，用于优化tiling。<br>第二个数代表A8W4可选使能weight，先转置的NZ格式。<br>第三个数代表允许额外使用的内存空间。详见<a href="#约束说明">约束说明</a>。</td>
          <td>兼容历史版本，用户如不使用该参数，不传（即为nullptr）即可。</td>
          <td>INT64</td>
          <td>-</td>
          <td>3</td>
          <td>-</td>
      </tr>
      <tr>
          <td>out</td>
          <td>输出</td>
          <td>公式中的输出<code>y</code>。</td>
          <td>最多支持128个tensor。</td>
          <td>FLOAT、FLOAT16、INT32<sup>1</sup>、INT8<sup>1</sup>、BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td></td>
      </tr>
      <tr>
          <td>activationFeatureOutOptional</td>
          <td>输出</td>
          <td>激活函数的输入数据，当前只支持传入nullptr。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dynQuantScaleOutOptional</td>
          <td>输出</td>
          <td>当前只支持传入nullptr。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>workspaceSize</td>
          <td>输出</td>
          <td>返回需要在Device侧申请的workspace大小。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>executor</td>
          <td>输出</td>
          <td>返回op执行器，包含了算子计算流程。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
  </tbody>
  </table>


  - <term>Ascend 950PR/Ascend 950DT AI处理器</term>：

    - 上表数据类型列中的角标“1”代表该系列不支持的数据类型。
    - 输入参数x、weight均不支持FLOAT、INT16类型，且x不支持int4类型；
    - 输出参数out不支持INT32、INT8类型。

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    - 上表数据类型列中的角标“2”代表该系列不支持的数据类型。
    - 不支持FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8、FLOAT8_E8M0类型。
    - 输入参数biasOptional不支持BFLOAT16；
    - 输入参数scaleOptional不支持INT64类型。

  - <term>Atlas 推理系列产品</term>：仅支持x、weight、out均为FLOAT16的场景，其中weight仅支持NZ数据格式。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一阶段接口完成入参校验，出现以下场景时报错。

  <table>
    <thead>
      <tr>
        <th style="width: 250px">返回值</th>
        <th style="width: 130px">错误码</th>
        <th style="width: 850px">描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="4"> ACLNN_ERR_PARAM_NULLPTR </td>
        <td rowspan="4"> 161001 </td>
        <td>传入参数是必选输入、输出或者必选属性，且是空指针。</td>
      </tr>
      <tr>
        <td>传入参数weight的元素存在空指针。</td>
      </tr>
      <tr>
        <td>传入参数x的元素为空指针，且传出参数out的元素不为空指针。</td>
      </tr>
      <tr>
        <td>传入参数x的元素不为空指针，且传出参数out的元素为空指针。</td>
      </tr>
      <tr>
        <td rowspan="6"> ACLNN_ERR_PARAM_INVALID </td>
        <td rowspan="6"> 161002 </td>
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
      <tr>
        <td>传入参数tuningConfigOptional的元素为负数，或者大于x的行数m。</td>
      </tr>
    </tbody>
  </table>

## aclnnGroupedMatmulV5

- **参数说明：**

  |参数名| 输入/输出   |    描述|
  |-------|---------|----------------|
  |workspace|输入|在Device侧申请的workspace内存地址。|
  |workspaceSize|输入|在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulV5GetWorkspaceSize获取。|
  |executor|输入|op执行器，包含了算子计算流程。|
  |stream|输入|指定执行任务的Stream。|

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 场景分类

<a id="场景分类"></a>

- GroupedMatmul算子根据计算过程中对输入数据（x, weight）和输出矩阵（out）的精度处理方式，其支持场景主要分为：非量化，伪量化，全量化。

  - <term>Atlas 推理系列产品</term>：

    |场景名|    x    |    weight       |   out | 约束说明|计算公式|
    |---------|---------|----------------|--------|--------|--|
    |非量化|FLOAT16|FLOAT16|FLOAT16|[非量化场景约束](#atlas推理系列产品)|[计算公式](#非量化场景)|

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    |场景名|    x    |    weight      |   out | 约束说明|计算公式|
    |---------|---------|----------------|--------|--------|--|
    |非量化|FLOAT32|FLOAT32|FLOAT32|[非量化场景约束](#非量化场景约束)|[计算公式](#非量化场景)|
    |非量化|BFLOAT16|BFLOAT16|BFLOAT16|[非量化场景约束](#非量化场景约束)|[计算公式](#非量化场景)|
    |非量化|FLOAT16|FLOAT16|FLOAT16|[非量化场景约束](#非量化场景约束)|[计算公式](#非量化场景)|
    |全量化-A8W8|INT8|INT8|BFLOAT16/FLOAT16/INT32/INT8|[A8W8场景约束](#a8w8场景约束)|[计算公式](#全量化场景)|
    |全量化-A4W4|INT4|INT4|BFLOAT16/FLOAT16|[A4W4场景约束](#a4w4场景约束)|[计算公式](#全量化场景)|
    |伪量化-A8W4|INT8|INT4|BFLOAT16/FLOAT16|[A8W4场景约束](#a8w4场景约束)|[计算公式](#a8w4伪量化场景)|
    |伪量化-A16W8|BFLOAT16/FLOAT16|INT8|BFLOAT16/FLOAT16|[A16W8场景约束](#a16w4场景约束)|[计算公式](#伪量化场景)|
    |伪量化-A16W4|BFLOAT16/FLOAT16|INT4|BFLOAT16/FLOAT16|[A16W4场景约束](#a16w4场景约束)|[计算公式](#伪量化场景)|

  - <term>Ascend 950PR/Ascend 950DT AI处理器</term>：

    详见[Ascend 950PR/Ascend 950DT AI处理器](#昇腾910_95AI处理器)
<a id="计算公式"></a>
- 计算公式
  <a id="非量化场景"></a>

  - **非量化场景：**

    $$
    y_i=x_i\times weight_i + bias_i
    $$

  <a id="全量化场景"></a>

  - **全量化场景（无perTokenScaleOptional）：**
    - x为INT8，bias为INT32

      $$
      y_i=(x_i\times weight_i + bias_i) * scale_i + offset_i
      $$

  - **全量化场景（有perTokenScaleOptional）：**
    - x为INT8，bias为INT32

      $$
      y_i=(x_i\times weight_i + bias_i) * scale_i * per\_token\_scale_i
      $$

    - x为INT8，bias为BFLOAT16

      $$
      y_i=(x_i\times weight_i) * scale_i * per\_token\_scale_i  + bias_i
      $$

    - x为INT4，无bias

      $$
      y_i=x_i\times (weight_i * scale_i) * per\_token\_scale_i
      $$

  <a id="伪量化场景"></a>

  - **伪量化场景：**

    - x为Float16、BFloat16，weight为INT4、INT8（仅支持x、weight、y均为单tensor的场景）。

      $$
      y_i=x_i\times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i
      $$

    <a id="a8w4伪量化场景"></a>

    - x为INT8，weight为INT4（仅支持x、weight、y均为单tensor的场景）。其中$bias$为必选参数，是离线计算的辅助结果，且 $bias_i=8\times weight_i  * scale_i$ ，并沿k轴规约。
    
      $$
      y_i=((x_i - 8) \times weight_i * scale_i+bias_i ) * per\_token\_scale_i
      $$

## 约束说明

- 确定性计算：
  - aclnnGroupedMatmulV5默认确定性实现。
<details>
<summary><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></summary>

  - **公共约束**
  <a id="公共约束"></a>
    - x和weight若需要转置，转置对应的tensor必须[非连续](../../../docs/zh/context/非连续的Tensor.md)。
    - x和weight中每一组tensor的最后一维大小都应小于65536。$x_i$的最后一维指当x不转置时$x_i$的K轴或当x转置时$x_i$的M轴。$weight_i$的最后一维指当weight不转置时$weight_i$的N轴或当weight转置时$weight_i$的K轴。
    - 当weight[数据格式](../../../docs/zh/context/数据格式.md)为FRACTAL_NZ格式时，要求weight的Shape满足FRACTAL_NZ格式要求。
    - perTokenScaleOptional：一般情况下，只支持1维且长度与x的M相同。仅支持x、weight、out均为单tensor（TensorList长度为1）场景。
    - groupListOptional：当输出中TensorList的长度为1时，groupListOptional约束了输出数据的有效部分，groupListOptional中未指定的部分将不会参与更新。
    - groupListType为0时要求groupListOptional中数值为非负单调非递减数列，表示分组轴大小的cumsum结果（累积和），groupListType为1时要求groupListOptional中数值为非负数列，表示分组轴上每组大小，groupListType为2时要求 groupListOptional中数值为非负数列，shape为[E, 2]，E表示Group大小，数据排布为[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]，其中groupSize为分组轴上每组大小，详见[groupListOptional配置示例](#grouplistoptional配置示例)。
    - groupType代表需要分组的轴，如矩阵乘为C[m,n]=A[m,k]xB[k,n]，则groupType取值-1：不分组，0：m轴分组，1：n轴分组，2：k轴分组。当前不支持n轴分组，详细参考<a href="#groupType-constraints">groupType支持场景</a>约束。
    - actType（int64\_t，计算输入）：整数型参数，代表激活函数类型。取值范围为0-5，支持的枚举值如下：
      * 0：GMMActType::GMM_ACT_TYPE_NONE；
      * 1：GMMActType::GMM_ACT_TYPE_RELU；
      * 2：GMMActType::GMM_ACT_TYPE_GELU_TANH；
      * 3：GMMActType::GMM_ACT_TYPE_GELU_ERR_FUNC（不支持）；
      * 4：GMMActType::GMM_ACT_TYPE_FAST_GELU；
      * 5：GMMActType::GMM_ACT_TYPE_SILU；

    <a id="a8w8场景约束"></a>
    
    <details>
    <summary>A8W8场景约束</summary>

    - **数据类型要求**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | INT8 | INT8 (ND) | INT32/null | UINT64 | null | null | null | null | INT64 | null | null | null | INT8 |
      | INT8 | INT8 (ND/NZ) | INT32/null |BFLOAT16| null | null | null | FLOAT/null | INT64 | null | null | null | BFLOAT16|
      | INT8 | INT8 (ND/NZ) | BFLOAT16/null |FLOAT/BFLOAT16| null | null | null | FLOAT/null | INT64 | null | null | null | BFLOAT16|
      | INT8 | INT8 (ND/NZ) | INT32/null | FLOAT | null | null | null | FLOAT/null | INT64 | null | null | null | FLOAT16 |
      | INT8 | INT8 (ND/NZ) | INT32/null | null | null | null | null | null | INT64 | null | null | null | INT32 |

    - **约束说明**

      除[公共约束](#公共约束)外，A8W8场景其余约束如下
      - 仅支持GroupType=0（M轴分组）
      - 当前仅支持x、weight、out均为长度为1的TensorList
      - x不支持转置
      - x仅支持2维Tensor，Shape为（M，K）
      - weight仅支持3维Tensor，Shape为（E，K，N）或（E，N，K）
      - 如果需要启用定轴算法以优化性能，需同时满足以下输入形状与参数配置条件：
        * 输入形状条件（满足任意一组即可）

          x 的 shape 为 (M, 7168)，weight 的 shape 为 (7168, 4096)。

          x 的 shape 为 (M, 2048)，weight 的 shape 为 (2048, 7168)。
        * 参数配置条件

          tuningConfigOptional 的第一个元素：设为大于 128 且小于 512。

          tuningConfigOptional 的第二个元素：设为 0。

          tuningConfigOptional 的第三个元素：设为 -1，或设为大于等于 M × N × 4 的数值。
          
    </details>

    <a id="a8w4场景约束"></a>

    <details>
    <summary>A8W4场景约束</summary>

    - **数据类型要求**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | null | null | null | FLOAT | INT64 | null | null | null | BFLOAT16|
      | INT8 | INT4 (ND/NZ) | FLOAT | UINT64 | FLOAT/null | null | null | FLOAT | INT64 | null | null | null | FLOAT16 |

    - **约束说明**

      除[公共约束](#公共约束)外，A8W4场景其余约束如下：
      - 仅支持GroupType=0（M轴分组），actType=0
      - 当前仅支持x、weight、out均为长度为1的TensorList
      - x不支持转置、weight不支持转置
      - x仅支持2维Tensor，Shape为（M，K）
      - weight默认支持3维Tensor，Shape为（E，K，N）
      - Bias为计算过程中离线计算的辅助结果，值要求为$8\times weight \times scale$，并在第1维累加，shape要求为$[E, N]$。
      - 当weight传入数据类型为INT32时，会将每个INT32视为8个INT4。
      - offset为空时
        - 该场景下仅支持groupListType为1（算子不会检查groupListType的值，会认为groupListType为1），k要求为quantGroupSize的整数倍，且要求k <= 18432。其中quantGroupSize为k方向上pergroup量化长度，当前支持quantGroupSize=256。
        - 该场景下要求n为8的整数倍。
        - 该场景下scale为pergroup与perchannel离线融合后的结果，shape要求为$[E, quantGroupNum, N]$，其中$quantGroupNum=k \div quantGroupSize$。
        - 该场景下，各个专家处理的token数的预期值大于n/4时，即tuningConfigOptional中第一个值大于n/4时，通常会取得更好的性能，此时显存占用会增加$g\times k \times n$字节（其中g为matmul组数即分组数）。
      - offset不为空时
        - scale为pergroup与perchannel离线融合后的结果，shape要求为$[E, 1, N]$。
        - 该场景下offsetOptional不为空。非对称量化offsetOptional为计算过程中离线计算辅助结果，即$antiquantOffset \times scale$，shape要求为$[E, 1, N]$，dtype为FLOAT32。
      - tuningConfigOptional数组第二个元素可置1，以使能A8W4场景中weight的特殊格式模板，以优化算子性能(性能优势的shape范围参考:K >= 2048 && N >= 2048)。需要说明的是，该模板要求weight的shape为（E，N，K）,然后再对其进行ND2NZ转换后作为算子输入。
    </details>

    <a id="a16w4场景约束"></a>

    <details>
    <summary>A16W4场景约束</summary>

    - **数据类型要求**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | FLOAT16 | INT4 (ND) | FLOAT16/null | null | null | FLOAT16 | FLOAT16 | null | INT64 | null | null | null | FLOAT16 |
      | BFLOAT16| INT4 (ND) | FLOAT/null | null | null | BFLOAT16 | BFLOAT16 | null | INT64 | null | null | null | BFLOAT16|

    - **约束说明**

      除[公共约束](#公共约束)外，a16w4场景其余约束如下：
      - x不支持转置
      - 仅支持GroupType=-1、0，actType=0，groupListType=0/1
      - weight中每一组tensor的最后一维大小都应是偶数，最后一维指weight不转置时$weight_i$的N轴或当weight转置时$weight_i$的K轴。
      - 对称量化支持perchannel和pergroup量化模式，若为pergroup，pergroup数G或$G_i$必须要能整除对应的$k_i$。
      - 非对称量化仅支持perchannel模式。
      - 在pergroup场景下，当weight转置时，要求pergroup长度$s_i$是偶数。
      - 若weight为多tensor，定义pergroup长度$s_i = k_i / G_i$，要求所有$s_i(i=1,2,...g)$都相等。
      - 伪量化参数antiquantScaleOptional和antiquantOffsetOptional的shape要满足下表（其中g为matmul组数，G为pergroup数，$G_i$为第i个tensor的pergroup数）：

        | 使用场景 | 子场景 | shape限制 |
        |:---------:|:-------:| :-------|
        | 伪量化perchannel | weight单 | $[E, N]$|
        | 伪量化perchannel | weight多 | $[N_i]$|
        | 伪量化pergroup | weight单 | $[E, G, N]$|
        | 伪量化pergroup | weight多 | $[G_i, N_i]$|
    </details>

    <a id="a16w8场景约束"></a>

    <details>
    <summary>A16W8场景约束</summary>

    - **数据类型要求**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | FLOAT16 | INT8 (ND) | FLOAT16/null | null | null | FLOAT16 | FLOAT16 | null | INT64 | null | null | null | FLOAT16 |
      | BFLOAT16| INT8 (ND) | FLOAT/null | null | null | BFLOAT16 | BFLOAT16 | null | INT64 | null | null | null | BFLOAT16|

    - **约束说明**

      除[公共约束](#公共约束)外，a16w8场景其余约束如下：
      - x不支持转置
      - 仅支持GroupType=-1、0，actType=0，groupListType=0/1
      - 仅支持perchannel量化模式。
      - 若weight为多tensor，定义pergroup长度$s_i = k_i / G_i$，要求所有$s_i(i=1,2,...g)$都相等。
      - 伪量化参数antiquantScaleOptional和antiquantOffsetOptional的shape要满足下表（其中g为matmul组数）：

        | 使用场景 | 子场景 | shape限制 |
        |:---------:|:-------:| :-------|
        | 伪量化perchannel | weight单 | $[E, N]$|
        | 伪量化perchannel | weight多 | $[N_i]$|
    </details>

    <a id="a4w4场景约束"></a>

    <details>
    <summary>A4W4场景约束</summary>

    - **数据类型要求**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | INT4 | INT4 (ND/NZ) | null | UINT64 | null | null | null | FLOAT/null | INT64 | null | null | null | FLOAT16/BFLOAT16|

    - **约束说明**

      除[公共约束](#公共约束)外，A4W4场景其余约束如下：
      - 仅支持GroupType=0（M轴分组），actType=0，groupListType=0/1
      - 当前仅支持x、weight、out均为长度为1的TensorList
      - x不支持转置，weight不支持转置
      - x仅支持2维Tensor，Shape为（M，K）
      - weight仅支持3维Tensor，Shape为（E，K，N）
      - weight的数据格式为ND时，要求n为8的整数倍。
      - 支持perchannel和pergroup量化。perchannel场景的scale的shape需为$[E, N]$，pergroup场景需为$[E, G, N]$。
      - pergroup场景下，$G$必须要能整除$k$，且$k/G$需为偶数。
    </details>

    <a id="非量化场景约束"></a>

    <details>
    <summary>非量化场景约束</summary>

    - **数据类型要求**

      | x | weight | bias | scale | offset | antiquantScale | antiquantOffset | perTokenScale | groupList | activationInput | activationQuantScale | activationQuantOffset | out |
      |---------|----------------|--------------|--------|------------|----------------|-----------------|---------------|-----------|-----------------|----------------------|-----------------------|---------|
      | FLOAT | FLOAT (ND) | FLOAT/null | null | null | null | null | null | INT64 | null | null | null | FLOAT |
      | FLOAT16 | FLOAT16 (ND/NZ) | FLOAT16/null | null | null | null | null | null | INT64 | null | null | null | FLOAT16 |
      | BFLOAT16| BFLOAT16(ND/NZ) | FLOAT/null | null | null | null | null | null | INT64 | null | null | null | BFLOAT16|

    - **约束说明**

      除[公共约束](#公共约束)外，非量化场景其余约束如下：
      - 支持GroupType=-1、0、2，actType=0，groupListType=0/1
    </details>

    <a id="groupType-constraints"></a>

    <details>
    <summary>groupType支持场景</summary>

    - a16w8、a16w4场景仅支持groupType为-1和0场景。
    - A8W8、A8W4、A4W4场景仅支持groupType为0场景中x tensor数为单。
    - x、weight、y的输入类型为aclTensorList，表示一个aclTensor类型的数组对象。下面表格支持场景用"单"表示由一个aclTensor组成的aclTensorList，"多"表示由多个aclTensor组成的aclTensorList。例如"单多单"，分别表示x为单tensor、weight为多tensor、y为单tensor。

      | groupType | x tensor数 | weight tensor数 | y tensor数 | splitItem| groupListOptional | 转置 | 其余场景限制 |
      |:---------:|:-------:|:-------:|:-------:|:--------:|:------------------|:--------| :-------|
      | -1 | 多个|多个|多个 | 0/1 | groupListOptional必须传空 | 1）x不支持转置；<br> 2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一| x中tensor要求维度一致，支持2维，weight中tensor需为2维，y中tensor维度和x保持一致 |
      | 0 | 单个|单个|单个 | 2/3 | 1）必须传groupListOptional；<br> 2）当groupListType为0时，最后一个值应小于等于x中tensor的第一维；当groupListType为1时，数值的总和应小于等于x中tensor的第一维；当groupListType为2时，第二列数值的总和应小于等于x中tensor的第一维；<br> 3）groupListOptional第1维最大支持1024，即最多支持1024个group |1）x不支持转置；<br> 2）支持weight转置，A8W4与A4W4场景不支持weight转置 |weight中tensor需为3维，x，y中tensor需为2维|
      | 0 | 单个|多个|单个 | 2/3 | 1）必须传groupListOptional；<br> 2）当groupListType为0时，最后一个值应小于等于x中tensor的第一维；当groupListType为1时，数值的总和应小于等于x中tensor的第一维；当groupListType为2时，第二列数值的总和应小于等于x中tensor的第一维；<br> 3）groupListOptional第1维最大支持128，即最多支持128个group|1）x不支持转置；<br> 2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一 |1）x，weight，y中tensor需为2维；<br> 2）weight中每个tensor的N轴必须相等 |
      | 0 | 多个|多个|单个 | 2/3 | 1）groupListOptional可选；<br> 2）若传入groupListOptional，当groupListType为0时，groupListOptional的差值需与x中tensor的第一维一一对应；当groupListType为1时，groupListOptional的数值需与x中tensor的第一维一一对应；当groupListType为2时，groupListOptional第二列的数值需与x中tensor的第一维一一对应；<br> 3）groupListOptional第1维最大支持128，即最多支持128个group |1）x不支持转置；<br> 2）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一|1）x，weight，y中tensor需为2维；<br> 2）weight中每个tensor的N轴必须相等 |
      | 2 | 单个|单个|单个 | 2/3 | 1）必须传groupListOptional；<br> 2）当groupListType为0时，最后一个值应小于等于x中tensor的第二维；当groupListType为1时，数值的总和与x应小于等于tensor的第二维；当groupListType为2时，第二列数值的总和应小于等于x中tensor的第二维；<br> 3）groupListOptional第1维最大支持1024， 即最多支持1024个group | x必须转置；<br> 2）weight不能转置 |1）x，weight中tensor需为2维，y中tensor需为3维；<br> 2）bias必须传空|
      | 2 | 单个|多个|多个 | 0/1 | groupListOptional必须传空 | 1）x必须转置；<br> 2）weight不能转置| 1）x，weight，y中tensor需为2维。<br> 2）weight长度最大支持128，即最多支持128个group；<br> 3）原始shape中weight每个tensor的第一维之和不应超过x第一维；<br> 4）bias必须传空 |
    </details>

    <a id="grouplistoptional配置示例"></a>

    <details>
    <summary>groupListOptional配置示例</summary>

    - shape信息
      M = 789、 K=4096、 N=7168 、E = 8（0,2,5个专家有需要处理的token，0处理123个token， 2/5处理333个token）
      X的shape是[[789, 4096]]
      W的shape是[[9, 4096, 7168]]
      Y的shape是[[789, 7168]]

    - groupListType为0时groupList配置如下
      - groupListOptional：`[123, 123, 456, 456, 456, 789, 789, 789, 789]`

    - groupListType为1时groupList配置如下
      - groupListOptional：`[123, 0, 333, 0, 0, 333, 0, 0, 0]`

    - groupListType为2时groupList配置如下
      - groupListOptional在该模式会将所有非0的group移动到前面，适用于非激活专家较多场景。
      - groupListOptional：`[[0, 123], [2, 333], [5, 333], [1, 0], [3, 0], [4, 0], [6, 0], [7, 0], [8, 0]]`
    </details>

    <a id="tuningConfigOptional配置示例"></a>

    <details>
    <summary>tuningConfigOptional配置示例</summary>

    - tuningConfigOptional入参是Host侧的aclIntArray，数组中存储INT64的元素。兼容历史版本，用户如不使用该参数，不传（即为nullptr）即可。
      * 第一个元素：

        语义：各个专家处理的token数的预期值，算子tiling时会按照数组中第一个元素进行最优tiling，性能更优。
        
        适用场景：[a8w4场景](#a8w4场景约束)与[a8w8场景](#a8w8场景约束)，且为x、weight、out均为单tensor场景。
      
      * 第二个元素：
        
        语义：是否使能weight为亲核格式，该格式需要先将weight内存排布转置，然后转为NZ格式。
        
        适用场景：[a8w4场景](#a8w4场景约束)。
      
      * 第三个元素：
        
        语义：允许额外使用的最大workspace空间，本算子部分场景使用了定轴算法，可以提升性能但会额外使用一部分内存空间，如果允许额外使用的内存空间允许使用定轴算法，会有性能提升，如果不在意使用的workspace空间上限可设置为-1。
        
        适用场景：[a8w8场景](#a8w8场景约束)。

    </details>

</details>

<a id="atlas推理系列产品"></a>

<details>
<summary><term>Atlas 推理系列产品</term></summary>


- 产品规格

  - groupType：整数型参数，代表需要分组的轴，当前只支持m轴分组。
  - groupListType：支持取值0、1。取值0: groupListOptional中数值为非负单调非递减数列，表示分组轴大小的cumsum结果（累积和），1: groupListOptional中数值为非负数列，表示分组轴上每组大小。
  - actType：当前只支持传入0，表示GMMActType::GMM_ACT_TYPE_NONE。
  - tuningConfigOptional：不支持此参数。
  - 输入输出只支持float16的数据类型，输出y的n轴大小需要是16的倍数。

- 支持场景

  | groupType | x tensor数 | weight tensor数 | y tensor数 | 场景限制 |
  |:---------:|:-------:|:-------:|:-------:| :------ |
  | 0 | 单个|单个|单个 |1）仅支持splitItem为2/3；<br>2）weight中tensor需为3维，x，y中tensor需为2维；<br>3）必须传groupListOptional，且当groupListType为0时，最后一个值与x中tensor的第一维相等，当groupListType为1时，数值的总和与x中tensor的第一维相等；<br>4）groupListOptional第1维最大支持1024，即最多支持1024个group；<br>5）支持weight转置，不支持x转置 |

</details>

<a id="昇腾910_95AI处理器"></a>

<details>
<summary><term>Ascend 950PR/Ascend 950DT AI处理器</term></summary>

  - 公共约束：

    - groupType：支持m轴分组和不分组，仅非量化支持k轴分组。
    - groupListType：支持取值0、1。
    - actType：当前只支持传入0，表示GMMActType::GMM_ACT_TYPE_NONE。
    - tuningConfigOptional：不支持此参数。
  - 非量化场景支持的数据类型为：

    - 以下入参为空：scaleOptional、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、perTokenScaleOptional、activationInputOptional、activationQuantScaleOptional、activationQuantOffsetOptional、actType、activationFeatureOutOptional
    - 不为空的参数支持的数据类型组合要满足下表

      |groupType| x       | weight  | biasOptional | out     |
      |:-------:|:-------:|:-------:| :------      |:------ |
      |-1/0/2   |BFLOAT16     |BFLOAT16     |BFLOAT16/FLOAT32/null    | BFLOAT16|
      |-1/0/2   |FLOAT16     |FLOAT16     |FLOAT16/FLOAT32/null    | FLOAT16|

  - 伪量化场景支持的数据类型为：

    - 以下入参为空：scaleOptional、offsetOptional、perTokenScaleOptional、activationInputOptional、activationQuantScaleOptional、activationQuantOffsetOptional
    - 不为空的参数支持的数据类型组合要满足下表

      |groupType| x       | weight  | biasOptional |antiquantScaleOptional|antiquantOffsetOptional| out     |
      |:-------:|:-------:|:-------:| :------      |:------|:------|:------|
      |-1/0   |BFLOAT16     |INT8/INT4     |BFLOAT16/FLOAT32/null| BFLOAT16 | BFLOAT16/null | BFLOAT16 |
      |-1/0   |FLOAT16     |INT8/INT4     |FLOAT16/null    | FLOAT16 | FLOAT16/null | FLOAT16 |
      |0   |BFLOAT16     |FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8 |BFLOAT16/FLOAT32/null| BFLOAT16 | null | BFLOAT16 |
      |0   |FLOAT16     |FLOAT8_E5M2/FLOAT8_E4M3FN/HIFLOAT8    |FLOAT16/null    | FLOAT16 | null | FLOAT16 |

    - 当weight的数据类型为FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8时，antiquantOffsetOptional仅支持传入空指针或空tensorList，weight仅支持转置。
    - 若weight的类型为INT4，则weight中每一组tensor的最后一维大小都应是偶数。$weight_i$的最后一维指weight不转置时$weight_i$的N轴或当weight转置时$weight_i$的K轴。
    - antiquantScaleOptional和非空的biasOptional、antiquantOffsetOptional要满足下表（其中E为matmul组数即分组数）：

      |groupType| 使用场景 | shape限制 |
      |:---------:|:---------:| :------ |
      |-1|weight多tensor|每个tensor 1维，shape为（$n_i$），不允许存在一个tensorList中部分tensor的shape为（$n_i$）部分tensor为空的情况 |
      |0|weight单tensor|每个tensor 2维，shape为（E, N）|

  - 不同groupType支持场景:

    - 支持场景中单表示单tensor，多表示多tensor，表示顺序为x，weight，out，例如单多单表示支持x为单tensor，weight多tensor，out单tensor的场景。

      | groupType | x tensor数 | weight tensor数 | y tensor数 | 场景限制 |
      |:---------:|:-------:|:-------:|:-------:| :------ |
      | -1 | 多个 | 多个|多个 |1）仅支持splitItem为0/1<br>2）非量化x，out中tensor需为2维， shape分别为（$m_i$, $k_i$）和（$m_i$, $n_i$）；伪量化场景x中tensor要求维度一致，支持2-6维，y中tensor维度和x保持一致；weight中tensor需为2维，shape为（$n_i$, $k_i$）或（$k_i$, $n_i$）；bias中tensor需为1维，shape为（$n_i$）<br>3） groupListOptional必须传空<br>4）支持weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>5）x不支持转置<br>6）仅支持ND进ND出|
      | 0 | 单个|单个|单个 |1）仅支持splitItem为2/3<br>2）weight中tensor需为3维，shape为（E, N, K）或（E, K, N）；x，out中tensor需为2维，shape分别为（M, K）和（M, N）；bias中tensor需为2维，shape为（E, N）<br>3）必须传groupListOptional，且当groupListType为0时，最后一个值不大于x中tensor的第一维，当groupListType为1时，数值的总和不大于x中tensor的第一维<br>4）groupListOptional第1维最大支持1024，即最多支持1024个group<br>5）支持x不转置，weight转置、不转置均支持<br>6）仅支持ND进ND出|
      | 0 | 单个|多个|单个 |1）仅支持splitItem为2/3<br>2）必须传groupListOptional， 且当groupListType为0时，最后一个值与x中tensor的第一维相等，当groupListType为1时，数值的总与x中tensor的第一维相等，长度最大为 128<br>3）x，out中tensor需为2维， shape分别为（M, K）和（M, N）；weight中tensor需为2维，shape为（N, K）或（K, N）；bias中tensor需为1维，shape为（N）<br>4）weight中每个tensor的N轴必须相等<br>5）支持weight转置，但weight的tensorList中每tensor是否转置需保持统一<br>6）x不支持转置<br>7）仅支持非量化<br>8）仅支持ND进ND出 |
      | 0 | 多个|多个|单个 |1）仅支持splitItem为2/3<br>2）x，out中tensor需为2维， shape分别为（M, K）和（M, N）；weight中tensor需为2维，shape为（N, K）或（K, N）；bias中tensor需为1维，shape为（N）<br>3）weight中每个tensor的N轴必须相等<br>4）若传入groupListOptional，当groupListType为0时，groupListOptional的差值需与x中tensor的第一维一一对应，当groupListType为1时，groupListOptional的数值需与x中tensor的第一维一一对应，且长度最大为128<br>5）支weight转置，但weight的tensorList中每个tensor是否转置需保持统一<br>6）x不支持转置<br>7）仅支持非量化<br>8）仅支持ND进ND出 |
      | 2 | 单个|单个|单个 |1）仅支持splitItem为2/3<br>2）x，weight中tensor需为2维，shape分别为（K, M）和（K, N）；out中tensor需为3维, shape为（E, M, N）<br>3）必须传groupListOptional，且当groupListType为0时，最后一个值不大于x中tensor的第一维，当groupListType为1时，数值的总和不大于x中tensor的第一维<br>4）groupListOptional第1维最大支持1024，即最多支持1024个group<br>5）仅支持x转置且weight不转置<br>6）仅支持非量化<br>7）仅支持ND进ND出|

</details>

## 调用示例
调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_grouped_matmul_v5.h"

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

      // 创建tuningconfig aclIntArray
      std::vector<int64_t> tuningConfigData = {512};
      aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigData.data(), 1);

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
      // 调用aclnnGroupedMatmulV5第一段接口
      ret = aclnnGroupedMatmulV5GetWorkspaceSize(x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale, groupedList, activationInput, activationQuantScale, activationQuantOffset, splitItem, groupType, groupListType, actType, tuningConfig, out, activationFeatureOut, dynQuantScaleOut, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用aclnnGroupedMatmulV5第二段接口
      ret = aclnnGroupedMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulV5 failed. ERROR: %d\n", ret); return ret);

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