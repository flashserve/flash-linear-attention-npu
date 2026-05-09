# CausalConv1d

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |


## 功能说明

- 算子功能：完成因果一维卷积（Causal Conv1d）计算，支持前向计算（runMode=0）和状态更新（runMode=1）两种运行模式。

- 计算公式：

  Causal Conv1d 是一种因果一维卷积算子，常用于序列建模中。在每个时间步 $t$，根据当前输入 $x_t$、卷积权重 $w$ 和历史状态，计算卷积输出 $y_t$。

  $$
  y_t = \text{Activation}\left(\sum_{j=0}^{W-1} w_j \cdot x_{t-j} + b\right)
  $$

  其中，$W$ 为卷积核宽度（支持2、3、4），$w_j$ 为卷积权重，$b$ 为偏置（可选），$\text{Activation}$ 为激活函数（可选，SiLU）。当 `activationMode=0` 时不使用激活函数，`activationMode=1` 时使用 SiLU 激活函数。

  算子同时维护卷积状态 `convStates`，用于在增量推理（runMode=1）时缓存历史输入，实现高效的状态更新。


## 参数说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
<col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>输入序列，公式中的x。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>输入</td>
    <td>卷积权重，公式中的w。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入</td>
    <td>偏置，公式中的b。可选输入，若不提供则默认为0。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>convStates</td>
    <td>输入&输出</td>
    <td>卷积状态，缓存历史输入用于因果卷积计算。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>queryStartLoc</td>
    <td>输入</td>
    <td>变长序列的起始位置索引，用于变长输入模式。可选输入。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cacheIndices</td>
    <td>输入</td>
    <td>序列到状态缓存的映射索引。可选输入。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>initialStateMode</td>
    <td>输入</td>
    <td>标记序列是否有初始状态。可选输入。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>numAcceptedTokens</td>
    <td>输入</td>
    <td>每个序列接受的token数量，用于投机解码。可选输入。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>卷积输出，公式中的y。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

### 属性说明

| 属性名 | 描述 | 数据类型 | 默认值 |
|:---:|:---:|:---:|:---:|
| activationMode | 激活函数模式。0：不使用激活函数，1：使用SiLU激活函数。 | INT64 | 0 |
| padSlotId | 无效缓存槽位的标记ID，用于跳过不需要计算的序列。 | INT64 | -1 |
| runMode | 运行模式。0：前向计算模式（fn），1：状态更新模式（update）。 | INT64 | 0 |

## 约束说明
- 输入tensor的shape大小需满足一定约束，具体见[aclnnCausalConv1d](./docs/aclnnCausalConv1d.md)。


## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_causal_conv1d.cpp](./examples/test_aclnn_causal_conv1d.cpp) | 通过[aclnnCausalConv1d](./docs/aclnnCausalConv1d.md)调用aclnnCausalConv1d算子 |
