# MoeGatingTopKSoftmaxV2

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term> Ascend 950PR/Ascend 950DT</term>               |      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明

-   算子功能：MoE计算中，如果renorm=0，先对x的输出做Softmax计算，再取topk操作；如果renorm=1，先对x的输出做topk操作，再进行Softmax操作。其中yOut为softmax的topk结果；expertIdxOut为topk的indices结果即对应的专家序号；如果对应的行finished为True，则expert序号直接填num\_expert值（即x的最后一个轴大小）。
-   计算公式：
1. renorm = 0,

    $$
    softmaxResultOutOptional=softmax(x,axis=-1)
    $$

    $$
    yOut,expertIdxOut=topK(softmaxResultOutOptional,k=k)
    $$

2. renorm = 1

    $$
    topkOut,expertIdxOut=topK(x, k=k)
    $$

    $$
    yOut = softmax(topkOut,axis=-1)
    $$

## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出/属性</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>公式中的`x`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>finishedOptional</td>
      <td>输入</td>
      <td>表示如果对应的行finished为True，则expert序号直接填num_expert值（即x的最后一个轴大小）。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>属性</td>
      <td>公式中的`k`，表示topk的k值。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>renorm</td>
      <td>属性</td>
      <td>公式中的`renorm`，表示renorm标记，取值0和1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputSoftmaxResultFlag</td>
      <td>属性</td>
      <td>表示是否输出softmax的结果。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>公式中的`yOut`，表示对x做softmax后取的topk值。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertIdxOut</td>
      <td>输出</td>
      <td>公式中的`expertIdxOut`，表示对x做softmax后取topk值的索引，即专家的序号。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmaxResultOutOptional</td>
      <td>输出</td>
      <td>公式中的`softmaxResultOutOptional`，表示计算过程中Softmax的结果。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

-   k的值不大于1024。
-   renorm的值只支持0和1。
-   x和finishedOptional的每一维大小应不大于int32的最大值2147483647。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_gating_top_k_softmax_v2.cpp](examples/test_aclnn_moe_gating_top_k_softmax_v2.cpp) | 通过[aclnnMoeGatingTopKSoftmaxV2](docs/aclnnMoeGatingTopKSoftmaxV2.md)接口方式调用MoeGatingTopKSoftmaxV2算子。 |