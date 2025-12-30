# MoeGatingTopK

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term> Ascend 950PR/Ascend 950DT</term>                  |    √     |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：MoE计算中，对输入x做Sigmoid或者SoftMax计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。
- 计算公式：

    对输入做Sigmoid或者SoftMax：

    $$
    if normType==1:
        normOut=Sigmoid(x)
    else:
        normOut=SoftMax(x)
    $$

    如果bias不为空：

    $$
    normOut = normOut + bias
    $$

    对计算结果按照groupCount进行分组，每组按照groupSelectMode取max或topk2的sum值对group进行排序，取前kGroup个组：

    $$
    groupOut, groupId = TopK(ReduceSum(TopK(Split(normOut, groupCount), k=2, dim=-1), dim=-1),k=kGroup)
    $$

    根据上一步的groupId获取normOut中对应的元素，将数据再做TopK，得到expertIdxOut的结果：

    $$
    y,expertIdxOut=TopK(normOut[groupId, :],k=k)
    $$

    对y按照输入的routedScalingFactor和eps参数进行计算，得到yOut的结果：

    $$
    yOut = y / (ReduceSum(y, dim=-1)+eps)*routedScalingFactor
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待计算输入，对应公式中的`x`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>可选属性</td>
      <td>与输入x进行计算的bias值，对应公式中的`bias`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输入</td>
      <td>topk的k值，对应公式中的`k`。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kGroup</td>
      <td>输入</td>
      <td>分组排序后取的group个数，对应公式中的`kGroup`。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupCount</td>
      <td>输入</td>
      <td>分组的总个数，对应公式中的`groupCount`。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>routedScalingFactor</td>
      <td>输入</td>
      <td>计算yOut使用的routedScalingFactor系数，对应公式中的`routedScalingFactor`。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>输入</td>
      <td>用于计算yOut使用的eps系数，对应公式中的`eps`。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>对x做norm、分组排序topk后计算的结果，对应公式中的`yOut`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expertIdxOut</td>
      <td>输出</td>
      <td>对x做norm、分组排序topk后的索引，对应公式中的`expertIdxOut`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normOut</td>
      <td>输出</td>
      <td>norm计算的输出结果，对应公式中的`normOut`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>groupSelectMode</td>
      <td>输入</td>
      <td>分组排序方式。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>renorm</td>
      <td>输入</td>
      <td>renorm标记。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>normType</td>
      <td>输入</td>
      <td>norm函数类型。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outFlag</td>
      <td>输入</td>
      <td>表示是否输出norm操作结果。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    
  </tbody></table>

## 约束说明

  * 输入shape限制：
      * x最后一维（即专家数）要求不大于2048。 
  * 输入值域限制：
      * 要求1 <= k <= x_shape[-1] / groupCount * kGroup。
      * 要求1 <= kGroup <= groupCount，并且kGroup * x_shape[-1] / groupCount的值要大于等于k。
      * 要求groupCount > 0，x_shape[-1]能够被groupCount整除且整除后的结果大于groupSelectMode，并且整除的结果按照32个数对齐后乘groupCount的结果不大于2048。
      * renorm仅支持0，表示先进行norm操作，再计算topk。
  * 其他限制：
      * groupSelectMode取值0和1，0表示使用最大值对group进行排序, 1表示使用topk2的sum值对group进行排序。
      * normType取值0和1，0表示使用Softmax函数，1表示使用Sigmoid函数。
      * outFlag取值true和false，true表示输出，false表示不输出。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_gating_top_k](examples/test_aclnn_moe_gating_top_k.cpp) | 通过[aclnnMoeGatingTopK](docs/aclnnMoeGatingTopK.md)接口方式调用MoeGatingTopK算子。 |
| 图模式 | [test_geir_moe_gating_top_k](examples/test_geir_moe_gating_top_k.cpp)  | 通过[算子IR](op_graph/moe_gating_top_k_proto.h)构图方式调用MoeGatingTopK算子。         |