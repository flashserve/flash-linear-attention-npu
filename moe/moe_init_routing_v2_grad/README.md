# MoeInitRoutingV2Grad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>    |    √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

-   **算子功能**：[aclnnMoeInitRoutingV2](../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)的反向传播，完成tokens的加权求和。
-   **计算公式**：

    $$
    gradX_i=\sum_{t=0}^{topK}gradExpandedX[expandedRowIdx[i * topK + t]]
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
      <td>gradExpandedX</td>
      <td>输入</td>
      <td>表示Routing过后的目标张量，对应公式中的`gradExpandedX`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expandedRowIdx</td>
      <td>输入</td>
      <td>表示token按照专家序排序索引，对应公式中的`expandedRowIdx`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>topK</td>
      <td>属性</td>
      <td>topk值，对应公式中的`topk值`。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropPadMode</td>
      <td>属性</td>
      <td>表示场景是否为Drop类。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activeNum</td>
      <td>属性</td>
      <td>表示是否为Active场景。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示Routing反向输出，对应公式中的`gradX`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

  无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_moe_init_routing_v2_grad](examples/test_aclnn_moe_init_routing_v2_grad.cpp) | 通过[aclnnMoeInitRoutingV2Grad](docs/aclnnMoeInitRoutingV2Grad.md)接口方式调用MoeInitRoutingV2Grad算子。 |