# GroupedMatmulAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |


## 功能说明

- 算子功能：实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。基本功能为矩阵乘，如$yRef_i[m_i,n_i]=x_i[m_i,k_i] \times weight_i[k_i,n_i]+y_i[m_i,n_i], i=1...g$，其中g为分组个数，$m_i/k_i/n_i$为对应shape。当前仅支持K轴分组。

  - k轴分组：$k_i$各不相同，但$m_i/n_i$每组相同。
- 计算公式：

  $$
  yRef_i=x_i\times weight_i + y_i
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
      <td style="white-space: nowrap">x</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的输入x。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">weight</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">公式中的weight。</td>
      <td style="white-space: nowrap">FLOAT16、BFLOAT16</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupList</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">表示输入K轴方向的matmul大小分布的cumsum结果（累积和）。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">y</td>
      <td style="white-space: nowrap">输入</td>
      <td style="white-space: nowrap">表示原地累加的输出矩阵。</td>
      <td style="white-space: nowrap">FLOAT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">transposeX</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">表示x矩阵是否转置。</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">transposeWeight</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">表示weight矩阵是否转置。</td>
      <td style="white-space: nowrap">BOOL</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupType</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">表示分组类型。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">groupListType</td>
      <td style="white-space: nowrap">属性</td>
      <td style="white-space: nowrap">表示分组groupList格式。</td>
      <td style="white-space: nowrap">INT64</td>
      <td style="white-space: nowrap">-</td>
    </tr>
    <tr>
      <td style="white-space: nowrap">yRef</td>
      <td style="white-space: nowrap">输出</td>
      <td style="white-space: nowrap">表示原地累加的输出矩阵。</td>
      <td style="white-space: nowrap">FLOAT32</td>
      <td style="white-space: nowrap">ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32的最大值2147483647。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_grouped_matmul_add](examples/test_aclnn_grouped_matmul_add.cpp) | 通过接口方式调用[GroupedMatmulAdd](docs/aclnnGroupedMatmulAdd.md)算子。 |