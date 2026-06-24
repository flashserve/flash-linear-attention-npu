# SolveTril

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

- 算子功能：计算单位下三角矩阵的逆矩阵，使用 MXR 算法（MCH + MBH 组合）。

- 计算公式：

  给定单位下三角矩阵 $L = I + A$（$A$ 为严格下三角矩阵），计算：

  $$
  output = L^{-1}
  $$

  其中 $L^{-1}$ 也是单位下三角矩阵。

  MCH（叶子块 16x16 求逆）：

  $$
  (I+A)^{-1} = (I-A)(I+A^2)(I+A^4) \cdots (I+A^{2^{m-1}})
  $$

  MBH（分块递归组装）：

  $$
  L^{-1} = \begin{bmatrix} L_{11}^{-1} & 0 \\ -L_{22}^{-1} L_{21} L_{11}^{-1} & L_{22}^{-1} \end{bmatrix}
  $$


## 参数说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 300px">
<col style="width: 150px">
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
    <td>input</td>
    <td>输入</td>
    <td>单位下三角矩阵，shape为[batch, n, n]或[n, n]。对角线元素为1，上三角部分为0。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>output</td>
    <td>输出</td>
    <td>逆矩阵结果L^{-1}，shape与input相同。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
</tbody>
</table>


## 约束说明

- 输入矩阵维度 n 必须为 16 的倍数，支持 n ∈ {16, 32, 64, 128}。
- 输入必须为单位下三角矩阵（对角线元素为1，上三角部分为0），算子不校验输入合法性。
- 支持 float16、float32 数据类型。
- 输入 tensor 维度为 2D 或 3D（最后两个维度必须相等，即方阵）。
- 不支持非连续 Tensor。
- 不支持空 Tensor（n >= 16）。
- 具体shape约束见 [aclnnSolveTril](./docs/aclnnSolveTril.md)。


## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_solve_tril.cpp](./examples/test_aclnn_solve_tril.cpp) | 通过[aclnnSolveTril](./docs/aclnnSolveTril.md)调用aclnnSolveTril算子 |
