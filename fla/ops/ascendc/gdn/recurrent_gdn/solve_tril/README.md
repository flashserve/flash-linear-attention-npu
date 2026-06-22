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

- 算子功能：对输入张量 A（shape 为 [B, S, H, BT] 或 [T, H, BT]）中的每个 BT×BT chunk 计算单位下三角矩阵的逆矩阵，使用 MXR 算法（MCH + MBH 组合）。

- 计算公式：

  给定单位下三角矩阵 $L = I + A$（$A$ 为严格下三角矩阵），计算：

  $$
  out = L^{-1}
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
    <td>A</td>
    <td>输入</td>
    <td>输入张量，shape为 [B, S, H, BT]（4D）或 [T, H, BT]（3D）。每个 chunk 包含一个 BT×BT 的单位下三角块，对角线元素为 1，上三角部分为 0。BT 必须属于 {16, 32, 64, 128}。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cu_seqlens</td>
    <td>输入（可选）</td>
    <td>变长序列的累积长度，用于标识每个 batch 中有效序列的边界。定长模式下可传入 nullptr。变长模式下必须同时提供 chunk_indices_out。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>chunk_indices_out</td>
    <td>输入（可选）</td>
    <td>变长模式下输出 chunk 的索引映射，其第一维大小决定总任务数。定长模式下可传入 nullptr。变长模式下必须同时提供 cu_seqlens。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出</td>
    <td>逆矩阵结果 $L^{-1}$，shape 与 A 相同。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>


## 属性说明

<table style="undefined;table-layout: fixed; width: 700px"><colgroup>
<col style="width: 150px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 150px">
</colgroup>
<thead>
  <tr>
    <th>属性名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
  </tr></thead>
<tbody>
  <tr>
    <td>layout</td>
    <td>输入（可选）</td>
    <td>输入张量的内存布局标识，默认值为 0。当前仅支持默认值。</td>
    <td>INT</td>
  </tr>
</tbody>
</table>


## 约束说明

- 输入张量 A 的最后一个维度 BT 必须属于 {16, 32, 64, 128}。
- 每个 BT×BT 块必须为单位下三角矩阵（对角线元素为 1，上三角部分为 0），算子不校验输入合法性。
- 仅支持 FLOAT16 数据类型。
- 输入 tensor 维度为 3D（[T, H, BT]）或 4D（[B, S, H, BT]）。
- 不支持非连续 Tensor。
- 不支持空 Tensor。
- 当 BT >= 64 时需要额外 Device 侧 workspace 空间。
- 支持定长模式和变长模式（通过 cu_seqlens 和 chunk_indices_out 可选输入控制）。
- 具体 shape 约束见 [aclnnSolveTril](./docs/aclnnSolveTril.md)。


## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | 调用示例占位，待开发阶段代码完成后补充 | 通过[aclnnSolveTril](./docs/aclnnSolveTril.md)调用 aclnnSolveTril 算子 |
