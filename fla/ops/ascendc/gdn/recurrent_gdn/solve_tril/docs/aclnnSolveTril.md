# aclnnSolveTril

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | x |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | x |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | x |
| <term>Atlas 推理系列产品</term> | x |
| <term>Atlas 训练系列产品</term> | x |

## 功能说明

- 接口功能：计算单位下三角矩阵的逆矩阵。输入为单位下三角矩阵 L = I + A（A 为严格下三角矩阵），输出为 L^{-1}。

- 算法：采用 MXR 算法（MCH + MBH 组合），利用严格下三角矩阵的幂零性质实现高效求逆。

- 计算公式：

  $$
  output = L^{-1}
  $$

  其中 L 为单位下三角矩阵（对角线元素全为 1）。

  MCH（叶子块 16x16 求逆）：

  $$
  (I+A)^{-1} = (I-A)(I+A^2)(I+A^4) \cdots (I+A^{2^{m-1}})
  $$

  MBH（分块递归组装）：

  $$
  L^{-1} = \begin{bmatrix} L_{11}^{-1} & 0 \\ -L_{22}^{-1} L_{21} L_{11}^{-1} & L_{22}^{-1} \end{bmatrix}
  $$

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnSolveTrilGetWorkspaceSize"接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用"aclnnSolveTril"接口执行计算。

```cpp
aclnnStatus aclnnSolveTrilGetWorkspaceSize(
    const aclTensor  *input,
    aclTensor        *output,
    uint64_t         *workspaceSize,
    aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnSolveTril(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnSolveTrilGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>input（aclTensor*）</td>
      <td>输入</td>
      <td>单位下三角矩阵，对应公式中L。</td>
      <td><ul><li>不支持空Tensor（n>=16）。</li><li>最后两个维度必须相等（方阵）。</li><li>矩阵维度n必须是16的倍数。</li><li>输入必须为单位下三角矩阵（对角线为1，上三角为0），算子不校验。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>x</td>
    </tr>
    <tr>
      <td>output（aclTensor*）</td>
      <td>输出</td>
      <td>逆矩阵结果，对应公式中L^{-1}。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与input一致。</li><li>shape与input相同。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>x</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>input、output存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>input的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>input的shape不满足约束（非方阵、n不是16的倍数、维度不在2-3范围内）。</td>
    </tr>
    <tr>
      <td>output的数据类型或shape与input不一致。</td>
    </tr>
  </tbody></table>

## aclnnSolveTril

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnSolveTrilGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码。

## 约束说明

- aclnnSolveTril默认确定性实现。
- 输入矩阵维度n必须是16的倍数（n = 16, 32, 64, 128, 256, ...）。
- 输入必须为单位下三角矩阵（对角线元素为1，上三角部分为0），算子不校验输入合法性。
- 不支持非连续Tensor。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持FLOAT、FLOAT16数据类型。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

```Cpp
// 调用示例占位，待开发阶段代码完成后补充
```
