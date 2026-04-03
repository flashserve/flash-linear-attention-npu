# 算子列表

> 说明：
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](./context/dir_structure.md)。
> - **算子执行硬件单元**：算子运行在AI Core。关于AI Core详细介绍参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中"概念原理和术语 > 硬件架构与数据处理原理"。

项目提供的所有GDA线性注意力算子分类和算子列表如下：

<table><thead>
  <tr>
    <th rowspan="2">算子分类</th>
    <th rowspan="2">算子目录</th>
    <th colspan="2">算子实现</th>
    <th rowspan="2">算子执行硬件单元</th>
    <th rowspan="2">说明</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
  </tr></thead>
<tbody>
  <tr>
    <td>chunk_gated_delta_rule</td>
    <td><a href="../../chunk_gated_delta_rule/chunk_bwd_dqkwg/">chunk_bwd_dqkwg</a></td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Chunk Gated Delta Rule训练反向传播中dq、dk、dw、dg的计算。</td>
  </tr>
  <tr>
    <td>chunk_gated_delta_rule</td>
    <td><a href="../../chunk_gated_delta_rule/chunk_bwd_dv_local/">chunk_bwd_dv_local</a></td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Chunk Gated Delta Rule训练反向传播中dv的局部计算。</td>
  </tr>
  <tr>
    <td>chunk_gated_delta_rule</td>
    <td><a href="../../chunk_gated_delta_rule/chunk_fwd_o/">chunk_fwd_o</a></td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Chunk Gated Delta Rule训练前向传播中输出o的计算。</td>
  </tr>
  <tr>
    <td>chunk_gated_delta_rule</td>
    <td><a href="../../chunk_gated_delta_rule/chunk_gated_delta_rule_bwd_dhu/">chunk_gated_delta_rule_bwd_dhu</a></td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Chunk Gated Delta Rule训练反向传播中dh和du的计算。</td>
  </tr>
  <tr>
    <td>chunk_gated_delta_rule</td>
    <td><a href="../../chunk_gated_delta_rule/chunk_gated_delta_rule_fwd_h/">chunk_gated_delta_rule_fwd_h</a></td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Chunk Gated Delta Rule训练前向传播中隐藏状态h的计算。</td>
  </tr>
  <tr>
    <td>chunk_gated_delta_rule</td>
    <td><a href="../../chunk_gated_delta_rule/prepare_wy_repr_bwd_da/">prepare_wy_repr_bwd_da</a></td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Chunk Gated Delta Rule训练中WY表示的反向传播da计算。</td>
  </tr>
  <tr>
    <td>chunk_gated_delta_rule</td>
    <td><a href="../../chunk_gated_delta_rule/prepare_wy_repr_bwd_full/">prepare_wy_repr_bwd_full</a></td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Chunk Gated Delta Rule训练中WY表示的完整反向传播计算。</td>
  </tr>
  <tr>
    <td>recurrent_gated_delta_rule</td>
    <td><a href="../../recurrent_gated_delta_rule/README.md">recurrent_gated_delta_rule</a></td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>增量推理场景的Recurrent Gated Delta Rule算子。</td>
  </tr>
</tbody>
</table>
