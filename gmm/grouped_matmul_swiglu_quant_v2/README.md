# GroupedMatmulSwigluQuantV2

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT AI处理器</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明
- 算子功能：融合GroupedMatmul 、dequant、swiglu和quant，详细解释见计算公式。
- 计算公式：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    <details>
    <summary>量化场景A8W8（A指激活矩阵，W指权重矩阵，8指INT8数据类型）：</summary>
    <a id="量化场景A8W8"></a>

      - **定义**：

        * **⋅** 表示矩阵乘法。
        * **⊙** 表示逐元素乘法。
        * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
        * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
      - **输入**：

        * $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
        * $W∈\mathbb{Z_8}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
        * $w\_scale∈\mathbb{R}^{E \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，N是输出维度。
        * $x\_scale∈\mathbb{R}^{M}$：激活矩阵（左矩阵）的逐 token缩放因子，M是总token数。
        * $grouplist∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
      - **输出**：

        * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
        * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。

      - **计算过程**

        - 1.根据groupList[i]确定当前分组的 token ，$i \in [0,Len(groupList)]$。

          >例子：假设groupList=[3,4,4,6]、groupListType=cumsum或groupList=[3,1,0,2]、groupListType=count。
          >
          >注：以上两种不同的分组方式，实际为相同的分组结果。
          >
          >第0个右矩阵`W[0,:,:]`，对应索引位置[0,3)的token`x[0:3]`（共3-0=3个token），对应`x_scale[0:3]`、`w_scale[0]`、`bias[0]`、`offset[0] `、`Q[0:3]`、`Q_scale[0:3]`、`Q_offset[0:3]`；
          >
          >第1个右矩阵`W[1,:,:]`，对应索引位置[3,4)的token`x[3:4]`（共4-3=1个token），对应`x_scale[3:4]`、`w_scale[1]`、`bias[1]`、`offset[1] `、`Q[3:4]`、`Q_scale[3:4]`、`Q_offset[3:4]`；
          >
          >第2个右矩阵`W[2,:,:]`，对应索引位置[4,4)的token`x[4:4]`（共4-4=0个token），对应`x_scale[4:4]`、`w_scale[2]`、`bias[2]`、`offset[2] `、`Q[4:4]`、`Q_scale[4:4]`、`Q_offset[4:4]`；
          >
          >第3个右矩阵`W[3,:,:]`，对应索引位置[4,6)的token`x[4:6]`（共6-4=2个token），对应`x_scale[4:6]`、`w_scale[3]`、`bias[3]`、`offset[3] `、`Q[4:6]`、`Q_scale[4:6]`、`Q_offset[4:6]`；
          >
          >注：grouplist中未指定的部分将不会参与更新。
          >例如当groupList=[12,14,18]、GroupListType=cumsum，X的shape为[30，:]时。
          >
          >则第一个输出Q的shape为[30，:]，其中Q[18:，：]的部分不会进行更新和初始化，其中数据为显存空间申请时的原数据。
          >
          >同理，第二个输出Q的shape为[30]，其中Q\_scale[18:]的部分不会进行更新或初始化，其中数据为显存空间申请时的原数据。
          >
          >即输出的Q[:grouplist[-1],:]和Q\_scale[:grouplist[-1]]为有效数据部分。

        - 2.根据分组确定的入参进行如下计算：

          $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ BroadCast} \odot w\_scale_{i\ BroadCast}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp;其中$Swish(x)=\frac{x}{1+e^{-x}}$

        - 3.量化输出结果

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>
    <details>
    <summary>MSD场景A8W4（A指激活矩阵，W指权重矩阵，8指INT8数据类型，4指INT4数据类型）：</summary>
    <a id="MSD场景A8W4"></a>
    
      - **定义**：
        * **⋅** 表示矩阵乘法。
        * **⊙** 表示逐元素乘法。
        * $\left \lfloor x\right \rceil$ 表示将x四舍五入到最近的整数。
        * $\mathbb{Z_8} = \{ x \in \mathbb{Z} | −128≤x≤127 \}$
        * $\mathbb{Z_4} = \{ x \in \mathbb{Z} | −8≤x≤7 \}$
        * $\mathbb{Z_{32}} = \{ x \in \mathbb{Z} | -2147483648≤x≤2147483647 \}$
      - **输入**：
        * $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
        * $W∈\mathbb{Z_4}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
        * $weightAsistMatrix∈\mathbb{R}^{E \times N}$：计算矩阵乘时的辅助矩阵（生成辅助矩阵的计算过程见下文）。
        * $w\_scale∈\mathbb{R}^{E \times K\_group\_num \times N}$：分组权重矩阵（右矩阵）的逐通道缩放因子，E是专家个数，K\_group\_num 是在K轴维 度上的分组数，N是输出维度。
        * $x\_scale∈\mathbb{R}^{M}$：激活矩阵（左矩阵）的逐token缩放因子，M是总token数。
        * $grouplist∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
      - **输出**：
        * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
        * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
      - **计算过程**
        - 1.根据groupList[i]确定当前分组的token，$i \in [0,Len(groupList)]$。
          - 分组逻辑与A8W8相同。
        - 2.生成辅助矩阵（weightAsistMatrix）的计算过程（请注意weightAsistMatrix部分计算为离线生成作为输入，并非算子内部完成）：
          - 当为per-channel量化（$w\_scale$为2维）：

            $weightAsistMatrix_{i} = 8 × weightScale × Σ_{k=0}^{K-1} weight[:,k,:]$

          - 当为per-group量化（$w\_scale$为3维）：

            $weightAsistMatrix_{i} = 8 × Σ_{k=0}^{K-1} (weight[:,k,:] × weightScale[:, ⌊k/num\_per\_group⌋, :])$

            注：$num\_per\_group = K // K\_group\_num$

        - 3.根据分组确定的入参进行如下计算：

          - 3.1.将左矩阵$\mathbb{Z_8}$，转变为高低位 两部分的$\mathbb{Z_4}$
            $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$
            $X\_low\_4bits_{i} = X_{i} \& 0x0f - 8$
          - 3.2.做矩阵乘时，使能per-channel或per-group量化
            per-channel：

            $C\_high_{i} = (X\_high\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

            $C\_low_{i} = (X\_low\_4bits_{i} \cdot W_{i}) \odot w\_scale_{i}$

            per-group：

            $C\_high_{i} = \\ Σ_{k=0}^{K-1}((X\_high\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

            $C\_low_{i} = \\ Σ_{k=0}^{K-1}((X\_low\_4bits_{i}[:, k * num\_per\_group : (k+1) * num\_per\_group] \cdot W_{i}[k *   num\_per\_group : (k+1) * num\_per\_group, :]) \odot w\_scale_{i}[k, :] )$

          - 3.3.将高低位的矩阵乘结果还原为整体的结果

            $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAsistMatrix_{i}) \odot x\_scale_{i}$

            $C_{i,act}, gate_{i} = split(C_{i})$

            $S_{i}=Swish(C_{i,act})\odot gate_{i}$  &nbsp;&nbsp; 其中$Swish(x)=\frac{x}{1+e^{-x}}$

        - 3.量化输出结果

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$
    </details>

  - <term>Ascend 950PR/Ascend 950DT AI处理器</term>：
    <details>
    <summary>MX量化场景：</summary>

      - **定义**：

        * **⋅** 表示矩阵乘法。
        * **⊙** 表示逐元素乘法。
      - **计算过程**
        - 1.根据groupList[i]确定当前分组的 token ，$i \in [0,Len(groupList)]$

        - 2.根据分组确定的入参进行如下计算：

          $C_{i} = (X_{i}\cdot W_{i} )\odot xScale_{i\ BroadCast} \odot wScale_{i\ BroadCast}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$，其中$Swish(x)=\frac{x}{1+e^{-x}}$

        - 3.量化输出结果

          $shared\_exp = \left\lfloor \log_2(max_i(|S_i|)) \right\rceil - emax$

          $QScale = 2 ^ {shared\_exp}$

          $Q_i = quantize\_to\_element\_format(S_i/Qscale), \space i\space from\space 1\space to\space blocksize$
          - $emax$: 对应数据类型的最大正则数的指数位。

            |   DataType    | emax |
            | :-----------: | :--: |
            | FLOAT8_E4M3FN |  8   |
            |  FLOAT8_E5M2  |  15  |
            |  FLOAT4_E1M2  |  1   |
            |  FLOAT4_E2M1  |  2   |
          - $blocksize$：指每次量化的元素个数，仅支持32。
    </details>

## 参数说明

<table style="undefined;table-layout: fixed;width: 1567px"><colgroup>
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
    <th style="white-space: nowrap">输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>x</td>
    <td rowspan="1">输入</td>
    <td>表示左矩阵，对应公式中的X。</td>
    <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E1M2、FLOAT4_E2M1、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td rowspan="1">输入</td>
    <td>表示权重矩阵，对应公式中的W。</td>
    <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E1M2、FLOAT4_E2M1、INT8、INT4、INT32</td>
    <td>ND、FRACTAL_NZ</td>
  </tr>
  <tr>
    <td>weightScale</td>
    <td rowspan="1">输入</td>
    <td>表示右矩阵的量化因子，公式中的wScale。</td>
    <td>FLOAT8_E8M0、UINT64、FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightAssistMatrix</td>
    <td rowspan="1">可选输入</td>
    <td>表示计算矩阵乘时的辅助矩阵，公式中的weightAssistMatrix。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td rowspan="1">可选输入</td>
    <td>表示矩阵乘计算的偏移值。</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>xScale</td>
    <td rowspan="1">输入</td>
    <td>表示左矩阵的的量化因子，公式中的xScale。</td>
    <td>FLOAT8_E8M0、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>smoothScale</td>
    <td rowspan="1">可选输入</td>
    <td>表示左矩阵的的量化因子。</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>groupList</td>
    <td rowspan="1">输入</td>
    <td>表示每个分组参与计算的Token个数，公式中的grouplist。</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dequantMode</td>
    <td rowspan="1">输入</td>
    <td>表示反量化计算类型，用于确定激活矩阵与权重矩阵的反量化方式。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dequantDtype</td>
    <td rowspan="1">输入</td>
    <td>表示中间GroupedMatmul的结果数据类型。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>quantMode</td>
    <td rowspan="1">输入</td>
    <td>表示量化计算类型，用于确定swiglu结果的量化模式。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>groupListType</td>
    <td rowspan="1">输入</td>
    <td>表示分组的解释方式，用于确定groupList的语义。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>tuningConfig</td>
    <td rowspan="1">输入</td>
    <td>用于算子预估M/E的大小，走不同的算子模板，以适配不不同场景性能要求。</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>output</td>
    <td rowspan="1">输出</td>
    <td>表示输出的量化结果，公式中的Q。</td>
    <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E1M2、FLOAT4_E2M1、INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>outputScale</td>
    <td rowspan="1">输出</td>
    <td>表示输出的量化因子，公式中的QScale。</td>
    <td>FLOAT8_E8M0、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td rowspan="1">输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>executor</td>
    <td rowspan="1">输出</td>
    <td>返回op执行器，包含了算子计算流程。</td>
    <td>-</td>
    <td>-</td>
  </tr>
</tbody>
</table>

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - x仅支持INT8量化数据类型、不支持其他数据类型。
    - weight仅支持非转置，支持INT8、INT4、INT32数据类型，ND格式shape形如{(E, K, N)}，NZ格式下，当weight数据类型是INT8时shape形如{(E, N / 32, K / 16, 16, 32)}，INT4时shape形如{(E, N / 64, K / 16, 16， 64)}，INT32时shape形如{(E, N / 64, K / 16, 16， 8)}。
    - weightScale，A8W8场景支持FLOAT、FLOAT16、BFLOAT16数据类型，shape只支持2维，形如{(E, N)}；A8W4场景支持UINT64数据类型，shape支持2维和3维，其中per-channel的shape形如{(E, N)}，per-group的shape形如{(E, KGroupCount, N)}。
    - 支持dequantMode参数：0表示激活矩阵per-token，权重矩阵per-channel；1表示激活矩阵per-token，权重矩阵per-group。
    - 不支持dequantDtype参数。
    - 不支持quantMode参数。
    - A8W8/A8W4场景，不支持N轴长度超过10240。
    - A8W8场景，不支持x的尾轴长度大于等于65536。
    - A8W4场景，不支持x的尾轴长度大于等于20000。
    - output仅支持数据类型INT8，shape支持2维，形如(M, N / 2)。
    - outputScale仅支持数据类型FLOAT，shape支持1维，形如(M,)。
- <term>Ascend 950PR/Ascend 950DT AI处理器</term>：
    - 仅支持FLOAT8、FLOAT4量化数据类型，不支持其他数据类型，支持weight转置。
    - x支持FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E1M2、FLOAT4_E2M1数据类型。
    - weight支持FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E1M2、FLOAT4_E2M1数据类型，非转置shape形如{(E, K, N)}，weight转置shape形如{(E, N, K)}。
    - weightScale支持FLOAT8_E8M0数据类型，shape支持4维：weightScale非转置shape形如{(E, ceil(K / 64), N, 2)}，weightScale转置shape形如{(E, N, ceil(K / 64), 2)}。
    - xScale: FLOAT8_E8M0数据类型：shape支持3维，形如(M, ceil(K / 64), 2)。
    - 支持dequantMode参数：当前仅支持取值2，2表示MX量化。
    - 支持dequantDtype参数：当前仅支持取值0，0表示DT_FLOAT。
    - 支持quantMode参数：当前仅支持取值2，2表示MX量化。
    - output支持数据类型FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E1M2、FLOAT4_E2M1，shape支持2维，形如(M, N / 2)。
    - outputScale支持数据类型FLOAT8_E8M0，shape支持3维，形如(M, ceil((N / 2) / 64), 2)。

## 约束说明
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - A8W8/A8W4量化场景下需满足以下约束条件：
        - 数据类型需要满足下表：
        <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
        <col style="width: 319px">
        <col style="width: 144px">
        <col style="width: 671px">
        </colgroup>
        <thead>
          <tr>
            <th>量化场景</th>
            <th>x</th>
            <th>weight</th>
            <th>weightScale</th>
            <th>xScale</th>
            <th>output</th>
            <th>outputScale</th>
          </tr></thead>
        <tbody>
          <tr>
            <td>A8W8</td>
            <td>INT8</td>
            <td>INT8</td>
            <td>FLOAT、FLOAT16、BFLOAT16</td>
            <td>FLOAT</td>
            <td>INT8</td>
            <td>FLOAT</td>
          </tr>
          <tr>
            <td>A8W4</td>
            <td>INT8</td>
            <td>INT4、INT32</td>
            <td>UINT64</td>
            <td>FLOAT</td>
            <td>INT8</td>
            <td>FLOAT</td>
          </tr>
        </tbody>
        </table>

      - A8W8场景下，不支持N轴长度超过10240，不支持x的尾轴长度大于等于65536。
      - A8W4场景下，不支持N轴长度超过10240，不支持x的尾轴长度大于等于20000。
      

  - <term>Ascend 950PR/Ascend 950DT AI处理器</term>：
    - MX量化场景下需满足以下约束条件：
        - 数据类型需要满足下表：
        <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
        <col style="width: 319px">
        <col style="width: 144px">
        <col style="width: 671px">
        </colgroup>
        <thead>
          <tr>
            <th>MX量化场景</th>
            <th>x</th>
            <th>weight</th>
            <th>weightScale</th>
            <th>xScale</th>
            <th>output</th>
            <th>outputScale</th>
          </tr></thead>
        <tbody>
          <tr>
            <td>MXFP8</td>
            <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
            <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
            <td>FLOAT8_E8M0</td>
          </tr>
          <tr>
            <td>MXFP4</td>
            <td>FLOAT4_E1M2、FLOAT4_E2M1</td>
            <td>FLOAT4_E1M2、FLOAT4_E2M1</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT8_E8M0</td>
            <td>FLOAT4_E1M2、FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
            <td>FLOAT8_E8M0</td>
          </tr>
        </tbody>
        </table>

      - MX量化场景下，需满足N为128对齐。
      - MXFP4场景不支持K=2。
      - MXFP4场景需满足K为偶数；当output的数据类型为FLOAT4_E1M2、FLOAT4_E2M1时，需满足N为大于等于4的偶数。

  - 确定性计算：
      - aclnnGroupedMatmulSwigluQuantV2默认为确定性实现。

## 调用说明

| 调用方式      | 调用样例                 | 说明                                                         |
|--------------|-------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_quant_grouped_matmul_swiglu_quant_V2](examples/test_aclnn_grouped_matmul_swiglu_quant_v2_a8w8.cpp) | 通过接口方式调用[GroupedMatmulSwigluQuantV2](docs/aclnnGroupedMatmulSwigluQuantV2.md)算子。 |