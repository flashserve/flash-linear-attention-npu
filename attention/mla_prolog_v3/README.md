# MlaPrologV3
## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明
-  **功能更新**：（相对与aclnnMlaPrologV2weightNz的差异）
    -  新增query与key的尺度矫正因子，分别对应qcQrScale（$\alpha_q$）与kcScale（$\alpha_{kv}$）。
    -  新增可选输入与参数，将cache_mode由必选改为可选。具体包括：
        - actualSeqLenOptional：用于BS合轴且CacheMode="PA_BLK_BSND"/"PA_BLK_NZ"时，指定当前batch中实际的序列长度。
        - kNopeClipAlphaOptional：表示对kv_cache做clip操作时的缩放因子。
        - queryNormFlag：表示是否输出query_norm，以及量化场景下的dequant_scale_q_norm。
        - weightQuantMode：表示weight_dq、weight_uq_qr、weight_uk、weight_dkv_kr的量化模式。
        - kvCacheQuantMode：表示kv_cache的量化模式。
        - queryQuantMode：表示query的量化模式。
        - ckvkrRepoMode：表示kv_cache和kr_cache的存储模式。
        - quantScaleRepoMode：表示量化scale的存储模式。
        - tileSize：表示per-tile量化时每个tile的大小。
        - queryNormOptional：公式中tokenX做rmsNorm后的输出tensor（对应$c^Q$）。
        - dequantScaleQNormOptional：query_norm的输出tensor的量化参数。
    -  调整cacheIndex参数的名称与位置，对应当前的cacheIndexOptional。
-  **算子功能**：推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程分为四路，首先对输入$x$乘以$W^{DQ}$进行下采样和RmsNorm后分为两路，第一路乘以$W^{UQ}$和$W^{UK}$经过两次上采样后得到$q^N$；第二路乘以$W^{QR}$后经过旋转位置编码（ROPE）得到$q^R$；第三路是输入$x$乘以$W^{DKV}$进行下采样和RmsNorm后传入Cache中得到$k^C$；第四路是输入$x$乘以$W^{KR}$后经过旋转位置编码后传入另一个Cache中得到$k^R$。
-  **计算公式**：

    RmsNorm公式

    $$
    \mathrm{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\mathrm{RMS}(x)}
    $$

    $$
    \mathrm{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    Query计算公式，包括下采样，RmsNorm和两次上采样

    $$
    c^Q = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$
    其中 $\alpha_q$ 是 Query 的尺度矫正参数。

    对Query进行ROPE旋转位置编码

    $$
    q^R = \mathrm{ROPE}(c^Q \cdot W^{QR})
    $$

    Key计算公式，包括下采样和RmsNorm，将计算结果存入cache

    $$
    c^{KV} = \alpha_{kv}\cdot\mathrm{RmsNorm}(x \cdot W^{DKV})
    $$

    $$
    k^C = \mathrm{Cache}(c^{KV})
    $$
    其中 $\alpha_{kv}$ 是 Key 的尺度矫正参数。

    对Key进行ROPE旋转位置编码，并将结果存入cache

    $$
    k^R = \mathrm{Cache}(\mathrm{ROPE}(x \cdot W^{KR}))
    $$

## 参数说明
| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| token_x                     | 输入      | 公式中计算Query和Key的输入tensor | INT8, BF16 | ND         |
| weight_dq                   | 输入      | 公式中计算Query的下采样权重矩阵$W^{DQ}$ <br> 不转置的情况下各个维度的表示：（k，n） | INT8, BF16 | FRACTAL_NZ |
| weight_uq_qr                 | 输入      | 公式中计算Query的上采样权重矩阵$W^{UQ}$和位置编码权重矩阵$W^{QR}$ <br> 不转置的情况下各个维度的表示：（k，n）| INT8, BF16 | FRACTAL_NZ |
| weight_uk                   | 输入      | 公式中计算Key的上采样权重$W^{UK}$ | FLOAT16, BF16       | ND         |
| weight_dkv_kr                | 输入      | 公式中计算Key的下采样权重矩阵$W^{DKV}$和位置编码权重矩阵$W^{KR}$ <br> 不转置的情况下各个维度的表示：（k，n）| INT8, BF16| FRACTAL_NZ |
| rmsnorm_gamma_cq             | 输入      | 计算$c^Q$的RmsNorm公式中$\gamma$参数 | FLOAT16, BF16       | ND         |
| rmsnorm_gamma_ckv            | 输入      | 计算$c^{KV}$的RmsNorm公式中$\gamma$参数 | FLOAT16, BF16       | ND         |
| rope_sin                    | 输入      | 旋转位置编码的正弦参数矩阵 | FLOAT16, BF16       | ND         |
| rope_cos                    | 输入      | 旋转位置编码的余弦参数矩阵 | FLOAT16, BF16       | ND         |
| kv_cache                 | 输入/ 输出| cache索引的aclTensor，计算结果原地更新（对应$k^C$）| FLOAT16, BF16, NT8 | ND         |
| kr_cache                 | 输入/ 输出| key位置编码的cache，计算结果原地更新（对应$k^R$） | FLOAT16, BF16, INT8 | ND         |
| cache_index                 | 输入      | 存储kvCache和krCache的索引 | INT64          | ND         |
| dequant_scale_x      | 输入      | token_x的反量化参数  | FLOAT          | ND         |
| dequant_scale_w_dq    | 输入      | weight_dq的反量化参数 | FLOAT          | ND         |
| dequant_scale_w_uq_qr  | 输入      | MatmulQcQr矩阵乘后反量化的per-channel参数 | FLOAT          | ND         |
| dequant_scale_w_dkv_kr | 输入      | weight_dkv_kr的反量化参数 | FLOAT          | ND         |
| quant_scale_ckv      | 输入      | KVCache输出量化参数 | FLOAT          | ND         |
| quant_scale_ckr      | 输入      | KRCache输出量化参数 | FLOAT          | ND         |
| smooth_scales_cq     | 输入      | RmsNormCq输出动态量化参数 | FLOAT          | ND         |
| actual_seq_len                 | 输入      | 预留参数，当前版本暂未使用，必须传入空指针 | INT32          | ND         |
| k_nope_clip_alpha    | 输入      | 对kv_cache做clip操作时的缩放因子  | FLOAT  | ND         |    
| rmsnorm_epsilon_cq           | 输入      | 计算$c^Q$的RmsNorm公式中$\epsilon$参数 | DOUBLE         | -          |
| rmsnorm_epsilon_ckv          | 输入      | 计算$c^{KV}$的RmsNorm公式中$\epsilon$参数 | DOUBLE         | -          |
| cache_mode          | 输入      | kvCache模式 | CHAR*          | -          |
| query_norm_flag           | 输入      | 表示是否输出query_norm，Host侧参数 | BOOL         | -          |
| weight_quant_mode          | 输入      | 表示weight_dq、weight_uq_qr、weight_uk、weight_dkv_kr的量化模式 | INT64         | -          |
| kv_cache_quant_mode           | 输入      | 表示kv_cache的量化模式 | INT64         | -          |
| query_quant_mode          | 输入      | 表示query的量化模式 | INT64         | -          |
| ckvkr_repo_mode           | 输入      | 表示kv_cache和kr_cache的存储模式 | INT64         | -          |
| quant_scale_repo_mode           | 输入      | 表示量化scale的存储模式 | INT64         | -          |
| tile_size          | 输入      | 表示per-tile量化时每个tile的大小，需要传入128 | INT64         | -          |
| qc_qr_scale          | 输入      | Query的尺度矫正参数，对应$\alpha_q$，默认传1.0 | DOUBLE         | -          |
| kc_scale          | 输入      | Key的尺度矫正参数，对应$\alpha_{kv}$，默认传1.0 | DOUBLE         | -          |
| query                   | 输出      | 公式中Query的输出tensor（对应$q^N$） | FLOAT16, BF16, INT8 | ND         |
| query_rope               | 输出      | 公式中Query位置编码的输出tensor（对应$q^R$） | FLOAT16, BF16, INT8       | ND |
| dequant_scale_q_nope | 输出     | 表示Query的输出tensor的量化参数   | FLOAT             | ND         |
| query_norm               | 输出      | 公式中tokenX做rmsNorm后的输出tensor（对应$c^Q$） | INT8, BF16 | ND |
| dequant_scale_q_norm | 输出     | query_norm的输出tensor的量化参数   | FLOAT | ND         |
                   
## 约束说明

-   shape约束
    -   若token_x的维度采用BS合轴，即(T, He)
        - rope_sin和rope_cos的shape为(T, Dr)
        - cache_index的shape为(T,)
        - dequant_scale_x的shape为(T, 1)
        - query的shape为(T, N, Hckv)
        - query_rope的shape为(T, N, Dr)
        - 全量化场景下，dequant_scale_q_nope的shape为(T, N, 1)，其他场景下为(1)
    - 若token_x的维度不采用BS合轴，即(B, S, He)
        - rope_sin和rope_cos的shape为(B, S, Dr)
        - cache_index的shape为(B, S)
        - dequant_scale_x的shape为(B*S, 1)
        - query的shape为(B, S, N, Hckv)
        - query_rope的shape为(B, S, N, Dr)
        - 全量化场景下，dequant_scale_q_nope的shape为(B*S, N, 1)，其他场景下为(1)
    -   B、S、T、Skv值允许一个或多个取0，即Shape与B、S、T、Skv值相关的入参允许传入空Tensor，其余入参不支持传入空Tensor。
        - 如果B、S、T取值为0，则query、query_rope输出空Tensor，kv_cache、kr_cache不做更新。
        - 如果Skv取值为0，则query、query_rope、dequant_scale_q_nope正常计算，kv_cache、kr_cache不做更新，即输出空Tensor。
-   特殊约束
    - per-tile量化模式下，ckvkr_repo_mode和quant_scale_repo_mode必须同时为1。
    - per-tile量化模式下，cache_mode只支持PA_BSND, BSND和TND。
    - 当ckvkr_repo_mode值为1时，kr_cache必须为空Tensor（即shape的乘积为0）。
-  aclnnMlaPrologV3WeightNz接口支持场景：
    <table style="table-layout: auto;" border="1">
    <tr>
      <th colspan="2">场景</th>
      <th>含义</th>
    </tr>
    <tr>
      <td colspan="2">非量化</td>
      <td>
          入参：所有入参皆为非量化数据 <br> 
          出参：所有出参皆为非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="3">部分量化</td>
      <td>kv_cache非量化 </td>
      <td>
          入参：weight_uq_qr传入pertoken量化数据，其余入参皆为非量化数据 <br>
          出参：所有出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td>kv_cache per-channel量化 </td>
      <td>
          入参：weight_uq_qr传入pertoken量化数据，kv_cache、kr_cache传入perchannel量化数据，其余入参皆为非量化数据 <br>
          出参：kv_cache、kr_cache返回perchannel量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td>kv_cache per-tile量化 </td>
      <td>
          入参：weight_uq_qr传入pertoken量化数据，kv_cache传入per-tile量化数据,其余入参皆为非量化数据 <br>
          出参：kv_cache_out返回pertile量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="3">全量化</td>
      <td> kv_cache非量化</td>
      <td>
          入参：token_x传入pertoken量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入perchannel量化数据，其余入参皆为非量化数据 <br>
          出参：所有出参皆为非量化数据
      </td>
    </tr>
    <tr>
      <td> kv_cache per-tensor量化 </td>
      <td>
          入参：token_x传入pertoken量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入perchannel量化数据，kv_cache传入pertensor量化数据，其余入参皆为非量化数据 <br>
          出参：query_out返回pertoken_head量化数据，kv_cache出参返回pertensor量化数据，其余出参范围非量化数据
      </td>
    </tr>
    <tr>
      <td> kv_cache per-tile量化 </td>
      <td>
          入参：token_x传入pertoken量化数据，weight_dq、weight_uq_qr、weight_dkv_kr传入perchannel量化数据，其余入参皆为非量化数据 <br>
          出参：query_out返回pertoken_head量化数据，kv_cache出参返回pertensor量化数据，其余出参范围非量化数据
      </td>
    </tr>
  </table>

## 调用说明

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_mla_prolog_v3.cpp">MlaPrologV3接口测试用例代码
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnMlaPrologV3WeightNz.md">aclnnMlaPrologV3WeightNz
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>