# 算子列表

> 说明：
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](./context/dir_structure.md)。
> - **算子执行硬件单元**：大部分算子运行在AI Core，少部分算子运行在AI CPU。默认情况下，项目中提到的算子一般指AI Core算子。关于AI Core和AI CPU详细介绍参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“概念原理和术语 > 硬件架构与数据处理原理”。
> - **算子接口列表**：为方便调用算子，CANN提供一套C API执行算子，一般以aclnn为前缀，全量接口参见[aclnn列表](op_api_list.md)。

项目提供的所有算子分类和算子列表如下：

<table><thead>
  <tr>
    <th rowspan="2">算子分类</th>
    <th rowspan="2">算子目录</th>
    <th colspan="2">算子实现</th>
    <th>aclnn调用</th>
    <th>图模式调用</th>
    <th rowspan="2">算子执行硬件单元</th>
    <th rowspan="2">说明</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
    <th>op_api</th>
    <th>op_graph</th>
  </tr></thead>
<tbody>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/attention_update/README.md">attention_update</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>将各SP域PA算子的输出的中间结果lse，localOut两个局部变量结果更新成全局结果。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_attention_score/README.md">flash_attention_score</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>使用FlashAttention算法实现self-attention（自注意力）的计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/flash_attention_score_grad/README.md">flash_attention_score_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>训练场景下计算注意力的反向输出，即FlashAttentionScore的反向计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/fused_infer_attention_score/README.md">fused_infer_attention_score</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>decode & prefill场景的FlashAttention算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../attention/gather_pa_kv_cache/README.md">gather_pa_kv_cache</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>根据blockTables中的blockId值、seqLens中key/value的seqLen从keyCache/valueCache中将内存不连续的token搬运、拼接成连续的key/value序列。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/incre_flash_attention/README.md">incre_flash_attention</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>增量推理场景的FlashAttention算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/kv_quant_sparse_flash_attention/README.md">kv_quant_sparse_flash_attention</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>在Sparse Flash Attention的基础上支持了[Per-Token-Head-Tile-128量化]输入。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/lightning_indexer/README.md">lightning_indexer</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>基于一系列操作得到每一个token对应的Top-k个位置。</td>
  </tr>
    <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_preprocess/README.md">mla_preprocess</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>推理MlaPreprocess算子</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_prolog/README.md">mla_prolog</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>推理MlaProlog算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_prolog_v2/README.md">mla_prolog_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>推理MlaPrologV2WeightNz算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/mla_prolog_v3/README.md">mla_prolog_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>推理MlaPrologV3WeightNz算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress/README.md">nsa_compress</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>训练场景下，使用NSA Compress算法减轻long-context的注意力计算，实现在KV序列维度进行压缩。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress_attention/README.md">nsa_compress_attention</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>NSA中compress attention以及select topk索引计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress_attention_infer/README.md">nsa_compress_attention_infer</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>实现Native Sparse Attention推理过程中，Compress Attention的计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress_grad/README.md">nsa_compress_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>aclnnNsaCompress算子的反向计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_compress_with_cache/README.md">nsa_compress_with_cache</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>实现Native-Sparse-Attention推理阶段的KV压缩。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_selected_attention_infer/README.md">nsa_selected_attention_infer</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>实现Native Sparse Attention推理过程中，Selected Attention的计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_selected_attention/README.md">nsa_selected_attention</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>训练场景下，实现NativeSparseAttention算法中selected-attention（选择注意力）的计算。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/nsa_selected_attention_grad/README.md">nsa_selected_attention_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/prompt_flash_attention/README.md">prompt_flash_attention</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>全量推理场景的FlashAttention算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/quant_lightning_indexer/README.md">quant_lightning_indexer</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>推理场景下，SparseFlashAttention前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/recurrent_gated_delta_rule/README.md">recurrent_gated_delta_rule</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>增量推理场景的Recurrent Gated Delta Rule算子。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/ring_attention_update/README.md">ring_attention_update</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>训练场景下，更新两次FlashAttention的结果。</td>
  </tr>
  <tr>
    <td>attention</td>
    <td><a href="../../attention/sparse_flash_attention/README.md">sparse_flash_attention</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>针对大序列长度推理场景的高效注意力计算模块。</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/ffn/README.md">ffn</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>提供MoeFFN和FFN的计算功能。</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/swin_attention_ffn/README.md">swin_attention_ffn</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>全量推理场景的FlashAttention算子。</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/swin_transformer_ln_qkv/README.md">swin_transformer_ln_qkv</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成fp16权重场景下的Swin Transformer 网络模型的Q、K、V 的计算。</td>
  </tr>
  <tr>
    <td>ffn</td>
    <td><a href="../../ffn/swin_transformer_ln_qkv_quant/README.md">swin_transformer_ln_qkv_quant</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Swin Transformer 网络模型 完成 Q、K、V 的计算。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul/README.md">grouped_matmul</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>实现分组矩阵乘计算。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_add/README.md">grouped_matmul_add</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_finalize_routing/README.md">grouped_matmul_finalize_routing</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_swiglu_quant/README.md">grouped_matmul_swiglu_quant</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>融合GroupedMatmul 、dequant、swiglu和quant。</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/grouped_matmul_swiglu_quant_v2/README.md">grouped_matmul_swiglu_quant_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>融合GroupedMatmul 、dequant、swiglu和quant，新增了MXFP8量化场景（仅Ascend 950PR/Ascend 950DT AI处理器支持）</td>
  </tr>
  <tr>
    <td>gmm</td>
    <td><a href="../../gmm/quant_grouped_matmul_inplace_add/README.md">quant_grouped_matmul_inplace_add</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>实现分组矩阵乘计算和加法计算。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/all_gather_matmul/README.md">all_gather_matmul</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成AllGather通信与MatMul计算融合。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/allto_all_all_gather_batch_mat_mul/README.md">allto_all_all_gather_batch_mat_mul</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成AllToAll、AllGather集合通信与BatchMatMul计算融合、并行。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/allto_allv_grouped_mat_mul/README.md">allto_allv_grouped_mat_mul</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合，先通信后计算。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/ffn_to_attention/README.md">ffn_to_attention</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>一个通信域内的FFN节点对Attention节点发送数据并写状态位，以检测通信链路是否正常。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/attention_to_ffn/README.md">attention_to_ffn</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>一个通信域内的Attention节点对FFN节点发送数据并写状态位，以检测通信链路是否正常。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/batch_mat_mul_reduce_scatter_allto_all/README.md">batch_mat_mul_reduce_scatter_allto_all</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>实现BatchMatMul计算与ReduceScatter、AllToAll集合通信并行。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/distribute_barrier/README.md">distribute_barrier</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成通信域内的全卡同步，xRef仅用于构建Tensor依赖，接口内不对xRef做任何操作。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/elastic_receivable_info_collect/README.md">elastic_receivable_info_collect</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>收集一个通信域内的所有卡发送的数据并整理输出，以检测通信链路是否正常。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/elastic_receivable_test/README.md">elastic_receivable_test</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对一个通信域内的所有卡发送数据并写状态位，以检测通信链路是否正常。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/grouped_mat_mul_all_reduce/README.md">grouped_mat_mul_all_reduce</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>在融合GroupedMatMul的基础上实现多卡并行AllReduce功能，实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/grouped_mat_mul_allto_allv/README.md">grouped_mat_mul_allto_allv</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成路由专家GroupedMatMul、Unpermute、AlltoAllv融合并实现与共享专家MatMul并行融合，先计算后通信。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/inplace_matmul_all_reduce_add_rms_norm/README.md">inplace_matmul_all_reduce_add_rms_norm</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成mm + all_reduce + add + rms_norm计算。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/matmul_all_reduce/README.md">matmul_all_reduce</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成MatMul计算与AllReduce通信融合。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/matmul_all_reduce_add_rms_norm/README.md">matmul_all_reduce_add_rms_norm</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td> 完成mm + all_reduce + add + rms_norm计算</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/matmul_reduce_scatter/README.md">matmul_reduce_scatter</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成mm + reduce_scatter_base计算。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine/README.md">moe_distribute_combine</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine_add_rms_norm/README.md">moe_distribute_combine_add_rms_norm</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加），之后完成Add + RmsNorm融合。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_combine_v2/README.md">moe_distribute_combine_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch/README.md">moe_distribute_dispatch</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_dispatch_v2/README.md">moe_distribute_dispatch_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_distribute_buffer_reset/README.md">moe_distribute_buffer_reset</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>故障检测流程中，对EP通信域做数据区与状态区的清理。</td>
  </tr>
  <tr>
    <td>mc2</td>
    <td><a href="../../mc2/moe_update_expert/README.md">moe_update_expert</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>完成每个token的topK个专家逻辑专家号到物理卡号的映射。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_compute_expert_tokens/README.md">moe_compute_expert_tokens</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_finalize_routing/README.md">moe_finalize_routing</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>MoE计算中，最后处理合并MoE FFN的输出结果。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_finalize_routing_v2/README.md">moe_finalize_routing_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>MoE计算中，最后处理合并MoE FFN的输出结果。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_finalize_routing_v2_grad/README.md">moe_finalize_routing_v2_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>aclnnMoeFinalizeRoutingV2的反向传播。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_gating_top_k/README.md">moe_gating_top_k</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_gating_top_k_softmax/README.md">moe_gating_top_k_softmax</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>MoE计算中，对x的输出做Softmax计算，取TopK操作。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_gating_top_k_softmax_v2/README.md">moe_gating_top_k_softmax_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>MoE计算中，如果renorm=0，先对x的输出做Softmax计算，再取topk操作；如果renorm=1，先对x的输出做topk操作，再进行Softmax操作。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing/README.md">moe_init_routing</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>MoE的routing计算，根据<a href="../../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md">aclnnMoeGatingTopKSoftmax</a>的计算结果做routing处理。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_quant/README.md">moe_init_routing_quant</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>MoE的routing计算，根据<a href="../../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md">aclnnMoeGatingTopKSoftmax</a>的计算结果做routing处理，并对结果进行量化。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_quant_v2/README.md">moe_init_routing_quant_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>MoE的routing计算，根据<a href="../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md">aclnnMoeGatingTopKSoftmaxV2</a>的计算结果做routing处理。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_v2/README.md">moe_init_routing_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>以MoeGatingTopKSoftmax算子的输出x和expert_idx作为输入，并输出Routing矩阵expanded_x等结果供后续计算使用。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_v2_grad/README.md">moe_init_routing_v2_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td><a href="../../moe/moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md">aclnnMoeInitRoutingV2</a>的反向传播，完成tokens的加权求和。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_init_routing_v3/README.md">moe_init_routing_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>MoE的routing计算，根据<a href="../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md">aclnnMoeGatingTopKSoftmaxV2</a>的计算结果做routing处理，支持不量化和动态量化模式。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_re_routing/README.md">moe_re_routing</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>MoE网络中，进行AlltoAll操作从其他卡上拿到需要算的token后，将token按照专家顺序重新排列。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute/README.md">moe_token_permute</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>MoE的permute计算，根据索引indices将tokens广播并排序。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_grad/README.md">moe_token_permute_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>aclnnMoeTokenPermute的反向传播计算。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_ep/README.md">moe_token_permute_with_ep</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>MoE的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_ep_grad/README.md">moe_token_permute_with_ep_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>aclnnMoeTokenPermuteWithEp的反向传播计算。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_routing_map/README.md">moe_token_permute_with_routing_map</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>aclnnMoeTokenPermuteWithRoutingMap的反向传播。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_permute_with_routing_map_grad/README.md">moe_token_permute_with_routing_map_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>MoE的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute/README.md">moe_token_unpermute</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘；最后进行累加求和，并输出计算结果。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_grad/README.md">moe_token_unpermute_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>aclnnMoeTokenUnpermuteGrad的反向传播。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_ep/README.md">moe_token_unpermute_with_ep</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>根据sortedIndices存储的下标位置，去获取permutedTokens中的输入数据与probs相乘，并进行合并累加。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_ep_grad/README.md">moe_token_unpermute_with_ep_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>aclnnMoeTokenUnpermuteWithEp的反向传播。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_routing_map/README.md">moe_token_unpermute_with_routing_map</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>对经过aclnnMoeTokenpermuteWithRoutingMap处理的permutedTokens，累加回原unpermutedTokens。根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘，最后进行累加求和，并输出计算结果。</td>
  </tr>
  <tr>
    <td>moe</td>
    <td><a href="../../moe/moe_token_unpermute_with_routing_map_grad/README.md">moe_token_unpermute_with_routing_map_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>aclnnMoeTokenUnpermuteWithRoutingMap的反向传播。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/apply_rotary_pos_emb/README.md">apply_rotary_pos_emb</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>执行旋转位置编码计算，推理网络为了提升性能，将query和key两路算子融合成一路。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/dequant_rope_quant_kvcache/README.md">dequant_rope_quant_kvcache</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入张量（x）进行dequant（可选）后，按`sizeSplits`（为切分的长度）对尾轴进行切分，划分为q、k、vOut，对q、k进行旋转位置编码，生成qOut和kOut，之后对kOut和vOut进行量化并按照`indices`更新到kCacheRef和vCacheRef上。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/interleave_rope/README.md">interleave_rope</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>针对单输入 x 进行旋转位置编码。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/rope_quant_kvcache/README.md">rope_quant_kvcache</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入张量的尾轴进行切分。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/rope_with_sin_cos_cache/README.md">rope_with_sin_cos_cache</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>推理网络为了提升性能，将sin和cos输入通过cache传入，执行旋转位置编码计算。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/rotary_position_embedding/README.md">rotary_position_embedding</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>执行单路旋转位置编码计算。</td>
  </tr>
  <tr>
    <td>posembedding</td>
    <td><a href="../../posembedding/rotary_position_embedding_grad/README.md">rotary_position_embedding_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>执行单路旋转位置编码的反向计算。</td>
  </tr>
</tbody>
</table>