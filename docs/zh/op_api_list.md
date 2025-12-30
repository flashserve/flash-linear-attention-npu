# 算子接口（aclnn）

##  使用说明

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

调用算子API时，需引用依赖的头文件和库文件，一般头文件默认在`${INSTALL_DIR}/include/aclnnop`，库文件默认在`${INSTALL_DIR}/lib64`，具体文件如下：

- 依赖的头文件：①方式1 （推荐）：引用算子总头文件aclnn\_ops\_\$\{ops\_project\}.h。②方式2：按需引用单算子API头文件aclnn\_\*.h。
- 依赖的库文件：按需引用算子总库文件libopapi\_\$\{ops\_project\}.so。

其中${INSTALL_DIR}表示CANN安装后文件路径；\$\{ops\_project\}表示算子仓（如math、nn、cv、transformer），请配置为实际算子仓名。

## 接口列表

> **确定性简介**：
>
> - 配置说明：因CANN或NPU型号不同等原因，可能无法保证同一个算子多次运行结果一致。在相同条件下（平台、设备、版本号和其他随机性参数等），部分算子接口可通过`aclrtCtxSetSysParamOpt`（参见[《acl API（C）》](https://hiascend.com/document/redirect/CannCommunityCppApi)）开启确定性算法，使多次运行结果一致。
> - 性能说明：同一个算子采用确定性计算通常比非确定性慢，因此模型单次运行性能可能会下降。但在实验、调试调测等需要保证多次运行结果相同来定位问题的场景，确定性计算可以提升效率。
> - 线程说明：同一线程中只能设置一次确定性状态，多次设置以最后一次有效设置为准。有效设置是指设置确定性状态后，真正执行了一次算子任务下发。如果仅设置，没有算子下发，只能是确定性变量开启但未下发给算子，因此不执行算子。
>   解决方案：暂不推荐一个线程多次设置确定性。该问题在二进制开启和关闭情况下均存在，在后续版本中会解决该问题。

算子接口列表如下：

|    接口名   |   说明     | 确定性说明（A2/A3） |   确定性说明（A5）   |
| ----------- | ------------------- | --------- | ------------- |
|[aclnnAllGatherMatmul](../../mc2/all_gather_matmul/docs/aclnnAllGatherMatmul.md)|完成AllGather通信与MatMul计算融合。|默认确定性实现||
|[aclnnAllGatherMatmulV2](../../mc2/all_gather_matmul_v2/doc/aclnnAllGatherMatmulV2.md)|aclnnAllGatherMatmulV2接口是对aclnnAllGatherMatmul接口的功能拓展。|默认确定性实现||
|[aclnnAlltoAllAllGatherBatchMatMul](../../mc2/allto_all_all_gather_batch_mat_mul/docs/aclnnAlltoAllAllGatherBatchMatMul.md)|完成AllToAll、AllGather集合通信与BatchMatMul计算融合、并行。|默认确定性实现||
|[aclnnAlltoAllvGroupedMatMul](../../mc2/allto_allv_grouped_mat_mul/docs/aclnnAlltoAllvGroupedMatMul.md)|完成路由专家AlltoAllv、Permute、GroupedMatMul融合并实现与共享专家MatMul并行融合。|默认确定性实现||
|[aclnnApplyRotaryPosEmb](../../posembedding/apply_rotary_pos_emb/docs/aclnnApplyRotaryPosEmb.md)|将query和key两路算子融合成一路。执行旋转位置编码计算，计算结果执行原地更新。|默认确定性实现||
|[aclnnApplyRotaryPosEmbV2](../../posembedding/apply_rotary_pos_emb/docs/aclnnApplyRotaryPosEmbV2.md)|将query和key两路算子融合成一路。执行旋转位置编码计算，计算结果执行原地更新。|默认确定性实现||
|[aclnnAttentionToFFN](../../mc2/attention_to_ffn/docs/aclnnAttentionToFFN.md)|将Attention节点上数据发往FFN节点。|默认确定性实现||
|[aclnnFFNToAttention](../../mc2/ffn_to_attention/docs/aclnnFFNToAttention.md)|将FFN节点上的token数据发往Attention节点。|默认非确定性实现||
|[aclnnFlashAttentionScore](../../attention/flash_attention_score/docs/aclnnFlashAttentionScore.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现||
|[aclnnFlashAttentionScoreV2](../../attention/flash_attention_score/docs/aclnnFlashAttentionScoreV2.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现||
|[aclnnFlashAttentionScoreV3](../../attention/flash_attention_score/docs/aclnnFlashAttentionScoreV3.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。对标竞品适配gptoss模型支持sink功能。|默认确定性实现||
|[aclnnFlashAttentionScoreV4](../../attention/flash_attention_score/docs/aclnnFlashAttentionScoreV4.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现||
|[aclnnFlashAttentionScoreGrad](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGrad.md)|训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启||
|[aclnnFlashAttentionScoreGradV2](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGradV2.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionScoreV2](../../attention/flash_attention_score/docs/aclnnFlashAttentionScoreV2.md)的反向计算。|默认非确定性实现，支持配置开启||
|[aclnnFlashAttentionScoreGradV3](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGradV3.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionScoreV3](../../attention/flash_attention_score/docs/aclnnFlashAttentionScoreV3.md)的反向计算。|默认非确定性实现，支持配置开启||
|[aclnnFlashAttentionScoreGradV4](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGradV4.md)|训练场景下计算注意力的反向输出，即[FlashAttentionScoreV4](../../attention/flash_attention_score/docs/aclnnFlashAttentionScoreV4.md)的反向计算。该接口query、key、value参数支持多个长度相等或者长度不相等的sequence。|默认非确定性实现，支持配置开启||
|[aclnnFlashAttentionUnpaddingScoreGrad](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionUnpaddingScoreGrad.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionVarLenScore](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScore.md)的反向计算。|默认非确定性实现，支持配置开启||
|[aclnnFlashAttentionUnpaddingScoreGradV2](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionUnpaddingScoreGradV2.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionVarLenScoreV2](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV2.md)的反向计算。|默认非确定性实现，支持配置开启||
|[aclnnFlashAttentionUnpaddingScoreGradV3](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionUnpaddingScoreGradV3.md)|训练场景下计算注意力的反向输出，即[aclnnFlashAttentionVarLenScoreV3](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV3.md)的反向计算。该接口相较于[aclnnFlashAttentionUnpaddingScoreGradV2](./aclnnFlashAttentionUnpaddingScoreGradV2.md)接口，新增queryRope、keyRope、dqRope和dkRope参数。|默认非确定性实现，支持配置开启||
|[aclnnFlashAttentionUnpaddingScoreGradV4](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionUnpaddingScoreGradV4.md)|训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启||
|[aclnnFlashAttentionUnpaddingScoreGradV5](../../attention/flash_attention_score_grad/docs/aclnnFlashAttentionUnpaddingScoreGradV5.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。增加`sinkInOptional`可选输入。|默认非确定性实现，支持配置开启||
|[aclnnFlashAttentionVarLenScore](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScore.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现||
|[aclnnFlashAttentionVarLenScoreV2](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV2.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现||
|[aclnnFlashAttentionVarLenScoreV3](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV3.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现||
|[aclnnFlashAttentionVarLenScoreV4](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV4.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。|默认确定性实现||
|[aclnnFlashAttentionVarLenScoreV5](../../attention/flash_attention_score/docs/aclnnFlashAttentionVarLenScoreV5.md)|训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。对标竞品适配gptoss模型支持sink功能。|默认确定性实现||
|[aclnnFusedInferAttentionScore](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScore.md)|适配增量&全量推理场景的FlashAttention算子。|默认确定性实现||
|[aclnnFusedInferAttentionScoreV2](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV2.md)|适配增量&全量推理场景的FlashAttention算子。|默认确定性实现||
|[aclnnFusedInferAttentionScoreV3](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV3.md)|适配增量&全量推理场景的FlashAttention算子。|默认确定性实现||
|[aclnnFusedInferAttentionScoreV4](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreV4.md)|适配Decode & Prefill场景的FlashAttention算子。|默认确定性实现||
|[aclnnFusedInferAttentionScoreVX](../../attention/fused_infer_attention_score/docs/aclnnFusedInferAttentionScoreVX.md)|适配增量&全量推理场景的FlashAttention算子。|默认确定性实现||
|[aclnnGatherPaKvCache](../../attention/gather_pa_kv_cache/docs/aclnnGatherPaKvCache.md)|根据blockTables中的blockId值、seqLens中key/value的seqLen从keyCache/valueCache中将内存不连续的token搬运、拼接成连续的key/value序列。|默认确定性实现||
|[aclnnGroupedMatmul](../../gmm/grouped_matmul/docs/aclnnGroupedMatmul.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatmulV2](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulV2.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatmulV3](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulV3.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatmulV4](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatmulV5](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatmulAdd](../../gmm/grouped_matmul_add/docs/aclnnGroupedMatmulAdd.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatmulAddV2](../../gmm/grouped_matmul_add/docs/aclnnGroupedMatmulAddV2.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatMulAlltoAllv](../../mc2/grouped_mat_mul_allto_allv/docs/aclnnGroupedMatMulAlltoAllv.md)|完成路由专家GroupedMatMul、Unpermute、AlltoAllv融合并实现与共享专家MatMul并行融合。|默认确定性实现||
|[aclnnGroupedMatmulFinalizeRouting](../../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRouting.md)|GroupedMatMul和MoeFinalizeRouting的融合算子。|默认非确定性实现，支持配置开启||
|[aclnnGroupedMatmulFinalizeRoutingV2](../../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRoutingV2.md)|GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作。|默认非确定性实现，支持配置开启||
|[aclnnGroupedMatmulFinalizeRoutingV3](../../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRoutingV3.md)|GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作。|默认非确定性实现，支持配置开启||
|[aclnnGroupedMatmulFinalizeRoutingWeightNz](../../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRoutingWeightNz.md)|GroupedMatMul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作，支持输入Weight为AI处理器亲和数据排布格式(NZ)。|默认非确定性实现，支持配置开启||
|[aclnnGroupedMatmulFinalizeRoutingWeightNzV2](../../gmm/grouped_matmul_finalize_routing/docs/aclnnGroupedMatmulFinalizeRoutingWeightNzV2.md)|GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作，支持w为AI处理器亲和数据排布格式(NZ)。|默认非确定性实现，支持配置开启||
|[aclnnGroupedMatmulSwigluQuant](../../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuant.md)|融合GroupedMatMul、Dequant、Swiglu和Quant。|默认确定性实现||
|[aclnnGroupedMatmulSwigluQuantV2](../../gmm/grouped_matmul_swiglu_quant_v2/docs/aclnnGroupedMatmulSwigluQuantV2.md)|融合GroupedMatmul 、dequant、swiglu和quant。|默认确定性实现|默认确定性实现|
|[aclnnGroupedMatmulSwigluQuantWeightNZ](../../gmm/grouped_matmul_swiglu_quant/docs/aclnnGroupedMatmulSwigluQuantWeightNZ.md)|融合GroupedMatMul、Dequant、Swiglu和Quant，输入权重Weight会被强制视为NZ格式。|默认确定性实现||
|[aclnnGroupedMatmulWeightNz](../../gmm/grouped_matmul/docs/aclnnGroupedMatmulWeightNz.md)|实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同，输入权重Weight会被强制视为NZ格式。|默认确定性实现|默认确定性实现|
|[aclnnIncreFlashAttention](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttention.md)|在全量推理的基础上实现增量推理。|默认确定性实现||
|[aclnnIncreFlashAttentionV2](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV2.md)|兼容[aclnnIncreFlashAttention](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttention.md)接口功能，在其基础上新增量化特性。|默认确定性实现||
|[aclnnIncreFlashAttentionV3](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV3.md)|兼容[aclnnIncreFlashAttentionV2](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV2.md)接口功能，在其基础上新增位置编码，page attention，KV cache反量化特性。|默认确定性实现||
|[aclnnIncreFlashAttentionV4](../../attention/incre_flash_attention/docs/aclnnIncreFlashAttentionV4.md)|在全量推理场景的FlashAttention算子的基础上实现增量推理。|默认确定性实现||
|[aclnnInplaceAttentionWorkerScheduler](../../attention/attention_worker_scheduler/docs/aclnnInplaceAttentionWorkerScheduler.md)|Attention和FFN分离部署场景下，Attention侧数据扫描算子。该算子接收来自FFNToAttention算子的输出数据，并对数据进行逐步扫描，确保数据准备就绪。|默认确定性实现||
|[aclnnInplaceFfnWorkerScheduler](../../ffn/ffn_worker_scheduler/docs/aclnnInplaceFfnWorkerScheduler.md)|Attention和FFN分离场景下，FFN侧数据扫描算子。该算子接收AttentionToFFN算子发送的数据，进行扫描并完成数据整理。|默认确定性实现||
|[aclnnInplaceMatmulAllReduceAddRmsNorm](../../mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。|默认非确定性实现，支持配置开启||
|[aclnnInplaceQuantMatmulAllReduceAddRmsNorm](../../mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceQuantMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。|默认非确定性实现，支持配置开启||
|[aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm](../../mc2/inplace_matmul_all_reduce_add_rms_norm/docs/aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。|默认非确定性实现，支持配置开启||
|[aclnnLightningGrad](../../attention/lightning_indexer_grad/docs/aclnnLightningGrad.md)|训练场景下，实现LightningIndexer反向，其中输入有Query, Key, Weights, Dy, Indices，反向主要利用正向计算的Indices从Key中提取TopK序列从而降低Matmul计算量。|默认非确定性实现，不支持配置开启||
|[aclnnMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnMatmulAllReduce.md)|完成MatMul计算与AllReduce通信融合。|默认非确定性实现，支持配置开启||
|[aclnnMatmulAllReduceV2](../../mc2/matmul_all_reduce/docs/aclnnMatmulAllReduceV2.md)|完成MatMul计算与AllReduce通信融合。|默认非确定性实现，支持配置开启||
|[aclnnMatmulAllReduceAddRmsNorm](../../mc2/matmul_all_reduce_add_rms_norm/docs/aclnnMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。|默认非确定性实现，支持配置开启||
|[aclnnMatmulReduceScatter](../../mc2/matmul_reduce_scatter/docs/aclnnMatmulReduceScatter.md)|完成mm + reduce_scatter_base计算。|默认非确定性实现，支持配置开启||
|[aclnnMatmulReduceScatterV2](../../mc2/matmul_reduce_scatter_v2/docs/aclnnMatmulReduceScatterV2.md)|aclnnMatmulReduceScatterV2接口是对[aclnnMatmulReduceScatter](../../mc2/matmul_reduce_scatter/docs/aclnnMatmulReduceScatter.md)接口的功能扩展。|默认非确定性实现，支持配置开启||
|[aclnnMlaPreprocess](../../attention/mla_preprocess/docs/aclnnMlaPreprocess.md)|Multi-Head Latent Attention前处理的计算 。|默认确定性实现||
|[aclnnMlaPreprocessV2](../../attention/mla_preprocess_v2/docs/aclnnMlaPreprocessV2.md)|推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程如下：|默认确定性实现||
|[aclnnMlaProlog](../../attention/mla_prolog/docs/aclnnMlaProlog.md)|Multi-Head Latent Attention前处理的计算 。|默认确定性实现||
|[aclnnMlaPrologV2WeightNz](../../attention/mla_prolog_v2/docs/aclnnMlaPrologV2WeightNz.md)|Multi-Head Latent Attention前处理的计算 。|默认确定性实现||
|[aclnnMlaPrologV3WeightNz](../../attention/mla_prolog_v3/docs/aclnnMlaPrologV3WeightNz.md)|Multi-Head Latent Attention前处理的计算 。|默认非确定性实现，支持配置开启||
|[aclnnMoeComputeExpertTokens](../../moe/moe_compute_expert_tokens/docs/aclnnMoeComputeExpertTokens.md)|MoE计算中，通过二分查找的方式查找每个专家处理的最后一行的位置。|默认确定性实现||
|[aclnnMoeDistributeCombine](../../mc2/moe_distribute_combine/docs/aclnnMoeDistributeCombine.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合。|默认确定性实现||
|[aclnnMoeDistributeCombineV2](../../mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV2.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AllToAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AllToAllV通信，最后将接收的数据整合。|默认确定性实现||
|[aclnnMoeDistributeCombineV3](../../mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV3.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合。|默认确定性实现||
|[aclnnMoeDistributeCombineV4](../../mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV4.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AllToAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AllToAllV通信，最后将接收的数据整合。|默认确定性实现||
|[aclnnMoeDistributeCombineAddRmsNorm](../../mc2/moe_distribute_combine_add_rms_norm/docs/aclnnMoeDistributeCombineAddRmsNorm.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合，之后完成Add + RmsNorm融合。|默认确定性实现||
|[aclnnMoeDistributeCombineAddRmsNormV2](../../mc2/moe_distribute_combine_add_rms_norm/docs/aclnnMoeDistributeCombineAddRmsNormV2.md)|当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合，之后完成Add + RmsNorm融合。|默认确定性实现||
|[aclnnMoeDistributeDispatch](../../mc2/moe_distribute_dispatch/docs/aclnnMoeDistributeDispatch.md)|对Token数据进行量化，当存在TP域通信时，先进行EP域的AllToAllV通信，再进行TP域的AllGatherV通信；当不存在TP域通信时，进行EP域的AllToAllV通信。|默认确定性实现||
|[aclnnMoeDistributeDispatchV2](../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV2.md)|对token数据进行量化，当存在TP域通信时，先进行EP域的AllToAllV通信，再进行TP域的AllGatherV通信；当不存在TP域通信时，进行EP域的AllToAllV通信。|默认确定性实现||
|[aclnnMoeDistributeDispatchV3](../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV3.md)|对token数据进行量化，当存在TP域通信时，先进行EP域的AllToAllV通信，再进行TP域的AllGatherV通信；当不存在TP域通信时，进行EP域的AllToAllV通信。|默认确定性实现||
|[aclnnMoeDistributeDispatchV4](../../mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV4.md)|对token数据进行量化，当存在TP域通信时，先进行EP域的AllToAllV通信，再进行TP域的AllGatherV通信；当不存在TP域通信时，进行EP域的AllToAllV通信。|默认确定性实现||
|[aclnnMoeFinalizeRouting](../../moe/moe_finalize_routing/docs/aclnnMoeFinalizeRouting.md)|MoE计算中，最后处理合并MoE FFN的输出结果。|默认确定性实现||
|[aclnnMoeFinalizeRoutingV2](../../moe/moe_finalize_routing_v2/docs/aclnnMoeFinalizeRoutingV2.md)|MoE计算中，最后处理合并MoE FFN的输出结果，支持配置dropPadMode。|默认确定性实现||
|[aclnnMoeFinalizeRoutingV2Grad](../../moe/moe_finalize_routing_v2_grad/docs/aclnnMoeFinalizeRoutingV2Grad.md)|aclnnMoeFinalizeRoutingV2的反向传播。|默认确定性实现||
|[aclnnMoeFusedTopk](../../moe/moe_fused_topk/docs/aclnnMoeFusedTopk.md)|MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。|默认确定性实现||
|[aclnnMoeGatingTopK](../../moe/moe_gating_top_k/docs/aclnnMoeGatingTopK.md)|MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。|默认确定性实现||
|[aclnnMoeGatingTopKSoftmax](../../moe/moe_gating_top_k_softmax/docs/aclnnMoeGatingTopKSoftmax.md)|MoE计算中，对x的输出做Softmax计算，取TopK操作。|默认确定性实现||
|[aclnnMoeGatingTopKSoftmaxV2](../../moe/moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)|MoE计算中，如果renorm=0，先对x的输出做Softmax计算，再取TopK操作；如果renorm=1，先对x的输出做TopK操作，再进行Softmax操作。|默认确定性实现||
|[aclnnMoeInitRouting](../../moe/moe_init_routing/docs/aclnnMoeInitRouting.md)|MoE的routing计算，根据aclnnMoeGatingTopKSoftmax的计算结果做Routing处理。|默认确定性实现||
|[aclnnMoeInitRoutingV2](../../moe/moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)|该算子对应MoE中的Routing计算，以MoeGatingTopKSoftmax算子的输出x和expert_idx作为输入，并输出Routing矩阵expanded_x等结果供后续计算使用。|默认确定性实现||
|[aclnnMoeInitRoutingV3](../../moe/moe_init_routing_v3/docs/aclnnMoeInitRoutingV3.md)|MoE的routing计算，根据[aclnnMoeGatingTopKSoftmaxV2](../../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)的计算结果做routing处理，支持不量化和动态量化模式。|默认确定性实现||
|[aclnnMoeInitRoutingQuant](../../moe/moe_init_routing_quant/docs/aclnnMoeInitRoutingQuant.md)|MoE的Routing计算，根据aclnnMoeGatingTopKSoftmax的计算结果做Routing处理，并对结果进行量化。|默认确定性实现||
|[aclnnMoeInitRoutingQuantV2](../../moe/moe_init_routing_quant_v2/docs/aclnnMoeInitRoutingQuantV2.md)|MoE的Routing计算，根据aclnnMoeGatingTopKSoftmaxV2的计算结果做Routing处理。|默认确定性实现||
|[aclnnMoeInitRoutingV2Grad](../../moe/moe_init_routing_v2_grad/docs/aclnnMoeInitRoutingV2Grad.md)|[aclnnMoeInitRoutingV2](../../moe/moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)的反向传播，完成Tokens的加权求和。|默认确定性实现||
|[aclnnMoeTokenPermute](../../moe/moe_token_permute/docs/aclnnMoeTokenPermute.md)|MoE的permute计算，根据索引indices将tokens广播并排序。|默认确定性实现||
|[aclnnMoeTokenPermuteGrad](../../moe/moe_token_permute_grad/docs/aclnnMoeTokenPermuteGrad.md)|[aclnnMoeTokenPermute](./aclnnMoeTokenPermute.md)的反向传播计算。|默认确定性实现||
|[aclnnMoeTokenPermuteWithEp](../../moe/moe_token_permute_with_ep/docs/aclnnMoeTokenPermuteWithEp.md)|MoE的permute计算，根据索引indices将tokens和可选probs广播后排序并按照rangeOptional中范围切片。|默认确定性实现||
|[aclnnMoeTokenPermuteWithEpGrad](../../moe/moe_token_permute_with_ep_grad/docs/aclnnMoeTokenPermuteWithEpGrad.md)|[aclnnMoeTokenPermuteWithEp](../../moe/moe_token_permute_with_ep/docs/aclnnMoeTokenPermuteWithEp.md)的反向传播计算。|默认确定性实现||
|[aclnnMoeTokenPermuteWithRoutingMap](../../moe/moe_token_permute_with_routing_map/docs/aclnnMoeTokenPermuteWithRoutingMap.md)|MoE的permute计算，将token和expert的标签作为routingMap传入，根据routingMaps将tokens和可选probsOptional广播后排序|默认确定性实现||
|[aclnnMoeTokenPermuteWithRoutingMapGrad](../../moe/moe_token_permute_with_routing_map_grad/docs/aclnnMoeTokenPermuteWithRoutingMapGrad.md)|[aclnnMoeTokenPermuteWithRoutingMap](../../moe/moe_token_permute_with_routing_map/docs/aclnnMoeTokenPermuteWithRoutingMap.md)的反向传播。|默认确定性实现||
|[aclnnMoeTokenUnpermute](../../moe/moe_token_unpermute/docs/aclnnMoeTokenUnpermute.md)|根据sortedIndices存储的下标，获取permutedTokens中存储的输入数据；如果存在probs数据，permutedTokens会与probs相乘；最后进行累加求和，并输出计算结果。|默认确定性实现||
|[aclnnMoeTokenUnpermuteGrad](../../moe/moe_token_unpermute_grad/docs/aclnnMoeTokenUnpermuteGrad.md)|[aclnnMoeTokenUnpermute](../../moe/moe_token_unpermute/docs/aclnnMoeTokenUnpermute.md)的反向传播。|默认确定性实现||
|[aclnnMoeTokenUnpermuteWithEp](../../moe/moe_token_unpermute_with_ep/docs/aclnnMoeTokenUnpermuteWithEp.md)|根据sortedIndices存储的下标位置，去获取permutedTokens中的输入数据与probs相乘，并进行合并累加。|默认确定性实现||
|[aclnnMoeTokenUnpermuteWithEpGrad](../../moe/moe_token_unpermute_with_ep_grad/docs/aclnnMoeTokenUnpermuteWithEpGrad.md)|[aclnnMoeTokenUnpermuteWithEp](../../moe/moe_token_unpermute_with_ep/docs/aclnnMoeTokenUnpermuteWithEp.md)的反向传播。|默认确定性实现||
|[aclnnMoeTokenUnpermuteWithRoutingMap](../../moe/moe_token_unpermute_with_routing_map/docs/aclnnMoeTokenUnpermuteWithRoutingMap.md)|对经过aclnnMoeTokenpermuteWithRoutingMap处理的permutedTokens，累加回原unpermutedTokens。|默认确定性实现||
|[aclnnMoeTokenUnpermuteWithRoutingMapGrad](../../moe/moe_token_unpermute_with_routing_map_grad/docs/aclnnMoeTokenUnpermuteWithRoutingMapGrad.md)|[aclnnMoeTokenUnpermuteWithRoutingMap](../../moe/moe_token_unpermute_with_routing_map/docs/aclnnMoeTokenUnpermuteWithRoutingMap.md)的反向传播。|默认确定性实现||
|[aclnnMoeUpdateExpert](../../mc2/moe_update_expert/docs/aclnnMoeUpdateExpert.md)|本API支持负载均衡和专家剪枝功能。经过映射后的专家表和Mask可传入MoE层进行数据分发和处理。|默认确定性实现||
|[aclnnNormRopeConcat](../../posembedding/norm_rope_concat/docs/aclnnNormRopeConcat.md)|transfomer注意力机制中，针对query、key和Value实现归一化（Norm）、旋转位置编码（Rope）、特征拼接（Concat）。|默认确定性实现||
|[aclnnNormRopeConcatBackward](../../posembedding/norm_rope_concat_grad/docs/aclnnNormRopeConcatBackward.md)|transfomer注意力机制中，针对query、key和Value实现归一化（Norm）、旋转位置编码（Rope）、特征拼接（Concat）融合算子功能反向推导。|默认非确定性实现，支持配置开启||
|[aclnnNsaCompress](../../attention/nsa_compress/docs/aclnnNsaCompress.md)|训练场景下，使用NSA Compress算法减轻long-context的注意力计算，实现在KV序列维度进行压缩。|默认确定性实现||
|[aclnnNsaCompressAttention](../../attention/nsa_compress_attention/docs/aclnnNsaCompressAttention.md)|NSA中compress attention以及select topk索引计算。|默认确定性实现||
|[aclnnNsaCompressAttentionInfer](../../attention/nsa_compress_attention_infer/docs/aclnnNsaCompressAttentionInfer.md)|Native Sparse Attention推理过程中，Compress Attention的计算。|默认确定性实现||
|[aclnnNsaCompressGrad](../../attention/nsa_compress_grad/docs/aclnnNsaCompressGrad.md)|[aclnnNsaCompress](../../attention/nsa_compress/docs/aclnnNsaCompress.md)算子的反向计算。|默认确定性实现||
|[aclnnNsaCompressWithCache](../../attention/nsa_compress_with_cache/docs/aclnnNsaCompressWithCache.md)|实现Native-Sparse-Attention推理阶段的KV压缩。|默认确定性实现||
|[aclnnNsaSelectedAttention](../../attention/nsa_selected_attention/docs/aclnnNsaSelectedAttention.md)|训练场景下，实现NativeSparseAttention算法中selected-attention（选择注意力）的计算。|默认确定性实现||
|[aclnnNsaSelectedAttentionGrad](../../attention/nsa_selected_attention_grad/docs/aclnnNsaSelectedAttentionGrad.md)|根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。|默认非确定性实现，支持配置开启||
|[aclnnNsaSelectedAttentionInfer](../../attention/nsa_selected_attention_infer/docs/aclnnNsaSelectedAttentionInfer.md)|Native Sparse Attention推理过程中，Selected Attention的计算。|默认确定性实现||
|[aclnnPromptFlashAttention](../../attention/prompt_flash_attention/docs/aclnnPromptFlashAttention.md)|全量推理场景的FlashAttention算子。|默认确定性实现||
|[aclnnPromptFlashAttentionV2](../../attention/prompt_flash_attention/docs/aclnnPromptFlashAttentionV2.md)|全量推理场景的FlashAttention算子。|默认确定性实现||
|[aclnnPromptFlashAttentionV3](../../attention/prompt_flash_attention/docs/aclnnPromptFlashAttentionV3.md)|全量推理场景的FlashAttention算子。|默认确定性实现||
|[aclnnQuantAllReduce](../../mc2/quant_all_reduce/docs/aclnnQuantAllReduce.md)|实现低比特数据的AllReduce通信，并且在通信的过程中对数据进行反量化，输出半精度或者全精度的通信结果。|默认非确定性实现，支持配置开启||
|[aclnnQuantGroupedMatmulDequant](../../gmm/quant_grouped_matmul_dequant/docs/aclnnQuantGroupedMatmulDequant.md)|对输入x进行量化，分组矩阵乘以及反量化。|默认确定性实现||
|[aclnnQuantGroupedMatmulInplaceAdd](../../gmm/quant_grouped_matmul_inplace_add/docs/aclnnQuantGroupedMatmulInplaceAdd.md)|实现分组矩阵乘计算和加法计算，基本功能为矩阵乘和加法的组合。|默认确定性实现|默认确定性实现|
|[aclnnQuantMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduce.md)|对量化后的入参x1、x2进行MatMul计算后，接着进行Dequant计算，接着与x3进行Add操作，最后做AllReduce计算。|默认非确定性实现，支持配置开启||
|[aclnnQuantMatmulAllReduceV2](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduceV2.md)|aclnnQuantMatmulAllReduceV2接口是对[aclnnQuantMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduce.md)接口的功能扩展。|默认非确定性实现，支持配置开启||
|[aclnnQuantMatmulAllReduceV3](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduceV3.md)|aclnnQuantMatmulAllReduceV3接口是对[aclnnQuantMatmulAllReduceV2](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduceV2.md)接口的功能扩展。|默认非确定性实现，支持配置开启||
|[aclnnQuantMatmulAllReduceV4](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduceV4.md)|兼容[aclnnQuantMatmulAllReduceV3](../../mc2/matmul_all_reduce/docs/aclnnQuantMatmulAllReduceV3.md)支持的功能，在此基础上新增perblock量化方式的支持。|默认非确定性实现，支持配置开启||
|[aclnnQuantMatmulAllReduceAddRmsNorm](../../mc2/matmul_all_reduce_add_rms_norm/docs/aclnnQuantMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。|默认非确定性实现，支持配置开启||
|[aclnnQuantReduceScatter](../../mc2/quant_reduce_scatter/docs/aclnnQuantReduceScatter.md)|实现quant + reduceScatter融合计算。|||
|[aclnnRainFusionAttention](../../attention/rain_fusion_attention/docs/aclnnRainFusionAttention.md)|RainFusionAttention稀疏注意力计算，支持灵活的块级稀疏模式，通过selectIdx指定每个Q块选择的KV块，实现高效的稀疏注意力计算。|默认确定性实现||
|[aclnnRecurrentGatedDeltaRule](../../attention/recurrent_gated_delta_rule/docs/aclnnRecurrentGatedDeltaRule.md)|完成变步长的Recurrent Gated Delta Rule计算。|默认确定性实现||
|[aclnnRingAttentionUpdate](../../attention/ring_attention_update/docs/aclnnRingAttentionUpdate.md)|将两次FlashAttention的输出根据其不同的softmax的max和sum更新。|默认确定性实现||
|[aclnnRingAttentionUpdateV2](../../attention/ring_attention_update/docs/aclnnRingAttentionUpdateV2.md)|指定softmax的输入排布，将两次FlashAttention的输出根据其不同的softmax的max和sum更新。|默认确定性实现||
|[aclnnRopeWithSinCosCache](../../posembedding/rope_with_sin_cos_cache/docs/aclnnRopeWithSinCosCache.md)|推理网络为了提升性能，将sin和cos输入通过cache传入，执行旋转位置编码计算。|默认确定性实现||
|[aclnnRotaryPositionEmbedding](../../posembedding/rotary_position_embedding/docs/aclnnRotaryPositionEmbedding.md)|执行单路旋转位置编码计算。|默认确定性实现||
|[aclnnRotaryPositionEmbeddingGrad](../../posembedding/rotary_position_embedding_grad/docs/aclnnRotaryPositionEmbeddingGrad.md)|单路旋转位置编码[aclnnRotaryPositionEmbedding](../../posembedding/rotary_position_embedding/docs/aclnnRotaryPositionEmbedding.md)的反向计算。|默认确定性实现||
|[aclnnScatterPaKvCache](../../attention/scatter_pa_kv_cache/docs/aclnnScatterPaKvCache.md)|更新KvCache中指定位置的key和value。|默认确定性实现||
|[aclnnSparseFlashAttentionGrad](../../attention/sparse_flash_attention_grad/docs/aclnnSparseFlashAttentionGrad.md)|根据topkIndices对key和value选取大小为selectedBlockSize的数据重排，接着进行训练场景下计算注意力的反向输出。|默认非确定性实现，不支持配置开启||
|[aclnnSparseLightningIndexerGradKLLoss](../../attention/sparse_lightning_indexer_grad_kl_loss/docs/aclnnSparseLightningIndexerGradKLLoss.md)|LightningIndexer的反向算子，再额外融合了Loss计算功能。|默认非确定性实现，不支持配置开启||
|[aclnnSwinTransformerLnQkvQuant](../../ffn/swin_transformer_ln_qkv_quant/docs/aclnnSwinTransformerLnQkvQuant.md)|Swin Transformer 网络模型 完成 Q、K、V 的计算。|默认确定性实现||
|[aclnnWeightQuantMatmulAllReduce](../../mc2/matmul_all_reduce/docs/aclnnWeightQuantMatmulAllReduce.md)|对入参x2进行伪量化计算后，完成MatMul和AllReduce计算。|默认非确定性实现，支持配置开启||
|[aclnnWeightQuantMatmulAllReduceAddRmsNorm](../../mc2/matmul_all_reduce_add_rms_norm/docs/aclnnWeightQuantMatmulAllReduceAddRmsNorm.md)|完成mm + all_reduce + add + rms_norm计算。|默认非确定性实现，支持配置开启||

## 废弃接口

|    废弃接口   |   说明     |
| --------------- | ----------------------- |
|[aclnnGroupedMatMulAllReduce](../../mc2/grouped_mat_mul_all_reduce/docs/aclnnGroupedMatMulAllReduce.md)|此接口后续版本会废弃，请勿使用该接口。|
