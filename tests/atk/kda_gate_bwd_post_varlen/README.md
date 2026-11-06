# kda_gate_bwd_post_varlen ATK 工程

本目录为 Ascend C 独立 OpDef 创建 Phase A 用例工程。**CPU 输出为全零 shape 占位，不是数值标杆，不能用于精度验收。** 当前源码只有私有 L0 launcher，没有独立 ACLNN 或 torch/Python 入口；executor 的 NPU 分支明确抛出 NotImplementedError。需要独立测试 launcher 和真实 CPU golden 后才能执行本算子的 NPU 精度验收。

## 接口与输出

OpDef 顺序为 `dg_act, raw_g, a_log, dt_bias, cu_seqlens, chunk_indices, chunk_size, safe_gate, lower_bound, input_scanned -> dg, dA, dbias`。输入 `dg_act/raw_g` 采用 HTK `[H,T,128]`；二者同 shape，dg_act 固定 FP32，raw_g 支持 FP16/BF16/FP32。a_log 为 FP32 `[H]`，dt_bias 为 FP32 `[H*128]`。本工程始终提供 dt_bias，尚未覆盖 bias 缺失分支。

CPU 占位严格按 OpDef 输出顺序返回3个 FP32 tensor：dg 与 raw_g 同 shape；dA=`[H]`；dbias=`[H*128]`。dA/dbias 的大小可由 kernel 中 `dAGm_[head]` 与 `dBiasGm_[head*128]` 的写入确认。

变长 cu_seqlens INT64 `[3]` 固定构造为 `[0,T//2,T]`；chunk_indices INT64 `[Nchunks,2]` 为按 sequence 递增、各 sequence 内 local_chunk 递增的 canonical packed pairs。generator 根据 T 和 C=64 修正 Nchunks，executor 构造内容，跳过空序列。每个 token 恰好被覆盖。 浮点数据直接使用 ATK 样本，executor 不重新随机采样。

## 约束来源清单

前缀 `S=fla/ops/ascendc/kda/chunk_kda_bwd`，下表来源均为当前 C++ 源码。

| 约束 | 来源 | YAML/generator/executor 映射 |
| --- | --- | --- |
| 输入顺序、固定 FP32 辅助与输出、raw_g 3种 dtype、ND | `S/internal_ops/kda_gate_bwd_post_varlen/op_host/kda_gate_bwd_post_varlen_def.cpp` | YAML 显式 inputs/dtypes；CPU 占位固定输出类型 |
| canonical BHTK 或 HTK、dg/raw 等元素数；正 B/H/T；K=128 | `S/op_host/chunk_kda_bwd_tiling.cpp` 的 Tiling4KdaGateBwdPost | YAML 固定 rank；generator 同 shape、只固定末维128 |
| chunk_size=64 | 同一 tiling 的 requires chunk_size=64 检查 | YAML 单值、generator 定点修正 |
| a_log 每 head 一项 | 同一 tiling 的 a_log GetShapeSize 校验 | generator 将 shape 修正为 [H] |
| rank3 必须同时有 cu/chunk；rank4 必须无元数据 | 同一 tiling 的 hasCuSeqlens/hasChunkIndices 校验 | executor 构造两个元数据 tensor |
| chunk pair 数为正且成对、sequence/local_chunk 合法、canonical packed 顺序覆盖全部 token | 同一 tiling 的 metadataElems 与 packedCursor 校验 | generator 修正 [Nchunks,2]；executor 生成 canonical pairs |
| input_scanned 实际固定为 false | `S/op_kernel/kda_gate_bwd_post.h:433-437` | YAML false；generator 修正为 false，不制造无效 true 覆盖 |
| dA=[H]、dbias=[H*K] | `S/op_kernel/kda_gate_bwd_post.h:607-612` | CPU 占位3输出正确形状与顺序 |
| 只有私有第二 launcher | `S/internal_ops/kda_gate_bwd_post/op_host/kda_gate_bwd_post_def.cpp:5-6`、`S/op_host/op_api/chunk_kda_bwd.cpp:75-89` | NPU 分支明确报缺少独立入口，未冒用父算子结果 |
| 总元素数不超过2^34 | ATK 精度标准 §1.2 | YAML max_length；当前候选远小于上限 |

Tiling 含 int32 字段的 narrowing，kernel 还包含32位索引；当前仅生成较小的结构用例，未设计超大索引溢出专项，不宣称覆盖该风险。

## 覆盖范围与限制

shape 候选含 H=1/2/4、T=1/63/64/65/127/128/129、K=128；raw_g 3种 dtype；safe_gate 两种分支；lower_bound=-5/-1；C=64。分布按本次读取的 ATK 标准提供50%均匀 [-5,5] 与50%正态（均值[-5,5]、标准差[0.1,2]）。空 tensor/标量不符合 tiling 的正尺寸/rank 契约，禁用；非有限值边界可生成。有限候选空间不表示所有组合已被穷举或验证。

注册 SoC 为 ascend910b/ascend910_93/ascend950；tiling 可达 key 为 safe_gate=true→1、false→3，实际选择尚未采集。未添加用户模型性能 case、mss/TilingKey 验收 JSON 或精度成功结论。占位结果只能检查输出结构，不能计算有意义的误差或通过率。

## 运行命令

```bash
# 仓库根目录：仅生成候选
bash tests/atk/run_test_cpu.sh -op=kda_gate_bwd_post_varlen -scope=gen_cases

cd tests/atk/kda_gate_bwd_post_varlen
timeout 60 atk case -f kda_gate_bwd_post_varlen.yaml -p gen_kda_gate_bwd_post_varlen.py -dt 1 -en 2
timeout 60 atk case -f kda_gate_bwd_post_varlen.yaml -p gen_kda_gate_bwd_post_varlen.py -s 20261030
```

输出为 `result/kda_gate_bwd_post_varlen/json/all_kda_gate_bwd_post_varlen.json`，审核后固化为 `atk_kda_gate_bwd_post_varlen.json`；运行输出不提交。CPU 结构占位可通过 ATK 发起：

```bash
atk node -b cpu task -c atk_kda_gate_bwd_post_varlen.json -p executor_kda_gate_bwd_post_varlen.py -s 0 -e 1 -tk accuracy
```

该命令仅确认占位路径能返回 tensor，不能作为正式精度验收。执行 NPU 前必须补齐独立测试 launcher、真实数值 golden、值域校准与正式验收三类用例包。

ATK 合成 upper_border 可能越过 YAML 离散候选并取 INT32_MAX。generator 读取 YAML 每维候选最大值，将该类边界投影回本工程声明的有限测试域；不在 constraint 中另设候选范围。大于该测试域的索引溢出场景需另行设计。

## 本次 Phase A 验证记录

- 环境：CPU Docker，ATK 26.9.8、PyTorch 2.7.1；未执行 NPU。
- ATK `case -dt 1 -en 0 -s 20261030` dry-run 通过；正式按 YAML `dtype_numbers=16`、`extra_numbers=3` 生成 52 条候选；按用户要求仅保留首条到 `atk_kda_gate_bwd_post_varlen.json`（共 1 条）。
- JSON 由 ATK 产生，归档时统一设置 `default_seed=20261030`；输入 shape/dtype/属性保持生成结果。`standard.acc=mixed_tolerance_bm` 遵循仓库测试契约。该包是 Phase A 候选，未满足正式验收三类用例包与每场景3固定种子的要求。
- `atk node -b cpu task -s 0 -e 1 -tk accuracy` 实测执行1条，日志为 `Total Task: 1, success 1, failed 0`。仅确认 CPU 路径能执行；未产生 CPU/NPU 精度通过率。
- 用归档 JSON 经 ATK `OpsDataset` 还原最小样本，核对实际 CPU 输出的 shape、输出数与 FP32 dtype。Python AST 与 YAML 解析检查通过。
