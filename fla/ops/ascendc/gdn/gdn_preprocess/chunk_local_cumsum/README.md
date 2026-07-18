# ChunkLocalCumsum 算子说明

`ChunkLocalCumsum` 是用于 Gated Delta Rule 中门控张量 `g` 的 chunk 内局部累加算子。该算子从 Triton `chunk_local_cumsum` 迁移为 AscendC 自定义算子，支持固定长度序列和基于 `cu_seqlens`、`chunk_indices_out` 的变长序列。

---

## 1. 算子功能

输入 `g` 的 shape 为 `[B, H, T, *]`。算子按 `chunk_size` 在时间维 `T` 上划分 chunk，并在每个 chunk 内执行局部累加，输出 shape 与输入一致。尾部 `*` 可为空或包含一个及以上维度，kernel 会将其展平为每个 token 的连续计算宽度。

正向模式 `reverse=false`:

```text
out[b, h, t, ...] = scale * sum(g[b, h, k, ...]), k = chunk_start ... t
```

反向模式 `reverse=true`:

```text
out[b, h, t, ...] = scale * sum(g[b, h, k, ...]), k = t ... chunk_end - 1
```

其中 `chunk_start = floor(t / chunk_size) * chunk_size`，`chunk_end = min(chunk_start + chunk_size, T)`。

---

## 2. 接口定义

### 2.1 ACLNN 接口

```cpp
// 获取执行所需的 workspace 大小
aclnnStatus aclnnChunkLocalCumsumGetWorkspaceSize(
    const aclTensor *g,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOutOptional,
    int64_t chunkSize,
    bool reverse,
    double scale,
    bool headFirst,
    char *outputDtypeOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

// 执行算子计算
aclnnStatus aclnnChunkLocalCumsum(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

详细接口说明见 [docs/aclnnChunkLocalCumsum.md](docs/aclnnChunkLocalCumsum.md)。

---

## 3. 输入参数

| 参数 | 数据类型 | 是否必须 | 描述 |
|------|----------|----------|------|
| g | FLOAT32 | 是 | 输入张量，shape 为 `[B, H, T, *]` |
| cu_seqlens | INT64 | 否 | 变长序列累积长度。固定长度模式传空 tensor |
| chunk_indices_out | INT64 | 否 | 变长模式下 block 到 `(seq_id, block_id)` 的映射，按二元组连续存储 |
| chunk_size | int64 | 是 | chunk 长度，必须为 2 的幂 |
| reverse | bool | 否 | 是否执行反向 chunk 内累加，默认 `false` |
| scale | double | 否 | 输出缩放系数，默认 `1.0` |
| head_first | bool | 否 | 当前实现仅支持 `true` |
| output_dtype | string | 否 | 当前仅支持 `float32`、`torch.float`、`torch.float32` |

---

## 4. 输入约束

1. **数据类型**：输入 `g` 和输出 `out` 仅支持 FLOAT32。
2. **输入维度**：`g` 必须为 rank >= 3 的 tensor，shape 为 `[B, H, T, *]`。
3. **chunk_size**：必须为 2 的幂，且 host tiling 计算得到的 `block_t` 不小于 `chunk_size`。
4. **数据布局**：当前 AscendC kernel 仅支持 `[B, H, T, *]`，`head_first=false` 会被显式拒绝。
5. **变长模式**：当 `cu_seqlens` 非空时，`B` 必须为 1，且 `chunk_indices_out` 必须非空、元素数为偶数。

---

## 5. 输出参数

| 输出 | 数据类型 | 描述 |
|------|----------|------|
| out | FLOAT32 | 输出张量，shape 与 `g` 一致 |

---

## 6. 算子实现

该算子使用 AIV kernel 实现。

固定长度模式下，kernel 以 `(batch * head, chunk, tail tile)` 为任务粒度分核，每个任务在一个 batch/head 的一个时间 chunk 内对一段尾部连续维度执行正向或反向累加。

变长模式下，kernel 从 `chunk_indices_out` 读取 `(seq_id, block_id)`，结合 `cu_seqlens` 计算当前序列边界，再在序列局部坐标内执行 chunk-local cumsum。

---

## 7. 目录结构

```text
chunk_local_cumsum/
├── docs/
│   └── aclnnChunkLocalCumsum.md
├── op_host/
│   ├── op_api/
│   │   ├── aclnn_chunk_local_cumsum.cpp
│   │   ├── aclnn_chunk_local_cumsum.h
│   │   └── chunk_local_cumsum.cpp
│   ├── chunk_local_cumsum_def.cpp
│   ├── chunk_local_cumsum_infershape.cpp
│   ├── chunk_local_cumsum_tiling.cpp
│   └── CMakeLists.txt
├── op_kernel/
│   ├── chunk_local_cumsum.cpp
│   └── chunk_local_cumsum_tiling_data.h
├── test/
│   └── test.py
├── CMakeLists.txt
└── README.md
```

---

## 8. 测试方法

```bash
cd /home/m00913889/codex08/flash-linear-attention-npu
bash build.sh --pkg --soc=ascend910b --ops=chunk_local_cumsum -j16
python3 fla/ops/ascendc/gdn/gdn_preprocess/chunk_local_cumsum/test/test.py
bash torch_custom/fla_npu/test/test.sh --device 0 --op chunk_local_cumsum
```

测试脚本会通过 `torch.ops.npu.npu_chunk_local_cumsum` 调用算子，并覆盖固定长度、反向累加、缩放系数以及变长 `cu_seqlens + chunk_indices_out` 场景。
