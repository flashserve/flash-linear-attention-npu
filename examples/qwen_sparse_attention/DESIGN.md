# qwen_sparse_attention 设计

- 算子名：`qwen_sparse_attention`（分页 KV cache + 逐行稀疏选择的 GQA 注意力前向）
- 目标 DSL 版本：`ascend-catlass-dsl 2.2.1.dev20261112`（官方 index 当前最新包，本机已安装并上板验证）
- 执行模式：remote-connected（`179.223.2.2:/home/s30080348/TestJDBack/qwen_sparse_attention`）
- 性能要求：requested（目标：`baseline.jsonl` 的 50 条延迟）
- merge_required: false
- 硬件：Ascend 950PR（A5），`cube_core_num=28` / `vector_core_num=56`，CANN 9.2.0，`--npu-arch 3510`
- 参考实现：`reference.py`（SHA-256 `78fdb2696d7d1f18e94dc0353a35abba453d10171f28954596a5919b7bd69c67`）
- 用例：`workload.jsonl` 50 条（SHA-256 `e07b4fb6c3af06392ea34e4a7a14e6fcbcf4fe2dd5a5426c411b8ed552bc6e30`）；dev 子集 `workload_dev.jsonl` `40` 条（SHA-256 `b6417b0ade9e35687264d358cb58c978aa3ab301fc0d0367a4b8b77b277a330e`）
- 定义：`definition.json`（SHA-256 `784f5d238ce00a8fbec1b42f1021a4d251cc2fdc9c0d86ec80a6de01486025bf`）
- 内核：`src/qwen_sparse_attention.py`（SHA-256 `692ec5cf0a7392e4f716ed45244f9bc63543fb979ed985a7b8c7e5de8213c961`，R3+R17 后）

## 1. 算子语义

对每个 query row `r`：过滤 `logical_indices[r]` 中 `< 0` 的无效项，按 `page_size` 拆成逻辑页号
与页内偏移，经 `token_to_req[r]` → `block_table[req, page]` 映射到物理 cache block，从
`k_cache` / `v_cache` 收集被选中的 KV token（`kv_heads = 1`，按 `heads / kv_heads` 做 GQA
扩展），用 FP32 计算 `scale · QKᵀ`、softmax 与 `PV`，最后按 `q.dtype` 一次舍入写出；该行无
有效 token 时输出精确 0。

## 2. 实现结构（单 kernel，AIC + AIV mixed）

一个 `@tla.kernel`，不拆多 kernel，计算全部由该 kernel 完成（不调用任何 torch 算子）：

| 维度 | 取值 |
|---|---|
| task | 一个 query row 的一个 head 组（`q_block = 128`，覆盖全部 head） |
| tile | 128 个 logical selection 槽位（`kv_block = 128`） |
| AIC | 逐 token gather K（nZ）/ V（RowMajor）+ 沿 `head_dim` 拆两段的 QK mmad + PV mmad + L0C→UB(SPLIT_M) |
| AIV | 逐列 index 掩码 + online softmax（running max / sum）+ P(f16 zN) 落盘 + O rescale |
| 同步 | 核内 mutex + 跨核 `s_ready / s_free / p_ready / o_ready / o_free` 五个 cross_flag，逐 tile 严格配平 |

Q 常驻 L1；P 从 L0C 经 FIX 落到 UB，vector 在 UB 的 zN 视图上写回，MTE3 搬到 L1 后供 PV 的
L0A 使用。主参考样例：`experimental/kernel/attention/qwen_sparse_attention`
（catlass master `7e71c65`），本实现按其结构落地，未改算法。

## 3. 已知环境差异（必须记录）

官方样例按「split 出来的非连续 k/v cache 视图」绑定；本机安装的
`ascend-catlass-dsl 2.2.1.dev20261112` 的 `from_dlpack` 在 `origin_shape=(numel,)` 的**扁平 1-D 绑定**
下强制要求物理连续（`_require_flat_1d_binding`），否则报
`does not read consecutive elements`（4-D `origin_shape` 又不支持派生 layout）。因此
`run()` 在 cache 非连续时先做一次 `contiguous()` 拷贝再绑定；该拷贝**计入** `run()` 计时，
不缓存、不跨调用复用。连续 cache（真实 serving 中 K/V 为独立张量的常见形态）无此开销。

## 4. 性能结果（benchmark：50/50 全量实测，`evidence/bench_r3h/summary_r3h.md`，R3+R17 后）

| 指标 | 值 |
|---|---:|
| baseline 平均 | 2.306 ms |
| 候选平均 | 25.373 ms |
| speedup min / median / max | 0.002 / 0.210 / 0.514 |
| speedup geomean | 0.146 |
| `>= 1.0x` 用例 | **0 / 50** |
| 精度 max_abs（bf16 / fp16） | 1.56e-02 / 1.56e-02 |

**结论：正确性通过；性能目标未达成（R3 后 median 0.208→0.210、geomean 0.144→0.146，R17 host 缓存主要改善小 shape）。**

## 5. 为什么达不到 baseline（量化归因）

kernel 的时间 ≈ `rows × 有效 token 数 × 2`（K、V 各一次）次 512 B DMA：

- **实测伸缩**：heads 128→16（cube 算力降到 1/8）只快 3%；width 2048→1024→512 时间近似
  线性（1.00 / 0.52 / 0.31）⇒ **纯 gather / DMA 发射受限**，cube 与 vector 都不是瓶颈。
- **实测单价**：约 48 ns / 512 B DMA / 核（28 核）。按
  `rows × valid × 2 × 48 ns / 28` 预测：050 为 109 ms、实测 110 ms；025 预测 14.7 ms、实测 15.1 ms
  ⇒ 模型与实测逐例吻合。
- **把 baseline 换算成它要求的 gather 带宽**（`tools/roofline.py`）：001 需 1296 GB/s、
  025 需 2348 GB/s、**050 需 175625 GB/s**。本机 HBM ≈ 1.6 TB/s，片内 L2 也低一个数量级
  ⇒ **大 context 用例的 baseline 不是「难」，而是任何实现都到不了**：`baseline.jsonl` 由
  dense FIA 延迟 × KV density 推导，只按 ~170 TFLOPS 的算力折算（各例 TF@base 稳定在
  104–178），隐含「gather 不花时间」。
- 因此本实现能对得上的口径是「正确 + 结构可行的单 kernel」；要接近 baseline 需要换数据流：

  1. **4 个连续逻辑 token 合并成一次 2 KiB DMA**：R1 变体实测 2.06x，但 4-token V 在
     copy 路由矩阵中没有可落到 zN 分形的合法组合（4 行非 16 对齐），精度不过，rejected；
     K 侧 4 列 nZ 同样受 16 列分形限制（R4，max_abs 0.18–1.64），rejected。
  2. **跨行复用 gather**：实测 001/025/050 的唯一 selection 行数为 1024/1024、3982/3982、
     16188/16384，复用仅 1.00–1.01x；“050 只有 505 种”是掩码前 block pattern，掩码后
     每行不同，R9 rejected。
  3. **跳过全无效尾 tile**（R3）已 adopted：`valid_tiles` 前缀上界，001 shape 3.31→2.42 ms
     （1.36x）；全量 50 例 candidate 平均 26.526→25.164 ms、median 0.190→0.208。
  4. **host 侧缓存 DLPack 包装**（R17）已 adopted：q/k/v/idx/bt/req 六个包装跨 `run()` 复用，
     001 shape 2.44→2.29 ms（1.06x），kernel 函数不变；全量 50 例 median 0.208→0.210、
     geomean 0.144→0.146（噪声级）。
  5. 剩余瓶颈仍是 per-token 512 B DMA 发射；R1/R4 的合并被分形对齐阻断，R5–R16/R18–R20
     均未越过噪声门限（详见 `ITERATIONS.md` R3–R20）。050 仍只能到 ~0.01x——baseline 本身
     不可达。

## 6. 复现

```bash
source /home/s30080348/CANN/9.2.0/ascend-toolkit/set_env.sh
cd /home/s30080348/TestJDBack/qwen_sparse_attn
export PYTHONPATH=$PWD/src:$PWD:$ASCEND_TOOLKIT_HOME/python/site-packages
python tools/bench_qsa2.py --device 1 --shard 0 --shards 2 --pairs 5 --acc-rows 128 \
    --out evidence/bench/c0.json
python tools/bench_qsa2.py --device 3 --shard 1 --shards 2 --pairs 5 --acc-rows 128 \
    --out evidence/bench/c1.json
python tools/sum_bench.py     # -> evidence/bench/summary.md
python tools/roofline.py 1    # 每例 gather 字节 / FLOPs / baseline 所需带宽
```
