# ChunkKdaFwd ATK 工程

本目录提供 `chunk_kda_fwd` 的 ATK 单算子工程。通用版本要求、case 范围、测试动作和
结果检查规则见 [`../README.md`](../README.md)。

## 输入限制

- `layout` 支持 `BSND/BNSD/TND/NTD`；`layout` 只描述输入，输出布局由接口固定约定。
- `BSND` 下 `q/k=[B,T,H_k,K]`，`v=[B,T,H_v,V]`，`g=[B,T,H_v,K]`，`beta=[B,T,H_v]`。
- `BNSD` 下 `q/k=[B,H_k,T,K]`，`v=[B,H_v,T,V]`，`g=[B,H_v,T,K]`，`beta=[B,H_v,T]`。
- `TND/NTD` 使用打包 token；`cu_seqlens` 从 `0` 开始、以总 token 数结束且单调不减。
- head 映射满足 `0 < H_k <= H_v <= 128` 且 `H_v % H_k == 0`。
- `q/k/v` 支持 `BFLOAT16/FLOAT16`；`g` 支持 `FLOAT/BFLOAT16`；`beta` 支持 `FLOAT/BFLOAT16`。
- `K/V` 只支持两档且必须同档：`K=V=64` 或 `K=V=128`；混合档（如 `K=64,V=128`）与其它
  取值（含 `V=256`）返回 `ACLNN_ERR_PARAM_INVALID`，负向用例覆盖同类拦截。
- `chunk_size` 只支持 `64`，其它取值（含 `128`）在参数校验阶段返回
  `ACLNN_ERR_PARAM_INVALID`，负向用例覆盖该类拦截；rank-4 变长输入要求 `B=1`，
  逻辑序列数最多 `1024`。
- `use_gate_in_kernel=true` 时必须提供 `A_log`；`safe_gate=true` 时 `lower_bound` 取 `[-5,0)`。
- `initial_state` 如提供，末两维由 `state_v_first` 解释为 `[K,V]` 或 `[V,K]`。

## 精度拓扑

精度标准统一使用 `mixed_tolerance_bm`。显式 CPU 节点调用 executor 的 FP64 路径生成
唯一 golden，显式 NPU 节点只运行 DUT。CPU golden 输出转换为 FP32，NPU 输出保持算子
原始 dtype，由 ATK 按混合容差规则比较。

```text
ATK accuracy task
|-- CPU FP64 golden (output FP32)
`-- NPU DUT
```

## 运行

在仓库根目录准备好 ATK、CANN、OPP 和 Python 包环境后执行：

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=accuracy
```

只验证前 5 个 case：

```bash
CASE_START=0 CASE_END=5 \
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=accuracy
```

确定性回归会重复运行 MSS 用例并逐位比较全部可见输出。MSS 包含 #440 的 63-token 单序列
尾块场景，通过 96 个 value heads 放大并行调度覆盖，同时启用 `disable_recompute=true`、最终状态
和全部中间量：

```bash
DC_LOOP_NUMS=100 \
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=determinism
```

性能、确定性、mssanitizer 和用例生成均通过统一脚本的对应 `-scope` 执行。

## 模型 case 与性能用例映射（评审稿 v2，2026-11-10）

按 `tests/atk/README.md` 的正式验收用例包要求建立三类映射。`atk_chunk_kda_fwd_perf.json`
已按工作区根 `target.md` §3 的 A5 Prefill 模型 case 重建（原文件只有 1 个 case）；
全部 case 标记 `review_status=draft_await_device_validation`，
**设备侧逐 case 跑通校验前不得作为正式验收依据**。

### 模型 case → perf case id

| 模型 case | Layout | B | H_k | H_v | T | chunk_size | perf case id | 预期路径 | 基线 | 目标 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| E | NTD | 1 | 32 | 32 | 65536 | 64 | 250 | V2（待实测） | 待实测 | 1.20x |
| F | BNSD | 1 | 1 | 2 | 16384 | 64 | 251 | V1/V2 待实测 | 待实测 | 1.20x |
| G | BSND | 64 | 8 | 8 | 2048 | 64 | 252 | V2（待实测） | 待实测 | 1.20x |

> **契约校准**：历史 BSND C9 的 `chunk_size=128` 与 NTD C33 的 `V=256` 已被本文「输入限制」的
> 参数校验拒绝，不能作为当前用例或基线；本映射全部按 `chunk_size=64`、`K=V=128` 重建，
> 历史数据只用于判断耗时占比。
> V1 融合与 V2 三算子是不同物理实现，路径须由 `FLA_NPU_STABLE_TRACE=1` 与
> `stable_chunk_kda_fwd.cpp` 门控实测确认，并分别记录基线与收益，不跨路径取平均。

### 另外两类映射（现状与缺口）

| 映射 | 现状 | 待办 |
| --- | --- | --- |
| 逻辑分支 → 精度 case id | `atk_chunk_kda_fwd.json` 48 个 case（250–297） | 逐分支映射待补，覆盖四 layout、`return_intermediate` 两模式、尾块与变长 |
| 可达 TilingKey → mss case id | `atk_chunk_kda_fwd_mss.json` 2 个 case（250 / 298） | 核对是否覆盖全部可达 TilingKey，缺项补齐后建映射 |

### 设备侧校验门禁

1. 每个 perf case 在 A5 上以 `-scope=performance` 单独运行，确认进入 NPU kernel、无参数校验拒绝、单次 <60s。
2. 校验 `chunk_size=64`、`K=V=128`、`0 < H_k <= H_v <= 128` 且 `H_v % H_k == 0`；
   rank-4 变长输入要求 `B=1`（E/F 满足），G 为非变长 dense。
3. 用 `FLA_NPU_STABLE_TRACE=1` 记录实际 kernel 数与命中路径，写回 `docs/validation.md` 追踪表。
