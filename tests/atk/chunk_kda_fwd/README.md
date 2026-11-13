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
- `chunk_size` 支持 `32` 和 `64`；chunk32 仅用于 Ascend950 的 `ChunkKdaFwdFused` 融合路径，
  并要求 `K=V=128`、BF16 q/k/v、safe gate、核内 q/k L2 norm 和 beta sigmoid。其它取值
  （含 `128`）在参数校验阶段返回 `ACLNN_ERR_PARAM_INVALID`，负向用例覆盖该类拦截；
  rank-4 变长输入要求 `B=1`，逻辑序列数最多 `1024`。
- `use_gate_in_kernel=true` 时必须提供 `A_log`；`safe_gate=true` 时 `lower_bound` 取 `[-5,0)`。
- `initial_state` 如提供，末两维由 `state_v_first` 解释为 `[K,V]` 或 `[V,K]`。

## Accuracy case 组成

正式 `atk_chunk_kda_fwd.json` 同时包含：

- 既有 48 条 case（ID 250–297），schema、seed 和 ID 保持不变；
- 从 `chunk32/atk_chunk_kda_fwd_chunk32.json` 转换的 135 条 case（ID 508–642），统一使用
  `cv_fused_double_benchmark`。转换保留原始 shape、dtype、layout、seed、state、varlen 和调用开关；
  既有 48 条保持同一容差标准。

`gen_chunk_kda_fwd.py` 是统一转换入口。用以下命令可幂等更新正式 accuracy JSON，不会重复追加：

```bash
python3 tests/atk/chunk_kda_fwd/gen_chunk_kda_fwd.py --merge-frozen-chunk32 --summary
```

执行时，所有正向 case 统一复用 `chunk32/executor_chunk_kda_fwd_chunk32.py` 的 legacy 输入、FP64 golden、公开调用和输出契约，避免两套 reference 数学路径发生漂移；负向 case 保留本目录的 ACLNN 拦截检查。容差统一使用 `cv_fused_double_benchmark`（max≤5、avg≤1.5、RMSE≤1.5）。

`chunk32/` 的独立 accuracy 套件仍保留用于专项对照；本目录的 accuracy 入口覆盖合并后的全部用例。

## 精度拓扑

精度标准统一使用 `cv_fused_double_benchmark`。显式 CPU 节点调用 executor 的 FP64 路径生成
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
