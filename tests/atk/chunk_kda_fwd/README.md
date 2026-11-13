# ChunkKdaFwd ATK 工程

本目录提供 `chunk_kda_fwd` 的 ATK 单算子工程。通用版本要求、case 范围、测试动作和
结果检查规则见 [`../README.md`](../README.md)。

## 输入限制

- `layout` 支持 `BSND/BNSD/TND/NTD`；`layout` 只描述输入，输出布局由接口固定约定。
- `BSND` 下 `q/k=[B,T,H_k,K]`，`v=[B,T,H_v,V]`，`g=[B,T,H_v,K]`，`beta=[B,T,H_v]`。
- `BNSD` 下 `q/k=[B,H_k,T,K]`，`v=[B,H_v,T,V]`，`g=[B,H_v,T,K]`，`beta=[B,H_v,T]`。
- `TND/NTD` 使用打包 token；`cu_seqlens` 从 `0` 开始、以总 token 数结束且单调不减。
- head 映射满足 `0 < H_k <= H_v <= 128` 且 `H_v % H_k == 0`；chunk64 组合入口要求 `1 <= H_v/H_k <= 16`。
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


## final_state precision category and host contracts

For BF16/FP16 input paths, `final_state` is stored in FP32 after matmuls with
BF16/FP16 intermediate operands. Its storage dtype does not promise a full-FP32
mathematical forward pass. Cases exporting it explicitly select the existing
ATK low-precision standard through `output_dtype_overrides` for output slot 1.
The golden and DUT tensors remain unchanged; no output cast or custom threshold
is applied. FP32 gate outputs retain their FP32 standard. A separate diagnostic
reconstructs each FP32 state update from actual device kg/v_new/gk outputs.

`akkOut` is OPTIONAL. Its historical null-output test is retained as a positive
workspace-validation test; `aqkOut` remains REQUIRED. Null mandatory inputs must
preserve ACLNN_ERR_PARAM_NULLPTR through the outer GetWorkspaceSize entry.


## A5 public-entry routing regression (PR 1179 resolution)

The Python/C++ Stable ABI work-item threshold was measured on A2. A5 eligible
BF16 K=V=128, chunk=64 calls now select `KdaFwdV2` inside the V1 ACLNN host entry
as well. This makes small dense tails and packed GVA cases use the A5 Prepare,
FwdH and Finalize implementation regardless of that threshold. Other scenarios
continue to use their existing route. Existing validation of required outputs,
layout, head ratio and contiguous output constraints runs before dispatch.

The old generic recurrent backend intermittently omitted parts of the last
rank-one update for T=65. A synchronized saved failure showed correct first
chunk state but final-state errors up to 0.0384. This routing fix avoids that
backend for eligible A5 calls; it does not claim a repair of its internal
pipeline for other devices or unsupported shapes. Validation keeps the original
FP64 golden and inputs and separately reconstructs FP32 state from device
kg/v_new/gk to detect missing updates independently of BF16 accuracy tolerance.


## 2026-11-12 A5 resolution validation

After the A5 public-entry route correction, the 720-case GVA matrix (ratios 1-16,
dense/packed layouts, gate/beta dtype combinations, state orientation and three
fixed seeds) passes 720/720. Two model shapes, B1 HK16 HV128 T11264 dense and
T16384 varlen (16 x 1024), pass across three seeds, 6/6. All 726 strict FP32 state
reconstructions pass; maximum absolute discrepancy is 2.23517418e-8 for the matrix
and 4.47034836e-8 for models. Full-chain bitwise determinism passes 13 x 50.
The 56 host contract checks pass (55 expected errors, one optional akkOut
acceptance). The precision-category qualification described above applies;
this does not claim passage under the former FP32 mathematical criterion.

Source fix: 263b6be. Raw reports, preserved original failures, frozen-input
integrity record and strict state audit are under validation_pr1179_a5/resolution
on the validation host. Finalize performance and memory results are recorded in
../chunk_kda_fwd_finalize/README.md. The broader A2/A3 and native performance-ratio
acceptance items are not completed by this A5 run.
