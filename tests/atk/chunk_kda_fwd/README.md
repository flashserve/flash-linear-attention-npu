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
  chunk64 已移除通用 L2 的 1024 条序列限制；chunk32 保留其独立限制。
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

## 2026-11-16：V2 chunk64 序列数专项验证

通用 L2 原先拒绝 `seqNum >= 1025`，本次删除该校验。共享校验的 chunk64 路径均受此改动影响；
专项精度实测采用 V2，chunk32 的独立限制不变。旧负向用例 `seq_gt_1024` 改为
`state_seq_count`，继续验证初始状态的 N 维必须与逻辑序列数一致。

- 源码基线：upstream/main `d146b98ce3933b28ae82f9117be5bda28959f6c1`。
- 测试件：[work_tool/atk-test](https://gitcode.com/chen-linxin4/work_tool/tree/atk-test/FLA_ATK/chunk_kda_fwd)，
  固定版本 `e11821d833e73cb9cf2b1fc10b9c8255b0838c63`。
- 设备：A2 / Ascend910B3；CANN 9.2.0（1017），Python 3.11、PyTorch/torch_npu 2.14.0、ATK 26.9.24。
- DUT 使用该源码基线的最新 `_stable.py`，替换测试件原有的私有 ctypes 调用；CPU FP64 golden
  与同精度 control 算法保持不变，测试副本删除旧的 1024 条序列拦截。
- 为确保小 work-items 场景也进入 V2，传入 `epsilon=2e-6`，同时关闭 Q/K L2norm；
  此时 epsilon 不参与数学运算，但会使 Stable adapter 选择 V2。逐 case 日志记录了序列数。
- 构建 `chunk_kda_fwd` 及自动解析的依赖，安装至隔离环境的 OPP overlay；实际加载的
  `libcust_opapi.so` MD5 与新构建一致，全部 19 个 kernel `.o` 与新构建一致。
- BF16 Q/K/V、K=V=128、chunk64、G 为 FP32；beta 交替 FP32/BF16。
  20 例覆盖四种 layout、HV=1/2、safe gate、两种 state 顺序、初始状态、最终状态和中间量。
  序列长度包含 1–5、63、65、129，覆盖尾块和多 chunk；显式与自动 chunk_indices 交替。
- 本专项沿用该外部测试件的 `cv_fused_double_benchmark`：max_re_ratio=5、avg_re_ratio=1.5、
  root_mean_squared_ratio=1.5；这与上文仓内正式套件的 mixed_tolerance_bm 路线分别记录。

| id | seqNum | tokens | layout | H/HV | 初始状态 | 全中间量 | 结果 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1024 | 3318 | TND | 1/1 | 无 | 否 | PASS |
| 1 | 1025 | 3320 | NTD | 1/1 | 无 | 否 | PASS |
| 2 | 1152 | 3702 | BSND | 1/1 | 有 | 否 | PASS |
| 3 | 1280 | 4085 | BNSD | 1/1 | 无 | 是 | PASS |
| 4 | 1408 | 4472 | TND | 1/2 | 无 | 否 | PASS |
| 5 | 1536 | 4853 | NTD | 1/1 | 无 | 否 | PASS |
| 6 | 1664 | 5238 | BSND | 1/1 | 无 | 否 | PASS |
| 7 | 1792 | 5622 | BNSD | 1/1 | 有 | 是 | PASS |
| 8 | 1920 | 6005 | TND | 1/1 | 无 | 否 | PASS |
| 9 | 2048 | 6392 | NTD | 1/2 | 无 | 否 | PASS |
| 10 | 2176 | 6773 | BSND | 1/1 | 无 | 否 | PASS |
| 11 | 2304 | 7158 | BNSD | 1/1 | 无 | 是 | PASS |
| 12 | 2432 | 7542 | TND | 1/1 | 有 | 否 | PASS |
| 13 | 2560 | 7925 | NTD | 1/1 | 无 | 否 | PASS |
| 14 | 2816 | 8693 | BSND | 1/2 | 无 | 否 | PASS |
| 15 | 3072 | 9462 | BNSD | 1/1 | 无 | 是 | PASS |
| 16 | 3328 | 10232 | TND | 1/1 | 无 | 否 | PASS |
| 17 | 3584 | 10998 | NTD | 1/1 | 有 | 否 | PASS |
| 18 | 3840 | 11765 | BSND | 1/1 | 无 | 否 | PASS |
| 19 | 4096 | 12533 | BNSD | 1/2 | 无 | 是 | PASS |

ATK 最终 Excel 的 20 行运行结果均为 SUCCESS、精度通过均为 True；汇总
`Total Task: 20, success 20, failed 0`、`acc_pass_result:Pass`。全部可见输出有限，逐输出比较通过。
额外执行修改后的 `state_seq_count` 负向入口，1/1 通过，错误码 161002 与
`state must be [N,HV,K,V]` 文案匹配。该入口只检查预期拒绝，不比较输出精度；
为适配当前 ATK 的比较器注册名，测试 JSON 的标准字段使用该发行版的 cv_fused_double_benchmark。

在固定测试件目录，准备归档中的 JSON、Stable loader 和 executor 适配后，运行：

```bash
atk node --name npu_dut --backend npu --devices 0 --output_path ./atk_output \
    node --name cpu_reference --backend cpu --output_path ./atk_output \
    task -c ./atk_chunk_kda_fwd_seq20.json --task accuracy --bm_device cpu \
    -p ./executor_chunk_kda_fwd_v2.py -s 0 -e 20 -sp -mt 1 -to 14400
```

结论范围：A2 V2 BF16/K=V=128/chunk64 的 1024–4096 条序列专项通过。
未执行 A3/A5 精度、性能或完整正式套件验收，不据此扩大到所有 shape 或无限序列。
原始 Excel、日志、用例 JSON、执行适配和逐输出指标随任务证据归档，未提交到源码仓库。
