# FusedRecurrentRwkv8 ATK 工程

本目录提供 `fused_recurrent_rwkv8`（RWKV-v8 WKV7 逐 token 递推前向）的 ATK 单算子工程，
包含 `executor_fused_recurrent_rwkv8.py`、`gen_fused_recurrent_rwkv8.py`、
`fused_recurrent_rwkv8.yaml`、`atk_fused_recurrent_rwkv8.json`。

## 输入约束

- 六个位置输入：`q/w/k/z/b=[B,H,T,K]`，`v=[B,H,T,V]`（io 布局 BHTC；K 与 V 独立，
  K==V 是特例）。
- `q/w/k/v/z/b` dtype 一致，支持 `FLOAT32/FLOAT16/BFLOAT16`；`K/V` 为 8 的倍数且 ≤ 128。
- `initial_state` 可选，`[B,H,K,V]` FLOAT32，缺省零初态。
- 属性：`scale`（q 读出缩放，默认 1.0）、`chunk_len`（s 快照间隔，默认 16，
  非 16 值与官方 backward 不兼容）、`output_chunk_state`/`output_sa`（可选输出开关）。
- `w` 为 log 域衰减参数（decay = exp(-exp(w))）；`z = -kk`、`b = kk * a` 参数化
  （kk 为 L2 归一化向量）——造数必须结构化，禁止无约束 randn 造 z/b。

## 标杆来源

CPU 标杆内嵌在 `executor_fused_recurrent_rwkv8.py`（ATK 规范：executor 自包含），
与 `fla/ops/ascendc/rwkv8/fused_recurrent_rwkv8/tests/pta/golden.py` 是同一份逻辑的
两份拷贝——**修改金标算法时两处必须同步**。语义锚点：BlinkDL/RWKV-LM @ 9521024
`RWKV-v8/cuda/wkv7_cuda.cu forward_kernel`；精度对拍锚点：fla @ a4a2624b DPLR。

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的
`executor_fused_recurrent_rwkv8.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`，可配合统一脚本的 `-soc=ascend910b` 使用。

## 默认用例

- FP32 用例：`{"name":"fp32_main","dtype":"fp32","B":2,"H":4,"T":64,"K":64,"V":64,"scale":1.0,"chunk_len":16,"initial_state":false,"output_chunk_state":false,"output_sa":false,"seed":42,"op":"fused_recurrent_rwkv8","case_id":0,"route":"ascendc","soc":"ascend910b"}`
- FP16 用例（K≠V + chunk_len=8 + s/sa 全开 + 非零初态）：`fp16_chunk8`，seed 51
- BF16 用例（scale=0.125 + 非零初态）：`bf16_init_scale`，seed 43

## 用例矩阵

| 文件 | 用例数 | 覆盖 |
|---|---:|---|
| `atk_fused_recurrent_rwkv8.json` | 200 | 按 TilingData 字段组合系统性构造：3 档 dtype、(K,V) 四组合 + 非 64 倍数、(B,T) 边界与多核排队、flags 8 组合、chunk_len {8,16}、scale {1.0,0.125} |
| `atk_fused_recurrent_rwkv8_perf.json` | 38 | 转测文档模板 7.1~7.6 全部代表 shape（典型场景 6 + dtype×(K,V) 9 + flags 8 + chunk_len 4 对 + scale 3 对 + 多核扩展 3） |
| `atk_fused_recurrent_rwkv8_mss.json` | 26 | 各 TilingKey/边角分支的代表用例，用于内存检测复跑 |

精度标准为 `mixed_tolerance_bm`（与 `../README.md` 统一，也与其他算子一致）。
CPU 节点以 FP64 递推生成唯一 golden，落盘前经 `_finite_tuple(golden=True)` 降为 FP32；
NPU 节点保持算子原始 dtype。ATK 按 (NPU dtype, golden FP32) 配对选阈值：

| 输出 | NPU | golden | 判定 |
|---|---|---|---|
| `o`（fp16/bf16 用例） | fp16/bf16 | fp32 | `supports_mixed_tolerance_dtype_pair` → 按 fp16/bf16 阈值 |
| `o`（fp32 用例） | fp32 | fp32 | dtype 相同 → fp32 阈值 |
| `s` / `sa` | fp32 | fp32 | fp32 阈值 |
| `s`（`T < chunk_len`） | numel=0 | numel=0 | 空张量直接 Pass |

阈值为「元素点通过率 ≥ 0.99 且绝对误差同时满足 `max_abs_error_limit` 或 32 倍 ULP」。

三个 JSON 的 `standard` 字段逐条一致，均为
`{"acc": "mixed_tolerance_bm", "mem": 1.1, "perf": "not_key"}`；`fused_recurrent_rwkv8.yaml`
的 `standard` 为 `{acc: mixed_tolerance_bm, perf: not_key}`。

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=fused_recurrent_rwkv8 -npu_device_id=0
bash tests/atk/run_test_cpu.sh -op=fused_recurrent_rwkv8 -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=fused_recurrent_rwkv8 -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=fused_recurrent_rwkv8 -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=fused_recurrent_rwkv8 -npu_device_id=0 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=fused_recurrent_rwkv8 -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，
生成器会把不支持 FP16 的算子改回合法 BF16 用例。

## 实测记录

| 日期 | 精度（200 条） | 确定性（200 条 × 20 轮） | 内存（200 条 memcheck） | 性能（38 条） |
|---|---|---|---|---|
| 2026-09-18 | 200/200 Pass（`mixed_tolerance_bm`，ATK 26.9.8，32m04s） | 200/200 Pass（2791 s，ATK 26.7.8） | 200/200 Pass，零非法访存（771 s，ATK 26.7.8） | 38/38 出数，137.9 µs ~ 21.93 ms |

环境：Ascend 910B4 / CANN 8.5.0 / ATK 26.9.8 / torch 2.10.0+cpu + torch_npu 2.10.0。
完整结论见 `docs/commit/【转测】fused_recurrent_rwkv8.md`。
