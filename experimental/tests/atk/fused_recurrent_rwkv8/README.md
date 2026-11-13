# FusedRecurrentRwkv8 experimental ATK 工程

> 当前实验 DUT 为 `fla_npu_experimental.fused_recurrent_rwkv8`，来源 PR #446。
> 当前构建/精度/隔离结论集中在 [验证记录](../../../validation.md)；来源 PR 的历史结果不累计为当前通过数。
> ATK 是手工复验备选，独立手动脚本直接比较 NPU 与 CPU reference。
> ATK worker 若提前导入 torch_npu，需在专用环境启用实验 bootstrap，详见[备选执行方式](#atk-备选执行方式)。
> 实验 `.run` / wheel 安装与调用方式见 [experimental README](../../../README.md)。
> 原 209 条用例不包含 reverse，只有最后 9 条明确覆盖 3 seeds；不据此声明完整正式验收。

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
与 `experimental/_tests/rwkv8_reference.py` 是同一份逻辑的
备选副本，手动独立对拍、PTA 和 dump 复用实验目录的 reference；**修改数学语义时须同步并通过一致性测试**。语义锚点：BlinkDL/RWKV-LM @ 9521024
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
| `atk_fused_recurrent_rwkv8.json` | 209 | 200 条按 TilingData 字段组合系统性构造：3 档 dtype、(K,V) 四组合 + 非 64 倍数、(B,T) 边界与多核排队、flags 8 组合、chunk_len {8,16}、scale {1.0,0.125}；另 9 条 blockDim 上限用例（3 个 shape：B*H = 65535 的 fp32 T=1、fp16 T=4 与 65528 的 bf16 H=8 T=8，每个 shape 3 个固定 seed） |
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

## 手动独立对拍

手动独立对拍不依赖 ATK。`experimental/_tests/accuracy.py` 直接比较实验 wheel 的 NPU 输出与
CPU reference，使用本目录精度 JSON（quick 前 3 条，full 全部 209 条），默认/inline 各执行一次。
共享输入与 reference 在 `experimental/_tests/rwkv8_reference.py`，契约测试检查造数 AST 和与自包含 executor 的 FP32/FP64 输出一致性。FP64 递推后转 FP32、输出契约和上述混合容差保持不变。
依赖在构建前检查，缺 ATK 仅告警；下列 ATK 工程保留为精度复验及确定性/MSS/性能备选。
独立脚本的实测结果与边界见 [手动对拍验证记录](../../../validation.md)。

在仅安装匹配实验 wheel 的独立环境中，启动新 shell、初始化 CANN 后手工执行：

```bash
export FLA_NPU_DISABLE_PTH=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
E="$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("fla_npu_experimental").origin).parent)')"
source "$E/opp/vendors/fla_npu_experimental_transformer/bin/set_env.bash"
python experimental/_tests/accuracy.py --mode quick --device 0 --output /tmp/rwkv8-queue.json
FLA_NPU_STABLE_LAUNCH=inline \
    python experimental/_tests/accuracy.py --mode full --device 0 --output /tmp/rwkv8-inline.json
```

## ATK 备选执行方式

ATK 不负责安装实验 wheel 或选择 vendor。先激活专用实验 Python 环境，source CANN 和实验
wheel 内的 vendor 环境脚本。下列命令显式选择实验目录，不使用正式算子的自动目录发现。
CPU 标杆保持独立，NPU executor 调用 `fla_npu_experimental`。

worker 可能提前导入 `torch_npu`，因此在专用环境中安装 `00_experimental_atk_bootstrap.pth`，
内容为 `import fla_npu_experimental`，并在任何 ATK/Python 进程启动前设置
`FLA_NPU_DISABLE_PTH=1` 和实验 vendor 环境。bootstrap 仅用于备选验证，不随 wheel 发布，
也不用于装载顺序矩阵；维护 wheel 时先禁用，装好匹配包并配置环境后再启用。
远程或常驻 worker 必须在自身启动环境完成同样准备，控制端后来 source 不会更新它们。
使用已安装 wheel，不能用源码 `PYTHONPATH` 替代；保留 CANN 自身的 `acl/tbe` 路径。

```bash
# 进入实验 ATK 工程，后续文件路径均相对此目录。
cd experimental/tests/atk/fused_recurrent_rwkv8
# 用 NPU 执行 DUT、CPU 执行标杆，比较全部精度用例。
atk node --name npu_dut --backend npu --devices 0 --output_path ./atk_output/accuracy \
    node --name cpu_golden --backend cpu --output_path ./atk_output/accuracy \
    task -c atk_fused_recurrent_rwkv8.json --task accuracy --bm_device cpu \
    -p executor_fused_recurrent_rwkv8.py --gm_init_flag -to 14400
# 仅测量 NPU 性能，保存 profiling 数据。
atk node --name npu_dut --backend npu --devices 0 --output_path ./atk_output/perf \
    task -c atk_fused_recurrent_rwkv8_perf.json --task performance_device \
    -p executor_fused_recurrent_rwkv8.py --save_data profile -sp -to 2000
# 重复执行固定输入，检查确定性。
atk node --name npu_dut --backend npu --devices 0 \
    task -c atk_fused_recurrent_rwkv8_mss.json --task accuracy_dc \
    -p executor_fused_recurrent_rwkv8.py --dc_loop_nums 50 -to 3600
# 生成候选精度用例；正式对拍仍使用已固化的 JSON。
atk case -f fused_recurrent_rwkv8.yaml -p gen_fused_recurrent_rwkv8.py -dt 100 -en 0 -s 20260813
```

候选生成使用 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，
生成器会把不支持 FP16 的算子改回合法 BF16 用例。

## 当前目录迁移验证

2026-11-12：A2 / CANN 9.2.1，直接 NPU/CPU reference 队列与 inline 各 209/209；
增量 wheel、wheel+run、禁用、恢复、仅 run 五种方式各 3/3。JSON 与数学标杆不变。
ATK 的完整精度、MSS、确定性及性能未在本轮执行；上述结果属于独立对拍脚本，详见
[本轮记录](../../../validation.md#2026-11-12-目录重构运行时精简与通用增量接入)。

## 来源 PR 历史实测记录

| 日期 | 精度 | 确定性 | 内存 | 性能 |
|---|---|---|---|---|
| 2026-09-18 | 200/200 Pass（`mixed_tolerance_bm`，ATK 26.9.8，32m04s） | 200 条 × 20 轮 Pass（2791 s，ATK 26.7.8） | 200 条 memcheck Pass，零非法访存（771 s，ATK 26.7.8） | 38/38 出数，137.9 µs ~ 21.93 ms |
| 2026-10-20 | 追加 9 条 blockDim 上限用例单独复跑：9/9 Pass（3 个 shape × 3 seeds；B*H = 65535 的 fp32 T=1 / fp16 T=4，B*H = 65528 的 bf16 H=8 T=8，ATK 26.9.8） | - | - | - |

环境：Ascend 910B4 / CANN 8.5.0 / ATK 26.9.8 / torch 2.10.0+cpu + torch_npu 2.10.0。
完整结论见 `docs/commit/【转测】fused_recurrent_rwkv8.md`。
