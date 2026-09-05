# ChunkFwdH ATK 工程

本目录提供 `chunk_fwd_h` 的 ATK 单算子工程，包含 `executor_chunk_fwd_h.py`、
`gen_chunk_fwd_h.py`、`chunk_fwd_h.yaml`、`atk_chunk_fwd_h.json`、
`atk_chunk_fwd_h_perf.json` 和 `atk_chunk_fwd_h_mss.json`。测试仅通过稳定入口
`fla_npu.ops.ascendc.chunk_fwd_h` 调用算子，不加载 legacy dispatcher。

## 输入约束

- `k/w/u` 均为 BF16 BNSD rank-4 tensor；`k=[B,HK,T,128]`，`w/u=[B,HV,T,128]`。
- `K` 和 `V` 固定为 `128`，`chunk_size` 固定为 `64`，`save_new_value` 固定为 `true`。
- `g` 与 `gk` 必须且只能提供一个；二者均支持 BF16/FP32。
- g-only 模式使用 `g=[B,HV,T]` 和 raw key，要求 `HV >= HK` 且 `HV % HK == 0`。
- gk-only 模式使用 `gk=[B,HV,T,128]`，且 `k` 为 prepared `kg`，要求 `HK=HV`。
- `initial_state` 支持空、BF16 或 FP32；`state_v_first=false/true` 时末两维分别为
  `[K,V]`/`[V,K]`。
- 变长模式使用 BNSD 容器且要求 `B=1`；`cu_seqlens` 从 0 开始、以 `T` 结束并严格递增。
- `chunk_indices` 为空时由稳定入口生成；非空时必须使用 sequence-major 规范顺序。
- 正向矩阵覆盖连续 tensor 和非连续 ND view；私有 NPU format 不属于本接口范围。

## 标杆来源

`torch_custom/fla_npu/test/test_fwd_h.py`；
`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/README.md`

CPU 标杆、输入构造、`run_cpu`、`run_npu` 和 `FunctionApi` 均在本目录的
`executor_chunk_fwd_h.py` 中实现；公共文件只提供基础工具函数。标杆按 sequence、value head
和 chunk 展开递推，并复现 kernel 的 BF16 写回、FP32 矩阵乘累加、state dtype 转换和
`use_exp2` 计算语义。CPU 与 NPU 节点使用相同 seed 和量化输入，ATK 使用
`mixed_tolerance_bm` 比较全部可见输出。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的
`-soc=ascend910b|ascend910_93|ascend950` 使用。提交的 JSON 用例使用 `soc=all`，同一矩阵
用于 A2、A3、A5。

## 默认用例

| 文件 | 用例数 | 用途 | 关键覆盖 |
| --- | ---: | --- | --- |
| `atk_chunk_fwd_h.json` | 200 | 精度 | 40 条功能/边界专项 + 5 组完整 32 项模板矩阵 |
| `atk_chunk_fwd_h_perf.json` | 44 | 性能 | 12 条模型/专项场景 + 32 项模板矩阵 |
| `atk_chunk_fwd_h_mss.json` | 32 | 确定性与内存检测 | 每个顶层模板实例一条 |

精度矩阵的 case 0-39 覆盖最小输入、分核、g/gk、state 生命周期、tail、变长和非连续 view；
case 40-71、72-103、104-135、136-167、168-199 分别覆盖 dense tail=1、整 chunk、
dense tail=63、terminal no-final 和 varlen explicit-index 的完整模板矩阵。性能矩阵包含
`B=2,HK=16,HV=32,T=11264` 目标场景、对应 B=1/initial-state 对照、完整模板矩阵和变长/模型
shape。`atk_chunk_fwd_h_mss.json` 同时供 determinism 与 mssanitizer scope 使用，默认均执行
完整 32 项模板矩阵。

### TilingKey 覆盖

顶层模板参数依次为 gate dtype、`V_DIM`、g/gk、exp/exp2、state dtype 和 state layout。
`V_DIM=128` 固定，其余五项各有两个取值，共 `2 x 1 x 2 x 2 x 2 x 2 = 32` 个实例。
下表数值来自当前 CANN 9.1 构建产物，不作为跨版本公共语义；选择条件依次表示
`gate dtype / gate mode / exponent / state dtype / state layout`。普通 accuracy 使用整 chunk，
边界 accuracy 列依次为 tail=1、tail=63、no-final 和 varlen case。A2/A3/A5 clean build
均确认 32-key metadata 与 wrapper 映射；实际选择只记录 host/runtime 已加载的 key，未执行的
组合明确标为未验证。

| TilingKey | 选择条件 | 普通 accuracy | 边界 accuracy | performance | determinism / MSS | 适用 SOC | 编译映射证据 | 实际选择证据 |
| ---: | --- | ---: | --- | ---: | ---: | --- | --- | --- |
| 10 | BF16 / g / exp / BF16 / `[K,V]` | 72 | 40/104/136/168 | 7 | 0 | A2/A3/A5 | A2/A3/A5 | A2 runtime |
| 4106 | BF16 / g / exp / BF16 / `[V,K]` | 73 | 41/105/137/169 | 8 | 1 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 2058 | BF16 / g / exp / FP32 / `[K,V]` | 74 | 42/106/138/170 | 9 | 2 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 6154 | BF16 / g / exp / FP32 / `[V,K]` | 75 | 43/107/139/171 | 10 | 3 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 1034 | BF16 / g / exp2 / BF16 / `[K,V]` | 76 | 44/108/140/172 | 11 | 4 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 5130 | BF16 / g / exp2 / BF16 / `[V,K]` | 77 | 45/109/141/173 | 12 | 5 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 3082 | BF16 / g / exp2 / FP32 / `[K,V]` | 78 | 46/110/142/174 | 13 | 6 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 7178 | BF16 / g / exp2 / FP32 / `[V,K]` | 79 | 47/111/143/175 | 14 | 7 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 522 | BF16 / gk / exp / BF16 / `[K,V]` | 80 | 48/112/144/176 | 15 | 8 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 4618 | BF16 / gk / exp / BF16 / `[V,K]` | 81 | 49/113/145/177 | 16 | 9 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 2570 | BF16 / gk / exp / FP32 / `[K,V]` | 82 | 50/114/146/178 | 17 | 10 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 6666 | BF16 / gk / exp / FP32 / `[V,K]` | 83 | 51/115/147/179 | 18 | 11 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 1546 | BF16 / gk / exp2 / BF16 / `[K,V]` | 84 | 52/116/148/180 | 19 | 12 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 5642 | BF16 / gk / exp2 / BF16 / `[V,K]` | 85 | 53/117/149/181 | 20 | 13 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 3594 | BF16 / gk / exp2 / FP32 / `[K,V]` | 86 | 54/118/150/182 | 21 | 14 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 7690 | BF16 / gk / exp2 / FP32 / `[V,K]` | 87 | 55/119/151/183 | 22 | 15 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 30 | FP32 / g / exp / BF16 / `[K,V]` | 88 | 56/120/152/184 | 23 | 16 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 4126 | FP32 / g / exp / BF16 / `[V,K]` | 89 | 57/121/153/185 | 24 | 17 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 2078 | FP32 / g / exp / FP32 / `[K,V]` | 90 | 58/122/154/186 | 25 | 18 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 6174 | FP32 / g / exp / FP32 / `[V,K]` | 91 | 59/123/155/187 | 26 | 19 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 1054 | FP32 / g / exp2 / BF16 / `[K,V]` | 92 | 60/124/156/188 | 27 | 20 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 5150 | FP32 / g / exp2 / BF16 / `[V,K]` | 93 | 61/125/157/189 | 28 | 21 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 3102 | FP32 / g / exp2 / FP32 / `[K,V]` | 94 | 62/126/158/190 | 29 | 22 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 7198 | FP32 / g / exp2 / FP32 / `[V,K]` | 95 | 63/127/159/191 | 30 | 23 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 542 | FP32 / gk / exp / BF16 / `[K,V]` | 96 | 64/128/160/192 | 31 | 24 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 4638 | FP32 / gk / exp / BF16 / `[V,K]` | 97 | 65/129/161/193 | 32 | 25 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 2590 | FP32 / gk / exp / FP32 / `[K,V]` | 98 | 66/130/162/194 | 33 | 26 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 6686 | FP32 / gk / exp / FP32 / `[V,K]` | 99 | 67/131/163/195 | 34 | 27 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 1566 | FP32 / gk / exp2 / BF16 / `[K,V]` | 100 | 68/132/164/196 | 35 | 28 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 5662 | FP32 / gk / exp2 / BF16 / `[V,K]` | 101 | 69/133/165/197 | 36 | 29 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 3614 | FP32 / gk / exp2 / FP32 / `[K,V]` | 102 | 70/134/166/198 | 37 | 30 | A2/A3/A5 | A2/A3/A5 | 未验证 |
| 7710 | FP32 / gk / exp2 / FP32 / `[V,K]` | 103 | 71/135/167/199 | 38 | 31 | A2/A3/A5 | A2/A3/A5 | A2 runtime |

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -npu_device_id=0
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -npu_device_id=0 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_h -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`，只执行 ATK case 模板生成。仓内三份冻结 JSON 由以下
命令重建：

```bash
python3 tests/atk/chunk_fwd_h/gen_chunk_fwd_h.py \
  --output-dir tests/atk/chunk_fwd_h \
  --summary
```

冻结 JSON 的 case id、`case_key` 和 seed 由生成器固定；修改矩阵后需要重建并复核三份 JSON。
