# chunk_delta_h_bwd_preprocess 测试

## 目录结构（`tests/atk` 规范）

| 文件 / 目录 | 说明 |
| --- | --- |
| `gen_chunk_delta_h_bwd_preprocess.py` 内联用例表 | 用例设计来源：12 条正向（K = V = 128、chunk_size = 64）+ 13 条反向拦截，由生成器展开成三份 `atk_*.json` |
| `chunk_delta_h_bwd_preprocess.yaml` | ATK 用例 schema（`atk case -f` 用；标准 `mixed_tolerance_bm`） |
| `gen_chunk_delta_h_bwd_preprocess.py` | 由 `gen_<op>.py` 内联用例表 展开出精度 / 性能 / 内存三份 ATK 用例矩阵；同时注册 `-scope=gen_cases` 用的 generator |
| `executor_chunk_delta_h_bwd_preprocess.py` | ATK executor：按 `case_spec` 确定性构造输入，NPU 走 `fla_npu.ops.ascendc`，CPU 走 `scripts/reference.py` 标杆 |
| `atk_chunk_delta_h_bwd_preprocess.json` | 精度矩阵（12 条，`standard.acc` 用 ATK 原生 `output_dtype_overrides` 按模型 dtype 判 `dhm`） |
| `atk_chunk_delta_h_bwd_preprocess_perf.json` | 性能精简矩阵（3 条：dense / varlen / 32 chunk 长链） |
| `atk_chunk_delta_h_bwd_preprocess_mss.json` | 内存检测与确定性矩阵（5 条，覆盖 4 个 tilingKey + varlen） |
| `scripts/reference.py` | CPU 全精度参考：`preprocess_reference` 给出 `dhm = [E_r \| P_r]`；`dh_scan_direct` 用非零 `dht` 反扫得到真实 `dh0`；`check_affine` 校验 `dh0 == P_r @ dht + E_r` |
| `atk_chunk_delta_h_bwd_preprocess.json` 的 13 条反向用例 | 由 `gen_<op>.py` 生成、带 `expected_return_code`，走 `atk ... --task run` 核对返回码（A5/A2 均 13/13） |

## ATK 一键执行（规范入口）

```bash
FLA_NPU_ENV=<custom opp>/bin/set_env.bash \
ATK_OUTPUT_ROOT=<output root> \
bash tests/atk/run_test_cpu.sh \
  -op=chunk_delta_h_bwd_preprocess -npu_device_id=0 -soc=ascend950 -scope=accuracy
```

`-scope` 可取 `all` / `accuracy` / `performance` / `determinism` / `mssanitizer` / `gen_cases`。
本算子目录的 `atk_*.json` 与生成器都取自 `gen_<op>.py` 内联用例表，改用例只需改 `gen_<op>.py` 内联用例表 后重跑
`python tests/atk/chunk_delta_h_bwd_preprocess/gen_chunk_delta_h_bwd_preprocess.py --summary`。

### ATK 精度结果（已执行）

| 平台 | 命令 | 结果 |
| --- | --- | --- |
| A5 / `ascend950` | `atk node --backend npu --devices <id> -o <out> node --backend cpu task -c ./atk_chunk_delta_h_bwd_preprocess.json --task accuracy -p ./executor_chunk_delta_h_bwd_preprocess.py -s 0 -e 12 -to 2000` | **12/12 通过**，`Total Task: 12, success 12, failed 0`，`acc_pass_result: Pass` |
| A2 / `ascend910b` | 同上（该机自带 ATK 版本低于 runner 要求，改用本机已有 ATK 26.9.8 运行） | **12/12 通过**，`Total Task: 12, success 12, failed 0`，`acc_pass_result: Pass` |
| A5 / `ascend950` | `... -scope=performance`（3 条：dense / varlen / 32 chunk） | **3/3 执行成功**，device 中位耗时 270.7 / 274.1 / 1066.7 µs（见下） |

上面两条精度行是把同一份 `atk_<op>.json` 用 `atk node ... --task accuracy` 直接执行的结果；同一份用例也可以走
`run_test_cpu.sh -scope=accuracy`。`fla_npu.ops.ascendc` 在本机稳定入口不可用时会自动回退 ctypes 参考通路，
两条通路共用同一份手写 aclnn，稳定入口本身另有验证（见下文"稳定入口"一节）。

`-scope=accuracy` 会走 `--bm_device cpu`，CPU 标杆的比较任务经 ATK 的 broker 队列派发；该 broker 默认占用
本机 `127.0.0.1:9090`。若 9090 已被其它服务占用（本机即是），runner 会在 compare 阶段报
`Unrecoverable error: JSONDecodeError('Expecting value: line 1 column 1 (char 0)')` 并卡住；runner 没有暴露端口参数。
此时用同一份用例显式指定空闲端口即可，实测 12/12 通过：

```bash
atk node --name npu_dut --backend npu --devices <device_id> -p <free_port> --output_path <out> \
  node --name cpu_golden --backend cpu -p <free_port> --output_path <out> \
  task -c ./atk_chunk_delta_h_bwd_preprocess.json --task accuracy --bm_device cpu \
  -p ./executor_chunk_delta_h_bwd_preprocess.py -s 0 -e 12 -to 2000
```

判据是 ATK 原生 `mixed_tolerance_bm`；`dhm` 虽是 FP32，但链上状态按设计用模型 dtype 传递，
所以用例的 `standard.acc` 用 ATK 原生的 `output_dtype_overrides` 声明按 `bf16`/`fp16` 判，
不在 executor 里自定义指标。

### ATK 性能结果（已执行，A5 / ascend950）

| 用例 | shape | device 中位耗时 |
| --- | --- | --- |
| `pos_01_none_gate_dense` | `B=1, Hk=Hv=4, T=512, K=V=128`（8 chunk） | 270.7 µs |
| `pos_05_gk_varlen_first_segment` | 同 shape，本 launch 只算 `[0,300)`（5 chunk） | 274.1 µs |
| `pos_13_long_nt_chain_accumulation` | 同 shape，`T=2048`（32 chunk） | 1066.7 µs |

数值是 ATK `performance_device` 的 device 中位耗时（同 shape 下单 chunk 约 33 µs，随 chunk 数近似线性）。
`-scope=determinism` / `-scope=mssanitizer` 使用同一份 `atk_<op>_mss.json`（5 条，覆盖 4 个 tilingKey），
本版未执行。

### ATK 反向拦截结果（已执行，`--task run`）

命令（NPU 单节点，不带 CPU 标杆节点）：

```bash
atk node --backend npu --devices <device_id> task \
  -c ./atk_chunk_delta_h_bwd_preprocess.json --task run \
  -p ./executor_chunk_delta_h_bwd_preprocess.py -s 12 -e 25
```

| 平台 | 结果 |
| --- | --- |
| A5 / `ascend950` | `Total Task: 13, success 13, failed 0`，13 条命中的 `aclnnStatus` 均为 161001 |
| A2 / `ascend910b` | `Total Task: 13, success 13, failed 0`，同上 |

判据：executor 先从算子异常文本里取 `aclnnStatus=<code>` 与用例声明的 `expected_return_code` 比对，
一致后才抛出与用例 `expected_error_msg` 完全一致的文本，交由 ATK 的"预期失败"判定（与
`tests/atk/chunk_kda_fwd` 同一口径）。返回码不符或算子没有拦截时，用例记为失败。

接入时踩到的两个 ATK 行为（记下来避免误判）：

- `BaseBackend.before_call` 会**丢掉所有 `dtype=non_param` 的输入**，`case_spec` 正是这一类；executor
  实际读到的是同一份元数据的**标量副本**。因此 `cu_seqlens` 按逗号分隔的 string 下发——用 `case_spec`
  或张量通道传都到不了 executor，会让 varlen 用例静默退化成 dense（dense 与 dense 互相比较仍会"通过"）。
- `compare_error_msg` 在字符串包含匹配未命中后会把 `expected_error_msg` 当**正则**编译；编译失败时会
  `raise` 一个非异常对象，报 `TypeError: exceptions must derive from BaseException`，掩盖真实比对结论。
  生成器因此保证下发的 `expected_error_msg` 本身是合法正则。

## 本地自检（无需 NPU）

```bash
python tests/atk/chunk_delta_h_bwd_preprocess/scripts/reference.py
```

该自检对无门控 / `USE_G` / `USE_GK` 三种模式分别构造随机输入，用**非零** `dht` 验证仿射恒等式。
只测 `dht = 0` 无法验证 `P_r`，因此自检固定使用非零 `dht`。

## 稳定入口（`fla_npu.ops.ascendc`）验证

`from fla_npu.ops.ascendc import chunk_delta_h_bwd_preprocess`（等价 `npu_chunk_delta_h_bwd_preprocess`）默认走 Stable-ABI 适配层（`csrc/src/stable_chunk_delta_h_bwd_preprocess.cpp` + `torch.ops.fla_npu_stable`），`libfla_npu_stable.so` 不可用时回退 ctypes 参考实现；两条通路与 aclnn 直调取数程序共用同一份输入与 CPU 标杆。已执行用例与结论（A5 / ascend950）：

| 用例 | E rel_norm | P rel_norm | 结论 |
| --- | --- | --- | --- |
| `pos_01_none_gate_dense`（无门控） | 5.562e-03 | 7.098e-03 | PASS |
| `pos_03_g_fp32_dense`（`g` 为 FP32） | 5.076e-03 | 5.466e-03 | PASS |
| `pos_04_gk_bf16_dense`（逐 K gate） | 6.280e-03 | 5.498e-03 | PASS |
| `pos_05_gk_varlen_first_segment`（gk + varlen） | 3.787e-03 | 4.515e-03 | PASS |
| `pos_14_fp16_inputs`（fp16） | 6.506e-04 | 6.721e-04 | PASS |
| `pos_15_head_contiguous_partition`（96 head 多 task） | 4.321e-03 | 4.356e-03 | PASS |

数值与 aclnn 直调口径逐项一致（同一份源码）。A2 / ascend910b 同口径复核也是 **6/6 PASS**
（`pos_01` 5.521e-03 / 7.169e-03、`pos_03` 5.068e-03 / 5.442e-03、`pos_04` 6.301e-03 / 6.572e-03、
`pos_05` 3.724e-03 / 4.519e-03、`pos_14` 7.124e-04 / 7.043e-04、`pos_15` 4.321e-03 / 4.357e-03）。

环境要点（复现该入口时容易踩）：稳定入口要求 custom OPP 指向**本算子 run 包的安装目录**
（`<install>/vendors/<vendor>`，同时把 `FLA_NPU_ENV` 指向该目录的 `set_env.bash`）。把 OPP
**复制**到别处再用会在 A2 上解析不到 tiling compile-info（`InitTilingParseCtx failed ...
compile info not contain [_pattern]`），从而在 `GetWorkspaceSize` 报 `aclnnStatus=161001`；
指回原安装目录即恢复正常。

## 精度判据

- `dhm` 为 FP32，但**链上状态按设计用模型 dtype 传递**（`Pc`/`PBf`/`dHBf` 为 bf16 或 fp16），因此绝对误差随
  序列长度放大（例如 32 chunk 用例 `max_abs` 可达 5.7e28）。判据采用**相对参考幅值**：
  `rel_norm = max_abs / max|参考|`，阈值 2%。
- 同时打印逐元素 `within2%`（相对误差小于 2% 的元素占比）作为辅助观察项；`max_rel` 会被参考中的极小值
  放大，不作为判据。
- 不允许通过收窄输入 range、跳过失败用例或放宽阈值来制造通过结论。
