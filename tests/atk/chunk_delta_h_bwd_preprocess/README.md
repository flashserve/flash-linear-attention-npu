# chunk_delta_h_bwd_preprocess 测试

## 目录结构（`tests/atk` 规范）

| 文件 / 目录 | 说明 |
| --- | --- |
| `cases.json` | **用例设计唯一来源**：12 条正向（全部 `K = V = 128`、`chunk_size = 64`）+ 13 条反向拦截 |
| `chunk_delta_h_bwd_preprocess.yaml` | ATK 用例 schema（`atk case -f` 用；标准 `mixed_tolerance_bm`） |
| `gen_chunk_delta_h_bwd_preprocess.py` | 由 `cases.json` 展开出精度 / 性能 / 内存三份 ATK 用例矩阵；同时注册 `-scope=gen_cases` 用的 generator |
| `executor_chunk_delta_h_bwd_preprocess.py` | ATK executor：按 `case_spec` 确定性构造输入，NPU 走 `fla_npu.ops.ascendc`，CPU 走 `scripts/reference.py` 标杆 |
| `atk_chunk_delta_h_bwd_preprocess.json` | 精度矩阵（12 条，`standard.acc` 用 ATK 原生 `output_dtype_overrides` 按模型 dtype 判 `dhm`） |
| `atk_chunk_delta_h_bwd_preprocess_perf.json` | 性能精简矩阵（3 条：dense / varlen / 32 chunk 长链） |
| `atk_chunk_delta_h_bwd_preprocess_mss.json` | 内存检测与确定性矩阵（5 条，覆盖 4 个 tilingKey + varlen） |
| `scripts/reference.py` | CPU 全精度参考：`preprocess_reference` 给出 `dhm = [E_r \| P_r]`；`dh_scan_direct` 用非零 `dht` 反扫得到真实 `dh0`；`check_affine` 校验 `dh0 == P_r @ dht + E_r` |
| `scripts/make_negative_case.py` + `scripts/run_negative.sh` | 反向拦截用例：按 `cases.json` 的 `negative_cases` 生成"非法但类型正确"的输入，调用 aclnn 并核对返回码 |

## ATK 一键执行（规范入口）

```bash
FLA_NPU_ENV=<custom opp>/bin/set_env.bash \
ATK_OUTPUT_ROOT=<output root> \
bash tests/atk/run_test_cpu.sh \
  -op=chunk_delta_h_bwd_preprocess -npu_device_id=0 -soc=ascend950 -scope=accuracy
```

`-scope` 可取 `all` / `accuracy` / `performance` / `determinism` / `mssanitizer` / `gen_cases`。
本算子目录的 `atk_*.json` 与生成器都取自 `cases.json`，改用例只需改 `cases.json` 后重跑
`python tests/atk/chunk_delta_h_bwd_preprocess/gen_chunk_delta_h_bwd_preprocess.py --summary`。

### ATK 精度结果（已执行）

| 平台 | 命令 | 结果 |
| --- | --- | --- |
| A5 / `ascend950` | `bash tests/atk/run_test_cpu.sh -op=chunk_delta_h_bwd_preprocess -npu_device_id=0 -soc=ascend950 -scope=accuracy`（stable 后端） | **12/12 通过**，`acc_pass_result: Pass` |
| A2 / `ascend910b` | 同上 `-soc=ascend910b`（ctypes 后端；该机 ATK 26.4.30 低于 runner 要求的 26.8.8，用本机已有 ATK 26.9.8 运行） | **12/12 通过**，`acc_pass_result: Pass` |
| A5 / `ascend950` | `... -scope=performance`（3 条：dense / varlen / 32 chunk） | **3/3 执行成功**，device 中位耗时 270.7 / 274.1 / 1066.7 µs（见下） |

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
