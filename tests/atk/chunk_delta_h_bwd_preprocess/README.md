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
| `scripts/make_case.py` | 由用例参数生成 NPU 侧输入（落盘原始字节）与 CPU 标杆 `expected_dhm.bin`，并写出 `run_case` 命令行 |
| `scripts/test_aclnn_chunk_delta_h_bwd_preprocess.cpp` | aclnn 直调取数：输入/输出全走文件，输出额外落 `workspace.bin`（用户区起点 16 MiB，可按 tiling 公式定位每个平面） |
| `scripts/compare.py` | 按 `E_r`/`P_r` 分平面给出 `max_abs`/`max_rel`/`rel_norm`，并给出 PASS/FAIL |
| `scripts/inspect_workspace.py` | 逐平面核对 workspace（NT=1）：slot（Q̄s/K̄/W/do/decayK）、T1、dVpre、dVhat、qterm、wterm、Pc、PBf、P 两个 parity |
| `scripts/run_accuracy.sh` | 上述"aclnn 直调取数 → 逐平面比对"的一键入口（用于定位与给出 `rel_norm` 口径） |
| `scripts/ctrl_case.py` | 受控实验：把某个 case 的 `W` 改成"仅第 0 行全 1"，据此可只手推出 `T1`/`P` 的期望结构，用于定位"左操作数转置语义/某个平面写错"这类问题 |
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

## 定位口径：直调 aclnn 的逐平面取数（`scripts/`）

1. 按 `cases.json` 生成用例；
2. NPU 侧调用 `fla_npu.ops.ascendc.chunk_delta_h_bwd_preprocess`，CPU 侧调用 `reference.py` 的同名口径；
3. 逐用例比较 `dhm`，并单独抽出 `dV_pre`、`T1`、`dV̂'`、`inc`、`dH`、`P_c`、`P` 与参考实现比对，
   用于区分"公式错误"与"ready/free 缺边导致的旧值/新值混用"；
4. 覆盖 A2/A5 两平台与 `USE_G`/`USE_GK`/无门控、dense/varlen、尾块、GVA；本版只支持 `K = V = 128`；
5. 反向用例只验证拦截：`g`/`gk` 同时非空、`K != 128`（64/256/512）、`V != 128`（96）、`Hv%Hk!=0`、
   `B>1`（dense/varlen）、`chunk_size!=64`、`g` shape 不匹配、`gk` FP32、`cu_seqlens` 过短、空 tensor。

### 一键执行（直调 aclnn，设备侧）

```bash
# 1) 编包并安装（A5 例：--soc=ascend950；A2 例：--soc=ascend910b）
bash build.sh --pkg --soc=<soc> --vendor_name=fla_npu --ops=chunk_delta_h_bwd_preprocess -j16
bash build_out/fla-npu-fla_npu_linux-*.run --install-path=<install_path>

# 2) 编取数程序并逐用例执行
cd tests/atk/chunk_delta_h_bwd_preprocess/scripts
g++ -std=c++17 -O2 test_aclnn_chunk_delta_h_bwd_preprocess.cpp -o run_case \
  -I<install_path>/vendors/fla_npu_transformer/op_api/include -I$ASCEND_HOME_PATH/include \
  -L<install_path>/vendors/fla_npu_transformer/op_api/lib -lcust_opapi \
  -L$ASCEND_HOME_PATH/lib64 -lascendcl -lnnopbase -lpthread -ldl
export ASCEND_CUSTOM_OPP_PATH=<install_path>/vendors/fla_npu_transformer
export LD_LIBRARY_PATH=<install_path>/vendors/fla_npu_transformer/op_api/lib:$LD_LIBRARY_PATH

python3 make_case.py --dir ./case_pos_13 --dtype bf16 --gate gk --Hk 4 --Hv 4 --T 2048 --K 128 --V 128
./run_case $(cat ./case_pos_13/run_case_args.txt)
python3 compare.py --dir ./case_pos_13
```

### 直调口径的已验证结果

`cases.json` 的 12 条正向用例（全部 `K = V = 128`、`chunk_size = 64`，本版唯一支持的场景）在
**A5（ascend950）与 A2（ascend910b）上均为 12/12 PASS**，`0` 编译错误。
下表数值来自**最终源码状态**下的完整重跑（两个平台的算子源码与本仓库工作区逐文件 md5 一致；
`build/autogen/inner/` 为空，确认 aclnn 走的是手写 exc 通路）。
arch35（A5）Vector 侧改为 RegBase `__simd_vf__` 融合后重跑，数值与改造前逐项一致（见下表）。

| 用例 | A5 `E/P rel_norm` | A2 `E/P rel_norm` |
| --- | --- | --- |
| `pos_01_none_gate_dense`（T=512） | 5.56e-3 / 7.10e-3 | 5.52e-3 / 7.17e-3 |
| `pos_02_g_bf16_dense` | 5.06e-3 / 5.18e-3 | 5.19e-3 / 5.17e-3 |
| `pos_03_g_fp32_dense` | 5.08e-3 / 5.47e-3 | 5.07e-3 / 5.44e-3 |
| `pos_04_gk_bf16_dense` | 6.28e-3 / 5.50e-3 | 6.30e-3 / 6.57e-3 |
| `pos_05_gk_varlen_first_segment` | 3.79e-3 / 4.52e-3 | 3.72e-3 / 4.52e-3 |
| `pos_06_tail_chunk`（T=200，尾块 8） | 4.01e-3 / 4.33e-3 | 4.01e-3 / 4.27e-3 |
| `pos_10_gva_hv_gt_hk`（Hk=4,Hv=8） | 4.52e-3 / 3.59e-3 | 4.52e-3 / 3.59e-3 |
| `pos_11_single_chunk`（T=64） | 2.08e-3 / 4.29e-3 | 2.08e-3 / 4.29e-3 |
| `pos_12_two_chunk_chain_direction` | 2.77e-3 / 2.98e-3 | 2.77e-3 / 2.99e-3 |
| `pos_13_long_nt_chain_accumulation`（32 chunk） | 1.15e-2 / 9.30e-3 | 9.50e-3 / 6.94e-3 |
| `pos_14_fp16_inputs` | 6.51e-4 / 6.72e-4 | 7.12e-4 / 7.04e-4 |
| `pos_15_head_contiguous_partition`（96 head / 多 task） | 4.32e-3 / 4.36e-3 | 4.32e-3 / 4.36e-3 |

`pos_07_k64_block32`（K=64）、`pos_08_k256_four_groups`（K=V=256）、`pos_09_v_tail_tile`（V=96）、
`pos_16_tile_split_partition`（K=V=256）四条历史用例已从正向矩阵移出：本版只支持 `K = V = 128`，
这些取值现在按不支持拦截，见下方反向用例 `neg_11`/`neg_12`/`neg_13`。

`rel_norm = max_abs / max|参考|`。上表的量级与"链上状态用模型 dtype（bf16/fp16）传递"这一设计一致：单
task / 单 chunk 场景约 2e-3，长链与多 task 场景最多 1.2e-2。重复执行（`pos_15`/`pos_13`/`gate_gk` 各 3~30 次）
结果逐位一致。

### 反向拦截（已执行）

```bash
bash tests/atk/chunk_delta_h_bwd_preprocess/scripts/run_negative.sh <work_dir>
```

`cases.json` 的 13 条反向用例在 **A2 与 A5 上均 13/13 PASS**（实际返回码与 `expected_return_code` 一致，
均为 `ACLNN_ERR_PARAM_INVALID` = 161001）：

| 用例 | 触发约束 |
| --- | --- |
| `neg_01_g_and_gk_both` | `g` 与 `gk` 同时非空（互斥） |
| `neg_02_k_too_large` | `K = 512`（本版只支持 `K = V = 128`） |
| `neg_03_hv_not_multiple_of_hk` | `Hk=3, Hv=4`（`Hv % Hk != 0`） |
| `neg_04_dense_b_greater_than_one` | dense 路径 `B = 2` |
| `neg_05_varlen_b_greater_than_one` | varlen 且 `B = 2` |
| `neg_06_chunk_size_not_64` | `chunk_size = 128` |
| `neg_07_g_shape_mismatch` | `g` 为 `[B,Hk,T]`（GVA：`Hk=2, Hv=4`） |
| `neg_08_gk_dtype_fp32` | `gk` 为 FP32 |
| `neg_09_cu_seqlens_too_short` | `cu_seqlens` 只有 1 项 |
| `neg_10_empty_tensor` | `T = 0` |
| `neg_11_k64_not_128` | `K = 64`（不是 128） |
| `neg_12_v96_not_128` | `V = 96`（不是 128） |
| `neg_13_k256_not_128` | `K = 256`（不是 128） |

反向用例只校验拦截与返回码，不做精度比较；脚本会 `grep` `run_case` 打印的 `GetWorkspaceSize failed <code>`
并比对期望值，全部通过才输出 `ALL_PASS`。

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
- `compare.py` 默认 `--tol 2e-2`，可用 `--tol` 收紧/放宽（阈值变更必须说明理由，不允许用阈值掩盖真实误差）。
- 不允许通过收窄输入 range、跳过失败用例或放宽阈值来制造通过结论。
