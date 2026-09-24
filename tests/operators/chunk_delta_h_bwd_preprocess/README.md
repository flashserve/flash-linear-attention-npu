# chunk_delta_h_bwd_preprocess 测试

## 内容

| 文件 | 说明 |
| --- | --- |
| `reference.py` | CPU 全精度参考：`preprocess_reference` 给出 `dhm = [E_r \| P_r]`；`dh_scan_direct` 用非零 `dht` 反扫得到真实 `dh0`；`check_affine` 校验 `dh0 == P_r @ dht + E_r` |
| `../../op_cases/chunk_delta_h_bwd_preprocess.json` | 用例设计唯一来源（正向 + 反向拦截） |
| `harness/make_case.py` | 由用例参数生成 NPU 侧输入（落盘原始字节）与 CPU 标杆 `expected_dhm.bin`，并写出 `run_case` 命令行 |
| `harness/test_aclnn_chunk_delta_h_bwd_preprocess.cpp` | aclnn 直调取数：输入/输出全走文件，输出额外落 `workspace.bin`（用户区起点 16 MiB，可按 tiling 公式定位每个平面） |
| `harness/compare.py` | 按 `E_r`/`P_r` 分平面给出 `max_abs`/`max_rel`/`rel_norm`，并给出 PASS/FAIL |
| `harness/inspect_workspace.py` | 逐平面核对 workspace（NT=1）：slot（Q̄s/K̄/W/do/decayK）、T1、dVpre、dVhat、qterm、wterm、Pc、PBf、P 两个 parity |
| `harness/run_accuracy.sh` | 上述取数 → 比对的一键入口 |
| `harness/ctrl_case.py` | 受控实验：把某个 case 的 `W` 改成"仅第 0 行全 1"，据此可只手推出 `T1`/`P` 的期望结构，用于定位"左操作数转置语义/某个平面写错"这类问题 |
| `harness/make_negative_case.py` + `harness/run_negative.sh` | 反向用例：按 `op_cases` 的 `negative_cases` 生成"非法但类型正确"的输入，调用 aclnn 并核对返回码 |

## 本地自检（无需 NPU）

```bash
python tests/operators/chunk_delta_h_bwd_preprocess/reference.py
```

该自检对无门控 / `USE_G` / `USE_GK` 三种模式分别构造随机输入，用**非零** `dht` 验证仿射恒等式。
只测 `dht = 0` 无法验证 `P_r`，因此自检固定使用非零 `dht`。

## 设备侧验证（已执行，见下）

1. 按 `op_cases/chunk_delta_h_bwd_preprocess.json` 生成用例；
2. NPU 侧调用 `fla_npu.ops.ascendc.chunk_delta_h_bwd_preprocess`，CPU 侧调用 `reference.py` 的同名口径；
3. 逐用例比较 `dhm`，并单独抽出 `dV_pre`、`T1`、`dV̂'`、`inc`、`dH`、`P_c`、`P` 与参考实现比对，
   用于区分"公式错误"与"ready/free 缺边导致的旧值/新值混用"；
4. 覆盖 A2/A3/A5 三条平台与 `USE_G`/`USE_GK`/无门控、dense/varlen、尾块、GVA、`K=64/128/256`；
5. 反向用例只验证拦截：`g`/`gk` 同时非空、`K>256`、`Hv%Hk!=0`、`B>1`、`chunk_size!=64`。

### 一键执行（设备侧）

```bash
# 1) 编包并安装（A5 例：--soc=ascend950；A2 例：--soc=ascend910b）
bash build.sh --pkg --soc=<soc> --vendor_name=fla_npu --ops=chunk_delta_h_bwd_preprocess -j16
bash build_out/fla-npu-fla_npu_linux-*.run --install-path=<install_path>

# 2) 编取数程序并逐用例执行
cd tests/operators/chunk_delta_h_bwd_preprocess/harness
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

### 已验证结果

`op_cases` 的 16 条正向用例在 **A5（ascend950）与 A2（ascend910b）上均为 16/16 PASS**，`0` 编译错误。
下表数值来自**最终源码状态**下的完整重跑（两个平台的算子源码与本仓库工作区逐文件 md5 一致；
`build/autogen/inner/` 为空，确认 aclnn 走的是手写 exc 通路）。

| 用例 | A5 `E/P rel_norm` | A2 `E/P rel_norm` |
| --- | --- | --- |
| `pos_01_none_gate_dense`（T=512） | 5.56e-3 / 7.10e-3 | 5.52e-3 / 7.17e-3 |
| `pos_02_g_bf16_dense` | 5.06e-3 / 5.18e-3 | 5.19e-3 / 5.17e-3 |
| `pos_03_g_fp32_dense` | 5.08e-3 / 5.47e-3 | 5.07e-3 / 5.44e-3 |
| `pos_04_gk_bf16_dense` | 6.28e-3 / 5.50e-3 | 6.30e-3 / 6.57e-3 |
| `pos_05_gk_varlen_first_segment` | 3.79e-3 / 4.52e-3 | 3.72e-3 / 4.52e-3 |
| `pos_06_tail_chunk`（T=200，尾块 8） | 4.01e-3 / 4.33e-3 | 4.01e-3 / 4.27e-3 |
| `pos_07_k64_block32` | 5.67e-3 / 4.62e-3 | 5.67e-3 / 4.63e-3 |
| `pos_08_k256_four_groups` | 4.06e-3 / 3.87e-3 | 4.07e-3 / 3.86e-3 |
| `pos_09_v_tail_tile`（V=96） | 4.26e-3 / 4.57e-3 | 4.61e-3 / 4.57e-3 |
| `pos_10_gva_hv_gt_hk`（Hk=4,Hv=8） | 4.52e-3 / 3.59e-3 | 4.52e-3 / 3.59e-3 |
| `pos_11_single_chunk`（T=64） | 2.08e-3 / 4.29e-3 | 2.08e-3 / 4.29e-3 |
| `pos_12_two_chunk_chain_direction` | 2.77e-3 / 2.98e-3 | 2.77e-3 / 2.99e-3 |
| `pos_13_long_nt_chain_accumulation`（32 chunk） | 1.15e-2 / 9.30e-3 | 9.50e-3 / 6.94e-3 |
| `pos_14_fp16_inputs` | 6.51e-4 / 6.72e-4 | 7.12e-4 / 7.04e-4 |
| `pos_15_head_contiguous_partition`（96 head / 多 task） | 4.32e-3 / 4.36e-3 | 4.32e-3 / 4.36e-3 |
| `pos_16_tile_split_partition`（K=V=256，T=1024） | 1.24e-2 / 1.06e-2 | 1.45e-2 / 1.12e-2 |

`rel_norm = max_abs / max|参考|`。上表的量级与"链上状态用模型 dtype（bf16/fp16）传递"这一设计一致：单
task / 单 chunk 场景约 2e-3，长链与多 task 场景最多 1.5e-2。重复执行（`pos_15`/`pos_13`/`gate_gk` 各 3~30 次）
结果逐位一致。

### 反向拦截（已执行）

```bash
bash tests/operators/chunk_delta_h_bwd_preprocess/harness/run_negative.sh <work_dir>
```

`op_cases` 的 10 条反向用例在 **A2 与 A5 上均 10/10 PASS**（实际返回码与 `expected_return_code` 一致，
均为 `ACLNN_ERR_PARAM_INVALID` = 161001）：

| 用例 | 触发约束 |
| --- | --- |
| `neg_01_g_and_gk_both` | `g` 与 `gk` 同时非空（互斥） |
| `neg_02_k_too_large` | `K = 512 > 256` |
| `neg_03_hv_not_multiple_of_hk` | `Hk=3, Hv=4`（`Hv % Hk != 0`） |
| `neg_04_dense_b_greater_than_one` | dense 路径 `B = 2` |
| `neg_05_varlen_b_greater_than_one` | varlen 且 `B = 2` |
| `neg_06_chunk_size_not_64` | `chunk_size = 128` |
| `neg_07_g_shape_mismatch` | `g` 为 `[B,Hk,T]`（GVA：`Hk=2, Hv=4`） |
| `neg_08_gk_dtype_fp32` | `gk` 为 FP32 |
| `neg_09_cu_seqlens_too_short` | `cu_seqlens` 只有 1 项 |
| `neg_10_empty_tensor` | `T = 0` |

反向用例只校验拦截与返回码，不做精度比较；脚本会 `grep` `run_case` 打印的 `GetWorkspaceSize failed <code>`
并比对期望值，全部通过才输出 `ALL_PASS`。

## 精度判据

- `dhm` 为 FP32，但**链上状态按设计用模型 dtype 传递**（`Pc`/`PBf`/`dHBf` 为 bf16 或 fp16），因此绝对误差随
  序列长度放大（例如 32 chunk 用例 `max_abs` 可达 5.7e28）。判据采用**相对参考幅值**：
  `rel_norm = max_abs / max|参考|`，阈值 2%。
- 同时打印逐元素 `within2%`（相对误差小于 2% 的元素占比）作为辅助观察项；`max_rel` 会被参考中的极小值
  放大，不作为判据。
- `compare.py` 默认 `--tol 2e-2`，可用 `--tol` 收紧/放宽（阈值变更必须说明理由，不允许用阈值掩盖真实误差）。
- 不允许通过收窄输入 range、跳过失败用例或放宽阈值来制造通过结论。
