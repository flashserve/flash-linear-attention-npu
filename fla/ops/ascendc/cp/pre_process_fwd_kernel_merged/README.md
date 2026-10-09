# pre_process_fwd_kernel_merged（AscendC 算子工程，WIP）

CP（context parallel）前处理算子：把一个 token 窗口压成仿射链 `(h | m)`，与竞品
`fla/ops/cp/chunk_delta_h.py::pre_process_fwd_kernel_merged` 数值对齐（对标 1.0x H20）。

完整语义、Stage 划分、内存分配与验收工程见 `docs/`（`api.md` / `design.md`）与
`tests/atk/pre_process_fwd_kernel_merged/`。

## 当前进度（2026-09-30）

| 部分 | 状态 |
| --- | --- |
| 01 接口 / 02 标杆 / 03 设计 | ✅ 冻结（`docs/api.md`、`docs/design.md`；标杆 `tests/atk/pre_process_fwd_kernel_merged/scripts/pre_process_fwd_kernel_merged_cpu.py` 已与 H20 `ieee` 对齐） |
| op_host | ✅ 已落地（`*_def.cpp` / `*_tiling.{h,cpp}` / `op_api/*`，含 aclnn 两段式） |
| op_kernel | ✅ 已落地（Cube + Vector 双核流水；Stage/同步协议见 `op_kernel/*.cpp` 顶部与 `docs/design.md`） |
| Python 接入 | ✅ ctypes（`_aclnn_ctypes.py`）+ stable ABI（`torch_custom/fla_npu/csrc/src/stable_pre_process_fwd_kernel_merged.cpp`）；`_stable.py` / `__init__.py` 已注册，11 条离线单测通过 |
| 单算子验证（A5 / A2-A3） | ✅ A5：L0 静态门禁、L1 位级 `BIT_IDENTICAL`、L2 smoke 10/10、L4 全量 41/41；A2/A3：单算子编译 + L0 + L2 10/10 + 位级复跑一致，性能无回退 |
| ATK 交付件 | ✅ 已就绪（`tests/atk/pre_process_fwd_kernel_merged/`：111 条精度、4 条性能、5 条内存/确定性；支持 CPU 双标杆与 GPU 双标杆），待上机执行并回填 README 的验收表 |

**范围**：GDN（`g`）与 KDA（`gk`）两条路径。**DPLR（`gk` + `bg`）不支持**：`bg` / `v`
必须传空，非空在 host tiling / aclnn / ctypes / stable 四个入口一致地被拒
（`docs/api.md` §6，`docs/design.md` 开头的范围变更）。

## 构建与验证

```bash
bash build.sh --opkernel --soc=ascend910b --ops=pre_process_fwd_kernel_merged   # 只编 kernel
bash build.sh --ophost   --ops=pre_process_fwd_kernel_merged                    # 只编 host 侧
```

精度与性能验收走 ATK 工程 `tests/atk/pre_process_fwd_kernel_merged/`（该目录 README 给出
用例规模、TilingKey 覆盖表与执行命令）。单算子自测入口是
`torch_custom/fla_npu/test/test_pre_process_fwd_kernel_merged.py`（离线，不需要 NPU）。

**只编译本算子（快）**：`build.sh` 支持 `--ops=` 白名单，例如

```bash
bash build.sh --opkernel --soc=ascend910b --ops=pre_process_fwd_kernel_merged   # 只编 kernel
bash build.sh --ophost   --ops=pre_process_fwd_kernel_merged                    # 只编 host 侧
```

容器里的 CANN 在 `/usr/local/Ascend/cann-9.1.0`（`ASCEND_HOME_PATH` 已设），可以离线编译。

## 关键契约（细节见 `docs/api.md`）

* `B ≡ 1`（varlen 打包）；`cu_seqlens` 必给且**允许子区间**（`bos > 0` / `eos < T`），与竞品调用形态一致；
* `K = V = 128`、`chunk_size = 64`、布局 BNSD `[1, H, T, D]`；
* `hm [Nseq, HV, K, V+K]` FP32，左 `[0,V)` 是 `h`（K×V）、右 `[V,V+K)` 是 `m`（K×K）；
* 三个舍入点：`h` 进 Cube 前降 BF16、`v_new` 进 Cube 前降 BF16、`m` 链 FP32（S4 用 FP32 原生 MMAD）。
