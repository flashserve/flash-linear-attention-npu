# ChunkKdaBwdRecompute

融合 kernel：`kda_gate_chunk_cumsum`（safe-gate + chunk cumsum）+ `recompute_w_u_fwd`。
Mix **1 AIC : 2 AIV**，仅 **ascend950 / A5**。

公开入口：`fla_npu.ops.ascendc.chunk_kda_bwd_recompute`。
输出顺序：`(gk, w, u, qg, kg)`。`gk` 为 FP32，其余 BF16。

远程分支：`lyq/chunk_kda_bwd_recompute`（不要用旧的 `lyq/work`，那里的 kernel 是未优化版本）。

---

## 另一台 A5 机器：拉代码、编译、测性能

下面命令在仓库根目录执行。把路径换成你这台机器上的实际位置。

### 0. 拉代码

```bash
git clone https://github.com/LiYuanqg/flash-linear-attention-npu.git
cd flash-linear-attention-npu
git fetch origin lyq/chunk_kda_bwd_recompute
git checkout lyq/chunk_kda_bwd_recompute
```

已有仓库：

```bash
git fetch origin lyq/chunk_kda_bwd_recompute
git checkout lyq/chunk_kda_bwd_recompute
git pull --ff-only origin lyq/chunk_kda_bwd_recompute
```

### 1. 环境

需要：

- Python 环境里已装 `torch` + `torch_npu`（和本机开发环境同一套 conda 即可）
- **CANN 9.1**。不要 source CANN 9.2
- NPU 是 A5 / Ascend950。`npu-smi info` 能看到卡

```bash
# 按你的环境改这两行
CONDA_ENV=wnc
CANN_HOME=/usr/local/Ascend/cann-9.1.0   # 或 ~/Ascend/cann-9.1.0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PATH="$(dirname "$(which python)")":$PATH

# conda 可能会改掉 CANN / OPP，必须在 conda activate 之后再 source 9.1
source "${CANN_HOME}/set_env.sh"
unset CANN_ENV
export FLA_NPU_DISABLE_PTH=1
export TMPDIR="${TMPDIR:-/tmp}"

which python
python -c "import torch, torch_npu; print(torch.__version__, torch.npu.is_available())"
npu-smi info
```

选一张空闲 A5。`ASCEND_RT_VISIBLE_DEVICES` 设成物理卡号后，进程里的设备永远是 `npu:0`：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0    # 改成你要测的物理卡号
```

大 shape 性能和 hugepage 有关。测 8K/16K 前看一下该卡 hugepage，不要拿 hugepage 很少的卡报正式数字。

### 2. 只编这个算子并装进 overlay

不要把 `build_out/*.run` 或 `fla_npu/opp/vendors/.../op_impl` 提交进 git。每台机器自己编。

```bash
REPO="$(pwd)"
export FLA_NPU_DISABLE_PTH=1

bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu --ops=chunk_kda_bwd_recompute -j8
```

成功后 `build_out/` 下会有 `fla_npu_linux-x86_64.run`（名字以实际为准）：

```bash
RUN_PKG="$(ls -1 build_out/fla_npu_linux-x86_64.run | tail -1)"
INSTALL_ROOT="${REPO}/fla_npu_opp"
mkdir -p "$INSTALL_ROOT"
bash "$RUN_PKG" --cann --quiet --install-for-all --install-path="$INSTALL_ROOT"

# 同步到 Python overlay，ctypes 从这里加载 libcust_opapi.so
rsync -a "${INSTALL_ROOT}/" "${REPO}/torch_custom/fla_npu/fla_npu/opp/"
```

### 3. 运行前环境变量（每次新开 shell 都要设）

必须在 `import torch_npu` 之前把自定义 OPP 配好。

```bash
REPO="$(pwd)"
OVERLAY="${REPO}/torch_custom/fla_npu"
VENDOR="${OVERLAY}/fla_npu/opp/vendors/fla_npu_transformer"

export PYTHONPATH="${OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_CUSTOM_OPP_PATH="${VENDOR}"
export FLA_NPU_OP_API_LIB="${VENDOR}/op_api/lib/libcust_opapi.so"
export FLA_NPU_DISABLE_PTH=1
unset CANN_ENV
```

冒烟：

```bash
python - <<'PY'
import torch, torch_npu
from fla_npu.ops.ascendc import chunk_kda_bwd_recompute
print("ok", torch.npu.get_device_name(0), chunk_kda_bwd_recompute)
PY
```

如果报找不到 `aclnnChunkKdaBwdRecompute` 或仍走到旧 OPP：确认 `PYTHONPATH` 指向仓内 `torch_custom/fla_npu`（在 site-packages 之前），并且 `libcust_opapi.so` 存在。

### 4. 精度冒烟（先做，再报性能）

脚本自带 CPU 标杆，不依赖仓外 golden 文件。

```bash
# 小 shape；ping-pong GM 需要每核 ≥3 tile，所以再跑一条 T=2048
python tests/atk/chunk_kda_bwd_recompute/check_npu_accuracy.py
python tests/atk/chunk_kda_bwd_recompute/check_npu_accuracy.py --hk 8 --hv 8 --tokens 2048
```

通过标准（max abs diff）：`gk/qg/kg < 0.05`，`w/u < 0.15`。`u` 因 BF16 GEMM 大约 0.12 是正常的。两次 launch 结果要一致，输出有限、`kg` 不能 Inf。

### 5. 性能（模型 shape）

对照 GPU 是两条 Triton 算子之和：`kda_gate_chunk_cumsum` + `recompute_w_u_fwd`。
NPU 是一条融合 kernel。用 Event 计时，不要用 ATK `performance_device` 报 8K/16K（ATK 每轮会 D2H）。

```bash
# 8K：B=1, H=96, T=8192, K=V=128, chunk=64, bf16
KDA_BWD_T=8192 KDA_BWD_H=96 \
  python tests/atk/chunk_kda_bwd_recompute/bench_npu_chunk_kda_bwd_recompute.py

# 16K
KDA_BWD_T=16384 KDA_BWD_H=96 \
  python tests/atk/chunk_kda_bwd_recompute/bench_npu_chunk_kda_bwd_recompute.py
```

看 JSON 里的 `median_us`。同时记下 `device` 和 `visible_devices`。

本机 A5（hugepage 正常的卡）参考，**另一台机器数字会漂，用来对齐量级**：

| shape | H20 两算子合计 | NPU 融合 median | NPU / H20 | 1.2× 目标 |
| --- | ---: | ---: | ---: | ---: |
| `B=1 H=96 T=8192 K=V=128 chunk=64` | **1069 µs** | **~1631 µs** | ~1.53× | &lt;1283 µs |
| 同上 `T=16384` | **2167 µs** | **~3134 µs** | ~1.45× | &lt;2600 µs |

请回报：`median_us`、`device`、CANN 版本、`npu-smi` 里该卡 hugepage。

### 6. 常见失败

| 现象 | 处理 |
| --- | --- |
| conda 之后编出来的 kernel 不对 / OPP 串包 | `conda activate` 之后重新 `source` CANN 9.1，`unset CANN_ENV`，`FLA_NPU_DISABLE_PTH=1`，`ASCEND_CUSTOM_OPP_PATH` 只留本仓 vendor |
| `GetCoreNumAic()==1` 导致 8K 变成几十毫秒 | tiling 已写死 fallback 28 AIC；确认跑的是本分支编出来的 OPP，不是旧 `lyq/work` |
| Mix 1:2 卡死 | 不要在 Mix 里用 `PipeBarrier<PIPE_ALL>()` |
| 精度脚本 `u` 到 0.12 | BF16 GEMM，限幅 0.15，可接受 |
| 16K 比 8K 慢很多且核利用率很低 | 换 hugepage 更充足的卡再测 |

---

## 输入约束

- 布局固定 dense BNSD：`q/k=[B,HK,T,K]`，`v=[B,HV,T,V]`，`g=[B,HV,T,K]`，`beta=[B,HV,T]`，`A=[B,HV,T,chunk_size]`。
- `A_log` 为 `[HV]` 的 `FLOAT`；`dt_bias` 可选，为 `[HV,K]` 的 `FLOAT`。
- `q/k/v/A` 仅 `BFLOAT16`；`g/beta` 支持 `BFLOAT16/FLOAT`。
- `K=V=128`，`chunk_size=64`；`HV % HK == 0`。
- `use_gate_in_kernel=true` 时必须提供 `A_log`；safe-gate 下 `lower_bound` 默认 `-5.0`。
- kernel 仅注册 `ascend950`。当前 ATK 用例为定长 dense，不含 `cu_seqlens/chunk_indices`。

## 标杆

CPU 标杆在 `executor_chunk_kda_bwd_recompute.py` 的 `_fused_ref`：safe-gate + chunk cumsum × RCP_LN2 + GQA 展开 + `A @ kbg/vb`。
同精度路径按 NPU cube 把 `A/kbg/vb` 量化到 bf16 再乘。

精度冒烟：`check_npu_accuracy.py`。ATK executor 与之同一套公式。

## ATK

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_recompute -npu_device_id=0 -scope=accuracy -soc=ascend950
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_recompute -npu_device_id=0 -scope=performance -soc=ascend950
```

- 精度 JSON：`atk_chunk_kda_bwd_recompute.json`
- 性能 JSON：`atk_chunk_kda_bwd_recompute_perf.json`（`B=1 HK=HV=4 T=512`，比模型 8K 小，只做 ATK 看护）
- 8K/16K 正式性能用第 5 节的 Event 脚本，不要用这条 ATK case 代替
