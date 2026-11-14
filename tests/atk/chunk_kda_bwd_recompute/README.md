# ChunkKdaBwdRecompute

融合 kernel：`kda_gate_chunk_cumsum`（safe-gate + chunk cumsum）+ `recompute_w_u_fwd`。
已有 A5 路径使用 Mix **1 AIC : 2 AIV**；新增 A6 源码路径使用 **1 AIC : 1 AIV**。
A6 已完成目标 SDK 编译、安装和产物一致性检查；修正后的最小 camodel 用例五个输出比较通过，
打印 PASS 后进程段错误退出（139），尚未完成正式验收。
下面原有 A5 记录不代表 A6 验证结果。

公开入口：`fla_npu.ops.ascendc.npu_chunk_kda_bwd_recompute`。
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
from fla_npu.ops.ascendc import npu_chunk_kda_bwd_recompute as chunk_kda_bwd_recompute
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
- kernel 注册 A2/A3/A5/A6，A6 走 arch92。原有 JSON 为定长 dense；A6 候选包包含变长元数据。

## A6 开发期候选用例

首次 camodel 冒烟使用 `B=HK=HV=1、T=64、seed=20260817`，回传结果为
`w max_diff=0.404297`、`qg max_diff=0.167969`、`kg max_diff=2.3125`，
gk/u 通过原脚本判据，全部输出两次调用的 `launch_diff=0`。
arch92 的 half Exp 结果转回 FP32 时误用 FP32 mask，会屏蔽高 64 通道；
已改用 half 源类型 mask，并按 Prepare 为零核数查询增加回退。相同最小用例的后续 camodel
复测五个输出均通过：gk/w/u/qg/kg 的 max_diff 分别为
`0.000076/0.001953/0.125000/0.000977/0.000977`，launch_diff 均为 0，阈值未改变。
打印 PASS 后发生段错误（退出码 139）；仅记录输出比较通过，不代表正常进程退出或正式验收通过。
当前候选参考 Merge 的 CompileInfo 传递：A6 TilingParse 默认 64/64，并用实际非零查询值覆盖；
运行期查询为零时读取 CompileInfo，对应字段不可用时用 64，再按两侧较小值组成 1:1 核对。
默认值依据目标 Ascend960DT_968B7 的平台配置；它不代表接口原始值已经恢复。
本轮已替换固定 SoC 补查，尚待远端重编译和复测。

复测相同最小用例：

```bash
python tests/atk/chunk_kda_bwd_recompute/check_npu_accuracy.py --batch 1 --hk 1 --hv 1 --tokens 64 --seed 20260817
```

Host Tiling 的临时核数打印和环境开关已删除，核数查询、CompileInfo 回退与配对规则保留。
`check_npu_accuracy.py` 保留设备、shape、五个输出的误差、两次调用差异和 PASS/FAIL，
移除调试用的 `--dump-dir` 数据导出。比较时先将输出回读 CPU 再转 FP32，避免额外的 NPU Cast。
此脚本沿用原有绝对误差与重复执行判据，不替代 ATK 混合容差验收。

### 独立单次调用冒烟

`scripts/recompute_a6_smoke.py` 按 Prepare 冒烟日志的 `[CPU] / [NPU] / [CMP]` 格式运行，
默认 `B=HK=HV=1、T=64、seed=20260817`。脚本自带输入构造和 CPU 公式，先一次性搬入八个输入，
仅调用一次公开 Recompute 接口并同步，随后在 CPU 上比较五个输出的 max_abs、RMSE、相对 L2。
输入值域及每个输出的绝对误差阈值沿用 `check_npu_accuracy.py`，不检查重复调用确定性。
根据用户提供的 Prepare 原脚本，对齐运行设置：导入 torch/torch_npu 前将
`TBE_PARALLEL_COMPILE_ENABLE/PARALLEL_COMPILE` 设为 0，CPU 单线程，
初始化 NPU 时设置 `allow_internal_format=False`、`jit_compile=False`，再设 device 0 并用 `.npu()` 搬输入。
Prepare 的 `tests/atk/.../scripts/cpu_golden.py` 是纯 CPU 标杆模块，引用其目录不表示启动 ATK；
本脚本的 CPU 公式内置于文件中。

```bash
python -u tests/atk/chunk_kda_bwd_recompute/scripts/recompute_a6_smoke.py --output <output.pt>
```

可用 `--batch/--hk/--hv/--tokens/--seed` 指定相同条件。输出文件先保存 `case/inputs/reference`，
完成 NPU 回读后追加 `npu`，便于 Host 失败时仍保留原输入和参考结果。
脚本打印实际 Python、包位置、环境路径和已加载的相关库。
需在已配置的运行环境中使用，且当前安装的 run 包应包含 Recompute。
精度比较失败返回 1，Host/运行时异常继续抛出；该脚本不替代 ATK 或正常进程退出验证。

`scripts/run_a6.py` 复用现有 JSON 的值域、`cv_fused_double_benchmark` 双标杆规则和 executor，
生成 200 条 A6 候选用例，由 ATK 发起实际调用。保留下面 28 类场景并按顺序循环，
每类执行 7 或 8 个不同 seed，case id 为 0–199，seed 为 20260817–20261016；
200 条表示不同的 `(场景, seed)` 组合。CPU executor 保留同精度参考和 FP64 高精度标杆路径，
比较阈值不变。脚本继承调用者的 Python/CANN/OPP 环境，不修改 PYTHONPATH。

```bash
python tests/atk/chunk_kda_bwd_recompute/scripts/run_a6.py --emit-only
python tests/atk/chunk_kda_bwd_recompute/scripts/run_a6.py --device 0
python tests/atk/chunk_kda_bwd_recompute/scripts/run_a6.py --device 0 --task determinism
```

| 分支 | 候选用例 |
| --- | --- |
| g/beta dtype key | BF16/BF16、BF16/FP32、FP32/BF16、FP32/FP32；各覆盖 T=64/65 |
| 尾块 | T=1/15/16/17/63/127/129 |
| head 覆盖 | HV=1/2/3/4/5，CG=1 |
| batch/GQA | B=2、HK=2、HV=4、T=65 |
| slot 回绕 | HV=3、T=8193；需设备执行记录确认每核 ordinal 和实际 blockDim |
| varlen | `[1,0,64,65]` 与 `[63,64]`，覆盖空 sequence 和不同 tail |
| 运行期模式 | 无 bias、自然指数、关闭 gate 的 exp2/exp、lower_bound=-1 |

Python executor 修正为当前公开 `npu_` 入口。CPU 标杆的定长数学保持原公式；新增变长用例
按各 sequence 切 chunk，关闭 gate 时与公开返回约定一致地省略 gk。

候选包已在本地检查 JSON、CPU shape、有限值和序列边界；未执行 ATK、值域新增场景校准、
确定性或 mssanitizer。四种 dtype key 的实际设备选择、1:1 bin json、同核 slot 回绕记录、
pipeline 重叠和 A2/A3/A5 回归均待有环境后执行。候选用例不构成正式验收包或 PASS 结论。

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
