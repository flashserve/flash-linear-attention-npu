# ChunkKdaFwdIntraSubChunk：msprof / msprof op simulator 采集指导

> 仓库实测用法汇总（2026-07）。门禁口径以本文为准。  
> 相关：`TARGET_1P5_ANALYSIS.md` §5（历史 Dual/MCH 片段）、`VEC_2WIN_PIPE.md`、`VECTOR_OPTIMAL_PIPELINE.md`。  
> 官方入门：[msOpProf 快速入门](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/optool/docs/zh/quick_start/msopprof_quick_start.md)

---

## 0. 先选哪一种

| 目的 | 命令 | Shape | 看什么 |
|------|------|-------|--------|
| **性能门禁**（Δ Task Dur） | 裸 `msprof` | 模型 `B1 H32 T8192 K128 BT64` | `op_summary` 中本算子 **Task Duration 中位** |
| 管道占比 / Insight 图 | `msprof op` | 同上 | `PipeUtilization_*.csv`、`visualize_data.bin` |
| **流水重叠 / instr** | `msprof op simulator` | **T=1024（或 smoke T=64）**，勿上 8192 | `instr_exe.csv`、`trace.json`、核 duration 表 |

门禁规则（本算子约定）：

1. 先精度 suite 全绿。  
2. 空闲卡上采 **Task Duration median**。  
3. 相对上一默认配置 **Δ ≤ −0.05 ms** 才 `default on`。  
4. **不要**用脚本打印的 host `avg=…ms`（含 sync/调度，常 ~3.0 ms）去和 Task Dur（常 ~2.2 ms）比。

---

## 1. 环境

```bash
# 推荐（本机已验证）
source /data/wnc/cann/ascend-toolkit/set_env.sh
# 或 tracy cann：source /data/wnc/cann/cann-9.1.0-beta.1/... 视当前 msprof 路径而定
conda activate fzy_atk

cd /workspace/fzy/code/kda/flash-linear-attention-npu

# 选空闲卡（npu-smi info）；多卡机建议绑一张，减少 hang
export ASCEND_RT_VISIBLE_DEVICES=1   # 物理卡号，按实际改
export ASCEND_DEVICE_ID=0            # 对进程可见后常为 0；脚本里 set_device 读此变量
```

说明：

- `prof_*.py` 用 `ASCEND_DEVICE_ID` 调 `torch.npu.set_device`。  
- 若只设 `ASCEND_DEVICE_ID=3` 不绑 `VISIBLE`，偶发整卡 msprof hang；优先 **VISIBLE=物理卡 + DEVICE_ID=0**。  
- Kernel 过滤名（`msprof op` / simulator）：**`ChunkKdaFwdIntraSubChunk`**（GE Op 前缀）。  
  **不要**用 C 符号 `chunk_kda_fwd_intra_sub_chunk` → 会全部 skip、无 dump。

仿真额外：

```bash
export LD_LIBRARY_PATH=/data/wnc/cann/cann-9.1.0-beta.1/aarch64-linux/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH
# 若该路径不存在，试：
# /data/wyf/cann/cann-9.1.0/cann-9.1.0-beta.1/aarch64-linux/simulator/Ascend910B3/lib
```

---

## 2. 板端门禁：裸 `msprof`（推荐日常）

### 2.1 命令

```bash
OUT=prof_l1a_dbuf   # 每次换目录名，避免覆盖
mkdir -p "$OUT"

ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 \
msprof --aic-metrics=PipeUtilization --output="$OUT" -- \
  python torch_custom/fla_npu/test/prof_chunk_kda_fwd_intra_sub_chunk_model.py \
  2>&1 | tee "$OUT/msprof_run.log"
```

脚本：`torch_custom/fla_npu/test/prof_chunk_kda_fwd_intra_sub_chunk_model.py`  
- Shape：`(1,32,8192,128)` BT=64 bf16  
- 内置 warmup=3；`FLA_NPU_PROF_ITERS` 控制计时迭代（默认 5）

### 2.2 产物

```text
$OUT/
  msprof_run.log
  PROF_*/mindstudio_profiler_output/
    op_summary_*.csv          ← 主表
    op_statistic_*.csv
    task_time_*.csv
    ...
  PROF_*/device_<id>/         ← 原始采样
```

### 2.3 读 Task Duration 中位

筛 `Op Name` 含 `ChunkKdaFwdIntraSubChunk`、`Task Type=MIX_AIC`，对 `Task Duration(us)` 取中位再 `/1000` → ms。

```bash
python3 - <<'PY'
import csv, glob, statistics
paths = sorted(glob.glob("prof_l1a_dbuf/PROF_*/mindstudio_profiler_output/op_summary_*.csv"))
assert paths, "no op_summary"
rows = list(csv.DictReader(open(paths[-1])))
td = sorted(float(r["Task Duration(us)"]) for r in rows
            if "ChunkKdaFwdIntraSubChunk" in r["Op Name"] and r["Task Type"] == "MIX_AIC")
print(paths[-1])
print(f"n={len(td)} med={statistics.median(td)/1000:.3f} ms  "
      f"min={td[0]/1000:.3f} max={td[-1]/1000:.3f}")
# 旁证 AIV vs AIC
r0 = next(r for r in rows if "ChunkKdaFwdIntraSubChunk" in r["Op Name"] and r["Task Type"]=="MIX_AIC")
print("sample aiv_time(us)=", r0["aiv_time(us)"], "aicore_time(us)=", r0["aicore_time(us)"])
PY
```

同表还可看：`aiv_scalar_ratio`、`aic_mte2_ratio`、`aic_fixpipe_ratio`、`aiv_mte2/mte3_ratio`（粗分瓶颈；细流水用仿真）。

---

## 3. 板端细采：`msprof op`

适合要 **按 launch 的 PipeUtilization / Insight** 时：

```bash
OUT=prof_msprof_op_model
mkdir -p "$OUT"

ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 \
msprof op \
  --kernel-name=ChunkKdaFwdIntraSubChunk \
  --output="$OUT" \
  --aic-metrics=PipeUtilization \
  --launch-count=8 --warm-up=3 \
  python torch_custom/fla_npu/test/prof_chunk_kda_fwd_intra_sub_chunk_model.py \
  2>&1 | tee "$OUT/msprof_op_run.log"
```

产物示意：

```text
$OUT/OPPROF_*/ChunkKdaFwdIntraSubChunk_*_mix_aic/<i>/
  PipeUtilization_*.csv
  visualize_data.bin          ← MindStudio Insight Import
```

也可用精度脚本包一层（会更慢、算子次数更多）：

```bash
msprof op --kernel-name=ChunkKdaFwdIntraSubChunk --output=prof_msprof_op_pytest \
  --aic-metrics=PipeUtilization \
  python torch_custom/fla_npu/test/test_npu_chunk_kda_fwd_intra_sub_chunk.py
```

---

## 4. 仿真流水：`msprof op simulator`

### 4.1 为何不用模型 shape

全量 `H=32,T=8192` 在 simulator 下极慢；本仓库约定：

| 脚本 | Shape | 用途 |
|------|-------|------|
| `prof_chunk_kda_fwd_intra_sub_chunk_sim_smoke.py` | `H=2,T=64` | 通路 / 旗时序 smoke |
| `prof_chunk_kda_fwd_intra_sub_chunk_sim_t1024.py` | `H=2,T=1024`（可用 env 改） | **流水分析主入口** |

`FLA_SIM_{B,H,T,K,BT}`、`FLA_SIM_WARMUP` 可覆盖 t1024 脚本。

### 4.2 命令

```bash
OUT=prof_msprof_op_sim_t1024_l1a
mkdir -p "$OUT"

export LD_LIBRARY_PATH=/data/wnc/cann/cann-9.1.0-beta.1/aarch64-linux/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH

ASCEND_RT_VISIBLE_DEVICES=1 ASCEND_DEVICE_ID=0 \
msprof op simulator \
  --kernel-name=ChunkKdaFwdIntraSubChunk \
  --soc-version=Ascend910B3 \
  --aic-metrics=PipeUtilization \
  --output="$OUT" \
  --launch-count=1 \
  python torch_custom/fla_npu/test/prof_chunk_kda_fwd_intra_sub_chunk_sim_t1024.py \
  2>&1 | tee "${OUT}_run.log"
```

smoke 把脚本换成 `..._sim_smoke.py`，`--output=prof_msprof_op_sim` 即可。

### 4.3 产物怎么读

```text
$OUT/OPPROF_*/
  simulator/
    core*.cubecore0/{instr_exe.csv,trace.json}
    core*.veccore0|1/{instr_exe.csv,trace.json}
  ...
${OUT}_run.log     ← 末尾有 Core operator results（duration_time 表）
```

| 文件 | 用法 |
|------|------|
| `*_run.log` 核 duration 表 | 比 **vec med vs cube med**（AIV-bound 判据） |
| `instr_exe.csv` | 按 `instr`/`pipe` 聚合 cycles（BAR / MTE2 / MMAD / WAIT…） |
| `trace.json` | Chrome 时间线：看 `MOV_OUT_TO_UB` ‖ `MMAD`、WAIT 后 gap |

快速聚合示例：

```bash
python3 - <<'PY'
import csv, glob
from collections import defaultdict
from pathlib import Path
root = Path("prof_msprof_op_sim_t1024_l1a")
tot = defaultdict(float)
for p in root.glob("OPPROF_*/simulator/*.veccore*/*instr_exe.csv"):
    with p.open() as f:
        for row in csv.DictReader(f):
            name = row["instr"].upper()
            fam = "OTHER"
            if "BAR" in name: fam = "BAR"
            elif "WAIT_FLAG" in name: fam = "WAIT_FLAG"
            elif "MOV_OUT_TO_UB" in name: fam = "MTE2"
            elif "MOV_UB_TO_OUT" in name: fam = "MTE3"
            elif "MOV_UB_TO_UB" in name: fam = "UB2UB"
            tot[fam] += float(row["cycles"])
s = sum(tot.values()) or 1
for k, v in sorted(tot.items(), key=lambda x: -x[1])[:8]:
    print(f"{k:12s} {100*v/s:5.1f}%")
PY
```

**注意（Vec2Win 现行 flag，勿与旧 MCH 文档混用）：**

| flag | 名 | 方向 |
|------|----|------|
| 4 | `S0_READY` | AIV → AIC |
| 2 | `CUBE_DONE` | AIC → AIV |
| 6/8/10/12 | `SLOT_FREE*` | 仅 Process 书挡 |

旧 Dual/MCH 的 `SOLVE_READY/DONE`（id 8/6）已随 Vector FwdSub 删除；`TARGET_1P5` §5.4–5.5 仅作历史对照。

---

## 5. 常见坑

| 现象 | 处理 |
|------|------|
| msprof hang / 一直不结束 | 换空闲卡；`ASCEND_RT_VISIBLE_DEVICES`+`DEVICE_ID=0`；勿 `tee`  alone 掩盖；HardEvent 避开 0/1（曾致 hang） |
| OPPROF 空、kernel 全 skip | `--kernel-name` 必须是 `ChunkKdaFwdIntraSubChunk` |
| 仿真极慢 | 用 T=1024 / smoke，不要 H32×8192 |
| 「我们更快」但和 0723 对不上 | 对齐 **Task Duration med**，别比 host wall |
| 精度绿但门禁不过 | 记 Δ；代码可留、宏 default off |
| root 跑 msprof 警告 | 可忽略，注意环境安全 |

---

## 6. 推荐工作流（单刀改码）

```text
1) 改码（单变量宏）
2) 精度：test_npu_chunk_kda_fwd_intra_sub_chunk.py（+ 需要时 varlen/GVA）
3) 门禁：§2 裸 msprof → Task Dur med → Δ≤−0.05？
4) 若要解释重叠：§4 simulator T=1024 → instr/trace
5) 记入 PERF / SCORE_TILE / VEC_2WIN 文档；决定 default on/off
```

目录命名建议：`prof_<tag>/`（板端）、`prof_msprof_op_sim_t1024_<tag>/`（仿真），并保留 `*_run.log`。

---

## 7. 脚本与历史产物索引

| 脚本 | 路径 |
|------|------|
| 模型板端 | `torch_custom/fla_npu/test/prof_chunk_kda_fwd_intra_sub_chunk_model.py` |
| 仿真 T=1024 | `.../prof_chunk_kda_fwd_intra_sub_chunk_sim_t1024.py` |
| 仿真 smoke | `.../prof_chunk_kda_fwd_intra_sub_chunk_sim_smoke.py` |
| varlen/GVA prof | `.../prof_chunk_kda_fwd_intra_sub_chunk_varlen_gva.py` |

| 示例产物 | 含义 |
|----------|------|
| `prof_vec_2win/` | Vec2Win 板端基线 |
| `prof_l1a_dbuf/`、`prof_ours_p1_d3/` | P1 L1A 板端 |
| `prof_msprof_op_sim_t1024_vec2win/` | 仿真：串行 W |
| `prof_msprof_op_sim_t1024_l1a/` | 仿真：P1 on（W‖MMAD1） |
