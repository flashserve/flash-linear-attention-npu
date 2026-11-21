# Issue #927：编译与增量安装验收报告

关联需求：[issue #927](https://github.com/flashserve/flash-linear-attention-npu/issues/927)。
核对日期：2026-11-16。当前主干实测版本为
[`134524a6`](https://github.com/flashserve/flash-linear-attention-npu/commit/134524a6aae55bc94101575146e88511856716e5)。

## 验收范围与结论

本报告覆盖完整 base wheel、单次 run/wheel 增量安装、适配层与 OPP 分步配套、
未覆盖算子隔离、状态查询及整体移除恢复。按本次确认的范围记录结果，
不据此宣称 issue 全部需求已闭环。

- 当前主干 A2/A5 各 157 项安装相关检查通过，无跳过；7 个 CI 调度场景通过。
- 当前主干 A5 单算子 run 实际构建成功；A2 本轮构建未通过，错误和边界见下文。
- A2/A5 已完成的数值验证继续作为证据：包括真实接口变化、L1–L4 生效、
  未覆盖算子不变，以及 base/overlay/remove 恢复。各份证据保留原测试版本。
- [PR1183 正式 CI](https://github.com/flashserve/flash-linear-attention-npu/actions/runs/43196008121)
  的六项 NPU 状态全部成功，含 A2/A5 全量 OPP、wheel 安装、GDR Example/ST。
  该结果属于 PR head `da7a92ed`，不标作当前 `134524a6` 的完整重跑。

## 验证准则与结果

| 验证项 | 通过准则 | 结果及对应命令 |
| --- | --- | --- |
| 完整 base wheel | OPP、Stable launcher 和 Python 入口配套；安装后可调用 | 已有 PR1183 双平台 CI 证据；本轮未重编全部算子。见命令 1。 |
| 单算子增量构建 | 选定算子及必要依赖生成 run；检查包内范围 | A5 通过，仅含 `chunk_local_cumsum` kernel 目录；A2 本轮未通过。见命令 2。 |
| 三类增量产物 | run 更新 OPP；standalone wheel 更新适配层；携带 OPP 的 wheel 更新两者 | 已有 A2/A5 组件安装和真实调用证据；本轮安装机制检查通过。见命令 2–4。 |
| 适配层与 OPP 分步配套 | 两种安装顺序配齐后，匹配的新接口可调用 | 已有 A2/A5 真实 ACLNN C ABI 变化及数值验收；不以 Python 参数变化代替 C ABI 验证。见命令 3–4。 |
| 未覆盖算子隔离 | base 含 A、B，增量仅更新 A；A 命中新实现，B 输出不变 | 已有 A2/A5 数值证据：A=`chunk_local_cumsum`，输出总和由 8320 变为 16640；B=`chunk_fwd_o` 输出不变，base 文件哈希不变。逐元素判据及命令见 4.1–4.3。 |
| 安装安全与恢复 | 重复安装可用，失败不破坏 base；移除后恢复原输出和文件 | 本轮机制检查及已有 A2/A5 三态数值证据通过；移除后 A 总和恢复 8320、B 不变。移除及结果检查命令见 4.3。 |
| 状态可观测 | 可查询当前组件、OPP 路径、启用状态和不匹配原因 | 当前安装检查覆盖；run/overlay wheel 成功后打印类型和 OPP 路径。见命令 4。 |
| 加载路径一致性 | `.pth` 与支持的提前 source 路径选择一致；新进程调用结果符合预期 | 已有 A2/A5 来源与数值证据，本轮另检查路径选择逻辑。见命令 5。 |
| 精度与端到端 | 按实际影响范围验证数值；保留原用例和阈值 | 已有 A2/A5 数值归档及 PR1183 GDR Example/ST CI；本轮没有重复全量数值测试。见命令 6。 |

## 编译与功能命令

先激活对应 Python 环境并初始化 CANN。当前实测 A2 使用 CANN 9.2、
torch/torch_npu 2.10 环境；A5 使用 CANN 9.2、`zzy` Python 环境。
安装和数值复现须使用与目标平台匹配的 torch/torch_npu、wheel 和 OPP。
Host 动态库和 wheel 在与目标 CPU 架构一致的主机上原生构建，
再通过 `--soc` 指定 Ascend 芯片型号。

```bash
# 在已初始化 CANN 的源码根目录执行；A5 将 SOC 改为 ascend950。
export SOC=ascend910b
export OPS_CPU_NUMBER=4 MAX_JOBS=4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OUT="$PWD/issue927-output"
mkdir -p "$OUT"
```

下列命令说明当前支持的入口；只有上表和实测记录明确通过的项目才计为已执行。
不同产物使用各自干净的源码工作区，避免旧 OPP 混入。

### 1. 完整 base wheel

功能：构建全部 OPP 和 Stable 适配层，供常规安装使用。

```bash
env -u FLA_NPU_OPS -u FLA_NPU_OVERLAY_OPS \
  FLA_NPU_SOC="$SOC" FLA_NPU_BUILD_ARGS='--ops=all -j4' \
  python scripts/build_wheel.py --wheel-dir "$OUT/base"
python -m pip install --force-reinstall --no-deps /absolute/path/to/base.whl
```

`base.whl` 须替换成构建输出的实际文件名。本轮未执行全量构建命令。

### 2. 单算子 OPP run

功能：只构建选定算子及必要依赖的 OPP，不更新 Python/Stable 适配层。

```bash
bash build.sh --pkg --soc="$SOC" --vendor_name=fla_npu \
  --ops=chunk_local_cumsum -j4
```

产物为 `build_out/fla_npu_linux-aarch64.run` 或
`build_out/fla_npu_linux-x86_64.run`，以实际平台和打包日志为准。
当前主干 A2/A5 本轮实际执行的就是这条命令。

### 3. 增量 wheel

只更新指定 Python/Stable 入口、保留当前 OPP：

```bash
# 干净工作区，torch_custom/fla_npu/fla_npu/opp 不得残留已有 OPP。
(
  cd torch_custom/fla_npu
  FLA_NPU_SOC="$SOC" FLA_NPU_OVERLAY_OPS=chunk_local_cumsum \
    python setup.py bdist_wheel --dist-dir "$OUT/adapter-only"
)
```

同时携带指定适配层和 OPP：

```bash
FLA_NPU_SOC="$SOC" FLA_NPU_OPS=chunk_local_cumsum \
FLA_NPU_OVERLAY_OPS=chunk_local_cumsum FLA_NPU_BUILD_ARGS='-j4' \
  python scripts/build_wheel.py --wheel-dir "$OUT/with-opp"
```

`FLA_NPU_OPS` 选择编译的 OPP，`FLA_NPU_OVERLAY_OPS` 声明覆盖的公共入口。

### 4. 安装、查询和恢复

先安装 base wheel，再按产物选择以下入口。安装、禁用、移除后均须重启业务进程，
让 OPP 路径在 CANN 初始化前就位。

```bash
# run 默认安装，无需 --overlay。
bash /absolute/path/to/update.run
# 对应的显式管理入口。
python -m fla_npu_opp_manage --overlay --source /absolute/path/to/update.run

# standalone wheel 或携带 OPP 的 wheel 使用相同管理入口。
python -m fla_npu_opp_manage --overlay --source /absolute/path/to/update.whl

python -m fla_npu_opp_manage --status
python -m fla_npu_opp_manage --remove-overlay
```

安装成功不等于新组合已经启用；应核对状态和诊断。不匹配组件可先保存，配齐后再验证调用。
完整使用说明见 [Wheel 增量安装指南](Wheel增量安装指南.md)。

#### 4.1. 准备能从结果区分的新版本

仅查询 `--status` 或看到 overlay 路径，不能证明算子执行了新实现。
以下使用已有验收的数值见证：base 的 A=`chunk_local_cumsum` 正常累加，
测试版 A 在实际 OPP 内将输出乘 2；未覆盖 B=`chunk_fwd_o` 保持不变。
调用者在三个阶段始终传相同输入、相同 `scale=1.0`，不能在调用端手工乘 2。
“移除后不生效”指新版本不再生效，base 算子仍应正常运行。

先安装未修改的完整 base wheel，再在独立测试源码中生成 ×2 的 run。
该改动只用于识别测试版本，不进入正式算子实现或发布包：

```bash
set -e
git worktree add --detach "$OUT/overlay-witness-src" 134524a6aae55bc94101575146e88511856716e5
(
  cd "$OUT/overlay-witness-src"
  python - <<'PY'
from pathlib import Path
p = Path("fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/op_host/op_api/chunk_local_cumsum.cpp")
text = p.read_text()
old = "float scaleFloat = static_cast<float>(scale);"
assert text.count(old) == 1, "Unexpected source; stop before changing it"
p.write_text(text.replace(old, "float scaleFloat = static_cast<float>(scale) * 2.0f;"))
PY
  bash build.sh --pkg --soc="$SOC" --vendor_name=fla_npu --ops=chunk_local_cumsum -j4
)
export UPDATE_RUN="$OUT/overlay-witness-src/build_out/fla_npu_linux-$(uname -m).run"
```

#### 4.2. 同一输入的算子结果检查命令

在当前 Bash 会话中定义以下函数；每次调用都会启动新的 Python 进程。
使用一个新证据目录，先运行 `base` 保存 B 的参考结果，再执行安装和移除。
测试 A 使用 FP32 全 1 输入 `[1,2,128]`，每个 chunk 长度为 64。
原始逐元素期望为重复的 `1…64`，新版为 `2,4,…128`，均要求零误差。
总和仅用于展示，实际断言比较每个元素。

```bash
export VERIFY_OUT="$(mktemp -d "$OUT/numeric-check-XXXXXX")"
verify_result() {
  env -u FLA_NPU_DISABLE_PTH -u FLA_NPU_OVERLAY_VENDOR \
    python - "$1" "$VERIFY_OUT" <<'PY'
import hashlib, json, math, os, sys
from pathlib import Path
import fla_npu_opp_env  # 在 torch_npu/CANN 初始化前配置路径。
import torch, torch_npu
from fla_npu.ops import ascendc as ops
import fla_npu_wheel_overlay as layers

phase, directory = sys.argv[1:]
assert phase in ("base", "overlay", "removed")
directory = Path(directory)
torch.set_num_threads(1)
torch.manual_seed(20260717)
plan = layers.runtime_plan(layers.package_dir())
states = layers.status(layers.package_dir())
if phase == "overlay":
    assert plan is not None and not plan["errors"], states
    assert not any(item["state"] == "pending" for item in states), states
else:
    assert plan is None and not states, states

# B 不在增量中：固定输入，检查其输出与安装前逐位一致。
b, h, t, k, v, cs = 2, 4, 129, 128, 128, 64
q = torch.randn(b, h, t, k, dtype=torch.bfloat16) * .1
key = torch.randn_like(q) * .1
value = torch.randn(b, h, t, v, dtype=torch.bfloat16) * .1
state = torch.randn(b, math.ceil(t / cs), h, k, v, dtype=torch.bfloat16) * .1
g = torch.zeros(b, h, t, dtype=torch.float32)
other = ops.npu_chunk_fwd_o(q.npu(), key.npu(), value.npu(), state.npu(),
    1 / math.sqrt(k), g=g.npu(), chunk_size=cs, transpose_state_layout=False).cpu()
b_hash = hashlib.sha256(other.float().numpy().tobytes()).hexdigest()

# A 的调用参数不变，新版的 ×2 必须由实际安装的实现产生。
x = torch.ones(1, 2, 128, dtype=torch.float32).npu()
actual = ops.npu_chunk_local_cumsum(x, chunk_size=64, scale=1.0).cpu()
expected = torch.arange(1, 65, dtype=torch.float32).repeat(1, 2, 2)
if phase == "overlay":
    expected = expected * 2
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
if phase != "base":
    saved = json.loads((directory / "base.json").read_text())
    assert b_hash == saved["B_sha256"], "Uncovered operator B changed"
result = {"phase": phase, "A_sum": float(actual.sum()), "B_sha256": b_hash,
          "max_abs": float((actual - expected).abs().max()),
          "OPP_paths": os.environ.get("ASCEND_CUSTOM_OPP_PATH"), "plan": plan, "passed": True}
(directory / (phase + ".json")).write_text(json.dumps(result, indent=2))
print(f"PASS {phase}: A_sum={result['A_sum']}, max_abs={result['max_abs']}, B_sha256={b_hash}")
PY
}
```

#### 4.3. 安装生效与移除失效的完整顺序

在仅安装 base 的测试环境中执行；失败时先检查日志，不继续把后续步骤计为通过。
每次 `verify_result` 都实际调用 A、B，不以安装命令返回成功代替结果验证。

```bash
set -e
verify_result base                         # A=8320；保存 B 的参考结果。
bash "$UPDATE_RUN"                        # 安装 4.1 构建的 ×2 新版本。
python -m fla_npu_opp_manage --status
verify_result overlay                      # A=16640；B 必须与 base 一致。

python -m fla_npu_opp_manage --remove-overlay
python -m fla_npu_opp_manage --status
verify_result removed                      # A=8320；B 不变；无 active/pending。
```

若验收的是 wheel overlay，将安装行替换为
`python -m fla_npu_opp_manage --overlay --source /absolute/path/to/update.whl`，
移除入口不变。用于本例的 wheel 也必须只更新 A、实现 ×2，才可使用同一检查函数：
standalone 在适配层实现 ×2；携带 OPP 的 wheel 可使用 4.1 的 ×2 OPP 和未改倍率的适配层。
不要将两处倍率同时改成 2。普通语义不变的优化包不会产生 ×2，不能拿它代替这个见证包。

| 阶段及命令 | A 的逐元素期望 / 总和 | B 的期望 | 已有 A2/A5 run 验证 |
| --- | --- | --- | --- |
| `verify_result base` | 每 chunk 为 `1…64`；8320 | 保存参考摘要 | 均通过 |
| 安装后 `verify_result overlay` | 每 chunk 为 `2,4,…128`；16640 | 与参考摘要相同 | 均通过 |
| `--remove-overlay` 后 `verify_result removed` | 恢复 `1…64`；8320 | 与参考摘要相同 | 均通过 |

上述实测结果来自 `default-run-install-20261112/full-alias/` 的 A2/A5
`base.json`、`default.json`、`restored-default.json`；各阶段 `max_abs=0`。
本节将既有验证逻辑整理为可复制命令，本次文档更新未重新构建见证包或重跑 NPU。

### 5. 安装机制和文件隔离检查

当前主干两台实际执行：

```bash
python -m unittest tests.test_wheel_overlay tests.test_wheel_environment \
  tests.test_opp_overlay tests.test_opp_abi_contract \
  ci.tests.test_install_workflows_wheel_layout -v
bash -n cmake/scripts/custom/install.sh
bash -n ci/run_checks.sh
bash ci/tests/test_run_checks_stage_pipeline.sh
```

每台 157 项通过、无跳过，CI 调度的 7 个场景通过。这些检查包含真实文件操作和子进程，
但使用测试产物；调度检查中的编译和 Example/ST 使用替代程序，不能计为 NPU 精度通过。

### 6. 数值和 Example/ST

已有数值验收覆盖：A 的修改结果生效、未覆盖 B 与 base 一致、移除恢复，
以及适配层/OPP 两种配套顺序、真实 C ABI、L1–L4 来源变化。
本轮没有重复执行这些数值实验，也没有修改用例或阈值。

后续按变更影响范围复跑时，可使用仓库已有入口，并在 base/overlay/remove 三态
保持输入、shape、dtype 和比较规则一致：

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_local_cumsum -soc="$SOC" -scope=all -npu_device_id=0
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -soc="$SOC" -scope=all -npu_device_id=0
python examples/flash_gated_delta_rule.py
```

## 本轮实测详情与未完成项

| 项目 | A2 | A5 |
| --- | --- | --- |
| 主机架构 / SoC | aarch64 / ascend910b | x86_64 / ascend950 |
| 权限与并发 | 非 root、4 核 CPU affinity、`-j4` | 非 root、4 核 CPU affinity、`-j4` |
| 安装相关检查 | 157/157，无跳过 | 157/157，无跳过 |
| 单算子 run 构建 | 退出码 2，未生成成功产物 | 退出码 0，165.22 秒 |
| 产物检查 | 未完成 | 3,017,687 bytes；1 个 kernel 目录，9 个 `.o` |

A2 首个明确错误发生在 CANN kernel PRECOMPILE：

```text
asc/impl/utils/sys_macros_impl.h:18:10: fatal error: 'cstdint' file not found
```

主机已存在 GCC/G++ 11.4.0 和对应标准头文件，因此应继续检查编译器搜索路径。
本轮未修正环境后重跑，不把该项标成成功，也不认定为 PR1183 或算子实现引入的问题。

A5 的 165.22 秒是依赖下载完成后一次成功命令的耗时，不含此前下载超时。
没有同机器、同并发、同缓存条件的全量构建对照，不能用它与 issue 中历史 884 秒直接计算加速比。

## 证据索引

当前主干的原始命令、退出码、日志、源码校验和 A5 run 保存在本地归档
`artifacts/issue927-main-20261116/`，大体积日志和产物不纳入仓库。
A5 成功 run 的 SHA256：

```text
406feaccc1ee6ca52d18e5e379849541b1d84aece77acab1bddb2f4757f62861
```

已有数值证据分别归档在：

- `artifacts/wheel-overlay-20261111/组件独立安装验证报告.md`：两种配套顺序、调用、禁用和恢复。
- `artifacts/wheel-overlay-20261111/真实CABI与L1-L4补充验证报告.md`：真实接口变化与各层数值生效，记录版本 `f87e0b38`。
- `artifacts/default-run-install-20261112/无参数run增量安装验证报告.md` 及 `full-alias/`：默认 run、显式安装、未覆盖算子和移除恢复。
- [PR1183](https://github.com/flashserve/flash-linear-attention-npu/pull/1183) 与
  [正式 CI](https://github.com/flashserve/flash-linear-attention-npu/actions/runs/43196008121)：全量构建、安装及 GDR Example/ST。

这些结果按各自版本和覆盖范围引用；后续是否补测由实际代码变化决定。
