# ChunkKdaFwd canonical 300 持久输入与执行入口

本目录保留 PR297 风格的 ATK executor。主精度拓扑现在是本机 `NPU DUT + CPU`：

- benchmark CPU 角色返回 Torch FP64 golden；
- 普通 CPU 角色返回按原输入 dtype 量化中间操作的同精度 benchmark；
- NPU 角色按 spec 执行 `ascendc`、ctypes `aclnn` 或 direct-launch 真实入口；
- 原 GPU FP64 truth 和 GPU Triton control 仍可兼容使用，但不再是主命令。

当前 [`atk_chunk_kda_fwd.json`](./atk_chunk_kda_fwd.json) 只有 `id=250-297` 共 48 条
PR297 子集，不能把它报告成设计矩阵的 176 条 accuracy 数值用例。后续 176 条必须由
[`canonical_case_adapter.py`](./canonical_case_adapter.py) 从 canonical 300 用例生成显式
executable spec，再交给本目录的缓存 CLI。

适配器固定输出 176 个 logical accuracy spec：`P001-P096` 映射为 `id=1001-1096`，
`G001-G080` 映射为 `id=2001-2080`，spec 内同时保留原始 `design_id`。它不解析设计表中的
自由文本；shape、dtype、GVA、varlen、state 和输出策略均由显式 ID 规则生成，并锁定 300 行
source digest。96 条 P case 只生成 `random`，80 条 G case 同时生成 `random` 和
`traceable_metamorphic`，因此共有 256 个独立数值 spec/cache entry。它们再按 SOC 和 route
投影：A5/A2 accuracy 分别是 509/329 个 physical task；只有实际执行后才能报告通过，不能用
176 个 logical spec、256 个缓存 entry 或旧 48 条子集代替。

其余 124 条由 [`canonical_execution_adapter.py`](./canonical_execution_adapter.py)
逐 ID 显式物化：92 条 `run`、20 条 `msopprof`、6 条 `stress`、6 条
`sanitizer`。适配器同时锁定 300 行 source digest、完整 case-row digest 和 ID 顺序；任何
canonical 行变化都会硬失败，不能继续套用旧 executable spec。

## 1. 缓存契约

300 条默认只读持久缓存。每个 cache entry 是一个目录；分片按执行类型声明：

```text
<cache_key>/
  manifest.json
  inputs-<sha256>.pt
  cpu_fp64-<sha256>.pt             # 仅 accuracy 数值 entry
  cpu_same_precision-<sha256>.pt   # 仅 accuracy 数值 entry
.<cache_key>.lock
catalog-<case_set_key>.json
```

`manifest.json` 和 entry cache key 严格绑定：

- 去除展示字段后的 normalized spec；
- 固定 runtime seed；
- 完整 executor SHA256，以及 golden/benchmark 两个域分离摘要；
- producer 的精确 `torch.__version__`（包括本地版本后缀）；
- canonical variant materializer schema 和对应 adapter 源码 SHA256；
- reference schema、输出名、每个输出的 shape/dtype/None schema；
- 输入张量的 shape/dtype 摘要；
- 每个 PT 分片的 SHA256。

124 条非精度 entry 只要求 inputs 分片；256 条 accuracy 数值 entry 要求三个分片。非精度
`inputs.pt` 可以在一个分片内保存多个去重后的 variant：L2 A/B 和输出 mask 共用同一输入；
mixed-tail、max-K/V 等确实改变输入的 variant 各生成一次。PT 分片先写临时文件、`fsync`
后按内容 SHA256 发布；声明的全部分片完成后才原子切换 manifest。`--force` 构建期间旧
manifest 和旧内容分片保持不变，因此已经打开的 reader 可以继续完成；中断构建只会留下未引用的
内容分片，不会破坏已发布 generation。锁使用 OS `flock`，进程退出后自动释放；锁文件本身持久
保留，不能用“文件存在”判断仍有构建进程。旧 generation 不自动清理，清理必须在确认无 reader
使用后由独立维护流程完成。
缺少 entry、metadata 过期、producer Torch 版本变化、executor 变化、schema 不一致或校验和错误
都会直接失败，不会在线重算或静默使用旧数据。catalog v2 另行绑定 producer Torch 版本、producer
executor/golden/benchmark 三个摘要、每个 entry 的 manifest generation 和全部 required shard
SHA256、输入 JSON 的 SHA256、准确 case ID 列表、logical count、cache entry count、adapter SHA256
和 variant schema，避免混淆
48/176/256/300/380 等不同口径。consumer 可以使用另一个经过交接验证的
Torch 版本和另一个 executor 文件加载固定 PT，但必须通过外部明确指定的 catalog key/path 取得完整
producer 身份，再用当前 spec、seed 和 adapter 重建 metadata；consumer Torch 版本和当前 executor
摘要只写验证回执，不进入 entry identity。普通无 `--catalog` build 仍以当前 Torch 和当前 executor
作为新的 producer，因此会生成新的 cache key。

缓存默认放在用户缓存目录，不放仓库：

```bash
export KDA_ATK_PERSISTENT_CACHE_DIR=<persistent_cache_dir>
export KDA_ATK_PERSISTENT_CACHE_CATALOG=catalog-<catalog_sha256>.json
```

cache 目录未设置时使用 `~/.cache/fla_npu/chunk_kda_fwd_atk`。只读 consumer 不会扫描目录猜测
catalog；必须设置 `KDA_ATK_PERSISTENT_CACHE_CATALOG`，或为命令传 `--catalog`。该值只能是
64 位 catalog key、`catalog-<sha256>.json` 文件名，或 cache 根目录内该文件的路径。本目录也忽略 `*.pt`，任何 PT、ATK
输出、xlsx 或日志都不得提交。确需临时放在仓内时只能使用本目录的 `.reference_cache/`，
该目录整体已加入 ignore。

## 2. 首次离线构建

在安装了 ATK 和 CPU Torch 的环境中进入本目录。先用一条小 case 建缓存并校验：

```bash
python ./build_reference_cache.py build \
  --case-json ./atk_chunk_kda_fwd.json \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --case-id 250

python ./build_reference_cache.py validate \
  --case-json ./atk_chunk_kda_fwd.json \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG" \
  --case-id 250
```

构建当前 48 条子集时删除 `--case-id`：

```bash
python ./build_reference_cache.py build \
  --case-json ./atk_chunk_kda_fwd.json \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR"

python ./build_reference_cache.py validate \
  --case-json ./atk_chunk_kda_fwd.json \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG"
```

`--case-id` 支持重复使用、逗号列表和闭区间，例如 `--case-id 250,251 --case-id 260-263`。
已存在且完整的 entry 会先验证后复用；只有明确需要替换时才传 `--force`。长序列 FP64
reference 的生成和落盘成本较高，建议按 case 分批构建并保留足够磁盘空间。
每次 build 成功后从 `CATALOG_WRITTEN file=catalog-<sha256>.json` 取得本次精确 catalog，设置
`KDA_ATK_PERSISTENT_CACHE_CATALOG` 后再 validate。不能用目录中“最新”文件代替这个外部 pin。

已有 catalog v1 可以在不重建 entry/PT 的前提下显式升级为 v2。升级命令必须把旧 v1 文件/key
作为 `--catalog` 传给 `build`，并使用与旧 catalog 完全相同的 JSON、adapter 和 case 选择：

```bash
python ./build_reference_cache.py build \
  --case-json <exact_source_json> \
  --case-adapter <exact_adapter_module:callable> \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --catalog catalog-<old_v1_sha256>.json \
  --case-id <exact_old_selection>
```

该路径先从全部旧 manifest 推导唯一的 producer Torch 与 executor/golden/benchmark 摘要，再逐项
校验旧 catalog、metadata、required shards 和全部 PT；只允许输出 `VALID`，任一 entry 缺失或冲突
都会失败，禁止配合 `--force`，也不会导入生成器。全部 entry 有效后才签发新的 v2 catalog；旧 v1
文件保留，不能从目录扫描或无 pin 推导 producer 身份。

构建完整 176 条 canonical logical accuracy 缓存。该命令实际生成 256 个数值 entry：

```bash
python ./build_reference_cache.py build \
  --case-json ../../tests/op_cases/chunk_kda_fwd.json \
  --case-adapter canonical_case_adapter:materialize \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR"

python ./build_reference_cache.py validate \
  --case-json ../../tests/op_cases/chunk_kda_fwd.json \
  --case-adapter canonical_case_adapter:materialize \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG"
```

完整 300 条统一构建和校验命令如下。该步骤是唯一允许生成确定性输入的阶段：

```bash
python ./build_reference_cache.py build \
  --case-json ../../tests/op_cases/chunk_kda_fwd.json \
  --case-adapter canonical_execution_adapter:materialize_all \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR"

python ./build_reference_cache.py validate \
  --case-json ../../tests/op_cases/chunk_kda_fwd.json \
  --case-adapter canonical_execution_adapter:materialize_all \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG"
```

catalog 必须精确显示 300 个 logical ID 和 380 个 cache entry；其中 256 个 accuracy 数值
entry 为三分片，124 个非精度 entry 为 input-only。
后续 `run`、性能、stress 和 sanitizer 全部只读这个 cache；缺失、陈旧、checksum 不匹配或
variant 不存在都会失败，不会现场随机生成。负向 case 只在缓存基础输入上施加 mutation，
mutation 参与 cache identity。

`a_log_dtype`、`dt_bias_dtype`、非连续输入策略和可追踪的 head/state 数据构造都属于 normalized
spec，会参与 cache identity。`route` 和 `soc` 不改变数值输入或 CPU 参考，因此不进入 cache
identity；同一个数值 variant 在 `ascendc`、`aclnn`、`direct_launch` 和目标 SOC 投影间复用
同一套 inputs/CPU reference 分片。`random` 与 `traceable_metamorphic` 的数值不同，分别缓存；
124 条非精度 case 不生成两个 CPU reference 分片。

### 2.1 跨 Torch 版本 CPU 交接检查

下面的独立检查不使用 NPU。producer 环境生成一个 accuracy 三分片 entry 和一个 input-only
entry；consumer 环境只读取同一 cache。命令中的 producer/consumer 版本应由实际环境打印，
不能通过修改 `torch.__version__` 伪造。

producer 环境：

```bash
python -c 'import torch; print(torch.__version__)'

python ./build_reference_cache.py build \
  --case-json ../../tests/op_cases/chunk_kda_fwd.json \
  --case-adapter canonical_execution_adapter:materialize_all \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --case-id 1001,5003

export KDA_ATK_PERSISTENT_CACHE_CATALOG=catalog-<sha256_from_CATALOG_WRITTEN>.json

python ./build_reference_cache.py validate \
  --case-json ../../tests/op_cases/chunk_kda_fwd.json \
  --case-adapter canonical_execution_adapter:materialize_all \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG" \
  --case-id 1001,5003
```

将 cache 根目录连同上面明确记录的 catalog 文件完整交给 consumer 后执行：

```bash
python -c 'import torch; print(torch.__version__)'

python ./build_reference_cache.py validate \
  --case-json ../../tests/op_cases/chunk_kda_fwd.json \
  --case-adapter canonical_execution_adapter:materialize_all \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG" \
  --case-id 1001,5003
```

consumer validate 必须加载所有声明分片并输出同一 producer、实际 consumer 和 `CATALOG_VALID`。
旧 catalog v1 只提供受信目录内的只读兼容：也必须明确 pin 文件/key，并且其全部 entry manifest
只能解析出一个完整 producer 身份（Torch 与 executor/golden/benchmark 摘要）。v1 不会从 cache
目录自动选择，存在零个或多个 producer 身份时直接失败；v1 没有 v2 的 manifest/PT 内容 pin，
不能用于不受信的 cache 副本。

为实际平台/路由生成 ATK JSON 时显式投影，不能把未生成或未执行的 route 算作覆盖：

```bash
python ./canonical_case_adapter.py \
  --source ../../tests/op_cases/chunk_kda_fwd.json \
  --soc ascend950 \
  --route ascendc \
  --output <a5_ascendc_case_json> \
  --summary

python ./canonical_case_adapter.py \
  --source ../../tests/op_cases/chunk_kda_fwd.json \
  --soc ascend910b \
  --route aclnn \
  --output <a2_aclnn_case_json> \
  --summary
```

不传 `--variant` 时会生成目标 SOC/route 声明的全部 physical variants；也可以分别生成两个
ATK JSON 后依次执行：

```bash
python ./canonical_case_adapter.py \
  --source ../../tests/op_cases/chunk_kda_fwd.json \
  --soc ascend950 \
  --route ascendc \
  --variant random \
  --output <a5_ascendc_random_case_json>

python ./canonical_case_adapter.py \
  --source ../../tests/op_cases/chunk_kda_fwd.json \
  --soc ascend950 \
  --route ascendc \
  --variant traceable_metamorphic \
  --output <a5_ascendc_traceable_case_json>
```

executor 按 spec 的 `route` 分别调用 `fla_npu.ops.ascendc.chunk_kda_fwd`、ctypes aclnn 入口或
`torch.ops.ascend_ops.chunk_kda_fwd_direct`。GVA 的 `random` 与
`traceable_metamorphic` 都具有独立 ATK case id、只读输入和双 CPU reference；traceable 变体
会在每个 sequence/chunk 起点注入可区分的 head 值，同时保留 head-distinct gate 和 state-pulse
等 case 自身构造。任何一个 physical variant 未执行时都不得报告为通过。

## 3. NPU + CPU accuracy

准备 ATK、CANN、当前构建的 OPP/Python 包，并只暴露目标 NPU。下面设备号使用进程内逻辑编号：

```bash
source <atk_environment>/bin/activate
source <cann_install_path>/set_env.sh
source <fla_npu_install_path>/vendors/fla_npu_transformer/bin/set_env.bash

export ASCEND_RT_VISIBLE_DEVICES=<physical_npu_id>
export PYTHONPATH=<repo_root>/torch_custom/fla_npu:<repo_root>:${PYTHONPATH:-}
export KDA_ATK_PERSISTENT_CACHE_DIR=<persistent_cache_dir>
export KDA_ATK_PERSISTENT_CACHE_CATALOG=catalog-<catalog_sha256>.json
export KDA_ATK_PERSISTENT_CACHE_MODE=readonly

cd <repo_root>/test/chunk_kda_fwd
```

先验证缓存，再跑单 case：

```bash
python ./build_reference_cache.py validate \
  --case-json ./atk_chunk_kda_fwd.json \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG" \
  --case-id 250

atk node --name npu_dut --backend npu \
    --devices 0 \
    --output_path <atk_output_dir> \
  node --name cpu_reference --backend cpu \
    --is_compare true \
  task \
    -c ./atk_chunk_kda_fwd.json \
    --task accuracy \
    --bm_device cpu \
    -p ./executor_chunk_kda_fwd.py \
    -s 0 \
    -e 1 \
    --save_data output \
    -sp \
    -to 2000
```

这版 ATK 的 `-s/-e` 是 JSON 列表下标，不是 case ID；`-s 0 -e 1` 对应 `id=250`。
跑当前 48 条时移除 `-s/-e`。spec、seed、executor、adapter 或 producer Torch 版本与 catalog
不一致时 readonly 初始化会报错。consumer Torch 版本可以不同，但仅限完成第 2.1 节交接检查并
使用同一个外部 pin 的 catalog；不能删除校验或临时切回目录扫描。

## 4. accuracy_lt 禁止固定缓存

`accuracy_lt --disable_id_seed` 的目的就是在固定结构下改变数值 seed，因此不得复用上述固定
PT。executor 在 `accuracy_lt` 检测到 `readonly` 会直接拒绝执行。复检前显式关闭：

```bash
export KDA_ATK_PERSISTENT_CACHE_MODE=off

atk node --name npu_dut --backend npu --devices 0 \
  node --name cpu_reference --backend cpu --is_compare true \
  task \
    -c ./atk_chunk_kda_fwd.json \
    --task accuracy_lt \
    --bm_device cpu \
    -p ./executor_chunk_kda_fwd.py \
    -wl '[250]' \
    --loop_nums 50 \
    --disable_id_seed \
    -mt 64 \
    -to 2000
```

复检不要加 `-sp`。关闭缓存后每轮 CPU golden/benchmark 都按当轮 runtime seed 重新计算，
不能用固定 PT 降低复检成本。

## 5. GPU 兼容拓扑

需要复现旧 PR297 分布式链路时，GPU benchmark 角色仍可返回 FP64 golden，普通 GPU 角色
仍调用 `KDA_ATK_TRITON_CALLABLE` 指定的同精度 Triton 实现。该在线 GPU 拓扑默认不使用
持久缓存，无需额外设置 `KDA_ATK_PERSISTENT_CACHE_MODE`。若显式启用 readonly cache，GPU
benchmark 直接读取 `cpu_fp64.pt` 并搬到 GPU，GPU Triton control 从 `inputs.pt` 恢复同一输入；
设置 `KDA_ATK_PERSISTENT_CACHE_MODE=off` 也会保持在线 GPU 计算行为。canonical 300 用例仍
强制要求预构建 readonly cache，不会因为默认值改变而回退到在线随机生成。

分布式 GPU server 的 ATK 版本、executor SHA256、producer identity、pinned catalog 和缓存内容
必须与发起端一致；consumer Torch 版本不同时也必须先完成第 2.1 节纯 CPU 交接检查。远端文件
同步仍按 ATK 的 `--syc_dataset` 约束处理。GPU 只是兼容入口，正式精度命令以第 3 节 CPU 双标杆为准。

## 6. Canonical run 负向契约

每个平台和路由分别生成 ATK JSON。`aclnn` 路由核对实际 ACLNN 数值返回码和
`aclGetRecentErrMsg`；`ascendc` 路由核对实际 Python 异常类型和消息。两项任一不匹配都失败，
不能用 shell 返回 0 代替契约通过。

```bash
python ./canonical_execution_adapter.py \
  --source ../../tests/op_cases/chunk_kda_fwd.json \
  --format run-atk --soc ascend950 --route aclnn \
  --output <a5_aclnn_run_json>

atk node --backend npu --devices 0 task \
  -c <a5_aclnn_run_json> \
  --task run \
  -p ./executor_chunk_kda_fwd.py \
  -sp -to 2000
```

将 `--route` 改为 `ascendc` 可执行 public Python 路由；A2 使用
`--soc ascend910b`。生成文件只含该 SOC/route 的适用 case，不能把另一条 route 记为已测。

## 7. msopprof 性能

先物化命令，再执行并解析结构化 profiler CSV/JSON。runner 只使用报告中的 device duration，
不读取 Python wall time。相对阈值 case 必须同时提供对应 dense baseline 报告。

```bash
python ./canonical_execution_runner.py msopprof-command \
  --design-id KDA-FWD-M006 --soc ascend950 --variant baseline \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG" \
  --output <profile_output> > <profile_command>

bash <profile_command> > <profile_log> 2>&1

python ./canonical_execution_runner.py parse-msopprof \
  --design-id KDA-FWD-M006 --soc ascend950 --variant baseline \
  --report <profile_output> --kernel-json <current_kernel_json> \
  --log <profile_log> \
  --baseline-report <m003_profile_output> \
  --baseline-kernel-json <baseline_kernel_json> \
  --baseline-log <m003_profile_log>
```

parser 从每份 kernel JSON 的唯一 primary MIX 入口解析精确 runtime name；多 stage
应用重复传入 `--kernel-json`。每个数字 replay 目录必须恰好包含各 stage 一条
`op_type=mix` 的数值记录，stage 时长先按 replay 求和，再计算应用统计。目录断号、
重复/缺失/未知 stage、数值 companion、profiler 故障标记和非有限时长都会使报告无效。

`l2_streaming_single_read_disabled` 子实验必须使用对应编译产物，并在运行前设置：

```bash
export KDA_CANONICAL_BUILD_VARIANT=l2_streaming_single_read_disabled
```

runner 会核对 build variant；未切换产物时不会把同一个 binary 伪装成 L2 A/B。

## 8. Stress 逐 bit

命令按 canonical repeat count 执行：S001-S003/G097 为 100 次，S004 为两种 mask 各 20 次，
G098 为两种 mask 各 100 次。每个 variant 比较全部 tensor 输出与 run 0；两种 mask 还比较
公共输出。输入只从 cache 读取。

```bash
python ./stress_npu_determinism.py \
  --design-id KDA-FWD-G098 --soc ascend950 \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG" \
  --device 0
```

复现旧 PR297 子集的具体 case 时使用独立兼容入口。先在 CPU 环境一次性构建并校验缓存：

```bash
python ./build_reference_cache.py build \
  --case-json ./atk_chunk_kda_fwd.json \
  --case-id 254,255 \
  --cache-dir <legacy_cache_dir>

python ./build_reference_cache.py validate \
  --case-json ./atk_chunk_kda_fwd.json \
  --case-id 254,255 \
  --cache-dir <legacy_cache_dir> \
  --catalog <legacy_catalog_sha256_or_filename>
```

设备侧先跑 20 次烟测，通过后再跑 100 次：

```bash
python ./stress_legacy_cached_cases.py \
  --case-id 254 --case-id 255 \
  --cache-dir <legacy_cache_dir> \
  --catalog <legacy_catalog_sha256_or_filename> \
  --device 0 --repeats 20

python ./stress_legacy_cached_cases.py \
  --case-id 254 --case-id 255 \
  --cache-dir <legacy_cache_dir> \
  --catalog <legacy_catalog_sha256_or_filename> \
  --device 0 --repeats 100
```

该入口校验完整 manifest 与全部分片 checksum，但只加载固定 `inputs` 分片；缺失或陈旧缓存会
直接失败，不会调用随机输入生成器。每轮同步后检查全部 12 个输出无 NaN/Inf，并与第 0 轮逐 bit
比较。

## 9. Sanitizer

四类工具分别执行。runner 先用 `nm` 验证算子 object 含 sanitizer 符号，再启动工具；日志必须
出现目标 kernel 的 `Start <tool> sanitizer on kernel ...`。出现
`No active sanitizer tool on kernel ...`、未命中目标 kernel 或工具错误均失败。

```bash
python ./canonical_execution_runner.py run-sanitizer \
  --design-id KDA-FWD-G099 --soc ascend950 --variant mixed_tail \
  --tool racecheck \
  --operator-object <sanitizer_operator_object> \
  --log <raw_sanitizer_log> \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG" \
  --device 0
```

A2/A5 不适用项输出 `status=not_applicable` 并使用退出码 3，独立记录，不能计为 PASS。

## 10. Host-only 契约测试

不需要 NPU、CANN、ATK server 或 GPU：

```bash
python -m pytest -q \
  tests/operators/chunk_kda_fwd/ut/op_host/test_atk_executor_contract.py \
  tests/operators/chunk_kda_fwd/ut/op_host/test_canonical_accuracy_adapter.py \
  tests/operators/chunk_kda_fwd/ut/op_host/test_canonical_execution_adapter.py
```

测试覆盖 cache identity、原子 PT round-trip、缺失/陈旧/损坏拒绝、显式 catalog pin、catalog v1/v2
producer 解析、producer/consumer Torch 版本解耦、`weights_only=True` fail-closed、路径越界拒绝、
input-only/三分片 load+signature、输入 FP64 promotion、CPU/GPU 角色兼容、`accuracy_lt` 禁用固定
缓存、176 条精确 ID/字段映射、route/SOC 投影复用，以及 300 条 catalog、92/20/6/6 精确计数、
A2/A5 投影、msopprof 报告解析、stress 逐 bit 和四类 sanitizer 命中/拒绝契约。

## 11. 其他定位入口

保存 ATK 输出后分析：

```bash
python ./analyze_atk_saved_outputs.py <atk_output_dir> --case-id <case_id>
```

公开 PR/issue 只记录测试项、case 范围和结论，不记录服务器、账号、绝对路径、环境路径、
缓存路径、token 或内部日志位置。
