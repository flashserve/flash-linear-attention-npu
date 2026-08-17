# ChunkKdaFwd canonical 300 执行入口

先约定两个名词：

- **设计用例**：[`tests/op_cases/chunk_kda_fwd.json`](../../tests/op_cases/chunk_kda_fwd.json)
  中 `design_matrix.cases` 的一行。该文件固定有 300 行。
- **执行项**：一条设计用例按平台、调用路径、输入变体或 sanitizer 工具展开后的一次实际运行。
  一条设计用例可能对应多个执行项，但仍只计为一条设计用例。

300 条设计用例固定分为：176 条 `accuracy`、92 条 `run`、20 条 `msopprof`、
6 条 `stress`、6 条 `sanitizer`。A5 适用其中 291 条：

| 段 | A5 设计用例 | A5 不适用 |
| --- | --- | --- |
| accuracy | `P001-P096`、`G001-G039`、`G041-G080`，共 175 条 | `G040` |
| run | `N001-N084`、`G081-G088`，共 92 条 | 无 |
| msopprof | `M001-M008`、`G089-G092`，共 12 条 | `M009-M012`、`G093-G096` |
| stress | `S001-S004`、`G097-G098`，共 6 条 | 无 |
| sanitizer | `S005-S008`、`G099-G100`，共 6 条 | 无 |

这 291 条 A5 设计用例实际展开为 649 次运行：accuracy 509 次、run 104 次、msopprof
14 次、stress 8 次、sanitizer 14 次。这里的 649 是执行次数，不是新增设计用例。

五段的含义：

| 段 | 验证内容 |
| --- | --- |
| accuracy | 正向调用、全部可见输出 finite、NPU 对 GPU control/FP64 golden 的精度 |
| run | 非法参数是否命中设计指定的返回码、异常类型和错误信息 |
| msopprof | `msopprof` 设备侧 kernel 时长；不使用 Python wall time |
| stress | 固定输入重复执行，检查所有输出是否逐 bit 稳定 |
| sanitizer | `racecheck`/`memcheck`/`initcheck`/`synccheck` 内存与同步检查 |

主要可执行 JSON：

- [`atk_chunk_kda_fwd.json`](./atk_chunk_kda_fwd.json)：A5/AscendC accuracy，175 条设计用例
  展开为 254 个执行项，其中 175 个 random、79 个 traceable。
- [`atk_chunk_kda_fwd_accuracy_aclnn.json`](./atk_chunk_kda_fwd_accuracy_aclnn.json)：A5
  ACLNN accuracy，234 个执行项。
- [`atk_chunk_kda_fwd_accuracy_direct_launch.json`](./atk_chunk_kda_fwd_accuracy_direct_launch.json)：
  A5 `<<<>>>` direct-launch accuracy，21 个执行项。
- [`atk_chunk_kda_fwd_run_aclnn.json`](./atk_chunk_kda_fwd_run_aclnn.json)：A5 ACLNN
  参数拦截，92 个执行项。
- [`atk_chunk_kda_fwd_run_ascendc.json`](./atk_chunk_kda_fwd_run_ascendc.json)：A5 public
  AscendC 参数拦截，12 个执行项；它们是上述 92 条中的第二调用路径，不是新增设计用例。
- [`atk_chunk_kda_fwd_pr297_48.json`](./atk_chunk_kda_fwd_pr297_48.json)：只用于复现旧 PR297，
  不属于完整 300 条入口。

AscendC accuracy JSON 的 254 个执行项中，235 个使用在线 GPU Triton 同精度对照和 GPU FP64 golden。
其余 19 个执行项来自 15 条 `chunk_size=128` 设计用例；上游 GPU Triton KDA 只支持
`chunk_size=32/64`，因此这些执行项显式使用 GPU Torch 同精度对照和 GPU FP64 golden，不能报告为
“Triton 对照通过”。

[`canonical_case_adapter.py`](./canonical_case_adapter.py) 和
[`canonical_execution_adapter.py`](./canonical_execution_adapter.py) 锁定 300 行摘要、完整
case-row 摘要和 ID 顺序。任何设计行变化都会使生成器和已跟踪 JSON 的契约测试失败。

重新生成五份 A5 ATK JSON：

```bash
python ./canonical_case_adapter.py \
  --source ../../tests/op_cases/chunk_kda_fwd.json \
  --soc ascend950 --route ascendc \
  --output ./atk_chunk_kda_fwd.json --summary

python ./canonical_case_adapter.py \
  --source ../../tests/op_cases/chunk_kda_fwd.json \
  --soc ascend950 --route aclnn \
  --output ./atk_chunk_kda_fwd_accuracy_aclnn.json --summary

python ./canonical_case_adapter.py \
  --source ../../tests/op_cases/chunk_kda_fwd.json \
  --soc ascend950 --route direct_launch \
  --output ./atk_chunk_kda_fwd_accuracy_direct_launch.json --summary

python ./canonical_execution_adapter.py \
  --source ../../tests/op_cases/chunk_kda_fwd.json \
  --format run-atk --soc ascend950 --route aclnn \
  --output ./atk_chunk_kda_fwd_run_aclnn.json

python ./canonical_execution_adapter.py \
  --source ../../tests/op_cases/chunk_kda_fwd.json \
  --format run-atk --soc ascend950 --route ascendc \
  --output ./atk_chunk_kda_fwd_run_ascendc.json
```

## 建议执行顺序

按下面顺序验收，前 3 项有真实问题时先修复，再进入第 4 项：

1. **报错**：第 3 节正向 accuracy 不得出现执行错误；第 6 节负向 run 必须精确命中预期
   返回码或异常。
2. **结构性精度错误**：accuracy 未达标时使用已保存输出判断索引、布局、边界、状态链或
   输出写回是否整体错误。确认只是普通数值误差时记录标注，不阻塞性能段。
3. **NaN/Inf**：executor 对全部可见输出逐 tensor 检查 finite；第 8、9 节再覆盖重复执行和
   sanitizer。
4. **性能**：只有前三项没有待修问题时，执行第 7 节的 12 条 A5 性能设计用例。

这里的“结构性精度错误”是指输出的索引、形状语义、布局映射、有效区、状态递推或整片数值关系
错误；单纯低精度舍入造成的小幅误差不归入这一类。

下文命令统一使用进程可见设备编号：

```bash
export KDA_NPU_DEVICE=0
```

## 1. 缓存契约

在线 GPU accuracy 和 ATK `run` 默认使用固定 seed 现场生成输入，不要求预建缓存。需要跨机器
逐 bit 复现、stress、sanitizer 或 msopprof 时，可以显式启用只读持久缓存。每个 cache entry
是一个目录；分片按执行类型声明：

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

本节是可选的离线缓存流程。在安装了 ATK 和 CPU Torch 的环境中进入本目录。复现旧 PR297
时可先用一条小 case 建缓存并校验：

```bash
python ./build_reference_cache.py build \
  --case-json ./atk_chunk_kda_fwd_pr297_48.json \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --case-id 250

python ./build_reference_cache.py validate \
  --case-json ./atk_chunk_kda_fwd_pr297_48.json \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
  --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG" \
  --case-id 250
```

构建旧 48 条子集时删除 `--case-id`：

```bash
python ./build_reference_cache.py build \
  --case-json ./atk_chunk_kda_fwd_pr297_48.json \
  --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR"

python ./build_reference_cache.py validate \
  --case-json ./atk_chunk_kda_fwd_pr297_48.json \
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
`canonical_execution_runner.py` 的 msopprof、stress 和 sanitizer 路径只读这个 cache；缺失、
陈旧、checksum 不匹配或 variant 不存在都会失败。直接使用已跟踪 ATK JSON 的 accuracy 和
`run` 路径不要求 cache。负向 case 的 mutation 仍由固定 seed 和明确 spec 决定。

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

## 3. A5 accuracy：NPU + GPU control + GPU FP64

accuracy 共运行 509 个执行项，覆盖 A5 适用的 175 条设计用例及其 AscendC、ACLNN、
direct-launch 路由。默认关闭持久缓存，NPU、GPU control 和 GPU FP64 都按 JSON 中的同一 seed
在线生成输入。

GPU 机器先启动 server：

```bash
export KDA_ATK_PERSISTENT_CACHE_MODE=off
export KDA_GPU_SERVER_PORT=9090
export KDA_GPU_DEVICE=0
export KDA_GPU_OUTPUT_DIR=/path/to/gpu_output

atk server \
  --host 0.0.0.0 \
  --port "$KDA_GPU_SERVER_PORT" \
  --devices "$KDA_GPU_DEVICE" \
  --name gpu_reference \
  --output_path "$KDA_GPU_OUTPUT_DIR" \
  --plugin_path ./executor_chunk_kda_fwd.py \
  --timeout 8000
```

NPU 机器依次执行三份 JSON；GPU server 保持运行：

```bash
export KDA_ATK_PERSISTENT_CACHE_MODE=off
export KDA_NPU_OUTPUT_ROOT=/path/to/npu_output
export KDA_GPU_SERVER_HOST=gpu-server.example.com
export KDA_GPU_SERVER_PORT=9090
export KDA_GPU_DEVICE=0

run_accuracy() {
  local route="$1"
  local case_json="$2"
  local end="$3"
  atk \
    node --name npu_dut --backend npu --devices "$KDA_NPU_DEVICE" \
         --output_path "$KDA_NPU_OUTPUT_ROOT/$route" \
    node --name gpu_reference --backend gpu \
         --host "$KDA_GPU_SERVER_HOST" --port "$KDA_GPU_SERVER_PORT" \
         --devices "$KDA_GPU_DEVICE" --is_compare true \
    task \
      -c "$case_json" \
      --task accuracy \
      --bm_device gpu \
      -p ./executor_chunk_kda_fwd.py \
      --save_data output \
      --syc_dataset \
      -s 0 -e "$end" \
      -mt 1 \
      -to 2000
}

run_accuracy ascendc ./atk_chunk_kda_fwd.json 254
run_accuracy aclnn ./atk_chunk_kda_fwd_accuracy_aclnn.json 234
run_accuracy direct_launch ./atk_chunk_kda_fwd_accuracy_direct_launch.json 21
```

direct-launch 段需要先安装本提交生成的 `ascend_ops` 扩展。需要定位时，`-s/-e` 使用 JSON 列表下标且 `-e`
为开区间，例如 `-s 0 -e 1` 只跑第一个执行项。所有可见输出都会检查 NaN/Inf。

509 个执行项中，467 个 `chunk_size=64` 执行项使用 GPU Triton 同精度对照和 GPU FP64 golden；
42 个 `chunk_size=128` 执行项的 `gpu_control_reference=torch_same_precision`，使用独立的 GPU
Torch 同精度对照和 GPU FP64 golden，原因是上游 Triton KDA 不支持 128。报告必须单独列出这 42 项，
不能把它们计作 Triton 通过。

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
    -wl '[1001]' \
    --loop_nums 50 \
    --disable_id_seed \
    -mt 64 \
    -to 2000
```

复检不要加 `-sp`。关闭缓存后每轮 CPU golden/benchmark 都按当轮 runtime seed 重新计算，
不能用固定 PT 降低复检成本。

## 5. 可选只读缓存拓扑

第 3 节在线 GPU 拓扑不需要 cache。只有需要跨环境逐 bit 复现时才设置
`KDA_ATK_PERSISTENT_CACHE_MODE=readonly`，并按第 2 节明确 pin catalog。此时 GPU FP64 从
`cpu_fp64.pt` 读取；GPU Triton 和 C128 的 GPU Torch 同精度对照都从同一 `inputs.pt` 恢复输入，
并在 GPU 上执行。缺失或摘要不一致会直接失败。

## 6. Canonical run 负向契约

两份已跟踪 JSON 覆盖 A5 的 92 条 run 设计用例。`aclnn` 路由核对实际 ACLNN 数值返回码和
`aclGetRecentErrMsg`；`ascendc` 路由核对实际 Python 异常类型和消息。两项任一不匹配都失败，
不能用 shell 返回 0 代替契约通过。负向测试只需要 NPU，不启动 GPU server。

```bash
atk node --name npu_dut --backend npu --devices "$KDA_NPU_DEVICE" \
  --output_path <aclnn_run_output_dir> task \
  -c ./atk_chunk_kda_fwd_run_aclnn.json \
  --task run \
  -p ./executor_chunk_kda_fwd.py \
  -s 0 -e 92 -sp -to 2000

atk node --name npu_dut --backend npu --devices "$KDA_NPU_DEVICE" \
  --output_path <ascendc_run_output_dir> task \
  -c ./atk_chunk_kda_fwd_run_ascendc.json \
  --task run \
  -p ./executor_chunk_kda_fwd.py \
  -s 0 -e 12 -sp -to 2000
```

92 个 ACLNN 执行项覆盖全部 run 设计用例；12 个 AscendC 执行项是公开 Python 路由补充覆盖，
不增加设计用例计数。A2 使用 `--soc ascend910b` 重新投影。不能把未执行的 route 记为已测。

## 7. msopprof 性能

本段 A5 设计用例为 `M001-M008`、`G089-G092`，共 12 条。`M001` 和 `M003` 各有
`baseline` 与 `l2_streaming_single_read_disabled` 两个执行项，因此实际运行 14 次。
先准备第 2 节完整只读 cache，再依次生成并执行 profiler 命令：

```bash
export KDA_PERF_OUTPUT_ROOT=<perf_output_root>
mkdir -p "$KDA_PERF_OUTPUT_ROOT"

for design_id in \
  KDA-FWD-M001 KDA-FWD-M002 KDA-FWD-M003 KDA-FWD-M004 \
  KDA-FWD-M005 KDA-FWD-M006 KDA-FWD-M007 KDA-FWD-M008 \
  KDA-FWD-G089 KDA-FWD-G090 KDA-FWD-G091 KDA-FWD-G092; do
  variant=baseline
  name="${design_id}_${variant}"
  output="$KDA_PERF_OUTPUT_ROOT/$name"
  mkdir -p "$output"
  python ./canonical_execution_runner.py msopprof-command \
    --design-id "$design_id" --soc ascend950 --variant "$variant" \
    --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
    --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG" \
    --device "$KDA_NPU_DEVICE" --output "$output/profile" \
    > "$output/run.sh"
  bash "$output/run.sh" > "$output/run.log" 2>&1
done
```

两个 L2-disabled 执行项必须切换到对应编译产物后单独运行；不能在 baseline binary 上仅设置环境
变量。安装对应产物后执行：

```bash
export KDA_CANONICAL_BUILD_VARIANT=l2_streaming_single_read_disabled

for design_id in KDA-FWD-M001 KDA-FWD-M003; do
  variant=l2_streaming_single_read_disabled
  name="${design_id}_${variant}"
  output="$KDA_PERF_OUTPUT_ROOT/$name"
  mkdir -p "$output"
  python ./canonical_execution_runner.py msopprof-command \
    --design-id "$design_id" --soc ascend950 --variant "$variant" \
    --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
    --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG" \
    --device "$KDA_NPU_DEVICE" --output "$output/profile" \
    > "$output/run.sh"
  bash "$output/run.sh" > "$output/run.log" 2>&1
done
```

每项完成后再用下方 parser 检查 device duration 和相对 baseline。没有相对阈值的 baseline 项只
记录设备侧时长，不把“成功生成报告”写成性能通过。

有相对阈值的映射固定为：`M005/M006/M007 -> M003`、`G090 -> G089`、`G092 -> G091`。
其余 baseline 项和两个 L2 A/B 项记录实测值，报告时明确写 `measured`，不伪造 PASS。

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
for design_id in \
  KDA-FWD-S001 KDA-FWD-S002 KDA-FWD-S003 KDA-FWD-S004 \
  KDA-FWD-G097 KDA-FWD-G098; do
  python ./stress_npu_determinism.py \
    --design-id "$design_id" --soc ascend950 \
    --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
    --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG" \
    --device "$KDA_NPU_DEVICE"
done
```

复现旧 PR297 子集的具体 case 时使用独立兼容入口。先在 CPU 环境一次性构建并校验缓存：

```bash
python ./build_reference_cache.py build \
  --case-json ./atk_chunk_kda_fwd_pr297_48.json \
  --case-id 254,255 \
  --cache-dir <legacy_cache_dir>

python ./build_reference_cache.py validate \
  --case-json ./atk_chunk_kda_fwd_pr297_48.json \
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

6 条 A5 sanitizer 设计用例展开为下面 14 个 variant/tool 执行项：

```bash
export KDA_SANITIZER_LOG_ROOT=<sanitizer_log_root>
mkdir -p "$KDA_SANITIZER_LOG_ROOT"

while read -r design_id variant tool; do
  python ./canonical_execution_runner.py run-sanitizer \
    --design-id "$design_id" --soc ascend950 --variant "$variant" \
    --tool "$tool" \
    --operator-object <sanitizer_operator_object> \
    --log "$KDA_SANITIZER_LOG_ROOT/${design_id}_${variant}_${tool}.log" \
    --cache-dir "$KDA_ATK_PERSISTENT_CACHE_DIR" \
    --catalog "$KDA_ATK_PERSISTENT_CACHE_CATALOG" \
    --device "$KDA_NPU_DEVICE"
done <<'EOF'
KDA-FWD-S005 dense_key2 racecheck
KDA-FWD-S005 mixed_tail_key2 racecheck
KDA-FWD-S006 dense_key2 memcheck
KDA-FWD-S006 mixed_tail_key2 memcheck
KDA-FWD-S006 max_kv_key1 memcheck
KDA-FWD-S007 tail_initial_none_hidden_outputs initcheck
KDA-FWD-S008 long_chain synccheck
KDA-FWD-S008 mixed_tail synccheck
KDA-FWD-G099 aligned racecheck
KDA-FWD-G099 aligned synccheck
KDA-FWD-G099 mixed_tail racecheck
KDA-FWD-G099 mixed_tail synccheck
KDA-FWD-G100 max_group_tail_initial_none memcheck
KDA-FWD-G100 max_group_tail_initial_none initcheck
EOF
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
