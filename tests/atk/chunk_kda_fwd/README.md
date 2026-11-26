# ChunkKdaFwd ATK 工程

本目录提供 `chunk_kda_fwd` 的 ATK 单算子工程。通用版本要求、case 范围、测试动作和
结果检查规则见 [`../README.md`](../README.md)。

## 输入限制

- `layout` 支持 `BSND/BNSD/TND/NTD`；`layout` 只描述输入，输出布局由接口固定约定。
- `BSND` 下 `q/k=[B,T,H_k,K]`，`v=[B,T,H_v,V]`，`g=[B,T,H_v,K]`，`beta=[B,T,H_v]`。
- `BNSD` 下 `q/k=[B,H_k,T,K]`，`v=[B,H_v,T,V]`，`g=[B,H_v,T,K]`，`beta=[B,H_v,T]`。
- `TND/NTD` 使用打包 token；`cu_seqlens` 从 `0` 开始、以总 token 数结束且单调不减。
- head 映射满足 `0 < H_k <= H_v <= 128` 且 `H_v % H_k == 0`。
- `q/k/v` 支持 `BFLOAT16/FLOAT16`；`g` 支持 `FLOAT/BFLOAT16`；`beta` 支持 `FLOAT/BFLOAT16`。
- `K/V` 只支持两档且必须同档：`K=V=64` 或 `K=V=128`；混合档（如 `K=64,V=128`）与其它
  取值（含 `V=256`）返回 `ACLNN_ERR_PARAM_INVALID`，负向用例覆盖同类拦截。
- `chunk_size` 只支持 `64`，其它取值（含 `128`）在参数校验阶段返回
  `ACLNN_ERR_PARAM_INVALID`，负向用例覆盖该类拦截；rank-4 变长输入要求 `B=1`，
  逻辑序列数最多 `1024`。
- `use_gate_in_kernel=true` 时必须提供 `A_log`；`safe_gate=true` 时 `lower_bound` 取 `[-5,0)`。
- `initial_state` 如提供，末两维由 `state_v_first` 解释为 `[K,V]` 或 `[V,K]`。

## 精度拓扑

精度标准统一使用 `mixed_tolerance_bm`。显式 CPU 节点调用 executor 的 FP64 路径生成
唯一 golden，显式 NPU 节点只运行 DUT。CPU golden 输出转换为 FP32，NPU 输出保持算子
原始 dtype，由 ATK 按混合容差规则比较。

```text
ATK accuracy task
|-- CPU FP64 golden (output FP32)
`-- NPU DUT
```

## 运行

在仓库根目录准备好 ATK、CANN、OPP 和 Python 包环境后执行：

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=accuracy
```

只验证前 5 个 case：

```bash
CASE_START=0 CASE_END=5 \
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=accuracy
```

确定性回归会重复运行 MSS 用例并逐位比较全部可见输出。MSS 包含 #440 的 63-token 单序列
尾块场景，通过 96 个 value heads 放大并行调度覆盖，同时启用 `disable_recompute=true`、最终状态
和全部中间量：

```bash
DC_LOOP_NUMS=100 \
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=0 -scope=determinism
```

性能、确定性、mssanitizer 和用例生成均通过统一脚本的对应 `-scope` 执行。

## 内部 checkpoint 导出参考

checkpoint 用例覆盖 packed/fixed batch、初态存在/缺失、VK/KV 布局、无效选点，
并以 H128 的多序列用例覆盖多轮 head/序列任务复用同一个 AIV 状态银行的情况；
后者检查 checkpoint 写回必须在下一轮状态更新前完成。每个精度场景保留三个固定种子，
相应 MSS 场景使用同一输入合同。

新增 checkpoint 用例的验收对象是 FwdH 的 FP32 状态导出及恢复，保持原 KDA 混合精度计算。
原有输出继续使用原数学参考；新增用例的末态与选点使用独立 CPU FP64 FwdH 递推，
显式保留 BF16 H/v_new 和 FP32 状态银行的存储舍入，混合容差阈值不变。

统一入口在 ATK 开始前生成共享 Prepare 输入张量，保存在结果目录的 `checkpoint_inputs/`，
通过 `KDA_ATK_CHECKPOINT_INPUT_DIR` 传给 CPU 参考。文件只包含 kg/w/u/gk、输入指纹和用例，
不包含 NPU 末态或 checkpoint。CPU 参考检查输入、seed和元数据一致后独立递推；
导出中间量与原运行模式的输出还必须逐位相同。这是 FwdH/导出合同验证，不是整模型精度验收。
独立参考位于同目录 `checkpoint_reference.py`，ATK CPU 节点和 NPU 合同测试共同复用；
参考模块只依赖 CPU PyTorch，不导入 NPU 后端或 pytest。

如需单独诊断整算子相对纯 FP64 数学参考的误差，可设
`KDA_ATK_CHECKPOINT_REFERENCE=math`；这会保留纯数学末态/选点参考，不能把该诊断与
状态导出验收混称。直接运行 ATK 时先执行同目录 executor 的
`--prepare-checkpoint-inputs <case.json> --device <id> -s <start> -e <end>`，
并为生成器和 ATK 显式设置同一个 `KDA_ATK_CHECKPOINT_INPUT_DIR`。

## A5 native FP32 checkpoint validation

Validated on Ascend950DT with CANN 9.3, Python 3.12, PyTorch 2.10 and
torch_npu 2.10.post7, against a pristine build of upstream `79164d12`.
The scope is the existing mixed-precision FwdH state export/resume contract,
not a change to the original KDA mathematical reference or whole-model accuracy.

| Check | Result |
| --- | --- |
| Native export, resume, layouts, invalid offsets, H128 bank reuse and graph replay | 34 passed |
| CPU wrapper validation | 8 passed |
| KDA / shared FwdH ATK accuracy | 63 / 200 passed |
| Determinism | All 7 profiles passed 50 bitwise repetitions |
| Complete 7-profile memcheck | ATK checker 7/7; 111 kernel starts/finishes; no ERROR, tool communication failure, fallback or lost records |
| All 5 new profiles, combined memcheck/racecheck | ATK checker 5/5; 76 kernel starts/finishes; no WARNING/ERROR, fallback or lost records |
| Linux CI contract | 338 unit tests and staged pipeline integration passed |

The complete memcheck is not a zero-warning result: the two unchanged legacy
fused profiles produce 18,596 ownership warnings, also reproduced by the
pristine upstream build. Their 16 kernel/metadata files are byte-identical and
14 visible tensors are bitwise equal. Profile 250 has 7,936 matching normalized
size/core/address findings; profile 298 has 10,660 matching access-size/core
counts, without an address-by-address equivalence claim. The changed Prepare
publication path separately reduces its previous 160 ownership warnings to zero.

Complete racecheck of the legacy fused path remains inconclusive. Combined
checking, disabling the PyTorch memory pool, and racecheck alone all encounter
the installed tool's 600-second response timeout in legacy profile 298.
The kernel completion record is missing, so zero ERROR in the partial raw trace
does not constitute a pass. This limitation is recorded separately from the
complete memcheck and the successful combined check of every new profile.

Same-card no-checkpoint composed-path graph measurements used seed 17,
BF16 Q/K/V, head dimension 128, chunk64 and the same initial-state clone.
Packed sequence boundaries were `[0, 65, 258, 275]` for H2 and
`[0, 1537, 4465, 4482]` for H12, with in-kernel Q/K normalization enabled.
Each of 40 samples contained 50 graph replays. H2 stayed near 39.6-39.7 us;
H12 changed from 327.30-327.40 us to 331.17-331.21 us in both measurement
orders, approximately +3.8 us / +1.2%. This regression is reported explicitly.
Default fused ATK device time changed from 1108.08 us to 1109.20 us (+0.10%).
These are component measurements, not an end-to-end throughput or TTFT claim.

For large legacy kernels, runtime instrumentation needed an explicit ATK
per-case budget of `-to 10800` seconds. The unified MSS wrapper does not override
ATK's default budget; the equivalent direct invocation is:

```bash
cd tests/atk/chunk_kda_fwd
mss_log="$PWD/kda-memcheck.log"
mssanitizer --tool=memcheck --log-file "$mss_log" -- \
  atk node --name npu_dut --backend npu --devices 0 \
  task -c atk_chunk_kda_fwd_mss.json -p executor_chunk_kda_fwd.py \
  --task run --mssanitizer -msl "$mss_log" -to 10800
python ../common/check_atk_result.py --type mssanitizer --output-root ./atk_output --op chunk_kda_fwd
```

The five new profiles additionally passed combined checking with
`PYTORCH_NO_NPU_MEMORY_CACHING=1`, default trace capacity, no block filter, and
ATK range `-s 2 -e 7`. A2/A5 repository CI, ABI-owner review and full-model
integration remain separate requirements; this report does not certify A2
or claim multi-host model validation.
