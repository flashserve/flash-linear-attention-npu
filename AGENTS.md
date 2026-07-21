# AGENTS.md

本文件是给 AI coding agent 的仓库级工作说明，适用于整个 `flash-linear-attention-npu` 仓库。若子目录后续出现更近的 `AGENTS.md`，以更近文件为准。

## 项目定位

`flash-linear-attention-npu` 是面向昇腾 NPU 的高性能线性注意力算子库，核心工作包括 Ascend C 算子、Tiling/InferShape/op_host、aclnn op_api、PyTorch 适配、Triton 适配、单算子测试和端到端 Example/ST。

优先阅读：

- `README.md`：构建、安装、调用、测试入口和目录结构。
- `CONTRIBUTING.md`：贡献流程和新增算子交付要求。
- `docs/repository-rules.md`：分支、ABI、NPU CI 和合入规则。
- `docs/agents/torch-npu-decoupled-architecture.md`：默认 `fla_npu.ops.ascendc` 解耦运行时、依赖确定与兼容性门禁、wheel 产物、多卡 device guard、stream、数据依赖、autograd 和 ACL 私有格式透传设计。
- `docs/agents/README.md`：面向 AI agent 的开发原理、方法论、验证和经验总结索引。
- `.github/pull_request_template.md`：PR 必填信息和验证矩阵。
- 当前修改算子的 `README.md`、`docs/aclnn*.md`、测试脚本和相邻算子实现。

## 工作原则

- 开始前先看 `git status --short`，不要回滚或覆盖用户已有改动。
- 先用 `rg` / `rg --files` 找代码和文档，再修改；不要凭记忆猜目录。
- 改动保持聚焦，避免无关格式化、批量重排和生成物噪声。
- 公共接口、shape/dtype/layout/range、预留参数、平台差异、返回码或报错文本变化，必须同步检查代码、README、aclnn 文档、PyTorch API 文档、测试和示例。
- 公开 PR、issue、评论和总结中不要暴露内网地址、机器名、用户名、绝对路径、临时目录、日志路径、token 或本地调测环境细节。
- 构建和测试默认面向 Linux + CANN + NPU 环境；其他平台只做静态阅读、文本编辑或格式检查，不把未验证命令写成已验证结论。

## 关键目录

- `fla/ops/ascendc/`：Ascend C 算子实现。
- `fla/ops/ascendc/common/`：公共 Ascend C 组件。
- `fla/ops/ascendc/gdn/`：GDN 相关算子。
- `torch_custom/fla_npu/`：PyTorch 自定义算子适配、YAML schema、Python 包和测试。
- `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py`：推荐 Python 稳定入口导出。
- `examples/`：端到端调用示例。
- `ci/`：NPU CI、Example/ST case 和本地 CI 脚本。
- `scripts/`：构建、打包、环境检查和代码生成辅助脚本。
- `tests/`：工程级 UT。

## 调用约定

安装后的 wheel 公开 Python import 面只使用 `fla_npu`。不要让新代码依赖顶层
`fla` 包；`fla/` 目录主要作为源码树内实现来源存在。

Ascend C 新代码优先使用稳定 Python 入口：

```python
from fla_npu.ops.ascendc import chunk_bwd_dv_local
```

Triton 算子同样使用 `fla_npu` 下的稳定入口：

```python
from fla_npu.ops.triton import chunk_local_cumsum
```

默认 Ascend C 调用路径必须保持与 `torch_npu` dispatcher、PyTorch C++ extension ABI、CPython ABI 和 C++ ABI 解耦。`fla_npu.ops.ascendc` 通过 Python ctypes 直调 `aclnn*`，不得在普通 import 或默认算子调用时 import `torch_npu`、注册 `torch.ops.npu`，或依赖 `custom_aclnn_extension_lib*.so`。

`torch.ops.npu.*` 是兼容旧调用的过渡路径，仅在兼容性测试或旧 API 验证中使用。需要旧路径时先确认 `fla_npu.load_legacy_torch_ops()` 的加载逻辑，并在 PR 中说明为什么不能使用 `fla_npu.ops.ascendc.<op>`。

修改 `torch_custom/fla_npu/fla_npu/ops/ascendc/_runtime.py`、`_aclnn_ctypes.py`、`torch_custom/fla_npu/setup.py` 或根目录 `setup.py` 时，必须同步检查 `docs/agents/torch-npu-decoupled-architecture.md`。涉及依赖确定阶段、版本或能力门禁、SOC/host/CANN 兼容范围、多卡 device guard、stream 感知、异步 launch 保活、正反向绑定、ACL 私有 format 透传、OPP wheel 安装位置或 legacy `torch_npu` 兼容路径的行为变化时，文档必须一起更新。

ctypes 算子如果会通过 data pointer 修改输入 tensor，必须在公共 wrapper 中显式维护 alias/mutation 契约：列出 mutated args，处理 eager autograd 版本计数，明确被修改状态的 grad 限制，并补充 mutation 测试。未增加 `torch.library` mutation schema、FakeTensor 和 `opcheck` 前，不得宣称该 mutable 路径支持 `torch.compile`、functionalization 或 `torch.export`；需要完整图编译支持时优先提供纯 Python custom-op 适配或返回新状态的 functional API，不得为此退回 PyTorch C++ extension。

## 算子开发交付 checklist

新增或修改 Ascend C 算子时，交付前逐项核对：

- [ ] `fla/ops/ascendc/**/op_host/*_def.cpp` 已同步输入、输出、属性、dtype、format、required/optional 定义。
- [ ] `fla/ops/ascendc/**/op_host/*tiling*` / `*infershape*` 已同步参数校验、shape 推导和 tiling。
- [ ] `fla/ops/ascendc/**/op_host/op_api/aclnn_*.h/.cpp` 已同步 aclnn 接口签名、返回和执行器逻辑。
- [ ] `fla/ops/ascendc/**/op_kernel/` 已同步 Kernel 实现、tiling data 和 tiling key。
- [ ] `torch_custom/fla_npu/npu_custom.yaml`、`test_native_functions.yaml`、`deprecated.yaml` 已同步 PyTorch schema 和适配生成输入。
- [ ] `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py` 已同步 Python 稳定导出。
- [ ] `torch_custom/fla_npu/test/test_npu_<op>.py` 已补充或更新单算子测试和参考实现。
- [ ] 当前算子的 `README.md`、`docs/aclnn*.md`、示例和 CI case 已同步更新。
- [ ] 参数校验、shape/dtype/layout/range、平台差异、预留参数语义和报错文本保持一致。
- [ ] 正向、反向、边界、异常、dense/varlen、关键 SOC 和目标 dtype/layout 场景已按改动风险覆盖。

ABI 敏感路径包括 `*_def.cpp`、`aclnn_*.h/.cpp`、`torch_custom/fla_npu/*.yaml` 和 `torch_custom/fla_npu/op_plugin/ops/opapi/**`。修改这些文件时，PR 需要明确说明 ABI 影响，并按 `.github/CODEOWNERS` 请求对应 owner 检视。

## prepare_wy_repr_bwd 专项约束

继续实现或修改 `prepare_wy_repr_bwd` 时，必须遵守以下已确认方案：

- 如果代码改动涉及计算阶段划分、Cube/Vector 依赖关系、workspace 布局、L1/L0/UB 分配、同步协议、分核逻辑、tiling key、tiling data 或公开接口语义变化，必须先说明方案修改点并询问用户确认；不得直接改代码。
- Cube/Vector 侧 process 类和通用文件名不要携带 `Stage0`，后续新增 stage 时复用通用类承载整体流程，例如 `PrepareWyReprBwdCubeProcess`、`PrepareWyReprBwdVectorProcess`。
- Vector 侧 scale 路径需要用语义化模板参数区分：生成 `Kbg` 使用 `beta * exp(g)`，生成 `Kbeta` 和 `Vb` 只使用 `beta`；模板布尔名使用类似 `APPLY_EXP_G_SCALE` 的正向语义名，不使用难解释缩写。
- Tiling data 不保存 `usedCoreNum`。host 侧 `SetBlockDim` 直接使用平台 AIC 核数；kernel 侧分核步长直接使用 `AscendC::GetBlockNum()`。
- workspace 大小按实际 `blockDim * workspaceCoreSize` 申请；每个 AIC core 保留两个 per-hv GM workspace slot，若某个中间量按 HK 作用域共享（如 `KKT`），其 cache buffer 计入 `workspaceCoreSize` 但不计入 per-hv `workspaceSlotSize`。
- `KKT = K @ K.T` 只依赖 `HK`，GVA 场景下不得按 `HV` 重复计算。stage0 debug 的 `debug_kkt` 输出 shape 使用 `[taskNum, HK, chunkSize, chunkSize]`；`Kbg/Vb/Kbeta/Dkbg/Dvb` 仍按 `[taskNum, HV, ...]` 输出。后续正式 workspace 中 KKT 应作为 HK-scoped cache，由当前 head slot 记录的 `kktSlotForSlot_[slot]` 映射读取，不放进每个 per-hv workspace slot 私有布局里重复生产。为覆盖 GVA 1:3 这类 2-head 窗口跨 group 边界的场景，每个 core 至少保留两块 KKT cache，`workspaceCoreSize = 2 * workspaceSlotSize + 2 * mBytes`。
- Cube 侧 K 需要独立 L1 resident ping/pang buffer。遇到新 `hk` 时才把 `K[BT,128]` 从 GM 搬入当前 `curKResidentSlot_`，同一 GVA group 内后续 `hv` 复用 `cachedKResidentSlot_`，不得用 `hv & 1` 推导。K resident ping/pang 需要独立核内事件组，不能复用 L1A/L1B operand ping/pong 的 EventID。`KKT=K@K.T` 使用同一块 resident K 同时构造 row-major `K` 视图和 column-major `K.T` 视图；resident free flag 必须等 K 和 K.T 两次 L1->L0 搬运都发出后再释放。
- Cube 侧 `Dvb = A^T @ du` 的 tile 配置必须区分模板 `V_DIM`：`V_DIM=128` 保持 L1/L0 的 MNK 为 `128,128,128`；`V_DIM=256` 时 L1 tile 使用 MNK `128,256,128`，L0 tile 使用 MNK `128,256,64`，在 K 方向循环完成两次 MMAD 累加后再统一 Fixpipe 写出。不得把 `V_DIM=256` 拆成 N=128 的两次输出，也不得在 K 循环内提前 Fixpipe。K 分片 MMAD 的 `unitFlag` 必须区分中间/最终分片：非最后 K 分片使用 `0b10`，最后 K 分片使用 `0b11`。
- Cube 侧使用 `TileMmadTla` 时，实际 `M/N/K` 必须由当前 `L0A/L0B/L0C` tensor 的 layout/originShape 表达，调用点只传 `initC/unitFlag`，不得退回手传显式 `m/n/k` 的写法。
- A2 上 Cube 侧 L0C->GM 的 Fixpipe copyout 对齐原 `prepare_wy_repr_bwd_full` block 级写法：`Dkbg/Dvb/KKT` 写 GM 时完整 tile、尾块和 small-M 都使用 `unitFlag=0b11`。stage0 已验证尾块 copyout 传 `0` 会导致后续 cube 任务卡死。
- Cube 侧同一个 L0C 计算结果只允许执行一次 Fixpipe copyout，不能从同一份 L0C 连续 Fixpipe 到 workspace 和 debug 等两个目的地。stage 调试需要中间结果时，只能选择一个 Fixpipe 目的地；若确实还需要另一份数据，必须从已落地 GM 结果走非 Fixpipe 路径另行复制，或在该 debug run 中用 debug 输出替代 workspace 输出。正式路径只写 workspace，不保留双写调试通路。
- Cube/Vector 分核按 `B * chunkNum` 或变长 `chunk_indices` 任务数分配，`HV` 在 core 内顺序遍历；不得用 `hv & 1` 或 `hv % 2` 推导 workspace slot。
- 新 `prepare_wy_repr_bwd` 的核间同步只允许使用 `Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_*>(...)` 和 `Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_*>(...)`。旧 `prepare_wy_repr_bwd_full` / `prepare_wy_repr_bwd_da` 中不带 reverse 的 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 只能作为历史实现阅读，不作为新算子同步写法参考。
- 新 `prepare_wy_repr_bwd` 的 Cube/Vector 双向同步使用方向独立的 `CrossCoreFlagWithReverse`：Vector->Cube 和 Cube->Vector 各一套 ready/reverse flag；workspace 两个 slot 不再额外拆 CrossCoreFlag。
- 新 `prepare_wy_repr_bwd` 的核内同步必须按实际 ping/pong buffer 拆事件组，不允许多个 ping/pong buffer 共用同一个 EventID：Vector input ping/pong 分别维护 `MTE2_V` 与 `V_MTE2`，Vector output ping/pong 分别维护 `V_MTE3` 与 `MTE3_V`；Cube 侧 L1A/L1B/L0A/L0B 等双缓冲事件也必须带 slot。
- `prepare_wy_repr_bwd` Vector 侧 UB input 分两套 ping/pong：`K/V` 等矩阵 tile 使用原 16KB matrix input ping/pong；`beta/g` 行向量使用独立 beta-g input ping/pong，每块只分配 `align32(chunk_size * 4)` 字节。两套 input ping/pong 必须分别维护当前 slot 类变量和 `MTE2_V`/`V_MTE2` 事件组，不能复用同一套 input buffer 或 EventID。
- `prepare_wy_repr_bwd` Vector 侧 `inputFp32`/`outputFp32` 矩阵临时区分别固定为 `2 * 16KB = 32KB`，对应一个固定 16KB half/bfloat16 matrix input/output tile cast 到 FP32 后的大小；不要按 `max(kVecRow, vVecRow) * max(K, V)` 分配。
- `stage_0` 的 Cube 与 Vector 都只依赖原始输入，二者在 `stage_0` 内独立向 GM workspace 生产中间结果，不能在 `stage_0` 尾部互相 `WaitFlagWithReverse`。跨 stage 的 `CrossCoreWaitFlagWithReverse` 必须放在下一阶段消费者入口，例如 `cube stage_1` 读取 `Kbg/Vb` 前等待 Vector stage0 ready，Vector 后续读取 `Dkbg/Dvb/KKT` 前等待 Cube stage0 ready。当前只实现 stage0 调试输出时，不发也不等跨核 ready，避免出现没有真实消费者的信号。
- stage 级精度验收必须临时导出该 stage 的中间产物并写专门 golden 逐项比对；不能只用最终 `dk/dv/dbeta/dg` 与 `da + full` 等价来判定某个 stage 内部正确。当前 stage0 验收时，需要临时把 stage0 写入 GM/workspace 的阶段输出导出，例如 Vector 侧 `Kbg/Vb/Kbeta` 以及 Cube 侧对应 stage0 产物；具体输出清单、shape、dtype、layout、有效区和无效区语义必须在写调试代码前说明清楚。
- stage 调试输出通路只允许作为临时验证代码存在。可以临时增加 debug-only 输出、独立 debug op、workspace dump 或额外 GM copy，但凡涉及 op schema、aclnn 接口、tiling/workspace 或 Python API 变化，都属于调试方案变更，编码前必须先说明并征得用户确认。stage golden 对比通过后，必须删除对应调试输出、调试 schema/aclnn/kernel/python 适配和临时测试入口，恢复正式接口；不得把调试输出通路提交或保留到最终 run 包。
- stage 中间产物 debug golden 使用临时 NPU PyTorch 公式 golden：actual 来自 `prepare_wy_repr_bwd_stage*_debug` 的 NPU 输出，expected 由同一用例输入在 NPU 上用 PyTorch 张量公式计算并逐中间产物比较；精度判定使用 `ct.single`，不要另写一套阈值或 bad-count 规则。这类脚本不能称作 ATK 对比；ATK 适合正式 API/全量泛化精度。
- stage 中间产物 debug golden 的超时结论必须分清 `kernel` 超时和 `golden/compare` 超时。测试脚本应先单独执行待测 op 并 `torch.npu.synchronize()`，打印 `kernel_sync` 耗时；如果没有打印该阶段结果，才按 kernel 未返回/疑似 kernel 超时定位。如果 `kernel_sync` 已返回但后续超时，应按 NPU PyTorch golden 生成或逐输出精度比对超时处理，不能误报为 kernel 超时。全量 run 期间如果日志超过 100 秒没有出现新的 case 行、`phase_time kernel_sync` 或结果行，可按最后一个未完成阶段判定卡死；最后日志已经进入某 case 但没有打印 `kernel_sync` 时，按该 case 的 kernel 阶段卡死处理。必要时先跑 `--kernel-only` 只确认 kernel 返回，再跑完整精度比对。
- 正式回归可保留最终四输出等价测试：先用 `fla_npu.ops.ascendc.prepare_wy_repr_bwd_da` 在 NPU 上计算 `dA`，再用 `fla_npu.ops.ascendc.prepare_wy_repr_bwd_full` 在 NPU 上计算 `dk/dv/dbeta/dg` 作为 golden，并与 `fla_npu.ops.ascendc.prepare_wy_repr_bwd` 的四个正式输出逐项 `ct.single` 比较。测试脚本路径为 `fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/test/test_stage0_golden.py`，该脚本只能作为 stage0 后的最终输出回归护栏，不能替代 stage0 临时中间产物 golden。stage0 测试默认输入为对称随机分布 `[-1, 1]`；如需缩小定位数值问题，可临时调整 `--input-scale`，但最终结论必须回到默认分布。
- stage0 以及后续每个 stage 的默认验证不跑完整 F1-F22、L1-L12；只有用户明确要求“全量验证”时，才运行 `test_stage0_debug_golden.py --suite all` 或对应正式回归全量。日常精度修复和阶段验收优先选择典型 case 集，必须覆盖 GVA 1:2、1:3，`chunk_size=64/128`，`V=128/256`，以及变长尾块；当前 stage0 debug 典型集可使用 `--cases F1,F3,F19,F20,L9`，其中 F19 覆盖 1:2/V256/chunk64，F20 覆盖 1:3/V256/chunk64，L9 覆盖 V256/chunk128/变长尾块，F1/F3 覆盖 V128 的 chunk64/chunk128。若怀疑执行卡死，先用同一典型集加 `--kernel-only` 确认 kernel 返回；只有典型集失败或用户要求时再扩大到全量。正式四输出入口接入后，再用 `test_stage0_golden.py` 按相同“典型优先、明确要求才全量”的规则验证正式输出。`--input-scale` 表示 `[-scale, scale]` 的半宽，默认 `1.0`。
- 用户说“测性能”时，`prepare_wy_repr_bwd` 默认只测 4 个 stage0 性能 case：`B=1, H=32, T=65536, K=128`，变长 64 个 sequence（即 `cu_seqlens` 长度为 65），`V=128/256` 与 `chunk_size=64/128` 两两组合；不要自动跑精度全量或其它大 case。若用户未额外指定 dtype，默认按当前调测主路径使用 `kType=bf16, gType=fp32`。性能结论以 `msprof` 为准，采集时打开 `--aic-metrics=PipeUtilization`，同时记录 operator task time 和 AIC/AIV pipe 指标；Python 同步耗时只作为辅助日志，不作为性能结论。性能数据汇报单位统一使用 `us`，不要把 `msprof` 导出的 `us` 换成 `ms`。
- stage0 性能脚本路径为 `fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/test/profile_stage0_perf.py`。运行前需确认已安装当前 scoped run 包，并加载 CANN/custom OPP 环境；从源码树调用时设置 `PYTHONPATH=<repo_root>/torch_custom/fla_npu:<repo_root>:$PYTHONPATH`、`FLA_NPU_OPP_PATH=<当前 wheel 的 opp 根目录>`、`TORCH_EXTENSIONS_DIR=/tmp/torch_ext_prepare_wy_repr_bwd_stage0`、`ASCEND_RT_VISIBLE_DEVICES=<device_id>`。
- stage0 性能采集命令模板：

```bash
export PREPARE_WY_STAGE0_PERF_WARMUP=0
export PREPARE_WY_STAGE0_PERF_REPEAT=1
# 可选：只跑单个或部分 case，例如 P1_V128_C64,P4_V256_C128
export PREPARE_WY_STAGE0_PERF_CASES=
msprof \
  --output=<profile_output_dir> \
  --task-time=l2 \
  --ai-core=on \
  --aic-mode=task-based \
  --aic-metrics=PipeUtilization \
  --application=<repo_root>/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/test/profile_stage0_perf.py
```

- stage0 性能结果读取 `mindstudio_profiler_output/op_summary_*.csv`，过滤 `OP Type == PrepareWyReprBwd`；按执行顺序映射 `P1_V128_C64`、`P2_V128_C128`、`P3_V256_C64`、`P4_V256_C128`。汇报字段至少包含 `Task Duration(us)`、`aicore_time(us)`、`aic_mac_time(us)`、`aic_scalar_time(us)`、`aic_mte1_time(us)`、`aic_mte2_time(us)`、`aic_fixpipe_time(us)`、`aiv_time(us)`、`aiv_vec_time(us)`、`aiv_scalar_time(us)`、`aiv_mte2_time(us)`、`aiv_mte3_time(us)` 和 `cube_utilization(%)`；需要核对 kernel task 时可辅助查看 `task_time_*.csv`。各 pipe 耗时存在 overlap，不能相加得到 task duration。
- A2/A3 调测 `prepare_wy_repr_bwd` 时，第一步允许使用仓库一键全量编包确认全工程生成链路：`bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu -j16`。后续定位优先只编相关三个算子的 run 包：`bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu --ops=prepare_wy_repr_bwd,prepare_wy_repr_bwd_da,prepare_wy_repr_bwd_full -j16`。
- 三算子 scoped run 包安装到当前 `fla_npu` wheel 内部 OPP 时，使用 `bash build/fla-npu-fla_npu_linux-aarch64.run --install --quiet`，不要带 `--install-path`。安装 scoped 包会替换当前 wheel 的共享 opapi/tiling/proto 库，因此该环境只保证包内三个算子可用于 stage0 golden 验证；验证前需要重新启动 Python 进程并 source CANN 环境。

### prepare_wy_repr_bwd 已踩坑

- 修改 `PrepareWyReprBwd` 的 op schema、输出数量或 `aclnnPrepareWyReprBwdGetWorkspaceSize` 参数表后，必须同步 `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` 的 `_GET_WORKSPACE_ARGTYPES`、Python wrapper 入参顺序和输出申请；否则 ctypes 会按旧 ABI 传参，常见表现是 `GetWorkspaceSize` 返回成功但 `workspaceSize=0`、`executor=nullptr`，随后 launch 报错或进程崩溃。
- stage 临时 debug 输出 shape 必须和 kernel 写回 layout 完全一致。fixed case 下 debug 输出第 0 维是 `B * chunk_num_per_b`，varlen case 下是 `chunk_indices_len / 2`；不要只按 `T / chunk_size` 申请，否则 batch 大于 1 时会出现 Python 索引越界或 kernel 写 debug GM 越界。
- 如果该算子还需要走 legacy C++/dispatcher 路径，必须同步检查 `torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuOpApi.cpp`、PyTorch schema/yaml 和导出注册。当前 stage 临时 debug 入口若只通过 `fla_npu.ops.ascendc` ctypes 调用，可以暂不接入 `op_plugin`，但验证说明中必须明确当前测试路径，避免误以为 legacy 路径也已适配。
- ABI 或输出数量变更后，不要只依赖增量编译结论。遇到 ABI 异常时需要检查 `libcust_opapi.so` 里的 `l0op::PrepareWyReprBwd` 符号签名是否与当前源码一致；若符号仍是旧签名，先清理 `build/` 或至少强制重编 host opapi，再重新生成 scoped run 包。
- 安装 run 包后必须确认实际加载的动态库和符号。建议检查 run 包、安装后 wheel OPP 中的 `libcust_opapi.so`：`nm -D -C <libcust_opapi.so> | grep 'l0op::PrepareWyReprBwd'`，确认输出参数数量与当前 aclnn/op_host 源码一致；必要时用 `strings <libcust_opapi.so> | grep PrepareWyReprBwd` 辅助确认。
- `bash build/fla-npu-fla_npu_linux-aarch64.run --install --quiet` 默认把 scoped run 包合入当前 Python 环境的 `site-packages/fla_npu/opp`。如果随后用源码树 `PYTHONPATH=<repo>/torch_custom/fla_npu` 调 `fla_npu.ops.ascendc`，需要确认源码 wrapper 实际解析到的 OPP 路径；源码 wrapper 调试时应显式设置 `FLA_NPU_OPP_PATH=<当前已安装 wheel 的 opp 根目录>`，或重新构建并安装包含最新 Python wrapper 的 wheel 后再验证。
- `fla_npu.ops.ascendc` 的 OPP 解析顺序会优先看 `FLA_NPU_OPP_PATH`，再看包内 embedded `opp`，然后看 `ASCEND_CUSTOM_OPP_PATH` / `ASCEND_OPP_PATH`。验证 ABI 问题时先在 Python 进程中打印 `fla_npu.__file__`、解析到的 vendor 目录和已加载 `CDLL._name`，避免源码 wrapper 与旧 OPP 库混用。
- `CrossCoreFlagWithReverse` 不会在每次 `Set/Wait` 后自动切到 `reverseId`；它只是在连续 `Set` 或连续 `Wait` 达到 `REVERSE_DEPTH` 后才用 `reverseId` 做反向握手防止 flag 溢出。涉及该类信号的方案变更时，必须在方案说明中列出每侧 `Set/Wait` 次数、方向和同步粒度，并询问用户确认。
- `KERNEL_TYPE_MIX_AIC_1_2` 下 Vector 存在 subBlock 维度。任何跨 AIC/AIV 的 stage/line 同步都要先明确同步粒度是“每个 value head 一次”还是“每个 subBlock 一次”。
- 每个 AI Core 都有自己独立的 UB，不同核之间不会互相踩踏 UB；一个 vec 的 UB 操作一定不会影响另一个核的 UB。排查精度问题时不要再做“关闭另一个核/vec/subBlock 观察 UB 是否互相影响”这类实验。优先检查核内 TBuf/LocalTensor 空间分配是否重叠、同一核内事件和 buffer 生命周期是否闭环，以及 workspace/debug GM 地址映射是否踩踏。
- 核内需要长期持有且彼此独立的 EventID 时必须用 `AllocEventID<HardEvent>()` 占用事件池，并在该事件生命周期结束时用同一个 `HardEvent` 模板调用 `ReleaseEventID<HardEvent>()` 释放；`AllocEventID` 和 `ReleaseEventID` 必须成对使用，不允许申请后遗漏释放、跨 `HardEvent` 模板释放，或把只查询空闲 ID 的 `FetchEventID` 当作持久事件分配接口。`FetchEventID` 不会占用事件池，连续调用可能得到同一个 ID，不能用于 ping/pong 或多 buffer 的持久事件组。
- Cube 侧手写 L0A/L0B 双缓冲时，`InitPipeFlags` 里预置的 `MTE1_MTE2`、`M_MTE1`、`FIX_M` 事件必须在 kernel 结束前被消费掉，避免同一 Python 进程下下一次 op 调用再次 `SetFlag` 已置位事件。stage0 已验证需要在每个 chunk task 结束后对全部 L0A/L0B free 事件做一次 `WaitFlag` 后重新 `SetFlag`，并在 `ProcessImpl` 退出前只 `WaitFlag` 不再 `SetFlag`。
- GVA 场景下 `KKT` 必须按 `HK` 去重生产：每个新 `hk` 只计算一次，debug/golden 按 `HK` 维度比对；同一 `hk` 组内后续 `hv` 不再重复做 KKT GEMM。KKT cache 不能只有每 core 一块，否则 GVA 1:3 的 `hv2/hv3` 双 head 窗口会在 `hv2` 仍需 `hk0` 时被 `hv3` 的 `hk1` 覆盖；应使用两块 HK-scoped KKT cache，并把每个 head workspace slot 映射到对应 `kktSlotForSlot_[slot]`。stage0 已验证 KKT 尾部需要同时闭环 L0C/fixpipe 和本次 KKT 使用的 L0A/L0B free 事件：`copyL0CToGm(KKT)` 后等待并重新设置 `FIX_M`，再等待并重新设置对应 `M_MTE1`，否则多 head 或多 chunk 下可能出现 `L0A/L0B read/write conflict`。
- `A/A.T`、`KKT` 的 L1/UB 分块驻留、`beta/g/exp(g)` 的 UB 复用、`Kbg/Kbeta/Vb` 的 L1 驻留都只是后续优化点；本轮只实现 K resident，不扩展其它 resident 方案。
- `V_DIM=256` 的 `Dvb` 会在 K 方向拆成多个 L0 MMAD 分片。stage0 已验证如果每个 K 分片都用 `unitFlag=0b11`，L9 类 `V=256/chunkSize=128/varlen` 用例会在 cube 侧卡死；必须按 block 级 unit flag 语义处理中间分片和最后分片。
- Vector 侧同一 V pipe 内也要显式保护 RAW 顺序：`CopyInRows` 将 GM 数据搬入 UB 并 cast 到 fp32 目标 tensor 后，如果后续马上用 `Adds/Mul/Brcb` 消费该目标 tensor，需要在消费前插入 `PipeBarrier<PIPE_V>()`。stage0 已验证 Vb 路径在 `CopyInRows(beta)` 后缺少消费前 V pipe barrier 会导致 L9 类 V=256/varlen 用例出现大量 `Vb` 精度失败；纯 beta broadcast 不需要先 `Adds(..., 0.0f)` 复制到 scale tensor，可直接 `Brcb(..., betaFp32Tensor_, ...)`。
- `fla_npu.ops.ascendc` ctypes runtime 会用 `_RECENT_LAUNCH_STORAGE` 保活最近 launch 的输出和 workspace。全量同进程跑 stage debug/正式回归时，每次 `torch.npu.synchronize()` 后应清理该保活队列，避免旧 case 的大 debug 输出继续占用 NPU 显存；这种 OOM 属于测试 wrapper 生命周期问题，不能归类为 kernel 超时。
- stage0 临时 debug op 当前只写 6 个中间产物输出，不写 `dk/dv/dbeta/dg`。在 `prepare_wy_repr_bwd_stage0_debug` 的 Python ctypes wrapper 中，这 4 个未写最终输出可使用 1 元素占位 tensor，避免 L12 这类大 shape 在 kernel launch 前因调试输出分配 OOM；`prepare_wy_repr_bwd_full` 等正式四输出 wrapper 仍必须使用真实 shape，不能套用该占位策略。

## 构建命令

先准备环境：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install -r requirements.txt
python scripts/check_npu_env.py --build-only
```

推荐的一体化 wheel 构建：

```sh
FLA_NPU_SOC=ascend910b python -m pip wheel --no-build-isolation --no-deps . -w dist
```

常用 SOC：

- A2：`ascend910b`
- A3：`ascend910_93`
- A5：`ascend950`

本地增量调试可以使用：

```sh
FLA_NPU_SOC=ascend910b FLA_NPU_INCREMENTAL_BUILD=1 python -m pip wheel --no-build-isolation --no-deps . -w dist
```

只构建部分算子用于定位时使用 `FLA_NPU_OPS`，不要和 `FLA_NPU_INCREMENTAL_BUILD` 同时使用：

```sh
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_fwd_o python -m pip wheel --no-build-isolation --no-deps . -w dist
```

分开编 OPP run 包和 `torch_custom` 适配时：

```sh
bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu
cd torch_custom/fla_npu
bash build.sh
```

一体化 wheel 和 `torch_custom/fla_npu` 单独编出的 wheel 使用相同 pip 项目名
`flash-linear-attention-npu`，Python import 名均为 `fla_npu`。单独编出的
standalone wheel 只提供 Python 适配和 OPP 骨架；配套 run 包使用 `--full` 或
`--install` 安装时，会把 run 包里的 `packages/vendors/fla_npu_transformer`
合并覆盖到当前 Python 环境的 `site-packages/fla_npu/opp/vendors/fla_npu_transformer`。

## 安装和验证

安装一体化 wheel：

```sh
python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
python scripts/check_packaged_wheel_api.py
```

运行 GDN 单算子测试：

```sh
cd torch_custom/fla_npu/test
bash test.sh --device 0
bash test.sh --device 0 --op causal_conv1d
```

`test.sh` 只覆盖已接入脚本的 GDN 算子。未纳入 `test.sh` 的算子按对应测试脚本直接运行，例如：

```sh
cd torch_custom/fla_npu/test
python3 test_npu_<op>.py
```

端到端 Example/ST：

```sh
python examples/flash_gated_delta_rule.py
python3 ci/run_example_st_cases.py --device 0 --cases-file ci/example_st_cases.json
```

本地复现 CI 主流程：

```sh
CI_MODE=quick CI_SOC=ascend910b CI_OPS=<op_name> bash ci/run_checks.sh
CI_MODE=full bash ci/run_checks.sh
```

如果缺少 CANN、NPU、`torch_npu`、`torchnpugen` 或 `triton-ascend`，不要伪造验证结论；在回复中说明未执行的命令和阻塞原因。

## 测试要求

- 修改参数校验、shape、dtype、layout、range、平台差异或预留参数语义时，补充反向测试和边界测试。
- 修改 Kernel 时至少覆盖对应单算子测试；涉及 GDN 端到端链路时跑 Example/ST。
- 修改 ABI、公共模块或共享路径时扩大回归范围，至少说明影响到的算子。
- 精度失败不能通过收窄输入 range、跳过 case、降低覆盖强度或放宽阈值来制造通过结论；应先定位误差来源，再修 kernel、标杆或后处理语义。
- 性能结论以合适的 profiling/CI 结果为准，不用 Python wall time 直接下结论。

## 构建产物与提交规范

不要提交构建、安装、调测和性能分析产物。重点检查：

- `build/`、`build_out/`、`output/`、`dist/`、`.ci-cache/`、`third_party/`
- `torch_custom/fla_npu/build/`、`torch_custom/fla_npu/dist/`、`torch_custom/fla_npu/torch_npu/`
- `torch_custom/fla_npu/test/test_output/`、`torch_custom/fla_npu/test/data/`
- `__pycache__/`、`*.pyc`
- `.tmp*`、`outputs/`、`PROF_*`、`OPPROF_*`、`extra-info`

提交前至少运行：

```sh
git status --short
git diff --check
```

## PR 和 CI

- PR 描述使用 `.github/pull_request_template.md`，不要自创栏目替代模板。
- PR 应关联 Issue，或在模板中说明无需 Issue 的原因。
- NPU CI 不会在每次 push 后自动跑；需要仓库 Admin 在 Actions 手动触发，或在 PR 评论 `/run-npu-ci quick` / `/run-npu-ci full`。
- 当前 head commit 需要通过 `NPU CI / 手动验证` 和 `NPU CI / 精度检查`，并满足 2 个 approval。
- push 新 commit 后旧 commit 的 CI 结果失效，需要重新触发。

## 给 AI 的最后检查

结束任务前确认：

- 改动是否只覆盖本次任务需要的文件。
- 文档、报错、返回码、schema、Python 导出和测试是否与代码语义一致。
- 是否遗漏对应 SOC、layout、dtype、varlen/dense、边界 case。
- 是否有未跟踪生成物或敏感信息混入。
- 是否清楚说明已执行和未执行的验证。
