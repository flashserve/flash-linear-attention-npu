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
- workspace 大小按实际 `blockDim * workspaceCoreSize` 申请；每个 AIC core 保留四个 per-hv GM workspace slot，两个相邻 slot 组成一个 2-head 窗口，窗口按 `0/1 -> 2/3 -> 0/1` 轮转。若某个中间量按 HK 作用域共享（如 `KKT`），其 cache buffer 计入 `workspaceCoreSize` 但不计入 per-hv `workspaceSlotSize`。
- `KKT = K @ K.T` 只依赖 `HK`。正式 workspace 中 KKT 应作为当前 2-head 窗口内的 HK-scoped cache，由当前 head slot 记录的 `kktSlotForSlot_[slot]` 映射读取，不放进每个 per-hv workspace slot 私有布局里重复生产。当前 4-slot 方案每个 core 保留四块 KKT cache，`workspaceCoreSize = 4 * workspaceSlotSize + 4 * mBytes`；同一窗口内相同 `hk` 复用 KKT cache，跨窗口不共享 KKT cache，避免下一窗口覆盖上一窗口 final output 仍在读取的 cache。
- Cube 侧 K 需要独立 L1 resident ping/pang buffer。遇到新 `hk` 时才把 `K[BT,128]` 从 GM 搬入当前 `curKResidentSlot_`，同一 GVA group 内后续 `hv` 复用 `cachedKResidentSlot_`，不得用 `hv & 1` 推导。K resident ping/pang 需要独立核内事件组，不能复用 L1A/L1B operand ping/pong 的 EventID。`KKT=K@K.T` 使用同一块 resident K 同时构造 row-major `K` 视图和 column-major `K.T` 视图；resident free flag 必须等 K 和 K.T 两次 L1->L0 搬运都发出后再释放。
- Cube 侧 `Dvb = A^T @ du` 的 tile 配置必须区分模板 `V_DIM`：`V_DIM=128` 保持 L1/L0 的 MNK 为 `128,128,128`；`V_DIM=256` 时 L1 tile 使用 MNK `128,256,128`，L0 tile 使用 MNK `128,256,64`，在 K 方向循环完成两次 MMAD 累加后再统一 Fixpipe 写出。不得把 `V_DIM=256` 拆成 N=128 的两次输出，也不得在 K 循环内提前 Fixpipe。K 分片 MMAD 的 `unitFlag` 必须区分中间/最终分片：非最后 K 分片使用 `0b10`，最后 K 分片使用 `0b11`。
- Cube 侧使用 `TileMmadTla` 时，实际 `M/N/K` 必须由当前 `L0A/L0B/L0C` tensor 的 layout/originShape 表达，调用点只传 `initC/unitFlag`，不得退回手传显式 `m/n/k` 的写法。
- A2 上 Cube 侧 L0C->GM 的 Fixpipe copyout 对齐原 `prepare_wy_repr_bwd_full` block 级写法：`Dkbg/Dvb/KKT` 写 GM 时完整 tile、尾块和 small-M 都使用 `unitFlag=0b11`。stage0 已验证尾块 copyout 传 `0` 会导致后续 cube 任务卡死。
- Cube 侧同一个 L0C 计算结果只允许执行一次 Fixpipe copyout，不能从同一份 L0C 连续 Fixpipe 到多个目的地。正式路径只写 workspace 或最终输出；若临时定位需要额外中间结果，必须先说明方案并获得用户确认，验证结束后删除临时输出通路。
- Cube/Vector 分核按 `B * chunkNum` 或变长 `chunk_indices` 任务数分配，`HV` 在 core 内顺序遍历；不得用 `hv & 1` 或 `hv % 2` 推导 workspace slot。
- 新 `prepare_wy_repr_bwd` 的核间同步只允许使用 `Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_*>(...)` 和 `Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_*>(...)`。旧 `prepare_wy_repr_bwd_full` / `prepare_wy_repr_bwd_da` 中不带 reverse 的 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 只能作为历史实现阅读，不作为新算子同步写法参考。
- 新 `prepare_wy_repr_bwd` 的 Cube/Vector 双向同步使用方向独立的 `CrossCoreFlagWithReverse`：Vector->Cube 和 Cube->Vector 各一套 ready/reverse flag；四个 GM workspace slot 不再额外拆 CrossCoreFlag。当前跨核同步只保留 6 个数据 ready 点，final output 完成后不再发送额外 V->C slot free 信号。
- 新 `prepare_wy_repr_bwd` 的核内同步必须按实际 ping/pong buffer 拆事件组，不允许多个 ping/pong buffer 共用同一个 EventID：Vector input ping/pong 分别维护 `MTE2_V` 与 `V_MTE2`，Vector output ping/pong 分别维护 `V_MTE3` 与 `MTE3_V`；Cube 侧 L1A/L1B/L0A/L0B 等双缓冲事件也必须带 slot。
- 当前 A2/A3 Ascend C 核内 EventID 只支持 `0~7`。新增 L1 resident 或其它长期复用 buffer 时，不能假设还有更大的 EventID 可用；若 `0~7` 不够，需要先分析现有 EventID 的生产/消费生命周期，在可证明安全的前提下复用现有 ID，并说明复用理由后再改代码。
- `prepare_wy_repr_bwd` Cube 侧 L1 resident buffer 必须在矩阵复用区独立新分，不能借用通用 scratch ping-pong 空间并延长其生命周期充当 resident。EventID 不够时只允许复用事件编号，不允许复用 resident 空间；当前布局为 `2 * 64KB` 通用 scratch ping/pong 加 `K/DW/DU/A` 等独立 resident buffer，按 512KB L1 总量校验。
- `prepare_wy_repr_bwd` Cube 侧当前已实现 `K/DW/DU/A` 四类独立 L1 resident：`K` 按当前窗口内的 HK/GVA group 驻留复用，供 `KKT` 和 stage3 `Dkb=D.T@K` 使用；每个 GM workspace slot 需要记录当次 head 对应的 `kResidentSlotForSlot_[slot]`，不得用 `hv & 1` 推导。`DW` 在 stage0 `Dkbg` 搬入并延续到 stage1 `DA1`，`DU` 在 stage0 `Dvb` 搬入并延续到 stage1 `DA2`。`DW/DU` resident 生命周期按 `residentSlot = curSlot_ & 1` 管理，释放点分别在 stage1 对应 L1->L0 搬运发出后，不能在 stage0 提前释放。
- `prepare_wy_repr_bwd` Cube 侧 `A` resident 按 `residentSlot = curSlot_ & 1` 管理：stage0 将当前 head 的 `A` 搬入 `aResident[residentSlot]`，stage0 的 `Dkbg/Dvb`、stage2 的 `DA5/DA6_T` 后续都复用同一份 L1 resident，不再把 `A` 放进通用 scratch 并跨阶段延长 scratch 生命周期。`A` resident 的 stage0 搬入事件临时复用对应 resident slot 的 scratch EventID，必须在 stage0 `Dvb` 最后一次 L1->L0A 搬运发出后恢复 free 状态，保证后续 stage 可以继续把同一 EventID 用于通用 scratch。
- `prepare_wy_repr_bwd` stage2 已确认采用 `DA6_T` 物理布局：数学公式仍是 `DA6=A.T@DA5`、`D=(-DA6*exp(min(g_i-g_j,0))*tril).T`，但 Cube 在 workspace `da6Offset` 写入 `DA6_T=DA6.T=DA5.T@A`，Vector 连续行读取后计算 `D[p,q]=-DA6_T[p,q]*exp(min(g[q]-g[p],0))*upper_tri(q>p)`，避免 Vector 侧转置散写。正式 golden 使用已有 NPU `prepare_wy_repr_bwd_da + prepare_wy_repr_bwd_full`，避免手写公式遗漏原算子的 g-factor 稳定化、尾块或 GVA 细节。
- `prepare_wy_repr_bwd` Cube 侧 stage2 当前只新增 `A` resident 复用，不新增 `DA4/DA5` resident：`DA5=DA4@A.T` 时 scratch 只承接 `DA4`，`A.T` 从 `A` resident 取视图；`DA6_T=DA5.T@A` 时 scratch 只承接 `DA5.T`，`A` 从同一份 `A` resident 取视图。stage2 内 `DA5` 仍由 Fixpipe 写入 GM 后被同一个 Cube 用 MTE2 读回做 `DA6_T`，`copyL0CToGm(DA5)` 后必须按当前窗口内两个 `residentSlot` 使用两套 `SetFlag<HardEvent::FIX_MTE2>`，并在后续 `DA6_T` 子阶段读取同一个 GM slot 的 `DA5.T` 前 `WaitFlag<HardEvent::FIX_MTE2>`；不能只用一套 id 后立刻 wait，否则当前窗口两个 head slot 会退化成纯串行，也不能只依赖 `FIX_M`，否则 `DA6_T` 可能读到旧值。
- `prepare_wy_repr_bwd` Vector 侧 UB input 分两套 ping/pong：`K/V` 等矩阵 tile 使用原 16KB matrix input ping/pong；`beta/g` 行向量使用独立 beta-g input ping/pong，每块只分配 `align32(chunk_size * 4)` 字节。两套 input ping/pong 必须分别维护当前 slot 类变量和 `MTE2_V`/`V_MTE2` 事件组，不能复用同一套 input buffer 或 EventID。
- `prepare_wy_repr_bwd` Vector 侧 matrix input 和 beta-g input 的 ping/pong 用于同一个 row tile 内多个 GM 输入的搬入流水，不做跨 row 预取。`CopyInRows`/`CopyInBetaGRows` 必须发起单个 GM 输入搬入后返回本次使用的 input slot，并立即切换对应 ping/pong；`CastInputRows`/`CastBetaGInputRows` 必须显式传入该 slot，等待该 slot 的 `MTE2_V`、转换到 FP32 计算区并释放该 slot。调用点应尽量在同一 row tile 内先发起两路不相关输入搬入，例如 `K/Dkb`、`Dvb/V`、`D/KKT`、`DA1/DA2`、`beta/g`，再按消费顺序 cast；当一个 slot cast 后释放，可以在后续 VEC 计算前提前发起第三路输入搬入。不要把 helper 写回 `CopyIn -> Wait -> Cast` 的单输入纯串行形态。
- `prepare_wy_repr_bwd` Vector 侧 HardEvent 生命周期覆盖整个 kernel：在 `Init()` 中按 matrix input、beta-g input、output 三类 UB ping/pong 统一 `AllocEventID` 并初始化 free flag；各 stage 只复用这些事件做当前 buffer 的生产/消费闭环，不能在 stage 入口或出口重新分配、重新初始化或释放。整个 `Process()` 全部任务结束后，再统一等待最终 free flag 并用匹配的 `HardEvent` 模板 `ReleaseEventID`。
- `prepare_wy_repr_bwd` Vector 侧当前已实现 `beta/g` UB resident，并且必须按 `residentSlot = curSlot_ & 1` 存两套：`ProcessVectorTask` 为当前 resident slot 一次搬入完整 chunk 的 `betaAll[residentSlot]` 和 `gRawAll[residentSlot]`，再生成 `gExpAll[residentSlot]`；后续 `ProcessDTask` 从 `gRawAll[residentSlot]` 取 raw `g_j-g_i`，`ProcessOutputTask` 从 `betaAll[residentSlot]` 和 `gExpAll[residentSlot]` 复用，不再重复 GM 搬入或重复 `Exp`。不得退回单块 `betaAll/gAll` 覆盖当前窗口两个 head，也不得在每个 `kVecRow/vVecRow` 内重复搬 `beta/g`。
- `prepare_wy_repr_bwd` Vector 侧 FP32 矩阵计算工作区命名为 `calcFp32A`/`calcFp32B`/`calcFp32C`，每块固定为 `2 * 16KB = 32KB`，对应一个固定 16KB half/bfloat16 matrix tile cast 到 FP32 后的大小；它们不是固定输入/输出语义，不要再命名为 `inputFp32`/`outputFp32`，也不要按 `max(kVecRow, vVecRow) * max(K, V)` 分配。`ProcessOutputTask` 的 K 维行循环应优先把 `K`、`Dkb`、`Dkbg` 在单个 row tile 内各搬入/转换一次，通过三块 `calcFp32*` 复用 `K`、`Dkb*beta`、`Dkbg*exp(g)` 等中间值，避免为 `dbeta/dg/dk` 重复从 GM/workspace 搬同一 tile 或重复 cast。
- `prepare_wy_repr_bwd` Vector 侧热路径 accumulator 不要写成 `Duplicate(0) -> Add(acc, acc, first)` 的形式。只要能证明当前 row/line 的首项一定完整覆盖所有有效元素，应把首项直接写入 accumulator，例如 `WholeReduceSum(acc, ...)`、`Mul(acc, ...)` 或其它首项覆盖写，后续项再 `Add` 累加。类似地，若只是为了下一条 vector 指令把同一 tensor `Adds(dst, src, 0.0f)` 复制一遍，应优先使用该指令的独立 `src/dst` 形式直接写目标。不能套用到语义必需的清零场景，例如 `Select` 使用的 zero 常量、尾块无效区域 padding 清零，或无法证明首项一定覆盖的条件累加。
- `prepare_wy_repr_bwd` Vector 侧 helper 的模板参数和运行期开关必须表达真实分支；如果所有调用点都固定为同一个值，或某个参数只保留历史分支但当前算子没有使用，应及时删除并简化调用点。例如矩阵 `CopyInRows` 只走普通 `DataCopy` 时，不应继续保留 `USE_DATA_COPY_PAD=false` 这类恒定模板参数；只有 beta/g 尾块这类确实需要 `DataCopyPad` 的路径才保留对应接口差异。做这类优化时必须主动扫描同文件/同算子的相似冗余，包括恒定模板参数、恒定 dtype 分支、历史验证死状态、无用 scalar 计算和可合并的重复初始化循环，不能只改用户点名的一处。
- `prepare_wy_repr_bwd` Vector 侧涉及模板常量列维度时，直接使用 `K_DIM`、`V_DIM`、`CHUNK_SIZE` 表达元素数、repeat stride 和列循环边界，不要先赋值到 `curCol_` 这类类成员再使用。函数间传递应显式传递 `elements`，例如 `CopyOutRows(..., curRow_ * K_DIM/V_DIM/CHUNK_SIZE)`，避免隐藏依赖可由编译期常量表达的状态，也减少后续 stage 复用时的误用风险。`CopyInRows`/`CopyInBetaGRows` 只允许表达从哪个 GM tensor 搬到哪个原始 dtype UB tensor，不允许内部包含 cast；原始 dtype UB 到 FP32 计算 tensor 的转换应在调用点或专门转换函数中显式完成。`CopyOutRows` 只允许表达从哪个 UB tensor 搬到哪个 GM tensor，不允许内部包含 cast；FP32 到 `kType` 的转换应在调用点或专门转换函数中显式完成，再把转换后的 `LocalTensor<kType>` 传给 `CopyOutRows`。
- `prepare_wy_repr_bwd` Vector 侧跨核同步应收在对应 `Process*Task` 函数内部，但必须放在函数内部 row/line 循环之外：消费者函数在独立标量初始化后、真正读取 GM workspace 的循环前执行 `CrossCoreWaitFlagWithReverse`；生产函数在全部 row/line 循环完成后执行 `CrossCoreSetFlagWithReverse`。不要把跨核 `Wait/Set` 放进热循环，也不要散落在外层 `Process()` 的 workspace slot 调度里。
- `stage_0` 的 Cube 与 Vector 都只依赖原始输入，二者在 `stage_0` 内独立向 GM workspace 生产中间结果，不能在 `stage_0` 尾部互相 `WaitFlagWithReverse`。跨 stage 的 `CrossCoreWaitFlagWithReverse` 必须放在下一阶段消费者入口，例如 `cube stage_1` 读取 `Kbg/Vb` 前等待 Vector stage0 ready，Vector 后续读取 `DA1/DA2`、`DA6_T`、`Dkb/DK` 前等待 Cube 对应 stage ready。`Dkbg/Dvb/KKT` 由 Cube stage0 生产，不单独新增 stage0 C->V ready；最终输出等待 `Dkb/DK` ready 后读取它们，依赖严格 stage 顺序保证可见。
- `prepare_wy_repr_bwd` 正式 API 只保留 `dk/dv/dbeta/dg` 四输出，不再维护 `stage*_debug` Python wrapper、临时 debug 输出、stage 中间产物 golden 或 stage perf 脚本。后续若需要重新导出中间产物定位，必须先说明临时输出清单、shape、op schema/aclnn 参数表和删除计划，并询问用户确认；验证结束后必须删除临时输出通路。
- 正式四输出精度验收使用 `fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/test/test_final_golden.py`：actual 来自 `fla_npu.ops.ascendc.prepare_wy_repr_bwd`，golden 使用 NPU `prepare_wy_repr_bwd_da + prepare_wy_repr_bwd_full`，比较使用 `ct.single`，不要另写一套阈值或 bad-count 规则。这类脚本不能称作 ATK 对比；ATK 适合正式 API/全量泛化精度。
- 正式四输出 golden 的超时结论必须分清 `kernel` 超时和 `golden/compare` 超时。测试脚本应先单独执行待测 op 并 `torch.npu.synchronize()`，打印 `kernel_sync` 耗时；如果没有打印该阶段结果，才按 kernel 未返回/疑似 kernel 超时定位。如果 `kernel_sync` 已返回但后续超时，应按 NPU golden 生成或逐输出精度比对超时处理，不能误报为 kernel 超时。全量 run 期间如果日志超过 100 秒没有出现新的 case 行、`phase_time kernel_sync` 或结果行，可按最后一个未完成阶段判定卡死；最后日志已经进入某 case 但没有打印 `kernel_sync` 时，按该 case 的 kernel 阶段卡死处理。必要时先跑 `--kernel-only` 只确认 kernel 返回，再跑完整精度比对。
- 正式回归可保留最终四输出等价测试：先用 `fla_npu.ops.ascendc.prepare_wy_repr_bwd_da` 在 NPU 上计算 `dA`，再用 `fla_npu.ops.ascendc.prepare_wy_repr_bwd_full` 在 NPU 上计算 `dk/dv/dbeta/dg` 作为 golden，并与 `fla_npu.ops.ascendc.prepare_wy_repr_bwd` 的四个正式输出逐项 `ct.single` 比较。正式四输出入口接入后，按“典型优先、明确要求才全量”的规则验证正式输出。阶段测试默认输入为对称随机分布 `[-1, 1]`；如需缩小定位数值问题，可临时调整 `--input-scale`，但最终结论必须回到默认分布。
- 默认精度验证不跑完整 F1-F22、L1-L12；只有用户明确要求“全量验证”时，才运行 `test_final_golden.py --suite all`。日常精度修复和验收优先选择典型 case 集，必须覆盖 GVA 1:2、1:3，`chunk_size=64/128`，`V=128/256`，以及变长尾块；默认使用 `test_final_golden.py --suite smoke`，该典型集使用较小 shape 覆盖上述组合，避免正式四输出和 full golden 在长序列大用例上默认占用过多显存。若怀疑执行卡死，先用同一典型集加 `--kernel-only` 确认 kernel 返回；只有典型集失败或用户要求时再扩大到全量。最终 golden 脚本默认隐藏逐输出 `ct.single` 明细，只打印失败摘要和进度；需要查看原始 CT 报告时再加 `--verbose-ct`。`--input-scale` 表示 `[-scale, scale]` 的半宽，默认 `1.0`。
- 用户说“测性能”时，`prepare_wy_repr_bwd` 默认只测当前最新 stage 的 4 个性能 case：`B=1, H=32, T=65536, K=128`，变长 64 个 sequence（即 `cu_seqlens` 长度为 65），`V=128/256` 与 `chunk_size=64/128` 两两组合；不要自动跑精度全量或其它大 case。若用户未额外指定 dtype，默认按当前调测主路径使用 `kType=bf16, gType=fp32`。性能结论以 `msprof` 为准，同时记录 operator task time 和 AIC/AIV pipe 指标；Python 同步耗时只作为辅助日志，不作为性能结论。性能数据汇报单位统一使用 `us`，不要把 `msprof` 导出的 `us` 换成 `ms`。默认只跑一轮测量，即 profile 脚本使用 `--repeat 1` 或默认 repeat，不做三轮均值；只有用户明确要求多轮统计时才增加 repeat 并汇报均值或方差。
- 旧 stage 脚本已删除；若用户要求测当前性能，使用 `profile_final_perf.py`。运行前需确认已安装当前 scoped run 包，并加载 CANN/custom OPP 环境；从源码树调用时设置 `PYTHONPATH=<repo_root>/torch_custom/fla_npu:<repo_root>:$PYTHONPATH`、`FLA_NPU_OPP_PATH=<当前 wheel 的 opp 根目录>`、`TORCH_EXTENSIONS_DIR=/tmp/torch_ext_prepare_wy_repr_bwd_final_perf`、`ASCEND_RT_VISIBLE_DEVICES=<device_id>`。
- A2/A3 上从源码树 `PYTHONPATH=<repo_root>/torch_custom/fla_npu:<repo_root>` 调用 `fla_npu` 时，必须显式设置 `FLA_NPU_OPP_PATH=<已安装 wheel 的 opp 根目录>`，例如当前调测环境为 `/data/miniconda3/envs/zsl/lib/python3.11/site-packages/fla_npu/opp`。否则 `_runtime` 会优先使用源码树内 `torch_custom/fla_npu/fla_npu/opp` 的旧 OPP，表现为已安装 run 包对象已更新但 runtime 日志 `bin path` 仍指向源码树 OPP，导致一直执行 stale kernel。
- stage 性能采集命令模板沿用 `prepare_wy_repr_bwd_da/test/run_da.sh` 的方式，使用简单 `msprof --output <dir> python3 <script>`。当前 A2 环境已验证：额外手动添加 `--task-time=l2 --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization` 可能导致只采到 host/API 数据，`device_0/data` 为空，无法导出 `op_summary_*.csv` 和 pipe 字段；因此不要在该环境中给 stage 性能采集叠加这些选项。默认导出的 `op_summary_*.csv` 已包含 `aic_*`、`aiv_*` 和 `cube_utilization(%)` 字段。

```bash
# 具体 cases 参数以当前最新 stage 的 profile 脚本为准；默认一轮测量，不额外加 repeat。
msprof \
  --output=<profile_output_dir> \
  python3 <repo_root>/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/test/profile_final_perf.py --mode both --repeat 1
```

- A2 流水可视化采集使用 simulator，不等同于常规性能结论。采集前需要确保当前算子的 Ascend C 编译选项包含 `-g`，并使用只调用当前算子的全 0 fixed shape 脚本：`B=1,H=4,T=2048,K=128,V=128,chunk_size=64`，脚本路径为 `fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/test/sim_zero_prepare_wy_repr_bwd.py`。脚本内不要引入当前算子以外的 NPU 算子或额外 NPU API：全 0 输入必须先在 CPU 上生成，只在调用 `fla_npu.ops.ascendc.prepare_wy_repr_bwd(...)` 时对输入使用 `.npu()`；不要在脚本里添加 `torch.npu.synchronize()`、`torch.npu.empty_cache()` 或其它 NPU op。运行前追加 simulator 动态库路径：`export LD_LIBRARY_PATH=<CANN包路径>/latest/tools/simulator/Ascend910B1/lib/:$LD_LIBRARY_PATH`，进入脚本目录后执行 `msprof op simulator python sim_zero_prepare_wy_repr_bwd.py`。simulator 流水采集可能很慢，运行中每 5 分钟检查一次即可，不按普通 kernel “日志 100 秒没更新大概率卡死”的规则判断；除非进程退出报错、用户要求停止，或明确出现重复错误日志，否则等待其自然完成。采集完成后在 profiler 输出目录中查找 `visualize_data.bin`，必须用 `scp` 拷回本地保存；debug 版流水文件可能达到数 GB，`scp` 传输也按慢任务处理，每 5 分钟检查一次本地文件大小或传输状态即可，不要因为几十秒无输出就中断或改用其它传输方式。本地文件与远端文件 `sha256sum` 校验一致后，删除服务器上本次对应的 `OPPROF_*` 流水目录，避免远端堆积大文件。若 `msprof op simulator` 失败、找不到 `visualize_data.bin`、动态库路径不明确或产物结构不符合预期，必须停下来询问用户，不要自行切换其它 simulator 参数、其它 shape、其它输入分布或其它采集方式；若出现 `Operation not permitted`，优先确认 CANN 包路径/版本是否正确。simulator 临时添加的 `-g` 只能用于流水取行号，做精度或性能验证前必须从 op_host CMake 编译选项中移除并重新编包。
- stage 性能结果读取 `mindstudio_profiler_output/op_summary_*.csv`；最终四输出性能对比过滤 `OP Type == PrepareWyReprBwd/PrepareWyReprBwdDa/PrepareWyReprBwdFull`，按脚本执行顺序映射 `P1_V128_C64`、`P2_V128_C128`、`P3_V256_C64`、`P4_V256_C128`。新算子耗时取 `PrepareWyReprBwd` 的 `Task Duration(us)`，原始基线取同一 case 下 `PrepareWyReprBwdDa + PrepareWyReprBwdFull` 的 `Task Duration(us)` 之和，优化比例为 `(baseline - fused) / baseline`。汇报字段至少包含 `Task Duration(us)`、`aicore_time(us)`、`aic_mac_time(us)`、`aic_scalar_time(us)`、`aic_mte1_time(us)`、`aic_mte2_time(us)`、`aic_fixpipe_time(us)`、`aiv_time(us)`、`aiv_vec_time(us)`、`aiv_scalar_time(us)`、`aiv_mte2_time(us)`、`aiv_mte3_time(us)` 和 `cube_utilization(%)`；需要核对 kernel task 时可辅助查看 `task_time_*.csv`。各 pipe 耗时存在 overlap，不能相加得到 task duration。
- A2/A3 调测 `prepare_wy_repr_bwd` 时，编包并发不允许低于 `-j64`，包括正式包、临时 `-g` 流水包和三算子 scoped run 包；不要因 `-g` 或日志较多主动降并发。第一步允许使用仓库一键全量编包确认全工程生成链路：`bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu -j64`。后续定位优先只编相关三个算子的 run 包：`bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu --ops=prepare_wy_repr_bwd,prepare_wy_repr_bwd_da,prepare_wy_repr_bwd_full -j64`。
- 三算子 scoped run 包安装到当前 `fla_npu` wheel 内部 OPP 时，使用 `bash build/fla-npu-fla_npu_linux-aarch64.run --install --quiet`，不要带 `--install-path`。安装 scoped 包会替换当前 wheel 的共享 opapi/tiling/proto 库，因此该环境只保证包内三个算子可用于当前阶段 golden 验证；验证前需要重新启动 Python 进程并 source CANN 环境。

### prepare_wy_repr_bwd 已踩坑

- 修改 `PrepareWyReprBwd` 的 op schema、输出数量或 `aclnnPrepareWyReprBwdGetWorkspaceSize` 参数表后，必须同步 `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` 的 `_GET_WORKSPACE_ARGTYPES`、Python wrapper 入参顺序和输出申请；否则 ctypes 会按旧 ABI 传参，常见表现是 `GetWorkspaceSize` 返回成功但 `workspaceSize=0`、`executor=nullptr`，随后 launch 报错或进程崩溃。
- 如果该算子还需要走 legacy C++/dispatcher 路径，必须同步检查 `torch_custom/fla_npu/op_plugin/ops/opapi/FLANpuOpApi.cpp`、PyTorch schema/yaml 和导出注册。当前正式入口优先通过 `fla_npu.ops.ascendc` ctypes 调用；验证说明中必须明确当前测试路径，避免误以为 legacy 路径也已适配。
- ABI 或输出数量变更后，不要只依赖增量编译结论。遇到 ABI 异常时需要检查 `libcust_opapi.so` 里的 `l0op::PrepareWyReprBwd` 符号签名是否与当前源码一致；若符号仍是旧签名，先清理 `build/` 或至少强制重编 host opapi，再重新生成 scoped run 包。
- 安装 run 包后必须确认实际加载的动态库和符号。建议检查 run 包、安装后 wheel OPP 中的 `libcust_opapi.so`：`nm -D -C <libcust_opapi.so> | grep 'l0op::PrepareWyReprBwd'`，确认输出参数数量与当前 aclnn/op_host 源码一致；必要时用 `strings <libcust_opapi.so> | grep PrepareWyReprBwd` 辅助确认。
- `bash build/fla-npu-fla_npu_linux-aarch64.run --install --quiet` 默认把 scoped run 包合入当前 Python 环境的 `site-packages/fla_npu/opp`。如果随后用源码树 `PYTHONPATH=<repo>/torch_custom/fla_npu` 调 `fla_npu.ops.ascendc`，需要确认源码 wrapper 实际解析到的 OPP 路径；源码 wrapper 调试时应显式设置 `FLA_NPU_OPP_PATH=<当前已安装 wheel 的 opp 根目录>`，或重新构建并安装包含最新 Python wrapper 的 wheel 后再验证。
- scoped run 包只替换当前 wheel 内的 OPP/opapi/kernel/proto 等产物，不会刷新已安装 wheel 的 Python wrapper。修改正式四输出 ABI 后，验证命令必须显式设置 `PYTHONPATH=<repo>/torch_custom/fla_npu:${PYTHONPATH}`，同时在 Python 启动前 prepend 当前 wheel OPP 的 `op_api/lib` 到 `LD_LIBRARY_PATH`；否则旧 installed wrapper 可能仍按旧 ABI 调用，表现为 `aclnnPrepareWyReprBwd ... not in libopapi.so`、参数错位、workspace 异常或进程崩溃。
- `fla_npu.ops.ascendc` 的 OPP 解析顺序会优先看 `FLA_NPU_OPP_PATH`，再看包内 embedded `opp`，然后看 `ASCEND_CUSTOM_OPP_PATH` / `ASCEND_OPP_PATH`。验证 ABI 问题时先在 Python 进程中打印 `fla_npu.__file__`、解析到的 vendor 目录和已加载 `CDLL._name`，避免源码 wrapper 与旧 OPP 库混用。
- `CrossCoreFlagWithReverse` 不会在每次 `Set/Wait` 后自动切到 `reverseId`；它只是在连续 `Set` 或连续 `Wait` 达到 `REVERSE_DEPTH` 后才用 `reverseId` 做反向握手防止 flag 溢出。涉及该类信号的方案变更时，必须在方案说明中列出每侧 `Set/Wait` 次数、方向和同步粒度，并询问用户确认。
- `KERNEL_TYPE_MIX_AIC_1_2` 下 Vector 存在 subBlock 维度。任何跨 AIC/AIV 的 stage/line 同步都要先明确同步粒度是“每个 value head 一次”还是“每个 subBlock 一次”。
- 每个 AI Core 都有自己独立的 UB，不同核之间不会互相踩踏 UB；一个 vec 的 UB 操作一定不会影响另一个核的 UB。排查精度问题时不要再做“关闭另一个核/vec/subBlock 观察 UB 是否互相影响”这类实验。优先检查核内 TBuf/LocalTensor 空间分配是否重叠、同一核内事件和 buffer 生命周期是否闭环，以及 workspace/GM 地址映射是否踩踏。
- 核内需要长期持有且彼此独立的 EventID 时必须用 `AllocEventID<HardEvent>()` 占用事件池，并在该事件生命周期结束时用同一个 `HardEvent` 模板调用 `ReleaseEventID<HardEvent>()` 释放；`AllocEventID` 和 `ReleaseEventID` 必须成对使用，不允许申请后遗漏释放、跨 `HardEvent` 模板释放，或把只查询空闲 ID 的 `FetchEventID` 当作持久事件分配接口。`FetchEventID` 不会占用事件池，连续调用可能得到同一个 ID，不能用于 ping/pong 或多 buffer 的持久事件组。
- Cube 侧手写 L0A/L0B 双缓冲时，`InitPipeFlags` 里预置的 `MTE1_MTE2`、`M_MTE1`、`FIX_M` 事件必须在 kernel 结束前被消费掉，避免同一 Python 进程下下一次 op 调用再次 `SetFlag` 已置位事件。stage0 已验证需要在每个 chunk task 结束后对全部 L0A/L0B free 事件做一次 `WaitFlag` 后重新 `SetFlag`，并在 `ProcessImpl` 退出前只 `WaitFlag` 不再 `SetFlag`。
- GVA 场景下 `KKT` 必须在当前 2-head 窗口内按 `HK` 去重生产：窗口内遇到新 `hk` 才计算一次，正式 golden 按窗口内 `HK` 语义覆盖；同一窗口内相同 `hk` 的后续 `hv` 不再重复做 KKT GEMM。当前 4-slot 方案使用四块 window-scoped KKT cache，并把每个 head workspace slot 映射到对应 `kktSlotForSlot_[slot]`；跨窗口不共享 KKT cache，避免下一窗口覆盖上一窗口 final output 仍在读取的 cache。stage0 已验证 KKT 尾部需要同时闭环 L0C/fixpipe 和本次 KKT 使用的 L0A/L0B free 事件：`copyL0CToGm(KKT)` 后等待并重新设置 `FIX_M`，再等待并重新设置对应 `M_MTE1`，否则多 head 或多 chunk 下可能出现 `L0A/L0B read/write conflict`。
- 当前 `A` 已经作为独立 L1 resident 复用到 stage0 `Dkbg/Dvb` 和 stage2 `DA5/DA6_T`，`Dkbg/Dvb/KKT` 仍在 stage0 生产。后续若继续改这些矩阵的计算顺序、resident 生命周期或输出位置，需要用正式四输出 golden 覆盖 `dk/dv/dbeta/dg` 中实际消费这些中间量的路径，不能只看局部中间结果。
- `V_DIM=256` 的 `Dvb` 会在 K 方向拆成多个 L0 MMAD 分片。stage0 已验证如果每个 K 分片都用 `unitFlag=0b11`，L9 类 `V=256/chunkSize=128/varlen` 用例会在 cube 侧卡死；必须按 block 级 unit flag 语义处理中间分片和最后分片。
- Vector 侧同一 V pipe 内也要显式保护 RAW 顺序：`CopyInRows`/`CopyInBetaGRows` 将 GM 数据搬入原始 dtype UB 后，必须通过显式 `CastInputRows`/`CastBetaGInputRows` 转到 fp32 目标 tensor；如果后续马上用 `Adds/Mul/Brcb` 消费该目标 tensor，需要在消费前插入 `PipeBarrier<PIPE_V>()`。stage0 已验证 Vb 路径在 beta/g 搬入并转换后缺少消费前 V pipe barrier 会导致 L9 类 V=256/varlen 用例出现大量 `Vb` 精度失败；纯 beta broadcast 不需要先 `Adds(..., 0.0f)` 复制到 scale tensor，可直接 `Brcb(..., betaFp32Tensor_, ...)`。
- `fla_npu.ops.ascendc` ctypes runtime 会用 `_RECENT_LAUNCH_STORAGE` 保活最近 launch 的输出和 workspace。全量同进程跑正式回归时，每次 `torch.npu.synchronize()` 后应清理该保活队列，避免旧 case 的大输出继续占用 NPU 显存；这种 OOM 属于测试 wrapper 生命周期问题，不能归类为 kernel 超时。
- 正式 `prepare_wy_repr_bwd` Python ctypes wrapper 必须只返回真实 `dk/dv/dbeta/dg` 四输出；不得再用 dummy 输出维持旧 ABI，也不得重新暴露 `stage*_debug` wrapper。

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
