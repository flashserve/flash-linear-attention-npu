# RecurrentKda Ascend950 RegBase 混合双发射设计

## 1. 背景与现状差距

当前 `recurrent_kda` 已通过 `__CCE_AICORE__ == 310` 将 Ascend950 入口路由到
`op_kernel/arch35/recurrent_kda.h`，但该实现仍为 `KERNEL_TYPE_AIV_ONLY`：

- A5 路径仅把部分 Vector 计算改成 MicroAPI RegBase；
- `S @ k`、`S @ q` 仍在 AIV 上用逐行乘加与规约完成；
- 没有 AIC/AIV 1:2 mixed kernel；
- 没有 L0C 结果经 Fixpipe 直接进入 AIV UB；
- 现有 `docs/design.md` 的平台表仍写明 A5“当前无算法差异”。

本设计以当前提交作为“改写前”基线，只调整 Ascend950 的 arch35 路径，保持
Ascend910B/Ascend910_93 公共实现和公开 ABI 不变。

## 2. 目标与非目标

### 2.1 目标

1. Ascend950 代码仅放在 `op_kernel/arch35/`，与 A2/A3 实现隔离。
2. A5 使用 `KERNEL_TYPE_MIX_AIC_1_2`：一个 AIC 与两个 AIV 子核协同执行。
3. `S_g @ k` 和 `S_g @ q` 使用 AIC Cube，FP32 累加。
4. Cube 输出通过 `PackedTileCopyTlaToUB<..., CopyL0CToUBMode::NO_SPLIT>`
   从 L0C 直接送入两个 AIV 的 UB。
5. AIV 使用 RegBase 完成 gate/beta、state 更新、输出合成及 GM 搬运。
6. 保持 BSND/TND、可选元数据、两种 state dtype/layout 和原位/非原位输出语义。
7. 按最新验收口径，以 PTA 精度脚本的 5 组场景全部通过为停止条件。

### 2.2 非目标

- 不修改 Ascend910B/Ascend910_93 kernel。
- 不修改 aclnn/Python 公共接口或支持范围。
- 不把测试阈值放宽，不删除既有注释和有效分支。
- 不引入第二个设备 kernel；AIC/AIV 工作在同一次 mixed launch 中。

## 3. 数学分解

原递推为：

```text
S_g   = exp(g_t) * S
delta = beta_t * (v_t - S_g @ k_t)
S'    = S_g + outer(delta, k_t)
o_t   = S' @ q_t
```

为避免 state 更新后再启动一次串行 Cube，输出等价改写为：

```text
sk    = S_g @ k_t
sq    = S_g @ q_t
kq    = dot(k_t, q_t)
delta = beta_t * (v_t - sk)
S'    = S_g + outer(delta, k_t)
o_t   = sq + delta * kq
```

其中 `sk` 与 `sq` 由 AIC 计算，`kq`、`delta`、outer update 和输出合成由
AIV RegBase 完成。该重排与原公式代数等价，并允许两个矩阵向量乘在 state 更新前完成。

## 4. Task 映射与双发射

逻辑任务仍为 `(sequence, value_head)`，不改变 host tiling 的语义：

- AIC block：处理一个 `(sequence, value_head)` 的完整 V 维。
- AIV sub-block 0：处理 V 维前半段。
- AIV sub-block 1：处理 V 维后半段。
- `coreIdx = GetBlockIdx() / GetSubBlockNum()` 用于映射 paired AIC/AIV。
- `GetSubBlockIdx()` 决定当前 AIV 的 V 分片。
- token 维保持串行，因为 `S_{t+1}` 依赖 `S_t`。
- 不同逻辑任务写入的 output/state 区间必须不重叠；显式 state slot 的既有校验保持不变。

主入口仅在 `__CCE_AICORE__ == 310` 时设置：

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
if ASCEND_IS_AIC { ... }
if ASCEND_IS_AIV { ... }
```

非 A5 继续使用原 `KERNEL_TYPE_AIV_ONLY`。

## 5. AIC/AIV 职责

### 5.1 AIV

1. 加载本 sub-block 的 state V 分片并在 UB 中保持 FP32 工作副本。
2. 加载/转换 q、k、v、gate、beta，完成 normalize、raw gate、safe gate 和 sigmoid 分支。
3. 对 state 做 gate decay，生成供 Cube 读取的 BF16 state mirror。
4. 将 state mirror 写入 system workspace，完成 MTE3 后发送 ready。
5. 计算 `kq = dot(k, q)`。
6. 等待 AIC 的 `sk/sq` L0C→UB 结果，并用 FP32 state 与 BF16 mirror 的差值补偿
   Cube 输入量化误差；补偿仍使用 RegBase 向量路径，不修改 FP32 state 工作副本。
7. 计算 delta、state outer update、`o = sq + delta * kq`。
8. 写出 out/final state，释放对应 UB slot。

两个 AIV 对 V 行做互斥分片；q/k/gate 可由两个子核各自读取，避免跨 AIV 共享 UB。

### 5.2 AIC

1. 等待两个 AIV 都完成当前 token 的 state mirror 写出。
2. 从 workspace 加载 BF16 `S_g[V,K]`。
3. 通过两个 Cube MMAD 计算 `S_g @ k` 与 `S_g @ q`，使用 FP32 L0C 累加。
4. 通过 `PackedTileCopyTlaToUB` 将两组结果按 V 行拆给两个 AIV sub-block。
5. 发送 result-ready；复用 L0C/UB slot 前等待 AIV result-free。

首版使用两个 MMAD，优先保证语义和同步正确。若 profiler 显示重复加载 state 成为主要开销，再评估把
k/q 打包为 K×16 的一次 MMAD；该优化不作为首版验收前提。

## 6. 内存规划

| 层级 | Buffer | 类型/布局 | 生产者 → 消费者 | 生命周期 |
|---|---|---|---|---|
| GM | 原输入/输出 | 保持现有布局 | Host → AIV/AIC | kernel 全程 |
| UB(AIV) | state residual temp | FP32，V 半片 × K | AIV 内部 | 每 token 复用 |
| system workspace | state mirror | BF16，`[active_task,V,K]` | AIV MTE3 → AIC MTE2 | 每 token 覆盖 |
| UB(AIV) | state work | FP32，V 半片 × K | AIV 内部 | 一个逻辑任务 |
| UB(AIV) | sk/sq slot | FP32，V 半片 | AIC Fixpipe → AIV V | 每 token ready/free |
| UB(AIV) | q/k/v/gate/beta/temp | 现有 dtype/FP32 | MTE2 → V | 按 token/任务复用 |
| L1/L0A/L0B | MMAD tile | BF16 | AIC MTE/Cube | 单次 MMAD |
| L0C | sk/sq accumulator | FP32 | Cube → Fixpipe | 双 slot 轮转 |

workspace 由 tiling 明确计算并通过现有 aclnn workspace 机制申请，不使用用户可见 tensor 作为中转。
AIV 中的 FP32 state 不因 Cube 输入而降精度；只对参与 dot 的 mirror 做 BF16 转换。

## 7. 同步协议

所有 flag 使用具名常量集中定义，禁止散落数字。每个队列 slot 都有 ready/free 双向闭环：

1. `state_free[slot]`：AIV 覆盖 workspace state mirror 前等待。
2. 两个 AIV 在 MTE3 完成后以同一 `CrossCoreFlagWithReverse<0x2>` 发布
   `state_ready`；AIC 在 FIX pipe 等待一次聚合事件。该方式与参考算子的 AIV→AIC
   workspace 可见性协议一致。
3. AIC 读取和 MMAD 完成后，分别向两个 AIV 子核发送 `state_free`。
4. AIC 复用 L0C/result slot 前 `wait result_free[slot]`。
5. Fixpipe 完成 L0C→UB 后 `set result_ready[slot]` 给两个 AIV。
6. AIV 在 V pipe 读取结果前 `wait result_ready[slot]`，消费结束后 `set result_free[slot]`。

使用参考算子的 `CrossCoreFlagWithReverse`/L0C→UB block 封装。参与核数量、PIPE 和 flag 方向固定：
AIV 写 GM 使用 `PIPE_MTE3` 通知 AIC；AIC 读取完成的反向通知使用实际消费 pipe；AIC Fixpipe
通知 AIV；AIV 读取/释放使用 V pipe。不会用 `SyncAll` 或核内 `PipeBarrier<PIPE_V>`
替代跨核同步。

## 8. 平台隔离与文件方案

预计最小改动文件：

- `op_kernel/recurrent_kda.cpp`：A5 分支改为 mixed AIC/AIV 调度，非 A5 不变。
- `op_kernel/arch35/recurrent_kda_common.h`：A5 常量、task/offset 和 flag 定义。
- `op_kernel/arch35/recurrent_kda_cube.h`：CATLASS Cube 与 L0C→UB。
- `op_kernel/arch35/recurrent_kda_vector.h`：AIV RegBase 与状态更新。
- `op_kernel/arch35/recurrent_kda_tiling_data_apt.h`：仅承载 A5 mixed kernel 新增的 tiling 字段。
- `op_host/op_tiling/arch35/recurrent_kda_tiling_a5.h/.cpp`：A5 专用 tiling、workspace 和 blockDim。
- 公共 tiling 入口、CMake、本设计文档、开发问题记录、测试报告和性能报告。

若现有 `arch35/recurrent_kda.h` 拆分会产生大规模无意义 diff，则保留文件名并只抽出必要
common/cube 辅助头；最终以可读性和最小 diff 共同决定。

### 8.1 A5 tiling 目录与路由

A5 mixed kernel 必须新增 workspace、AIC blockDim 和专用 tiling data，因此确认需要修改 tiling。
目录严格参考 `prepare_wy_repr_bwd_da`：

```text
op_host/
├── op_tiling/
│   ├── recurrent_kda_tiling.cpp
│   ├── recurrent_kda_tiling.h
│   └── arch35/
│       ├── recurrent_kda_tiling_a5.cpp
│       └── recurrent_kda_tiling_a5.h
└── ...
op_kernel/
└── arch35/
    └── recurrent_kda_tiling_data_apt.h
```

现有公共 tiling 文件位于 `op_host/` 根目录。实施时按最小修改原则迁入 `op_host/op_tiling/`，
同步修正 CMake/相对 include；A2/A3 的 `RecurrentKdaTiling` 和 processor 逻辑保持不变。
不得把 A5 workspace、AIC/AIV 分片或 mixed flag 字段塞入 A2/A3 公共 processor。

公共入口按参考实现读取平台并分派：

```cpp
const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
if (platform.GetCurNpuArch() == NpuArch::DAV_3510) {
    RecurrentKdaTilingA5 tilingA5;
    return tilingA5.SetTiling(context) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}
```

A5 tiling 的职责：

1. 使用 `GetCoreNumAic()` 设置 mixed kernel 的 blockDim，不能继续沿用公共路径的 AIV core 数。
2. 设置独立 A5 tiling key，并写入 `RecurrentKdaTilingDataA5`。
3. 计算两个 AIV 的 V 分片、state mirror workspace、L0C→UB result slot 和队列深度。
4. workspace 大小为 `GetLibApiWorkSpaceSize() + userWorkspaceSize`，system workspace 与用户中转区分开。
5. 仅在所用 CATLASS 模板确实要求时设置 batch schedule mode；不以 `SyncAll` 代替 ready/free 协议。
6. 继续复用公共 shape/dtype/layout/属性校验语义，避免 A5 与 A2/A3 接口范围漂移。

### 8.2 Ascend950 编译选项

`op_host/CMakeLists.txt` 沿用参考算子的 SOC 条件写法：
7. 为容纳 FP32 state residual 临时矩阵，A5 专属 `vStep` 上限为 32；V=128/256
   通过多轮双 AIV 分片覆盖，A2/A3 的公共 tiling 不受影响。

```cmake
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME RecurrentKda
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
    )
endif()
```

公共 `--cce-auto-sync=off` 保持不变；新增的所有 AIC/AIV、MTE/Fixpipe 依赖必须在 arch35
代码中显式同步。

## 9. 精度与兼容风险

1. **BF16 Cube mirror 误差**：已通过 FP32 state residual 补偿解决。全 FP32 Cube 虽可编译，
   但设备侧耗时不可接受，因此保留 BF16 Cube 主计算和 FP32 AIV 补偿，测试阈值未放宽。
2. **公式重排误差**：`S'@q` 改为 `sq + delta*kq` 会改变浮点累加次序。若误差不稳定，
   保留 state 更新后的第二次 MMAD 作为正确性优先方案。
3. **V=128/256 分片**：两个 AIV 分别处理 64/128 行，尾片读算写 mask 必须一致。
4. **空序列和 padding tail**：保持现有 device metadata 校验与有效 token 语义。
5. **state layout**：V-first/K-first 的 GM offset 独立验证，workspace 内统一转成 row-major `[V,K]`。
6. **flag 复用**：使用 bounded ready/free，禁止 producer 连续 set 未被消费的同一 flag。

## 10. 实施与确认门

1. 保存当前提交的 A5 构建、精度和 msprof 基线。
2. 先完成 mixed kernel 的最小 BSND case。
3. 再覆盖 TND、可选元数据、state dtype/layout 和输出语义。
4. 每次编译/运行失败立即写入 `docs/开发问题记录.md`，保存可搜索文本、真实终端截图、根因和修复。
5. 执行最新指定的 PTA 精度验收；5 组场景全部通过后停止任务。

## 11. 构建与验证计划

环境：

```bash
source <CANN set_env.sh>
conda activate yzq
```

构建命令严格使用仓库 README 方式 B 和参考算子的参数顺序/命名：

```bash
# 仓库根目录：只编 recurrent_kda 的 Ascend950 run 包
bash build.sh --soc=ascend950 --pkg --vendor_name=fla_npu --ops=recurrent_kda

# Python wrapper 有改动时才重编 runtime wheel
cd torch_custom/fla_npu
python3 setup.py bdist_wheel
```

run 包安装同样遵循 README 方式 B，不自行拼接 vendor 后缀：

```bash
./build_out/fla-npu-*.run --install
python -m pip install --force-reinstall --no-deps \
    torch_custom/fla_npu/dist/flash_linear_attention_npu-*.whl
```

仅 Kernel/tiling 改动时不强制重编 Python wheel；但最终验收仍记录 run 包和 wheel 的实际构建状态。
每次构建都重新加载用户指定 CANN 环境并进入 `yzq` 环境，不复用长期 SSH 会话中的旧环境变量。

验收顺序：

```bash
python fla/ops/ascendc/kda/recurrent_kda/tests/pta/test_accuracy.py
python torch_custom/fla_npu/test/test_npu_recurrent_kda.py
pytest -q tests/operators/recurrent_kda
```

性能仅使用 msprof/msopprof 设备侧 kernel duration；报告 warm-up、采样数、mean、median、min、max、
CV、block dim、输入 shape，并明确不包含编译、数据生成和 CPU reference。

## 12. 回退条件

- mixed 基础能力或 L0C→UB 在指定 CANN 环境无法编译：记录真实报错并以参考算子最小复现确认环境，
  不把环境问题误判为 kernel 问题。
- Cube mirror 或公式重排无法在既有阈值内通过：保留 mixed launch，回退到更高精度 Cube 输入或
  state 更新后的第二次 MMAD。
- mixed 版本稳定慢于改写前基线：保留 A5 tiling key 的 AIV fallback，并根据 shape 选择更优路径；
  不影响 A2/A3。

