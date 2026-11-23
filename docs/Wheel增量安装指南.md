# 增量安装指南：run 与 wheel

增量安装用于更新部分算子，同时保留 base 和此前增量包中未被覆盖的内容。
本文先说明“安装一次、再安装一次会发生什么”，再给出构建命令和状态规则。

## 1. 准备 base

先激活实际使用的 Python 环境，并初始化对应的 CANN 环境，再安装 base wheel：

```bash
python -m pip install --force-reinstall --no-deps /path/to/base.whl
```

base 必须包含支持多包增量安装的管理器；增量 wheel 也须由支持该功能的源码构建。
旧 base 需要先升级，安装增量包不会升级 base 的管理器。

**安装或移除前停止业务进程，完成后重新启动。** 增量内容在新 Python 进程中生效，
不支持在已初始化 CANN 的进程内热切换。默认保留 wheel 安装的 `.pth` 启动配置。

## 2. 三种增量包分别更新什么

算子由两部分配合工作：**适配层**负责 Python 接口及参数传递，**OPP** 提供 NPU 执行产物。

| 增量包 | 本次更新 | 保留什么 |
| --- | --- | --- |
| run-only（`.run`） | 包内算子的 OPP | 已有适配层，以及未覆盖的 OPP |
| 携带 OPP 的 wheel | 声明的适配层入口和包内 OPP | 未覆盖的适配层入口及 OPP |
| standalone wheel（不带 OPP） | 声明的适配层入口 | 已有 OPP，以及未覆盖的适配层入口 |

**普通 `pip install 新wheel` 更新整个 base。** 要把 wheel 作为增量安装，使用下文的
`python -m fla_npu_opp_manage --overlay --source ...`，两种 wheel 的命令相同。

## 3. 安装、查询和移除

以下各场景均从一个含算子 A、B 的完整 base 开始。A0、B0 表示 base 版本；
A1、B1 表示增量版本。假设 A1 包只更新 A，B1 包只更新 B，且依赖齐全、接口匹配。

示例路径请替换为实际产物路径，wheel 保留构建时生成的文件名。
**wheel 只更新 A 时，构建必须声明 `FLA_NPU_OVERLAY_OPS=A的入口名`**；
不声明会默认覆盖全部公共入口，不能按文件名或“只改过 A 的源码”推断范围。

### 3.1 安装一次 run，再安装第二次 run

```bash
# 第一次：安装只含 A1 的 OPP
bash /path/to/A1.run

# 第二次：继续安装只含 B1 的 OPP
bash /path/to/B1.run
```

| 安装到哪一步 | A 使用的 OPP | B 使用的 OPP |
| --- | --- | --- |
| 仅有 base | A0 | B0 |
| 安装 A1.run 后 | A1 | B0 |
| 再安装 B1.run 后 | **A1，继续保留** | **B1** |

两次安装都不更新适配层，A、B 仍使用 base 的适配层。
OPP 查找顺序为 **B1 → A1 → base**：B1 中没有 A，就继续查找 A1，A 不会退回 A0。

上述裸执行适用于支持默认增量安装的新 run。也可以明确使用管理入口：

```bash
python -m fla_npu_opp_manage --overlay --source /path/to/A1.run
```

### 3.2 安装一次带 OPP 的 wheel，再安装第二次

```bash
python -m fla_npu_opp_manage --overlay --source /path/to/A1-with-opp.whl
python -m fla_npu_opp_manage --overlay --source /path/to/B1-with-opp.whl
```

| 安装到哪一步 | A 使用的版本 | B 使用的版本 |
| --- | --- | --- |
| 安装 A1 wheel 后 | A1 适配层 + A1 OPP | B0 适配层 + B0 OPP |
| 再安装 B1 wheel 后 | **A1 适配层 + A1 OPP，均保留** | **B1 适配层 + B1 OPP** |

适配层入口和 OPP 分别按 **B1 → A1 → base** 查找。
即使 A1 的接口与 A0 不同，只要最终选中的适配层与 OPP 匹配，也可以使用。

### 3.3 安装一次 standalone wheel，再安装第二次

```bash
python -m fla_npu_opp_manage --overlay --source /path/to/A1-standalone.whl
python -m fla_npu_opp_manage --overlay --source /path/to/B1-standalone.whl
```

| 安装到哪一步 | A 使用的版本 | B 使用的版本 |
| --- | --- | --- |
| 安装 A1 wheel 后 | A1 适配层 + A0 OPP | B0 适配层 + B0 OPP |
| 再安装 B1 wheel 后 | **A1 适配层 + A0 OPP** | **B1 适配层 + B0 OPP** |

这里只改变适配层。若安装 standalone 之前已经有 run 增量，继续使用那些 OPP，
不会把 OPP 换回 base。新适配层暂时没有匹配 OPP 时，按第 5 节保存为待启用组合。

### 3.4 混合安装与重复更新

三类包可以混用。例如先安装 A1.run，再安装只更新 B 的 standalone wheel：
A 使用 base 适配层 + A1 OPP，B 使用新适配层 + base OPP。

只更新 A 的 standalone wheel 与只更新 A 的 run 也可以分两步安装；
两步完成、依赖匹配后得到新适配层 + 新 OPP，先装哪一类都可以。

**同名内容以后安装的为准，不比较版本号。** 例如已有 A1.run、B1.run，再安装 A2.run，
A 使用 A2，B 仍使用 B1。wheel 同理，但只替换它声明的入口。
如果 B 的包实际也携带了 A 或共享依赖，这部分同样参与覆盖；以包的实际内容为准。
同名 OPP 接口不匹配时，不会跳过新包去调用旧实现，处理规则见第 5 节。

### 3.5 查询结果与移除

```bash
python -m fla_npu_opp_manage --status

# 移除全部增量包，保留 base；当前不支持只移除其中一层
python -m fla_npu_opp_manage --remove-overlay
```

安装成功时会打印产物类型，以及 base、此前增量、本次安装的 OPP 路径，例如：

```text
[fla-npu] Installation successful: mode=overlay, artifact=run
[fla-npu] installed OPP: /.../wheel_layers/<本次层ID>/fla_npu/opp/vendors/fla_npu_transformer
```

overlay wheel 显示 `artifact=wheel`；standalone 未携带 OPP，会显示本次 OPP 为 `none`，
已有 OPP 仍保留。路径与状态用于确认选择；要验证算子效果，请在新进程运行相同输入的算子用例，
检查是否得到新版本的预期结果。移除后再开新进程，A、B 均恢复 base 版本。

升级或卸载 base 前，先执行上述移除命令。

## 4. 如何构建这三类包

以下以 A2 的 `chunk_fwd_o` 为例。将算子名换成实际更新对象；
目标 SoC 可选 `ascend910b`（A2）、`ascend910_93`（A3）或 `ascend950`（A5），
使用本机 Host 架构及对应的构建环境。

**run：** 在仓库根目录构建指定算子的 OPP。

```bash
bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu --ops=chunk_fwd_o
```

**带 OPP 的 wheel：** 在仓库根目录同时指定 OPP 范围和适配层入口范围。

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_fwd_o \
FLA_NPU_OVERLAY_OPS=chunk_fwd_o python scripts/build_wheel.py
```

**standalone wheel：** 在干净源码目录构建，确保包目录 `torch_custom/fla_npu/fla_npu/opp`
没有此前遗留的 OPP 二进制。

```bash
cd torch_custom/fla_npu
FLA_NPU_SOC=ascend910b FLA_NPU_OVERLAY_OPS=chunk_fwd_o \
  python setup.py bdist_wheel
```

`FLA_NPU_OPS` 决定构建哪些 OPP 算子及其依赖；
`FLA_NPU_OVERLAY_OPS` 决定 wheel 替换哪些公共适配层入口，可用逗号指定多个入口。

## 5. 什么情况下启用，状态怎么变化

安装器检查本次增量影响的调用关系：选中的适配层依赖是否齐全、与选中的 OPP 接口是否匹配。
安装成功不一定立即启用；`active` 表示已启用组合，`pending` 表示已保存但尚未配齐的组合。

| 条件或操作 | 状态变化 | 新业务进程使用什么 |
| --- | --- | --- |
| 安装后受影响的依赖齐全、接口匹配 | 新组合成为 active，清空 pending | 新组合 |
| 候选 wheel 的私有 ABI 自检报错 | 命令返回 1，打印 wheel 名和具体错误，保留原 active 和 pending | 原 active；没有则使用 base |
| 无可用 OPP、缺少所需接口、ABI 不匹配或依赖要求冲突 | 新组合保存为 pending；安装返回成功，原 active 保留 | 原 active；没有则使用 base |
| 已有 pending，继续安装其他包但仍未配齐 | 在最新 pending 上累积，更新 pending | 原 active 或 base |
| 已有 pending，继续安装后配齐 | 完整的新组合成为 active，清空 pending | 新组合 |
| 包格式不合法、校验失败或复制失败 | 安装失败，保留原有选择 | 原 active 或 base |
| 移除 overlay | 清除全部增量及 active/pending 记录 | base |

因此 active 和 pending 可以同时存在。重启不会自行配齐 pending，需要继续安装匹配组件。

例如，当前使用 A，准备安装含私有接口的 B。安装器校验 wheel 的 `RECORD` 后，在子进程中运行 B 的 `_op_abi.py`。若 B 声明了第一阶段却漏掉第二阶段，命令会指出 Python 算子名和缺失的符号，A 继续提供服务。若 B 自检通过，但当前 OPP 的签名与 B 要求的签名不同，B 进入 `pending`，可以补装匹配的 OPP。查询状态和解析运行时组合时，由已安装的管理器读取清单、核对实际 OPP，无需再次运行 B 的自检。自检代码具有安装用户的文件权限，请只安装可信来源的 wheel。

已保存的状态或文件损坏时，查询会显示 `damaged` 或 `errors`；先移除增量，再重新安装。

## 6. 其他情况

- **骨架 base（不带 OPP）也支持增量安装：** 带 OPP 的 wheel 可以补齐相应算子；
  standalone 还需配套 OPP。base 原本缺失且没有补齐的其他算子仍不可用。
- **临时禁用：** 设置 `FLA_NPU_OVERLAY_VENDOR=''` 后启动新业务进程，只使用 base，
  不删除已安装增量。恢复使用前取消该设置并重启。
- **覆盖边界：** wheel 增量覆盖声明的 `fla_npu.ops.ascendc` 入口，不更新顶层 `fla`、
  Triton、安装管理器或依赖包。
- 手动环境配置见[开发者指南](开发者指南.md#overlay-启动时序)；
  配套 OPP 的 `--opp-source`、旧版本兼容和详细设计见[增量安装设计](OPP叠加层增量设计.md#9-增量安装)。
