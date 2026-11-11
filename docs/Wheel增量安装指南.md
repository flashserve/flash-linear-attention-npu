# 用 wheel 增量更新算子

当你既修改了算子的 Python 参数或 Stable adapter，又希望保留其他算子的旧版本时，
可以将新 wheel 安装为 overlay。普通 `pip install 新wheel` 仍然是替换整个 base，
增量安装使用 `python -m fla_npu_opp_manage --overlay --source 新wheel`。

例如 base 中有 A 和 B，overlay 只声明 A：A 的公开 Python 函数、参数默认值、
Stable schema 和 launcher 来自 overlay；B 继续使用 base。移除 overlay 后 A 也恢复 base。
这里覆盖的是 `fla_npu.ops.ascendc` 公共入口，不替换顶层 `fla` 包、Triton 实现、
安装管理器或依赖包，也不执行增量 wheel 自带的 `.pth`。

## 1. 准备 base

base 和增量 wheel 都需要由支持 wheel layer 协议的源码构建，包含 `layer_manifest.json`
和支持显式注册的 Stable launcher。历史发布包不会自动获得这项能力，须先升级 base。
base 可以带 OPP，也可以是等待补齐 OPP 的骨架 wheel。

```bash
# 先激活实际使用的 Python 环境，并 source 对应 CANN set_env.sh。
python -m pip install --force-reinstall --no-deps /path/to/base.whl
```

装卸前停止使用这个 Python 环境的业务进程，完成后重新启动。CANN 初始化之后不支持热切换。

## 2. 构建增量 wheel

`FLA_NPU_OVERLAY_OPS` 声明要覆盖的**公共算子入口**，用逗号分隔，可带或省略 `npu_`。
它与 `FLA_NPU_OPS` 不同：后者决定编译哪些 OPP 算子，前者决定运行时替换哪些 Python 入口。
未设置 `FLA_NPU_OVERLAY_OPS` 时，wheel 声明全部公共 Ascend C 入口；仅更新 A 时务必显式指定。

### 只改 Python / Stable adapter，复用 base OPP

在干净的源码目录构建 standalone wheel；确保源码包下 `fla_npu/opp` 没有以前遗留的 OPP 二进制。

```bash
cd torch_custom/fla_npu
FLA_NPU_SOC=ascend910b FLA_NPU_OVERLAY_OPS=chunk_fwd_o \
  python setup.py bdist_wheel
```

这会编译匹配的新 launcher，生成不带 OPP 的骨架 wheel。Python 参数或 dispatcher schema
可以改变；其实际调用的 ACLNN 接口仍须与复用的 base OPP 一致。

### 连同 OPP 一起更新

从仓库根目录编译对应算子的 wheel：

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_fwd_o \
FLA_NPU_OVERLAY_OPS=chunk_fwd_o python scripts/build_wheel.py
```

将 SoC 改成目标机器的 `ascend910b`、`ascend910_93` 或 `ascend950`。只编译所需算子及其依赖。
若更新的共享 OPP 同时被 B 使用，须把 B 的匹配 adapter 也纳入 overlay；安装器会给出依赖入口。
一个 wheel 内的 OPP 不能额外覆盖声明范围以外的算子。共享依赖不能仅靠关闭 ABI 检查绕过。

## 3. 安装、查询和移除

下面将 `update.whl` 替换为准确的构建产物路径。

```bash
# 骨架或带 OPP 的 wheel 使用同一入口。
python -m fla_npu_opp_manage --overlay --source /path/to/update.whl
python -m fla_npu_opp_manage --status

# 骨架的 adapter 需要新 OPP 时，在同一次安装中配齐。
python -m fla_npu_opp_manage --overlay --source /path/to/update.whl \
  --opp-source /path/to/matching.run --force
# --opp-source 也可以是 vendor 目录或只含一个 vendor 的父目录。

# 恢复 base；之后重新启动业务进程。
python -m fla_npu_opp_manage --remove-overlay
```

| 查询状态 | 含义 |
| --- | --- |
| `active` | 依赖已配齐，新进程启用增量入口 |
| `pending` | wheel 已保存，但缺少匹配的 OPP 或存在覆盖冲突；命令返回 2，不启用它 |
| `damaged` 或非空 `errors` | 状态或产物损坏；可用独立管理入口移除恢复 |

已有 overlay 时用 `--force` 整体替换。新包复制、校验或状态切换失败不会破坏旧 active；
新包进入 pending 也保留旧 active。修正产物后重新执行上述安装命令，不会自动下载依赖。
移除会清理保存的历史层。升级、卸载 base 前也应先移除 overlay；overlay 不登记进 pip 的 RECORD。

## 4. 骨架与完整 wheel 的组合

| base | 增量 wheel | 可执行条件 |
| --- | --- | --- |
| 带 OPP | 带 OPP | 新 adapter 与新 OPP 配套，未覆盖入口仍使用 base |
| 带 OPP | 骨架 | 新 adapter 能复用 base OPP；否则用 `--opp-source` 配齐 |
| 骨架 | 带 OPP | 增量范围内所需 OPP 已配齐；不承诺 base 未安装的算子突然可用 |
| 骨架 | 骨架 | 用 `--opp-source` 配齐，否则保持 pending |

默认通过 `.pth` 在 CANN 初始化前选定 OPP 路径。禁用 `.pth` 时，先 source **base** 的
`fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash`，再启动业务进程；不要直接 source
私有层中从新 wheel 提取的脚本。`FLA_NPU_OVERLAY_VENDOR=''` 会同时禁用增量 Python 入口和增量 OPP。

当前只有一个 wheel overlay 槽位，使用默认名称；不与旧的 run-only overlay 混装，切换类型前先移除。
历史 `.run` / vendor 增量安装继续可用，仍要求与 base 的 ACLNN ABI 兼容。
wheel 模式允许被覆盖算子的接口变化，检查的是**新 adapter 与实际选中 OPP**，不是要求它等于旧 adapter。

机制、启动约束及验收标准见 [OPP 叠加层增量设计](OPP叠加层增量设计.md#9-wheel-携带适配层的增量安装)。
