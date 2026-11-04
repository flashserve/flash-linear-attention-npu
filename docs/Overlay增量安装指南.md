# Overlay 增量安装指南

Overlay 用来在保留完整安装的前提下，只更新部分算子。完整 wheel 提供基础实现，称为 **base**；单独安装的算子更新称为 **overlay**。

例如，base 有 A、B 两个算子，overlay 只包含 A 的新实现：安装后调用 A 使用 overlay，调用 B 仍使用 base；移除 overlay 后，A、B 都恢复使用 base。应用继续使用原有 `fla_npu` Python 接口。

本指南先演示“已有完整 wheel 和增量包，如何安装”，再说明如何自己编译增量包。架构和兼容规则见 [OPP 叠加层增量设计](OPP叠加层增量设计.md)。

## 1. 开始前准备什么

| 需要准备 | 说明 |
| --- | --- |
| Python 与 CANN 环境 | 完成 [README](../README.md) 的硬件、CANN 和 Python 依赖准备；以下命令使用 Bash |
| 完整 wheel | 必须包含完整基础 OPP 和 overlay 加载能力；只含部分算子的 wheel 不适合作为本指南的 base |
| 增量 `.run` 包 | 只包含要更新的算子，且与 base 的 NPU 型号、CPU 架构、CANN 环境和公开接口匹配 |
| 安装目录写权限 | 使用安装 base 的同一 conda/venv 环境，并能写入该环境的包目录 |

**Overlay 适合更新已有算子的实现，不能单独更新 Python 接口。** 如果新增算子入口，或改变参数、返回值、ACLNN 接口或 Python/Stable 适配代码，应重新构建并安装匹配的完整 wheel。不同分支、不同版本的 base 和增量不能仅凭“安装成功”推断全部兼容；初次使用建议从同一份源码构建 base，再在接口不变的前提下修改目标算子。

较早发布的 wheel（例如未包含此功能的原始 `26.9.1` 包）不能直接按此流程使用 overlay。版本号本身不代表功能齐全，先确认包的来源，并检查下面的管理命令可用。

安装、替换、移除或切换 overlay 前，先退出所有使用这个 Python 安装目录的程序，包括 Notebook 内核和 ATK worker；完成后启动新进程。Overlay 不支持进程内热更新。

## 2. 安装完整基础包

激活目标 conda/venv 环境，加载 CANN 环境后安装完整 wheel。将两个路径换成实际位置：

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
BASE_WHEEL=/path/to/full-package.whl
python -m pip install --force-reinstall --no-cache-dir --no-deps "$BASE_WHEEL"

# 确认当前 Python 能找到管理入口，并查看基础安装
python -m fla_npu_opp_manage --help
python -m fla_npu_opp_manage --status
```

`--no-deps` 表示不自动安装依赖，torch、torch_npu 等需事先准备好。已经安装合适 base 的用户可以跳过 pip 安装，直接检查管理命令。正常的新安装应能看到类型为 `base` 的 `fla_npu_transformer`；如果没有 vendor，应先完成基础包安装。

如果提示 `No module named fla_npu_opp_manage`，先确认用了正确的 Python 环境；若包不支持此功能，换用包含 overlay 功能的完整 wheel。不要只复制一个管理脚本到旧包中使用。

## 3. 安装增量包

以下步骤在**同一个终端、同一个 Python 环境**执行。将 `OVERLAY_RUN` 改为拿到的实际 `.run` 文件路径：

```sh
OVERLAY_RUN=/path/to/operator-update.run
STAGING_DIR="$(mktemp -d /tmp/fla-npu-overlay.XXXXXX)"

# 仅安装到临时目录，不覆盖 base
bash "$OVERLAY_RUN" --install-path="$STAGING_DIR" --quiet

# 使用默认的 overlay 自动发现规则
unset FLA_NPU_OVERLAY_VENDOR
python -m fla_npu_opp_manage --overlay --source "$STAGING_DIR/vendors"
python -m fla_npu_opp_manage --status
```

每条命令成功后再继续下一条。`STAGING_DIR` 是临时中转目录；管理器会把文件复制到当前 wheel 的 overlay 目录，安装成功后运行程序不依赖该临时目录。默认增量名称为 `fla_npu_overlay_transformer`。

如果拿到的已经是解出的 vendor 目录，例如 `/path/to/vendors/fla_npu_transformer`，可以直接作为 `--source`，不再执行 `.run`。该目录应包含 `op_api/lib/libcust_opapi.so`、公开头文件、`abi_manifest.json` 和配套 OPP 资源。`--source` 也可以指向只包含一个这样的 vendor 的父目录。

**不要将单算子 `.run --install` / `--full` 当作 overlay 安装命令。** 它们走的是原有 base 覆盖流程，可能影响基础包内其他算子。本节的 `--install-path` 只负责中转，也不需要 source 中转目录里提示的环境脚本；真正启用的是管理器复制到 wheel 内的 overlay。

## 4. 如何确认安装并使用了更新

查看 `python -m fla_npu_opp_manage --status` 输出，核对三件事：

1. `package OPP root` 指向目标 Python 环境，base 条目为 `fla_npu_transformer`。
2. 默认增量显示为类型 `overlay`、名称 `fla_npu_overlay_transformer`，`ops=` 列表包含预期算子。
3. 该 overlay 显示 `ABI: OK`；默认单 overlay 场景下，`config.ini` 中它排在 base 前面。

正常使用时，wheel 随附的 Python 启动配置（`.pth`）会在新进程启动时准备 OPP 路径，无需修改业务代码。保持 `FLA_NPU_DISABLE_PTH` 未设置，可在新进程中检查环境和导入：

```sh
python -c "import os; print(os.environ.get('ASCEND_CUSTOM_OPP_PATH', ''))"
python -c "import fla_npu; print('fla_npu import ok')"
```

路径中应先看到当前安装目录内的 `fla_npu_overlay_transformer`，再看到 `fla_npu_transformer`。随后重新启动自己的程序或测试，调用更新的算子，并保留一个未更新算子的用例确认 base 仍可使用。测试入口见 [README Step 3](../README.md#step-3-验证与测试)。

`--status` 和 import 成功只能帮助确认安装、接口检查和加载状态，不能替代算子数值验证。接口检查报错时应先解决包的匹配问题。

如果主动关闭了 `.pth`，需要在程序初始化 CANN **之前**手动配置安装目录内 vendor 的 `set_env.bash`，详见 [启动契约](OPP叠加层增量设计.md#3-来源选择和启动契约)。不要在已经运行的 Python 或 Notebook 中补设路径后继续使用。

## 5. 替换、临时禁用和移除

### 替换同名增量

退出相关进程，把新的 `.run` 按第 3 节装入一个新的临时目录，再执行：

```sh
python -m fla_npu_opp_manage --overlay --source "$STAGING_DIR/vendors" --force
python -m fla_npu_opp_manage --status
```

`--force` 表示用新内容替换整个同名 overlay，不是追加文件，也不会跳过接口校验。若旧 overlay 包含 A、B，而新包只有 A，替换后 B 回到 base。管理器先复制、校验再切换；失败时查看报错，若提示保留了恢复目录，按提示保留材料并处理。完成后重新启动程序。

### 仅让下一次运行使用 base

保留已安装的 overlay，但在启动新进程前显式设置空值：

```sh
# 将 your_program.py 替换为自己的程序
FLA_NPU_OVERLAY_VENDOR= python your_program.py
```

这只影响该次命令；没有该覆盖变量的新进程仍按默认规则发现 overlay。如果曾在终端中用 `export` 设置此变量，执行 `unset FLA_NPU_OVERLAY_VENDOR` 后再启动新进程即可恢复默认选择。

### 移除增量，恢复 base

先退出相关进程，再执行：

```sh
python -m fla_npu_opp_manage --remove-overlay
python -m fla_npu_opp_manage --status
```

默认 overlay 被移除，base 保留；重新启动程序后使用基础实现。Overlay 文件由此管理命令维护，升级或重装 base 前也应先移除已有 overlay，避免残留文件与新 base 不匹配。正常 import 因 overlay 损坏而失败时，仍可尝试这个独立管理入口查询、移除。

本指南使用单个默认 overlay。多个增量或自定义名称的选择、优先级与移除方式见 [设计文档](OPP叠加层增量设计.md#5-安装卸载与恢复)。

## 6. 没有增量包，如何自己编译

在已准备构建环境的仓库根目录，先安装完整 base。没有完整 wheel 时，按 [README Step 2](../README.md) 构建；不要设置只选部分算子的 `FLA_NPU_OPS`：

```sh
unset FLA_NPU_OPS
FLA_NPU_SOC=ascend910b python scripts/build_wheel.py
```

用第 2 节的 pip 命令安装本轮生成的完整 wheel 后，修改目标算子的实现，保持公开接口不变。以下以 A2 的 `chunk_fwd_o` 为例，只构建它的 run 包：

```sh
bash build.sh --pkg --soc=ascend910b --ops=chunk_fwd_o -j32
```

`-j32` 是本次编译的并发数，可按机器资源调小。使用构建输出提示的本轮 `.run` 准确路径，代入第 3 节的 `OVERLAY_RUN`，完成安装和验证。

| NPU 产品 | `--soc` / `FLA_NPU_SOC` |
| --- | --- |
| A2 | `ascend910b` |
| A3 | `ascend910_93` |
| A5 | `ascend950` |

多个算子使用 `--ops=算子A,算子B`。修改共享实现时，还需把需要重新编译的依赖算子纳入构建范围；overlay 不会自动更新未重新编译的产物。此示例使用普通 `--pkg` 流程，不添加 `--jit`。

## 7. 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 找不到 `fla_npu_opp_manage` | 检查 Python 环境和完整 wheel 是否支持 overlay |
| 提示同名 overlay 已存在 | 确认要替换后使用 `--force`，或先移除；不要直接覆盖 base |
| 缺清单、ABI 不匹配或 `ABI: INVALID` | 使用配套增量包；如果变更了接口，重新构建完整 wheel，`--force` 不能绕过检查 |
| `--source` 提示没有 vendor 或有多个候选 | 指向含 `op_api/lib/libcust_opapi.so` 的准确 vendor 目录 |
| 显示 `overlay?` 或仍使用 base | 检查 `FLA_NPU_OVERLAY_VENDOR` 是否为空或排除了该名称；恢复默认设置后启动新进程 |
| 安装后已有程序行为没变 | 完全重启 Python / Notebook 内核 / worker，不只重新 import |
| ATK 没有使用增量 | worker 需使用同一安装环境，并在启动前配置路径；使用仓内 FLA executor，详见 [ATK 启动说明](OPP叠加层增量设计.md#31-仓内-atk-用例的兼容方式) |

返回 [项目 README](../README.md)；需要了解实现与完整兼容边界时，阅读 [OPP 叠加层增量设计](OPP叠加层增量设计.md)。
