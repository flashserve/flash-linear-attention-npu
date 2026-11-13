# 增量更新适配层和 OPP

当你既修改了算子的 Python 参数或 Stable adapter，又希望保留其他算子的旧版本时，
可以将新 wheel 安装为 overlay。普通 `pip install 新wheel` 仍然是替换整个 base，
增量安装使用 `python -m fla_npu_opp_manage --overlay --source 新wheel`。

例如 base 中有 A 和 B，overlay 只声明 A：A 的公开 Python 函数、参数默认值、
Stable schema 和 launcher 来自增量适配层；B 继续使用 base adapter。OPP 单独选择，
未被增量 OPP 覆盖的 B 继续使用 base OPP。移除后恢复 base。
这里覆盖的是 `fla_npu.ops.ascendc` 公共入口，不替换顶层 `fla` 包、Triton 实现、
安装管理器或依赖包，也不执行增量 wheel 自带的 `.pth`。

## 1. 准备 base

base 和增量 wheel 都需要由支持 wheel layer 协议的源码构建，包含 `layer_manifest.json`
和支持显式注册的 Stable launcher。base 还须包含本指南的组件管理器；仅支持旧版整层安装的
base 也需要升级。安装增量包不会更新 base 管理代码。
base 可以带 OPP，也可以是等待补齐 OPP 的骨架 wheel。

适配层和 OPP 分别记录：后安装的同类组件优先，未携带的组件保留。standalone wheel 只更新
适配层，run-only 只更新 OPP，携带 OPP 的 wheel 更新两部分。不同组件的安装顺序不影响最终
组合；重复更新同一组件时，以最后安装的为准，不比较版本号。

```bash
# 先激活实际使用的 Python 环境，并 source 对应 CANN set_env.sh。
python -m pip install --force-reinstall --no-deps /path/to/base.whl
```

装卸前停止使用这个 Python 环境的业务进程，完成后重新启动。CANN 初始化之后不支持热切换。

## 2. 构建增量 wheel

`FLA_NPU_OVERLAY_OPS` 声明要覆盖的**公共算子入口**，用逗号分隔，可带或省略 `npu_`。
它与 `FLA_NPU_OPS` 不同：后者决定编译哪些 OPP 算子，前者决定运行时替换哪些 Python 入口。
未设置 `FLA_NPU_OVERLAY_OPS` 时，wheel 声明全部公共 Ascend C 入口；仅更新 A 时务必显式指定。

### standalone wheel overlay：只更新适配层

在干净的源码目录构建 standalone wheel；确保源码包下 `fla_npu/opp` 没有以前遗留的 OPP 二进制。

```bash
cd torch_custom/fla_npu
FLA_NPU_SOC=ascend910b FLA_NPU_OVERLAY_OPS=chunk_fwd_o \
  python setup.py bdist_wheel
```

这会编译匹配的新 launcher，生成不带 OPP 的骨架 wheel。Python 参数或 dispatcher schema
可以改变；安装后保留此前选择的 OPP，没有增量 OPP 才复用 base。即使暂时不匹配也可安装，
状态为 pending；稍后独立安装匹配 OPP 即可启用，无需重装 standalone wheel。

### 携带 OPP 的 wheel overlay：配套更新

从仓库根目录编译对应算子的 wheel：

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_fwd_o \
FLA_NPU_OVERLAY_OPS=chunk_fwd_o python scripts/build_wheel.py
```

将 SoC 改成目标机器的 `ascend910b`、`ascend910_93` 或 `ascend950`。只编译所需算子及其依赖。
若更新的共享 OPP 同时被 B 使用，安装器会检查 B 当前的 adapter，并在状态中列出受影响入口。
B 的接口兼容时可以复用原 adapter；不兼容时组合保持 pending，须补齐适配层。OPP 范围可以
大于适配层声明范围，但安装 OPP 不会自动新增 Python 入口。

## 3. 安装、查询和移除

三种输入可连续安装，未携带的组件不会被清空。安装前停止业务进程，完成后重启。
下面的路径须替换为准确的产物路径。

```bash
# 单独更新适配层；保留已经安装的 OPP
python -m fla_npu_opp_manage --overlay --source /path/to/standalone.whl

# 单独更新 OPP；保留已经安装的适配层。可先执行这一条，再执行上一条。
python -m fla_npu_opp_manage --overlay --source /path/to/update.run
# --source 也可以是 vendor 目录或只含一个 vendor 的父目录。
# 新构建的 run 也可直接执行，默认进行相同的增量安装，无需 --overlay：
bash /path/to/update.run
# 显式 --install、--full 与裸执行相同，都保留 base 和未覆盖算子。

# 一次同时更新两部分
python -m fla_npu_opp_manage --overlay --source /path/to/with-opp.whl
# 或指定配套 OPP；该显式来源优先于同一 wheel 内的 OPP
python -m fla_npu_opp_manage --overlay --source /path/to/standalone.whl \
  --opp-source /path/to/matching.run

python -m fla_npu_opp_manage --status
# 整体恢复 base；之后重新启动业务进程。
python -m fla_npu_opp_manage --remove-overlay
```

每次 run 或 overlay wheel 安装成功后，会打印安装类型、原有和本次安装的 OPP 路径，
以及新进程的选择。例如在已有 overlay 上再次安装 run：

```text
[fla-npu] Installation successful: mode=overlay, artifact=run
[fla-npu] base OPP: /.../fla_npu/opp/vendors/fla_npu_transformer
[fla-npu] previous overlay OPP (active): /.../wheel_layers/<旧ID>/fla_npu/opp/vendors/fla_npu_transformer
[fla-npu] installed OPP: /.../wheel_layers/<新ID>/fla_npu/opp/vendors/fla_npu_transformer
[fla-npu] combination: active
[fla-npu] selection for a new process: overlay (restart Python processes before use)
```

`artifact=wheel` 表示增量 wheel；`artifact=run` 也包含直接传入已解包 OPP 目录的情况。
骨架 base 没有 OPP 时显示 `base OPP: none`。standalone wheel 未携带 OPP 时，
`installed OPP` 显示 `none`，并说明沿用原选择，不把原 OPP 误报为本次新增。
原有增量 OPP 的 `active` / `pending` 标注对应本次安装所继承的组合。
新组合为 pending 时明确提示“已安装、尚未启用”，新进程仍用原 active，没有原 active 则用 base；
显式禁用 overlay 时，新进程选择同样为 base。路径是实际保存位置，安装后仍须重启业务进程。
普通 `pip install base.whl` 不增加此摘要；run 的外部 CANN 安装则单独标为 `mode=CANN`，
打印目标位置原有和本次安装的 OPP 路径，不把它标成 wheel 的 base 或 overlay。

| 查询状态 | 含义 |
| --- | --- |
| `active` | 当前可运行组合，新进程使用它 |
| `pending` | 最新组件已安装，但组合缺少匹配依赖；安装命令返回 0，并提示尚未启用 |
| `damaged` 或 active 中非空 `errors` | 状态或产物损坏；使用独立管理入口移除恢复 |

active 和 pending 可以同时存在：未配齐的新组合不替换旧 active；后续安装基于最新 pending
保留另一组件，匹配后自动启用并清空 pending。没有旧 active 时继续使用 base。
状态中的 `components.adapter` / `components.opp` 是各组件的产物 ID，null 表示 base；
`affected_base_operators` 列出本次 OPP 选择影响的 base adapter 入口。

组件安装不再要求 `--force`；旧命令带此参数仍可用，但不绕过 ABI 或完整性校验。
损坏产物、复制或提交失败不改变已有选择。移除统一清理所有保存的组件，base 原文件保留。
升级或卸载 base 前先移除 overlay；overlay 不登记进 pip RECORD。

例如当前为 P1+O1，安装 standalone P2 得到 P2+O1；再安装 run O2 得到 P2+O2；
安装完整 wheel P3+O3 后两部分同时更新。同一组件按完整产物替换，不合并多个历史 P 或 O 的范围。

## 4. 骨架与完整 wheel 的组合

| base | 增量 wheel | 可执行条件 |
| --- | --- | --- |
| 带 OPP | 带 OPP | 选中的新 P/O 匹配，未覆盖部分回落 base |
| 带 OPP | 骨架 | 新 P 与已选择的 O 匹配；无增量 O 时复用 base，否则等待补齐 |
| 骨架 | 带 OPP | 增量依赖配齐；base 未安装的其他算子仍不可用 |
| 骨架 | 骨架 | 先装、后装或配套提供 OPP 均可，未配齐保持 pending |

默认 `.pth` 在 CANN 初始化前配置 active 的 OPP 路径。禁用 `.pth` 时，先 source **base** 的
`fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash`；不直接 source 私有层脚本。
骨架 base 没有脚本时保留默认 `.pth`。`FLA_NPU_OVERLAY_VENDOR=''` 同时禁用增量 P/O。

| 产物 | 更新范围 | 未携带组件 |
| --- | --- | --- |
| run-only overlay | OPP L1–L4 配套产物 | 保留当前适配层 |
| standalone wheel overlay | 指定 Python/Stable 入口 | 保留当前 OPP |
| 携带 OPP 的 wheel overlay | 适配层和 OPP | 两部分均按新包替换 |

上述组合要求新版 base 控制器。无 layer 协议的旧 base 继续使用原 run-only 路径：须先解包 vendor，
同名 ABI 不兼容直接拒绝，不能混用新组件流程。新版控制器可接管单个已有 vendor overlay，以及
旧格式的完整 wheel active/pending 状态。当前组件模式使用默认名称，不支持自定义 overlay 名。

设计、覆盖范围和验收准则见 [增量安装](OPP叠加层增量设计.md#9-增量安装)。
