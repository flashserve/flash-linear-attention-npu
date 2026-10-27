# Overlay 修复与 Stable 后端日落验证（2026-10-27）

## 结论与基线

本轮已实现 overlay 的 ABI 门禁、损坏恢复入口和取消构建清理，并完成 ctypes 算子执行后端及 legacy 扩展的日落。在已执行的前后对照中没有发现新增失败。**不能据此声明全算子正式验收或发布门禁通过**：仍有相同的基线失败，当前 A5 编译环境也未满足发布所需的 libstdc++ 上限。

- 当前主干基线：`c9e2f1dbe6733d7d74c6bc4e1567e136f77da509`。
- 修改前：`5c53a2a312588363e63d0a1418e97e573324e202`，即该主干加原始 overlay 改动，保留 `a877211a` 的 overlay 设计，不混入 PR961 的其他修复。
- 设备：Ascend 950，CANN 9.2.0；构建用 conda `zzy`，最终设备对照使用已有 ATK 26.9.8 环境，固定 CPU 线程数为 1。
- 设备对照使用隔离安装目录与独立进程。首次 zzy 默认线程环境的对照超时，已单独保留；不计入通过结果。

## 实际修改边界

相对修改前共 99 个文件：78 个代码/构建/测试文件和 21 个文档，其中删除 10 个旧后端资产。代码按 overlay 修复和后端日落分成两笔本地提交；第一笔 `8ad65ad2` 已单独验证。新增公共策略辅助模块与冻结契约/测试用于消除重复规则及验证迁移，不涉及算子计算实现。

| 范围 | 修改与兼容策略 |
| --- | --- |
| ABI 门禁 | `_op_abi.py` 保留指针层数，校验 GetWorkspaceSize/执行两阶段、导出覆盖及库/头文件哈希；CMake 打包生成 `abi_manifest.json`，安装与运行时复核。旧 base 可读取公共头，新增 overlay 必须带清单。 |
| 恢复入口 | 新增顶层 `python -m fla_npu_opp_manage`，不导入损坏的 `fla_npu` runtime，也不加载 vendor 库；支持状态查询和移除 overlay。 |
| 取消构建 | `ci/pkg/make_packages.py` 以进程组而非仅父进程判断清理完成；远端增加进程身份检查和终止升级；清理未确认时禁止重试。Docker 路径保留容器归属检查。 |
| 构建与发布 | 两个 setup、构建脚本和发布检查强制携带匹配的 Stable launcher/构建戳；删除纯 ctypes 构建豁免、legacy CppExtension/codegen/YAML、torchnpugen CI 依赖；拒绝旧选项和 `--skip-build` 绕过。 |
| Python runtime | 删除 `_aclnn_ctypes.py`、算子 `_runtime.py`、legacy 加载和自动回退；launcher 缺失、构建戳不符、算子注册缺失均明确失败。 |
| overlay 后端策略 | 检查实际加载的 launcher，包括 `FLA_NPU_STABLE_LIB` 指向的库；有 overlay 时要求导出 `fla_npu_stable_overlay_order_version() >= 1`，不再自动切 ctypes 或容许混用来源。 |
| 公共接口 | 保留主干 38 个算子、76 个导出名；`__signature__` 统一来自 Stable wrapper。保留 `BACKENDS`、恒为空的 `FALLBACKS` 及显式 `install_torch_npu_ops_compat()` 纯 Python 别名。 |
| 调用方与测试 | 迁移旧 `torch.ops.npu` 调用及私有 runtime 依赖；正常测试走公共 Stable 入口。原始 ABI 负例使用产品包外、不打包的测试模块 `tests/atk/common/aclnn_error_probe.py`，仅调用 GetWorkspaceSize，不执行第二阶段或分配 workspace。 |
| 覆盖门禁 | `stable_coverage.py` 改用独立冻结的公开契约，校验 38 个 wrapper/schema/adapter 及签名、参数转发；增加旧模式拒绝、真实签名、旧调用方依赖和打包绕过测试。 |

测试探针位于**产品包之外、仓库之内**，不是恢复 ctypes 执行后端。标准库 ctypes 的动态库加载、构建戳/能力查询与 stream 诊断继续保留。

没有修改任何算子的 kernel、host、tiling 或 ACLNN 计算实现。对 38 个 Stable 公共包装函数做 AST 对照，除文档和纯参数辅助模块导入外，函数体无变化；三个签名的兼容调整如下。

| 包装函数 | 处理 |
| --- | --- |
| `npu_causal_conv1d_fn` | 将原公共签名的参数注解、默认值、关键字参数和返回注解复制到 Stable wrapper。 |
| `npu_causal_conv1d_update` | 同上；保持原地状态与 `out` 语义。 |
| `npu_pre_process_fwd_kernel_merged` | 签名默认值统一为原公共接口的 `chunk_size=64`；显式 `None` 仍按原逻辑解释为 64。 |

KDA `implementation="legacy"` 是算法选择，**保留并记作遗留事项**；本轮未改变其选择策略或主干新增的 KDA 反向行为。未引入此前 PR961 的 ATK golden、输入配置或 CT 参数修复。

## 已执行验证

| 检查 | 结果 |
| --- | --- |
| Linux 公共单测 | 170 PASS、4 SKIP；3 个设备前置条件跳过项随后在真实设备补测通过，另 1 个是源码未保留生成构建戳；安装态构建戳检查另行通过。 |
| 单独 overlay 修复提交 | `8ad65ad2` 独立源码树：93 PASS、1 个生成构建戳前置条件 SKIP；不依赖后端日落提交。 |
| CI Python 回归 | 150 PASS；Linux 进程组清理测试包含在公共单测中。 |
| CI shell 回归 | stage pipeline 与 A2/A5 设备解析通过；stage fixture 屏蔽系统 CANN 初始化对 fake 环境的覆盖，此项不等同于真实 Docker 构建。 |
| 静态门禁 | 38 个公开算子、38 个 adapter；schema/ABI 转发无不一致；无生产旧后端导入/调用；修改后的 Python 源码语法检查通过。 |
| A5 全量 OPP 构建 | 全量编译完成；实际 OPP 公共头与 Stable adapter 的 31 个 ACLNN 调用点检查通过，0 mismatch；生成有效 ABI 清单。 |
| 两种 wheel 构建入口 | 独立 setup 构建成功；根目录 `scripts/build_wheel.py` 使用 cumsum 限定范围构建成功；均安装到隔离目录。 |
| 实际安装/重复安装 | 独立 wheel 加全量 `.run` 安装、再次安装与 RECORD 更新成功；38 个实际注册算子、76 个导出及兼容别名检查通过。 |
| 全量签名对照 | 两个实际安装版本的 76 个初始公开签名完全相同。旧版 3 个包装函数解析后的签名与初始签名不同；新版 38 个解析结果均与公开签名一致。 |
| 旧模式/缺库拒绝 | 实际 `ctypes`、`legacy`、`STABLE_VALIDATE=1` 设置被拒绝；缺失或错误 launcher 不回退。 |
| 前后数值对照 | 已登记的 5 个配方逐输出 hash 相同，均为 Stable、fallback=0；其余 33 个算子没有此工具的数值配方，不计作通过。 |
| NPU 回归套件 | 修改前后各 261 PASS、3 FAIL；失败名称和错误文本相同，见下表。 |
| 原地修改/生命周期/stream | 两版均通过；包含 requires_grad/version 契约、200 轮 fresh 输入生命周期，以及 8 线程各自 stream 的正例与错误 stream 自检。 |
| KDA 反向 prepare 负例 | 两版均通过 30 个原始 ACLNN 错误与公共 Stable 异常传播检查。其正常布局套件在相同 A5 限制处失败。 |
| ATK 精度抽查 | 两版 cumsum 2/2、KDA fwd 3/3、KDA finalize 3/3 通过；原始 ctypes finalize executor 单独跑 3/3 通过。KKT 2 例与 KDA prepare 1 例在两版均失败。 |
| KDA fwd 旧负例调用方迁移 | 使用现有 generator 的 54 个 A5 负例，对比原始 ctypes executor 与新探针：均 41 个预期匹配、13 个不匹配；逐 case 错误码与结果一致，不把 13 个不匹配计为通过。 |
| ATK 确定性抽查 | cumsum 1 例、KDA finalize 2 例，20 轮，两版通过。 |
| 真正损坏 overlay 的恢复 | 清单/头漂移导入失败；独立管理模块仍能 status/remove；移除后重新导入成功。 |
| 实际 launcher 覆盖路径 | 构建匹配 hash 但能力符号缺失/版本为 0 的测试库，通过覆盖路径加载时均按预期拒绝；未错误地依据包内 launcher 放行。 |
| overlay 生命周期 | 新全量 OPP：base、安装 cumsum overlay、卸载后 3 个状态各 5 配方一致；base 所有文件哈希不变；有效清单下的 `double`→`double*` 漂移也被拒绝。 |

ATK 性能抽查为同机同参数的 device profiling，未覆盖全部算子或完整模型性能，不将抽样波动解释为优化收益：

| 算子 / case | 修改前 µs | 修改后 µs | 变化 |
| --- | ---: | ---: | ---: |
| cumsum / 0 | 9.1031 | 8.4409 | -7.27% |
| KDA finalize / 0 | 200.1246 | 199.7163 | -0.20% |
| KDA finalize / 1 | 389.7589 | 392.9844 | +0.83% |

## 仍存在的问题与验收限制

| 项目 | 修改前后证据 / 处理 |
| --- | --- |
| conv1d varlen + device metadata + head_num | 同一测试均 `max_abs_diff=11.381172`；保留，未扩大到算子行为修复。 |
| conv1d update 的 max_query_len 负例 | 两版均 `ValueError not raised`；保留。 |
| conv1d bwd 的 `BSH` 布局 | 两版均被现有 Stable wrapper 拒绝；保留。 |
| KDA bwd prepare 正常布局套件 | 两版同样在 A5 的 `BSND` case 返回 561103，tiling 明确仅支持 fixed BNSD / variable NTD。 |
| KKT / KDA prepare ATK | 相同精度/调用方失败，未修改 golden、关键字或测试期望来掩盖失败。 |
| KDA fwd 原始负例期望 | 13/54 两版不匹配；例如空指针实际 161002、期望 161001。探针迁移未新增差异。 |
| 发布工具链 | `check_pypi_wheel.py` 失败：OPP 需要 `GLIBCXX_3.4.32`，仓库上限 `3.4.29`。原始基线、当前全量和增量 OPP 均有该依赖；需用仓库固定的发布工具链重新构建，不放宽门禁。 |
| 全量验收 | 本轮没有完成 38 算子全量 ATK accuracy/performance/DC/MSS、A2/A3 真机及完整框架版本矩阵。不能据抽查或双方同时失败宣布正式通过。 |
| Docker 取消清理 | 已做 mock 回归；A5 账号无 Docker 访问权限，未执行真实容器取消验证。 |

## 证据位置

- A5：`/home/zzy1205/overlay-sunset-20261027/evidence/`。
- 本地任务：`artifacts/overlay-sunset-implementation-20261027/evidence/validation-summary.json`，含逐 case、失败、性能、构建退出码和版本对照；`reports/` 保存 ATK 原始报告。
- 外部验证驱动与构建脚本位于同一个任务 artifact 目录，不打入 wheel，不引入产品依赖。
- 本轮不创建新 PR；提交与验证记录供原有变更审查，发布失败未被豁免。
