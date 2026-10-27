# Overlay 修复与 Stable 后端日落验证（2026-10-27）

本记录包含两个阶段。最新主干已同步到 `c54ed595`；当前复验见文末“主干六笔提交同步与复验”。第一阶段的范围与结果保留为历史记录，不与本次结果累加。

## 第一阶段：overlay 修复与后端日落

本轮已实现 overlay 的 ABI 门禁、损坏恢复入口和取消构建清理，并完成 ctypes 算子执行后端及 legacy 扩展的日落。在已执行的前后对照中没有发现新增失败。**不能据此声明全算子正式验收或发布门禁通过**：仍有相同的基线失败，当前 A5 编译环境也未满足发布所需的 libstdc++ 上限。

- 第一阶段主干基线：`c9e2f1dbe6733d7d74c6bc4e1567e136f77da509`。
- 修改前：`5c53a2a312588363e63d0a1418e97e573324e202`，即该主干加原始 overlay 改动，保留 `a877211a` 的 overlay 设计，不混入 PR961 的其他修复。
- 设备：Ascend 950，CANN 9.2.0；构建用 conda `zzy`，最终设备对照使用已有 ATK 26.9.8 环境，固定 CPU 线程数为 1。
- 设备对照使用隔离安装目录与独立进程。首次 zzy 默认线程环境的对照超时，已单独保留；不计入通过结果。

### 实际修改边界

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

### 已执行验证

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

### 仍存在的问题与验收限制

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

### 证据位置

- A5：`/home/zzy1205/overlay-sunset-20261027/evidence/`。
- 本地任务：`artifacts/overlay-sunset-implementation-20261027/evidence/validation-summary.json`，含逐 case、失败、性能、构建退出码和版本对照；`reports/` 保存 ATK 原始报告。
- 外部验证驱动与构建脚本位于同一个任务 artifact 目录，不打入 wheel，不引入产品依赖。
- 本轮不创建新 PR；提交与验证记录供原有变更审查，发布失败未被豁免。


## 主干六笔提交同步与复验（2026-10-27）

本节为后续主干同步阶段，以上 `5c53a2a3` → `71a29c56` 的记录保留为第一阶段历史结果，不能与本节累加或视为当前提交的全量验收。

### 合并基线与冲突检查

- 同步前：PR878 `71a29c563d9b4d58b0af511cac616b55e5afaca3`，基于 `c9e2f1db`。
- 同步主干：`c54ed595b155b7f7877023e7f232ab00d468ecf9`，恰好新增以下六笔提交。
- 合并提交：`1764c2b55606028232c6cbc7b0e112b1a195ce22`。保留原有三笔提交，没有重放 PR961，也没有新建 PR。

| 主干提交 | 内容 |
| --- | --- |
| `1668f762` | 新增四层检视规范及工程样例 |
| `fa5493f1` | KDA A2 recompute 尾块处理 |
| `cb6cdc4a` | GDN bwd intra A2/A3 性能与公共元数据校验 |
| `9611b5dc` | CI 存储、代理回退及默认 16 路构建 |
| `b856e8c8` | 代理环境重复注入修复 |
| `c54ed595` | 代理环境大小写不敏感契约 |

自动合并无文本冲突。六笔提交涉及 89 个文件，与原 PR 修改重叠的仅 `ci/run_checks.sh`、`docs/Fla-npu仓CI部署教程.md`、`docs/agents/04-算子开发.md`。已检查三处合并结果，保留主干 CI 并行上限/代理行为和本 PR 的 Stable-only 依赖策略。合并点另有 86 个仅主干修改的文件与主干逐文件相同、106 个仅本 PR 修改的文件与同步前相同。`torch_custom` 在合并前后完全相同。

### 本次额外修正的范围

在六笔主干之外，仅修改 10 个文件：1 个测试夹具、8 个文档/样例注释文件、本验证记录。未额外修改生产 kernel、host、tiling、ACLNN、launcher 或 runtime；主干自身的算子更新原样引入。

| 文件 | 修改理由 |
| --- | --- |
| `ci/tests/test_make_packages_process_control.py` | 后台 shell 会继承 `SIGINT=SIG_IGN`，导致测试父进程无法按测试场景退出；夹具显式恢复父进程 SIGINT handler，子进程仍忽略 INT/TERM，以继续验证真正的进程组清理。同步前版本在同样后台环境复现相同失败。生产清理逻辑不变。 |
| `docs/agents/skills/operator-review/references/engineering-structure-review.md` | 新检视规范仍列出 ctypes/legacy 执行路径，改为 Stable-only 与显式 Python 兼容别名。 |
| 同目录 `documentation-review.md` | 删除旧后端作为正常文档入口的要求。 |
| 同目录 `engineering-example/engineering-structure.md` | 工程结构说明对齐 Stable-only，删除已不存在的 `_LAUNCHER_ONLY_OPS` 指引。 |
| 同目录 `engineering-example/L2组合算子示例/README.md` | 可选输出策略的上层入口从 Python/legacy 改为 Python/Stable。 |
| 同目录 `engineering-example/L2独立算子示例/fla/ops/ascendc/ops_classify/op_name/docs/api.md` | 样例 API 的原地更新契约改为 Stable schema 与 Python 登记一致，删除 ctypes 回退说明。 |
| 上述样例 `op_host/op_api/aclnn_op_name.cpp`、`.h` | 仅修正文中旧后端引用，ACLNN 代码不变。 |
| 同目录独立样例 `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py` | 仅修改模块说明，Python AST 去除 docstring 后一致。 |
| 本文件 | 分开记录当前复验、已有失败、工具限制与未完成验证。 |

标准库 ctypes 的库加载、构建戳/能力查询和 stream 诊断继续保留；KDA `implementation="legacy"` 算法选项继续作为遗留事项。

### 当前合并态验证

对照为隔离的 `71a29c56` 旧安装与 `1764c2b5` 合并代码重新构建的全量 OPP / wheel；均使用 Ascend 950、CANN 9.2.0、ATK 26.9.8 和固定 CPU 线程数。实际测试源码已逐文件校验（1857 个跟踪文件，CRLF→LF 归一化）。样例后续修改经 AST/去注释比较确认没有可执行逻辑变化。

| 检查 | 当前合并态结果 |
| --- | --- |
| Linux 公共单测 | 170 PASS、4 SKIP（设备/生成构建戳前置条件）；真实设备 ABI 和安装态构建戳另行验证通过。初跑为 169 PASS、1 FAIL、4 SKIP，已修复后台 SIGINT 夹具并重跑。 |
| CI Python / shell | 173 PASS；stage pipeline、A2/A5 设备解析通过。初跑 172 PASS、1 FAIL 为同一个 SIGINT 夹具问题。仅 fixture/mock，不冒充真实 Docker 取消验证。 |
| 覆盖与 ABI parity | 38 个公开算子 / 38 adapters，13 stack slots、0 mismatch。初次误把 `--check-parity` 传给另一工具退出 2；已用 `op_abi_parity.py` 正确重跑通过，原日志保留。 |
| A5 全量构建 | `bash build.sh --pkg --soc=ascend950 -j16` 通过；真实 OPP 公共头与 Stable adapter 的 31 个 ACLNN 调用点为 0 mismatch。 |
| wheel / 安装 | 独立 setup、根 `scripts/build_wheel.py` 的 cumsum 限定入口均构建成功；全量 `.run` 安装/重装、RECORD 更新、安装包 API 和真实 launcher 拒绝检查通过。发布门禁仍失败，见下表。 |
| 安装态公共契约 | 76 个公开签名前后相同；38 个注册算子、Stable 后端与零回退检查通过，两版解析后签名均无漂移。实际 ctypes/legacy/validate 旧模式被拒绝。 |
| 五配方数值对照 | 两版实际 OPP 库哈希不同，5 个已登记配方的 shape/dtype/逐输出 SHA256 相同；均走 Stable、fallback=0。其余 33 个算子无此配方，不计为通过。原 `compare-versions` 因 Stable Python 源文件未变而拒绝，独立证据校验使用隔离安装路径、提交身份与实际 OPP 二进制哈希；没有修改原工具放行。 |
| 原生 NPU 套件 | 两版各 261 PASS、3 FAIL，失败名称与错误文本完全相同。 |
| ABI / 原地 / 生命周期 / stream | 两版真实设备 ABI 12 项、requires_grad/version、200 轮 fresh 输入、多线程独立 stream 正例与错误 stream 自检均通过。 |
| KDA bwd prepare | 两版 30 个负例通过；正常布局套件仍在相同 A5 BSND case 失败。 |
| GDN bwd intra ATK | 两版精度各 18/18 通过；确定性各 18/18、每例 20 轮通过；3 个 device profiling 用例完成，见性能表。 |
| GDN 新增元数据门禁 | 缺失/长度错误/末项错误/非单调 cu_seqlens、sequence/chunk 越界、重复 chunk 等 8 个异常场景按预期拒绝。 |
| KDA recompute 尾块 | 两版原生 smoke 通过；9 个 FP32 beta/g 场景（T=1/7/8/9/63/64/65/97/2049）的 5 个输出均满足现有原生 LIMITS，逐输出 hash 相同。使用既有 ATK executor 的输入与 CPU 标杆、原种子，未改容差；该诊断不替代被 ATK 规则缺失阻塞的正式验收。 |
| 其他 ATK 精度抽查 | 两版 cumsum 2/2、KDA fwd 3/3、finalize 3/3 通过；KKT 2 例、KDA prepare 1 例均失败，前后逐 case 结果相同。 |
| 内存抽查 | 修正外部命令日志参数后，在空闲 NPU 7 的独立副本上 GDN 4/4 通过；KDA 4 例 sanitizer 日志未报告错误，但 ATK 后处理仍因规则缺失失败，计为未通过。首次缺日志导致的失败保留。两者都不是全量 MSS。 |
| overlay 生命周期 | 新全量 base / 安装 cumsum overlay / 卸载后三个状态各 5 配方一致，base 文件哈希不变；ABI 标量→指针漂移、实际覆盖 launcher 缺能力或版本为 0 均拒绝；损坏 overlay 时独立 status/remove 恢复成功。 |

GDN bwd intra 在同一张物理 NPU 1、相同参数下的 ATK device profiling 抽查如下。服务器为共享环境，未执行完整模型或跨平台性能验收，不把小幅波动解释为优化收益。

| case | 同步前 µs | 同步后 µs | 变化 |
| --- | ---: | ---: | ---: |
| 0 | 36.85545 | 36.63920 | -0.587% |
| 1 | 35.73475 | 35.89360 | +0.445% |
| 2 | 35.30975 | 35.29580 | -0.040% |

KDA recompute 性能任务两版均因 `cv_fused_double_benchmark` 未注册而失败，没有有效性能报告，不计为通过。

### 当前失败与未完成项

以下失败没有被当作通过，也没有改动 golden、输入种子、容差或发布上限来消除失败。

| 项目 | 结果与归因 |
| --- | --- |
| conv1d 三个原生失败 | 同步前后相同：varlen/device metadata/head_num 的 `max_abs_diff=11.381172`；update `max_query_len` 负例未抛 ValueError；bwd `BSH` 布局被现有 wrapper 拒绝。 |
| KDA bwd prepare 正常布局 | 两版在 A5 BSND case 返回 561103；当前 tiling 仅支持 fixed BNSD / variable NTD。 |
| KKT / KDA fwd prepare ATK | KKT 2 例、prepare 1 例在两版均失败，逐 case 状态相同。 |
| KDA recompute ATK 规则缺失 | ATK 26.9.8 环境未注册 `cv_fused_double_benchmark`；两版原有精度 3 例、FP32 beta/g 尾块精度 9 例、DC 3 例均失败；性能任务无有效报告。MSS 重跑也被该后处理错误阻塞。属于已在同步前环境复现的验收阻塞，直接数值诊断不替代正式 ATK 验收。 |
| 发布工具链 | 根入口 cumsum wheel 的实际发布检查失败；`readelf` 证实旧全量、新全量、新增量 OPP 都要求 `GLIBCXX_3.4.32`，高于仓库上限 `3.4.29`。须用规定工具链重建，未放宽门禁。 |
| 已修正的测试夹具 | 初次公共/CI 测试因后台继承忽略 SIGINT 超时；旧提交复现，夹具修正后全量重跑通过。该问题属于本 PR 已有测试问题，不是六笔主干新增算子问题。 |
| 验证命令限制与错误 | 版本比较工具仅以 `_stable.py` 哈希识别版本，不能直接处理此次 Python 未变而 OPP 更新的情况；独立对照补充真实 OPP 哈希。第一次 ABI 命令参数错误已纠正；第一次 MSS 缺 `--log-file` 导致 ATK 找不到日志，已修正并重跑。首次失败日志均保留，不计通过。 |

本阶段未完成 38 算子全量 ATK accuracy/performance/DC/MSS、其余 33 个算子的数值配方、A2/A3 真机和完整框架矩阵、完整 GDN Example/ST、真实 Docker 取消清理，以及规定发布工具链的重构建。A2/A3 专属主干修改不能凭 A5 抽查宣布通过。历史 preprocess backward MSS GM 写越界和第一阶段 KDA fwd 13/54 负例期望不匹配仍保留，未在本阶段复验完整矩阵。

正式 CI 和评审须针对新 head 重跑/确认：A2+A5 七项 NPU CI、自动 `CI 契约测试`、两个 approval 和 ABI Code Owner 检视。以上本地/A5 检查不能替代该门禁，PR 保持 Draft。

### 本阶段证据

- A5：`/home/zzy1205/pr878-main-sync-20261027/evidence/`。
- 本地：`artifacts/pr878-main-sync-20261027/evidence/sync-summary.json`、原始日志/XML/JSON 及 `reports/` 的 ATK 原始工作簿。
- `merge-audit.json` 保存父提交、重叠文件和两侧保留检查；`source-verification.json` 保存实际验证源码校验；`opp-version-parity.json` 保存安装路径、OPP 二进制哈希与输出哈希。
- 原始失败、脚本调用错误及修正后的重跑日志均保留，未覆盖或把命令返回 0 直接当作 ATK case 通过。
