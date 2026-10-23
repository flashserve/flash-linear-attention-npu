# Stable ABI 单后端收敛

本设计以主干 `8762b6139de96e62205cefb1595aa41e38497e82` 为修改前基线。
目标是移除 ctypes 算子执行后端和 legacy PyTorch 扩展，所有 AscendC 公共入口只使用
`libfla_npu_stable.so`。Triton、KDA 的原 fused/V2 算法选择和设备 kernel 不在本次变更范围。
本次也修复审查中确认的 Stable 接口缺口，包括 Python 参数规范化、输入预检以及 cumsum
C++ 适配层的输出分配；因此不能将本变更描述为“37 个计算体完全不变”或只有加载路径变化。

## 1. 完整性分析

基线有 37 个公开 `npu_*` 入口，均有 Stable Python wrapper。公共 wrapper 首次调用前
从 ctypes 取得 `__signature__`，实际执行仍优先选 Stable；首次调用会把模块属性绑定为
解析出的函数。两种签名来源有三处差异：Conv1d fn/update 的类型及返回注解，以及
pre_process_fwd_kernel_merged 的 chunk_size 默认值（64 与 None）。先把 Stable wrapper
补齐为原公共契约，再统一签名来源；显式 None 的原有处理继续保留。

`_stable.py` 和 `_kda_policy.py` 引用的 optional_bool/optional_float 与 ACL format
常量先迁移到无执行能力的公共参数模块，再删除 `_runtime.py` 和 `_aclnn_ctypes.py`。
标准库 ctypes 仍用于动态库加载、构建戳和能力探测；它不再构造并执行算子请求。

KDA `implementation="legacy"` 表示原 fused 算法，仍经 Stable launcher 调用。optimized
目前限制 Ascend950、K=V=128 和特定 gate 配置，且 dq/dk/db 的输出 dtype 与原路径存在
差别；删除原算法会缩小支持范围并改变返回契约，因此本次保留两种算法及 auto 策略。

## 2. 运行时

公开函数保持名称、参数、默认值、返回结构、autograd 和 mutation 契约；既有 Stable 实现
遗漏的已公开行为按下述四项补齐，而不是将基线错误固化为新契约。所有算子只解析
Stable wrapper 及其真实 dispatcher 注册；缺库、缺注册、缺失或不匹配的构建戳均保留
根本原因并直接报错。不存在自动回退和 legacy 扩展加载 API。

正常 import 保持延迟加载 torch 的边界，实际调用必须完成 launcher 验证。不得因删除
ctypes 注释或代码而删除版本计数、requires_grad 检查、每次取 stream、设备 guard 或
输入生命周期保障。`FLA_NPU_STABLE_ABI=ctypes` 和 `FLA_NPU_STABLE_VALIDATE=1` 明确报错。

审查确认并纳入本次实现的接口修复如下：

| 接口缺口 | 修改及边界 | 验证要求 |
|---|---|---|
| `npu_chunk_local_cumsum` 的 Stable 路径只接受 float32/bfloat16 | 恢复既有 `output_dtype` 字符串别名、torch dtype 的字符串形式、显式 None 对应 float32，以及 float16 和 same/same_as_input/input 输出策略；Python 规范化后编码，C++ 按相应 dtype 分配输出 | CPU 检查规范化、编码和非法值拒绝；重新编译 launcher 后执行原 NPU dtype/数值用例，不能把基线拒绝 float16 当期望结果 |
| Conv1d fn/update 遗漏预检 | `validate_data=True` 时执行独立的 query_start_loc 边界、类型、设备及 max_query_len 检查；update 保留 max_query_len 的基本约束。拒绝发生在下发与状态修改之前 | CPU 检查非法元数据不下发、不修改状态，以及关闭可选预检时不读回设备元数据；原 NPU update/变长测试仍须执行 |
| Conv1d fn/update 混淆显式 None 与默认 null slot | 显式 `null_block_id=None` 转发 -1，以关闭 null-slot 过滤；省略该参数或显式传 0 仍保持默认过滤 slot 0。原 Stable 错误地把 None 转成 0，会跳过实际缓存槽 | CPU 分别检查省略、0、正整数和 None 的转发；重跑原 head_num/变长 CPU golden 用例，确认原 `max_abs_diff=11.500977` 问题消失 |
| `npu_causal_conv1d_bwd` 拒绝文档中的 BSH 别名 | 将 BSH 规范化为已有 BSND 编码，保持 dispatcher schema 与 C++ 枚举不变 | CPU 检查别名转发；原 BSH × dtype/activation 的 CPU golden 用例须在 NPU 重跑 |

`tests/test_stable_only_contracts.py` 纳入普通 CPU CI 和 NPU CI 的环境契约阶段。
`tests/stable_public_api.json` 只更新 cumsum dtype 和 Conv1d fn/update null_block_id 的三个
精确 dispatcher 表达式，并记录变更理由；其余参数位置/表达式仍严格校验。
`tests/fixtures/stable_public_api.json`
保留原公开签名快照，不因修复行为而重生成。枚举名、顺序和整数编码继续双向核对。

## 3. 构建和交付

根目录与 torch_custom 子目录的 setup.py、build.sh、scripts/build_wheel.py 使用同一
构建策略。launcher、匹配的 `_stable_hash.py` 和所需运行依赖声明是 wheel 必选内容。
删除 pure-ctypes wheel、legacy 扩展编译、torchnpugen 生成链和 `--without-launcher` 豁免。
禁止将历史 `custom_aclnn_extension_lib*.so` 残留带入新产物。

`FLA_NPU_BUILD_STABLE_ABI=0` 和 `FLA_NPU_BUILD_LEGACY_EXTENSION=1` 拒绝执行；旧生成
开关移除。保留 SoC、host 架构、scoped OPP 和离线包构建能力。wheel 使用平台相关的
`py3-none-<platform>` 标签，不再提供 legacy CPython 标签分支。

实际构建器使用 vendored Stable C 面头文件，但仍使用 torch 安装中的链接库；不能宣称
完全无需 torch 构建。删除的是 torch_npu/torchnpugen 的 legacy 生成依赖，目标环境仍需
匹配的 torch/torch_npu、CANN 和驱动提供设备、内存分配及 stream 能力。

## 4. Overlay 后续适配

PR #878 单独跟进本变更。overlay 仍只带 OPP vendor，复用 wheel 根目录的唯一 launcher。
启用前同时校验公共 aclnn 契约、manifest 完整性与实际 launcher 的有序多库能力。
能力不足直接拒绝 overlay；删除自动改为 ctypes 和告警后混用 base host/overlay kernel。
保留 overlay→base→CANN 的按符号查找及独立管理工具的恢复能力。

launcher 能力应通过明确版本和实际加载库核验；不能仅扫描字符串，也不能在允许
FLA_NPU_STABLE_LIB 覆盖时只检查包内另一份库。新增算子或 ABI/schema 变化仍需完整 wheel。

## 5. 验证要求

| 范围 | 必须验证 |
|---|---|
| API | 全部 37 个入口、别名和 autograd 包装；首次调用前后签名；位置/关键字、默认值、None 和异常 |
| 运行时 | 所有算子命中 Stable；缺库、缺戳、缺注册及旧参数失败；mutation、梯度、设备/stream、生命周期 |
| 覆盖门禁 | 正常树通过；缺 wrapper/schema/run/注册/include、参数/枚举漂移、公共入口丢失、零解析均失败 |
| 打包 | 根/子目录入口；完整与 scoped 构建；无 launcher/hash/pin 和遗留扩展均拒绝；安装后真实 API 检查 |
| 调用方 | 仓内旧调用迁移；原输入、CPU 标杆、混合容差保持；ATK、Example/ST 和安装流程实际执行 |
| 数值回归 | 修改前后使用同一输入、环境、平台与设备 kernel；逐输出比较类型/shape/数值及状态，再对独立 CPU 标杆；上述接口修复单独验证预期行为变化，不豁免数值失败 |
| 平台 | A2/A3/A5 按各算子支持范围验证；单平台结果不能替代跨平台结论 |
| Overlay | base→overlay→移除生命周期，多层优先级、未覆盖算子、ABI 拒绝、旧 launcher 拒绝及损坏恢复 |

`scripts/verify_stable_only_results.py` 为既有测试入口记录公开调用的输入/输出摘要，比较
修改前后相同输入与逐字节结果。它补充独立 CPU/ATK 验证，不代替精度门禁。没有调用、
测试失败、调用集合变化或非有限输出均失败；不能通过跳过缺失设备或旧后端来宣称通过。

验证结果记录实际执行项、失败项与未执行项。只有相应平台全部要求完成后才标记该平台
验收完成；PR 的正式合入继续遵守仓库要求的两位批准和 A2+A5 NPU CI。

## 6. 首轮 A5 证据与未完成项

2026-10-23 首轮执行在上述接口修复前完成了 13 对 before/after 记录。对应
`*-before.json`、`*-after.json`、`*-compare.log` 及测试日志显示 3 对比较通过、10 对失败。
这只是首轮证据，不能作为接口修复后版本或全部 37 个算子的验收结论；后续重跑应记录
实际代码、launcher 构建戳和独立结果，不覆盖或混淆本轮结论。

| 首轮用例 | 观察到的结果与限制 |
|---|---|
| gelu、bwd-dqkwg | 两侧测试退出 0，记录比较通过 |
| prepare-da | 两侧各 2 次调用有限且一致；原脚本只计算 CPU golden，没有比较/断言，因此仅证明此次前后一致 |
| fwd-h、fwd-o | 两侧缺少 `ml_dtypes`，均为 0 次调用，必须补齐测试依赖后重跑 |
| recompute-wu、bwd-dv | 两侧缺少 `ct`，均为 0 次调用；bwd-dv 原有双 CPU golden 与明确的 `ct.dual(...)["success"]` 断言尚未执行 |
| prepare-full | 两侧各一次输出有限且一致，之后导入 `ct` 失败；四组 CPU 验证未执行，原脚本也未显式检查 `ct.dual` 返回值 |
| kkt | 首例 B=1、Hk=Hv=1、T=1、K=8、BT=16、FP16，两侧 GetWorkspaceSize 均以 561103/EZ1008 报 tiling 失败；有效 CPU golden、误差及零区域检查未执行到 |
| bwd-dhu | 两侧三次调用的 dh/dv2 均有非有限值，输入有限；原脚本只打印形状，没有 CPU 精度断言。尚无证据证明这是 padding 或记录工具误报 |
| cumsum | 两侧第三次调用均拒绝 float16；这是本次修复的接口缺口，须重编译并重跑 |
| conv1d | 两侧均出现变长 reshape 的 CPU golden 差异 `max_abs_diff=11.500977`，以及 update 预期 ValueError 未抛；前者的 None/null-slot 转发缺口已修复，仍需 NPU 重跑，不能由 CPU 参数测试代替数值验收 |
| conv1d-bwd | 两侧九个 BSH 子例均被 Stable 枚举拒绝；别名修复后须完整重跑 |

首轮比较没有报告修改前后结果或元数据差异，但“共同失败”并不等于通过。缺调用、未执行
CPU 检查、非有限输出和已知数值差异必须分别保留为验证缺口。设备 kernel 未修改也不能
据此推导数值正确，A5 单平台局部记录不能替代 ATK、Example/ST、全 37 入口或 A2+A5 CI。

`ct` 是测试工具依赖，其仓内指定来源为
[`精度对比与定位`](agents/reference/精度对比与定位.md#共同准备) 中的
[CT Tool v0.9.1](https://gitcode.com/Wei_NaChuan/ct/releases/v0.9.1)，安装包名为
`ct_tool-0.9.1-py3-none-any.whl`，安装后用 `ct --version` 核验。根 `requirements.txt`
已声明 `expecttest`，但没有声明 `ct_tool` 或 `ml_dtypes`；安装 `expecttest` 和 `ml_dtypes`
不能解决 `import ct` 失败。不得创建替代 `ct` 模块或放宽原金标/容差来使这些用例通过。

## 7. 修复后验证与提交状态（2026-10-23）

主干实现提交为 `521e1a63`、`48a9bf22`，测试依赖修复为 `50eab8c2`，对应
PR #961；overlay 已正常合入这些提交，代码提交为 `620ced27`，对应 PR #878。
下列结果来自实际 Ascend950/CANN 9.2.0、Python 3.12、torch/torch_npu 2.13 环境。
本节后的纯文档更新不改变已测试代码或 launcher 构建戳。

| 检查 | 实测结果 |
|---|---|
| 主干核心与 CI 契约 | Linux 核心 160 项通过，无跳过；CI unittest 150 项通过，无跳过 |
| Overlay Linux 回归 | 相同 core+overlay 套件 200 项通过，CI 完整套件 158 项通过，均无跳过；禁用 NPU 自动加载执行 |
| 公共签名及覆盖门禁 | 37 个入口、74 个公开名称的冻结签名与首次解析检查通过；strict coverage 37/37；缺失注册/入口、参数/枚举漂移等负向门禁通过 |
| 实际安装 | Stable-only 子目录 wheel、根目录 scoped wheel 原生构建成功；安装后 37 个实际 dispatcher 注册齐全 |
| Conv1d 前向/更新 | 原有 16 项测试通过，包含 head_num reshape、状态更新与 validate_data 负向检查；显式 None 与默认槽 0 的区别已独立复现 |
| Conv1d backward | 原有 3 组测试通过，包含先前被拒绝的 BSH 子例 |
| cumsum | 原有 35 个场景通过，覆盖三类输入 dtype、输出 dtype、same、定长/变长、尾块、reverse 与 scale |
| SolveTri | 原有 128 次调用的 CPU 检查通过；修改前后输入、逐输出与元数据比较通过 |
| KDA prepare | A5 支持的 BNSD 定长/NTD 变长组合共 16 个场景、48 个精确输出比较通过；同输入前后比较通过，不代表其他布局已支持 |
| 补充三个入口 | FastGELU forward、backward 零点导数及 KDA gate 的 3 项测试、4 次调用通过；前后比较通过。backward 零点不代表一般值域精度 |
| fwd-o / fwd-h 一致性 | fwd-o 完整 CLI 默认用例、fwd-h 既有 T=64/V=64 小 recipe 的前后记录通过；二者没有独立精度验收结论，小 recipe 不替代长用例 |
| 其他已有设备检查 | mutation、stream、输入生命周期、GDN 输出契约、preprocess、KDA prepare 负向和 launcher 失败行为通过；KDA intra 236 项通过 |
| 完整 OPP ABI | 实际头文件检查 29 个调用点，无不匹配；保留 SolveTri 已有公共头文件例外 |
| Overlay | 原生库的 capability=1，源码/Python/库内 stamp 一致，37 个注册齐全；base→overlay→移除的 cumsum 校验值为 8320→16640→8320，未覆盖算子哈希相同，base 文件哈希不变 |
| Overlay 拒绝与恢复 | 缺 manifest 安装被拒绝；损坏后 import 被拒绝；独立 status/remove 管理和恢复通过 |

初次编排中的测试依赖、漏传命令行参数、源码路径遮蔽已安装包，以及 SolveTri 的 300 秒
时限不足，均在原始日志中保留；修正编排后的结果单独记录，不能将编排失败作为产品回归。
elementwise 的门控 CPU reference 本身未改变，仅将无关的完整 fused reference 导入延迟
到实际使用处。Conv1d 的预期异常单独运行，严格结果记录器仍拒绝调用异常，不豁免负向记录。

**当前结论是可继续代码审查，尚未完成全面验收；两份 PR 保持 Draft。** 剩余条件包括：

- 尚无全部 37 个入口完整成功的前后数值记录；部分用例只证明前后一致，没有独立精度断言。
- KKT 小形状首例的 tiling 失败、DHU 非有限输出为修改前后共同失败，不能计作通过。
  KDA prepare 的完整通用矩阵含 A5 不支持的 BSND/TND 场景，完整矩阵仍失败；支持范围内
  的 16 项成功记录与完整矩阵结果分别保留。旧 fwd-h 独立脚本 T=256/V=128
  在基线出现 AIcore timeout（507014），小规模 recipe 通过不能豁免该失败。
- 本节执行时缺少仓内指定 CT Tool v0.9.1 和 ATK 可执行环境；当时 CT 下载返回
  HTTP 401。后续已取得官方工具并补跑，参见第 8 节；工具安装成功不等于算子验收通过。
- A5 主机 GCC 13 生成的 OPP host 库要求 GLIBCXX_3.4.32，高于 wheel 声明上限
  3.4.29，发布检查按预期拒绝。需用规定工具链重新构建，不能放宽 watermark；overlay
  launcher 自身的 ELF 审计通过不代表整包发布检查通过。
- A2/A3、最低 torch 版本及完整 dtype/layout/可选参数组合未全部实测。GitHub CI 当前
  为 `action_required`，正式 A2+A5 NPU CI 和两位批准尚未取得；本机测试不能替代这些条件。

## 8. 官方 ATK/CT 补验与验收脚本修复

从 [MindStudio ATK](https://gitcode.com/Ascend/ATK) 固定提交
`9504bee6076820ca74259fb919a790a12ef88cf8` 构建 ATK 26.9.8，满足最低版本 26.8.8。
wheel SHA256 为 `4e14fa2b66d7031b408165409576283ce4bd01b0bcc351753b25faf94e0f5836`。
`mixed_tolerance_bm` 使用官方公开实现，真实 CLI、executor 接口及 CPU 正反例检查通过。
从 [CT v0.9.1 官方发布页](https://gitcode.com/Wei_NaChuan/ct/releases/v0.9.1)
取得 wheel，其 SHA256 `0bdd83003c172f5dac8a010aff889400cc91f2caa6ba49f4c24127980f9b25c8`
与官方公布值一致。两者安装到独立 venv；原 zzy 的完整 distribution 清单前后相同，
torch/torch_npu 未变。`ct --version`、CLI 及 CPU 图形诊断已实际运行。

本次发现并修复以下测试问题；这些修改不改变设备 kernel、JSON 输入、随机种子或容差：

- ATK 汇总不能将缺报告、零用例、缺失字段和部分执行视为通过。四阶段现在要求本次新鲜
  报告、精确用例 ID/数量和逐例状态；性能必须有真实有限正耗时。
- 原 runner 未将 `MSS_LOG_PATH` 传给外层 mssanitizer，ATK 会把空日志判为成功。
  现在显式连接 `--log-file` 和 `-msl`，同时要求目标 OPP kernel 的开始与无错误完成记录。
  修复前的 XLSX `True` 不计内存验收；日志本身无 case ID，不能声称具备逐 case 日志关联。
- KDA fwd prepare executor 的 `a_log` 改为 Stable 的 `A_log`；preprocess backward
  负向检查识别 Stable 的 `GetWorkspaceSize failed` 状态码，仍拒绝无关异常或错误状态码。
- KKT ATK 标杆误将 beta 按列广播，已按接口 `beta[t]` 和既有独立 CPU reference
  修为按行广播。手算非等 beta 用例通过；BF16/FP16 × 3 seed 的同输入校准中，
  修正参考按官方混合容差 6/6 通过，原错误轴负控制 6/6 失败，正确参考最大误差
  不超过 `2.80e-9`。原始两条 NPU ATK 精度失败及基线同样失败的记录保留。
- CT `isclose` 和 `dual` 比较失败通常只返回状态。Recompute 的 w/u、prepare-full
  的四个输出现在均断言真实成功；bwd-dv 的双标杆参数修为 `test, gt=高精度真值,
  bench=低精度对照`。原来退出 0 却未检查比较结果的运行不能算精度通过。

资产审计发现 28 个 ATK 目录直接覆盖 28/37 个公共入口，固化 JSON 包含精度 5951 条、
性能 174 条和内存/确定性 578 条。仍有 9 个入口没有直接 ATK 工程，部分目录缺少
`all` 必需的性能或 MSS JSON。Conv1d 的状态/参数修复、BSH 别名、cumsum 新增 dtype
行为仍须保留第 7 节的专项测试；现有 ATK 的固定参数不能替代这些证据。

`all` 执行现有 JSON 的结果与覆盖完整性分别记录。完整公共运行时验收、正式 CI 和发布
工具链条件仍然有效，不因补齐 ATK/CT 或收紧测试门禁而豁免。
