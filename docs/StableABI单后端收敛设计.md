# Stable ABI 单后端收敛

本设计以主干 `8762b6139de96e62205cefb1595aa41e38497e82` 为修改前基线。
目标是移除 ctypes 算子执行后端和 legacy PyTorch 扩展，所有 AscendC 公共入口只使用
`libfla_npu_stable.so`。Triton、KDA 的原 fused/V2 算法选择和算子 kernel 不在本次变更范围。

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

公开函数保持名称、参数、默认值、返回结构、autograd 和 mutation 契约。所有算子只解析
Stable wrapper 及其真实 dispatcher 注册；缺库、缺注册、缺失或不匹配的构建戳均保留
根本原因并直接报错。不存在自动回退和 legacy 扩展加载 API。

正常 import 保持延迟加载 torch 的边界，实际调用必须完成 launcher 验证。不得因删除
ctypes 注释或代码而删除版本计数、requires_grad 检查、每次取 stream、设备 guard 或
输入生命周期保障。`FLA_NPU_STABLE_ABI=ctypes` 和 `FLA_NPU_STABLE_VALIDATE=1` 明确报错。

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
| 数值回归 | 修改前后使用同一输入、环境、平台与 kernel；逐输出比较类型/shape/数值及状态；再对独立 CPU 标杆 |
| 平台 | A2/A3/A5 按各算子支持范围验证；单平台结果不能替代跨平台结论 |
| Overlay | base→overlay→移除生命周期，多层优先级、未覆盖算子、ABI 拒绝、旧 launcher 拒绝及损坏恢复 |

`scripts/verify_stable_only_results.py` 为既有测试入口记录公开调用的输入/输出摘要，比较
修改前后相同输入与逐字节结果。它补充独立 CPU/ATK 验证，不代替精度门禁。没有调用、
测试失败、调用集合变化或非有限输出均失败；不能通过跳过缺失设备或旧后端来宣称通过。

验证结果记录实际执行项、失败项与未执行项。只有相应平台全部要求完成后才标记该平台
验收完成；PR 的正式合入继续遵守仓库要求的两位批准和 A2+A5 NPU CI。
