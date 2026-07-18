# 经验与注意事项

## 文档和代码必须双向一致

文档里写“不支持”“仅支持”“必传”“预留参数不支持非空”等约束时，代码应有对应拦截；代码新增或收紧拦截时，文档应有同等语义说明。

检查时要双向走一遍：从文档逐条找代码拦截，从代码新增拦截逐条找文档说明。

## ABI 变更要显式说明

修改 `*_def.cpp`、`aclnn_*.h/.cpp`、PyTorch schema 或 opapi 适配代码时，可能影响 ABI 或调用兼容性。PR 中应说明影响范围、兼容策略和回归范围，并按 `.github/CODEOWNERS` 请求对应 owner 检视。

## 生成代码不要手改后遗忘来源

`torch_custom/fla_npu` 下部分文件由 YAML 和生成脚本产出。修改生成结果前，先确认是否应该改生成输入、生成脚本或模板。否则下一次生成可能覆盖手工修改。

## 旧调用路径只做兼容

先根据算子实现类型选择稳定 Python 入口：Ascend C 算子使用 `fla_npu.ops.ascendc`，Triton 算子使用 `fla_npu.ops.triton`，两者不是同一算子的必选双通路。`torch.ops.npu.*` 是可选 legacy 入口，只在实际实现时显式加载并做通路验证，不作为新增算子的主文档或主测试入口。

## wheel 公开 import 面只用 fla_npu

一体化 wheel 和 `torch_custom/fla_npu` standalone wheel 的 pip 项目名都应是
`flash-linear-attention-npu`，安装后的公开 import 名只应是 `fla_npu`。不要在
wheel 运行路径里依赖顶层 `fla` 包。

Triton 算子也应通过 `from fla_npu.ops.triton import 算子名` 暴露；如果代码里
出现 `from fla.ops...`，需要确认它只是源码树内部引用，不会进入安装后 wheel
的运行依赖。standalone wheel 只包含 Python 适配和 OPP 骨架，run 包覆盖后的
`site-packages/fla_npu/opp/vendors/fla_npu_transformer` 才是实际 Ascend C 产物入口。

## 变长序列、layout 和可选状态最容易漏

线性注意力类算子经常同时支持定长/变长序列、多种 layout、不同 head 关系、可选 initial/final state 和特殊 gate 语义。新增 case 时优先覆盖这些分支的组合，而不是只测默认 shape。

## head ratio 不要隐式假设为 1

反向和局部梯度算子常见输出 head 数与 Q/K head 数不相等的场景。具体符号名使用所属模型根 README 的权威符号表；实现中应显式推导并校验 head ratio，用输出 head 映射回 Q/K head，测试中也要覆盖 ratio 大于 1 的情况。否则小 shape 或默认 head 数能过，但 GVA/grouped 场景会读错 head、写错 workspace slot 或复用错误中间结果。

## workspace slot 是协议，不只是地址

多阶段 AIC/AIV 算子里，workspace slot 同时承载数据和同步协议。slot 的数量、复用顺序、ready/free flag、tail chunk 空任务都必须匹配。不要让 AIV 跳过空任务而 AIC 仍等待，也不要让 AIC 覆盖 AIV 尚未消费的 tile。

## 提交前清理产物

构建目录、wheel、run 包、测试输出、profiling 输出、临时 patch、缓存和本地调测脚本不应进入提交。提交前查看 `git status --short`，确认未跟踪文件中没有生成物或敏感信息。
