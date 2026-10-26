# 已完成 issue 检视分类

- 仓库：`flashserve/flash-linear-attention-npu`
- 已完成 issue：188
- 生成方式：`python3 scripts/sync_issue_corpus.py`
- 约束：每条 `COMPLETED` issue 必须至少映射一个 rule；`NOT_PLANNED` 单独列在 JSON 快照中，不纳入已解决覆盖。

| Issue | 标题 | 检视规则 |
| --- | --- | --- |
| [#2](https://github.com/flashserve/flash-linear-attention-npu/issues/2) | No module named 'torchnpugen' | ENG-BUILD |
| [#12](https://github.com/flashserve/flash-linear-attention-npu/issues/12) | aclnnCausalConv1d.md 中 initialStateMode 用法不清晰，不知道在那种应用场景下使用哪种取值 | FUNC-STATE, DOC-CONSTRAINT |
| [#16](https://github.com/flashserve/flash-linear-attention-npu/issues/16) | No module named 'ct' | ENG-BUILD |
| [#21](https://github.com/flashserve/flash-linear-attention-npu/issues/21) | 使用ms-swift框架训练Qwen3.5-4B模型时出现了精度异常 | FUNC-PRECISION |
| [#39](https://github.com/flashserve/flash-linear-attention-npu/issues/39) | v_new溢出 | FUNC-PRECISION, FUNC-STATE |
| [#41](https://github.com/flashserve/flash-linear-attention-npu/issues/41) | Triton L2Norm recompiles for dynamic sequence lengths | FUNC-VARLEN |
| [#43](https://github.com/flashserve/flash-linear-attention-npu/issues/43) | The write address of the MTE instruction is out of range | FUNC-MEMORY |
| [#63](https://github.com/flashserve/flash-linear-attention-npu/issues/63) | 编译安装的依赖信息提示 | ENG-BUILD |
| [#68](https://github.com/flashserve/flash-linear-attention-npu/issues/68) | Feature Request: Support Prefix Caching | FEATURE-DELIVERY |
| [#77](https://github.com/flashserve/flash-linear-attention-npu/issues/77) | Docs] Clarify torch_npu and fla-linear-attention-npu version requirements in README | DOC-FORMAT |
| [#102](https://github.com/flashserve/flash-linear-attention-npu/issues/102) | 更新 flash_gated_delta_rule 示例使用 AscendC causal_conv1d 和 solve_tri | ENG-STRUCTURE |
| [#105](https://github.com/flashserve/flash-linear-attention-npu/issues/105) | v26.1.0速度提升数倍，但是训练过程梯度易爆炸 | PERF-REGRESSION |
| [#113](https://github.com/flashserve/flash-linear-attention-npu/issues/113) | 新增 KDA forward-only AscendC 算子支持 | FEATURE-DELIVERY |
| [#116](https://github.com/flashserve/flash-linear-attention-npu/issues/116) | chunk_fwd_h支持gk衰减 | FUNC-STATE |
| [#123](https://github.com/flashserve/flash-linear-attention-npu/issues/123) | 修复 wheel 打包安装后算子导出与 runtime 配置别名缺失问题 | ENG-STRUCTURE, ENG-BUILD |
| [#126](https://github.com/flashserve/flash-linear-attention-npu/issues/126) | 同步 v26.6.0 分支 NPU CI workflow 与主线 | FUNC-SYNC, CI-WORKFLOW |
| [#128](https://github.com/flashserve/flash-linear-attention-npu/issues/128) | 修复 NPU CI 精度检查失败时重复触发被跳过 | FUNC-PRECISION, CI-WORKFLOW |
| [#132](https://github.com/flashserve/flash-linear-attention-npu/issues/132) | CausalConv1dBwd开启算子异步下发报错 | FUNC-SYNC |
| [#134](https://github.com/flashserve/flash-linear-attention-npu/issues/134) | 修复 NPU CI 使用过期 merge ref 和 v26.6.0 用例未同步 | FUNC-SYNC, TEST-COVERAGE, CI-WORKFLOW |
| [#141](https://github.com/flashserve/flash-linear-attention-npu/issues/141) | 统一主线与稳定分支 wheel 命名规则 | ENG-STRUCTURE, ENG-BUILD |
| [#144](https://github.com/flashserve/flash-linear-attention-npu/issues/144) | v26.6.0 A5 example 在 solve_tri 不可用时需要回退到 Triton | ENG-COMPATIBILITY |
| [#147](https://github.com/flashserve/flash-linear-attention-npu/issues/147) | 构建多 SoC wheel 时重复编译 torch_custom 且部分环境导入探测过慢 | ENG-COMPATIBILITY, ENG-BUILD |
| [#149](https://github.com/flashserve/flash-linear-attention-npu/issues/149) | v26.6.0 A2: NPU out of memory. Tried to allocate more than 1EB memory | FUNC-MEMORY |
| [#155](https://github.com/flashserve/flash-linear-attention-npu/issues/155) | 【缺陷】chunk_bwd_dv_local 在 GVA-FIX4 场景执行超时 | FUNC-SYNC, FUNC-SHAPE |
| [#165](https://github.com/flashserve/flash-linear-attention-npu/issues/165) | 修复fwd_h在A5编译失败的问题 | ENG-BUILD |
| [#171](https://github.com/flashserve/flash-linear-attention-npu/issues/171) | aarch64 910b / 910_93 wheel包在cann9.0.0上安装后运行测试验证命令报错 | ENG-BUILD, TEST-COVERAGE |
| [#175](https://github.com/flashserve/flash-linear-attention-npu/issues/175) | 补充长序列 bf16 Example/ST 禁用用例 | FUNC-DTYPE-FORMAT, TEST-COVERAGE |
| [#182](https://github.com/flashserve/flash-linear-attention-npu/issues/182) | 增加 AI agent 仓库开发指南 | FEATURE-DELIVERY |
| [#185](https://github.com/flashserve/flash-linear-attention-npu/issues/185) | RecurrentGatedDeltaRule、CausalConv1d支持state非连续 | FUNC-DTYPE-FORMAT |
| [#187](https://github.com/flashserve/flash-linear-attention-npu/issues/187) | aarch64 910b  wheel包在cann8.5.2上安装torch_custom报No module named 'torchnpugen' | ENG-BUILD |
| [#188](https://github.com/flashserve/flash-linear-attention-npu/issues/188) | A5 GDN BSND 定长反向出现 non-finite | FUNC-PRECISION |
| [#189](https://github.com/flashserve/flash-linear-attention-npu/issues/189) | 新增 PrepareWyReprBwd 独立算子 | FEATURE-DELIVERY |
| [#203](https://github.com/flashserve/flash-linear-attention-npu/issues/203) | NPU CI 在维护分支 PR 上误用默认分支 CI 脚本 | CI-WORKFLOW |
| [#205](https://github.com/flashserve/flash-linear-attention-npu/issues/205) | docs: 补充 fla_npu 解耦运行时架构说明 | ENG-STRUCTURE |
| [#207](https://github.com/flashserve/flash-linear-attention-npu/issues/207) | Ascend950 上 GDN 反向算子可能发生同步超时及重复 S/V 同步 | FUNC-SYNC, ENG-COMPATIBILITY |
| [#210](https://github.com/flashserve/flash-linear-attention-npu/issues/210) | fix: v26.6.0 NPU CI 精度报告缺少触发元数据 | FUNC-PRECISION, CI-WORKFLOW |
| [#221](https://github.com/flashserve/flash-linear-attention-npu/issues/221) | [文档] 建立统一的算子开发规范与模板 | DOC-FORMAT |
| [#223](https://github.com/flashserve/flash-linear-attention-npu/issues/223) | 补充 prepare_wy_repr_bwd 融合算子支持 | ENG-STRUCTURE |
| [#225](https://github.com/flashserve/flash-linear-attention-npu/issues/225) | 测试与文档：补齐全量 Ascend C 算子三平台交付门禁 | DOC-FORMAT, TEST-COVERAGE, FEATURE-DELIVERY |
| [#226](https://github.com/flashserve/flash-linear-attention-npu/issues/226) | GDN triton_core kernel 失败：声明的最低版本 triton-ascend 3.2.1 缺失 tl.insert_slice | ENG-COMPATIBILITY |
| [#227](https://github.com/flashserve/flash-linear-attention-npu/issues/227) | [Feature]: 完善 KDA 正向算子的多平台与标准调用能力 | FEATURE-DELIVERY |
| [#230](https://github.com/flashserve/flash-linear-attention-npu/issues/230) | 支持 KDA fused recurrent 前向 API | ENG-STRUCTURE |
| [#234](https://github.com/flashserve/flash-linear-attention-npu/issues/234) | [Bug]: A3+CANN 9.0.0 算子验证失败 | FUNC-REGRESSION |
| [#236](https://github.com/flashserve/flash-linear-attention-npu/issues/236) | [Bug]: A5 chunk_gated_delta_rule_fwd_h 固定输入输出不确定 | FUNC-DETERMINISM, FUNC-OPTIONAL-OUTPUT |
| [#238](https://github.com/flashserve/flash-linear-attention-npu/issues/238) | [构建] v26.6.0 方式 A 无法在受限网络环境一键编包 | ENG-BUILD |
| [#242](https://github.com/flashserve/flash-linear-attention-npu/issues/242) | [缺陷] solve_tri TND 非 16 对齐尾块可能越界读取并产生 NaN | FUNC-PRECISION, FUNC-MEMORY, FUNC-VARLEN |
| [#248](https://github.com/flashserve/flash-linear-attention-npu/issues/248) | test: 增加 GDN 100 层纯异步稳定性复现脚本 | FEATURE-DELIVERY |
| [#250](https://github.com/flashserve/flash-linear-attention-npu/issues/250) | [Bug]: prepare_wy_repr_bwd_full 检测失败 | FUNC-REGRESSION |
| [#251](https://github.com/flashserve/flash-linear-attention-npu/issues/251) | [Bug]: RecomputeWUFwd 源码命名与运行时配置名不一致 | ENG-STRUCTURE, FUNC-REGRESSION |
| [#255](https://github.com/flashserve/flash-linear-attention-npu/issues/255) | [Bug] 一键 wheel 的自定义 libopapi.so 遮蔽 CANN 运行库 | ENG-BUILD |
| [#258](https://github.com/flashserve/flash-linear-attention-npu/issues/258) | [Bug]: AscendC ctypes runtime 连续调用长期持有 aclnn workspace | FUNC-MEMORY, FUNC-DTYPE-FORMAT |
| [#261](https://github.com/flashserve/flash-linear-attention-npu/issues/261) | [Feature]: 建立算子原型和公开接口兼容性红线 | ENG-STRUCTURE, ENG-COMPATIBILITY, FEATURE-DELIVERY |
| [#263](https://github.com/flashserve/flash-linear-attention-npu/issues/263) | [Feature]: 优化 A5 KDA chunk64 K128/V128 前向性能 | PERF-REGRESSION, FEATURE-DELIVERY |
| [#267](https://github.com/flashserve/flash-linear-attention-npu/issues/267) | [Bug]: 基于Q3版本的CANN编译FLA时报错 | ENG-BUILD, FUNC-REGRESSION |
| [#272](https://github.com/flashserve/flash-linear-attention-npu/issues/272) | [Bug]: 强化学习推理正常，训练失败 | FUNC-REGRESSION |
| [#278](https://github.com/flashserve/flash-linear-attention-npu/issues/278) | [Bug]: A5安装该库后执行readme里的测试命令也会报错 | ENG-BUILD, DOC-FORMAT, TEST-COVERAGE, FUNC-REGRESSION |
| [#281](https://github.com/flashserve/flash-linear-attention-npu/issues/281) | [Bug]: A5 KDA BF16 sub-16 尾块输出不确定且精度异常 | FUNC-PRECISION, FUNC-DETERMINISM, FUNC-VARLEN, FUNC-DTYPE-FORMAT, FUNC-OPTIONAL-OUTPUT |
| [#285](https://github.com/flashserve/flash-linear-attention-npu/issues/285) | [Bug]: 主线上AscendC算子chunk_bwd_dqkwg报错 | FUNC-REGRESSION |
| [#289](https://github.com/flashserve/flash-linear-attention-npu/issues/289) | [Bug]: 主线ascendc_causal_conv1d_bwd报错 | FUNC-REGRESSION |
| [#290](https://github.com/flashserve/flash-linear-attention-npu/issues/290) | [Feature]: 优化 chunk_gated_delta_rule_bwd_dhu 并补充 A5 实现 | PERF-REGRESSION, FEATURE-DELIVERY |
| [#293](https://github.com/flashserve/flash-linear-attention-npu/issues/293) | [Bug]: 安装过程中算子recompute_wu_fwd测试失败 | ENG-BUILD, TEST-COVERAGE, FUNC-REGRESSION |
| [#303](https://github.com/flashserve/flash-linear-attention-npu/issues/303) | [Bug]: A5算子调用报错RuntimeError: aclnnCausalConv1dGetWorkspaceSize failed with aclnnStatus=161001. | FUNC-MEMORY |
| [#304](https://github.com/flashserve/flash-linear-attention-npu/issues/304) | [Bug]: 安装过一遍fla后，重新进入conda环境运行报错Unable to find FLA NPU custom OPP. | ENG-BUILD, FUNC-REGRESSION |
| [#305](https://github.com/flashserve/flash-linear-attention-npu/issues/305) | [Bug]: chunk_bwd_dv_local算子标杆和底层实现不同 | TEST-COVERAGE, FUNC-REGRESSION |
| [#307](https://github.com/flashserve/flash-linear-attention-npu/issues/307) | [Bug]: [bwd_dhu] FP16 多 chunk 用例的 dv2 因小值域 ERR_COUNT_ratio 集中失败 | FUNC-DTYPE-FORMAT, TEST-COVERAGE |
| [#308](https://github.com/flashserve/flash-linear-attention-npu/issues/308) | [Bug]: [ chunk_bwd_dv_local]精度通过率整体较低（90±3％） | FUNC-PRECISION |
| [#309](https://github.com/flashserve/flash-linear-attention-npu/issues/309) | [Bug]: [chunk_gated_delta_rule_bwd_dhu]负向边界问题 | FUNC-REGRESSION |
| [#310](https://github.com/flashserve/flash-linear-attention-npu/issues/310) | [Bug]: 客户SFT训练接入FLA_NPU，训练结束后会抛double free异常 | FUNC-MEMORY |
| [#312](https://github.com/flashserve/flash-linear-attention-npu/issues/312) | [Bug]: ChunkGatedDeltaRuleBwdDhu 的 h0/dht shape 文档要求与 NPU Runtime 要求不一致 | FUNC-SHAPE, FUNC-STATE, DOC-FORMAT |
| [#313](https://github.com/flashserve/flash-linear-attention-npu/issues/313) | [Bug]: ChunkGatedDeltaRuleBwdDhu 在合法 K/V 尺寸下触发 AICore MTE 对齐异常 | FUNC-MEMORY, FUNC-SHAPE |
| [#315](https://github.com/flashserve/flash-linear-attention-npu/issues/315) | [Bug] 根目录一键 wheel 漏打 fla_npu Triton 算子源码 | ENG-STRUCTURE, ENG-BUILD |
| [#321](https://github.com/flashserve/flash-linear-attention-npu/issues/321) | [Bug]: A3 + CANN9.2.0.B020 + torchnpu2.10.0.post5.dev20260815 安装成功，但测试算子失败。 | ENG-BUILD, TEST-COVERAGE, FUNC-REGRESSION |
| [#323](https://github.com/flashserve/flash-linear-attention-npu/issues/323) | [Bug]: 26.6.0分支编译报错--重复链接 | ENG-BUILD, FUNC-REGRESSION |
| [#325](https://github.com/flashserve/flash-linear-attention-npu/issues/325) | [Bug]: chunk_kda_fwd 边界场景存在流水同步与尾块精度问题 | FUNC-SYNC, FUNC-PRECISION, FUNC-VARLEN |
| [#326](https://github.com/flashserve/flash-linear-attention-npu/issues/326) | [Bug]: 将 torch_npu/torchnpugen 支持判断从版本门禁拆分为能力探测 | ENG-COMPATIBILITY, FUNC-REGRESSION |
| [#337](https://github.com/flashserve/flash-linear-attention-npu/issues/337) | [Bug]: A5 ChunkKdaFwd 多 chunk 场景可能触发设备任务超时 | FUNC-SYNC |
| [#343](https://github.com/flashserve/flash-linear-attention-npu/issues/343) | [Feature]: 为 ChunkGatedDeltaRuleBwdDhu 增加 use_exp2 门控参数 | FEATURE-DELIVERY |
| [#349](https://github.com/flashserve/flash-linear-attention-npu/issues/349) | [Bug]: A5 ChunkKdaFwd 短尾块输出存在 bitwise determinism 不一致 | FUNC-PRECISION, FUNC-DETERMINISM, FUNC-VARLEN |
| [#351](https://github.com/flashserve/flash-linear-attention-npu/issues/351) | [Bug]: ChunkGatedDeltaRuleBwdDhu 三头任务在 A5 上可能无法完成 | FUNC-SHAPE |
| [#360](https://github.com/flashserve/flash-linear-attention-npu/issues/360) | [Bug]: A5 ChunkScaledDotKkt 重复调用存在同步事件复用风险 | FUNC-SYNC |
| [#362](https://github.com/flashserve/flash-linear-attention-npu/issues/362) | [Bug]: recurrent_gated_delta_rule 稳定入口、非连续 state 与尾块场景验证不完整 | FUNC-VARLEN, FUNC-DTYPE-FORMAT, FUNC-STATE, ENG-STRUCTURE |
| [#365](https://github.com/flashserve/flash-linear-attention-npu/issues/365) | [Bug]: ATK 公共 runner 四 scope 调用存在跨版本兼容与结果误判 | ENG-COMPATIBILITY, TEST-COVERAGE, CI-WORKFLOW, FUNC-REGRESSION |
| [#367](https://github.com/flashserve/flash-linear-attention-npu/issues/367) | [Bug]: A2 变长 TND 场景 fwd_h v_new 精度回归 | FUNC-PRECISION, FUNC-VARLEN, FUNC-STATE |
| [#369](https://github.com/flashserve/flash-linear-attention-npu/issues/369) | [Feature]: 按 Stage 模块化重构 FwdH 并建立逐阶段精度闭环 | FUNC-PRECISION, FEATURE-DELIVERY |
| [#371](https://github.com/flashserve/flash-linear-attention-npu/issues/371) | [Bug]: vllm-ascend 和fla-npu同时安装时GDN算子报错 | ENG-BUILD, FUNC-REGRESSION |
| [#384](https://github.com/flashserve/flash-linear-attention-npu/issues/384) | [Bug]: examples README 与仓库实际示例不一致 | DOC-FORMAT, FUNC-REGRESSION |
| [#385](https://github.com/flashserve/flash-linear-attention-npu/issues/385) | [Feature]: ATK 测试资产对齐开源混合容差单标杆 | TEST-COVERAGE, FEATURE-DELIVERY |
| [#403](https://github.com/flashserve/flash-linear-attention-npu/issues/403) | [Feature]: 新增 GDN 反向 finalize Ascend 950 算子 | FEATURE-DELIVERY |
| [#405](https://github.com/flashserve/flash-linear-attention-npu/issues/405) | [Feature]: 新增独立 chunk_fwd_h Ascend C 入口并按有效 head task 动态分核 | FUNC-SHAPE, FEATURE-DELIVERY |
| [#407](https://github.com/flashserve/flash-linear-attention-npu/issues/407) | [Bug]: GDN Phase6 final_state 精度在尾块仅含 1–15 个有效 token 时显著下降 | FUNC-PRECISION, FUNC-VARLEN |
| [#408](https://github.com/flashserve/flash-linear-attention-npu/issues/408) | [Feature] ChunkBwdDqkwg性能优化 | PERF-REGRESSION, FEATURE-DELIVERY |
| [#410](https://github.com/flashserve/flash-linear-attention-npu/issues/410) | [Feature]: 精简算子开发 Agent 文档结构 | DOC-FORMAT, FEATURE-DELIVERY |
| [#412](https://github.com/flashserve/flash-linear-attention-npu/issues/412) | [Feature]: 限定仓内 Agent 流程来源并隔离新算子设计 | FEATURE-DELIVERY |
| [#418](https://github.com/flashserve/flash-linear-attention-npu/issues/418) | [Feature]: 新增 ChunkGdnBwdIntra Ascend 950 融合算子 | FEATURE-DELIVERY |
| [#427](https://github.com/flashserve/flash-linear-attention-npu/issues/427) | [Question]: A3服务器FLA_NPU_SOC=ascend910_93 python -m pip wheel --no-build-isolation --no-deps . -w dist失败 | ENG-COMPATIBILITY, ENG-BUILD |
| [#428](https://github.com/flashserve/flash-linear-attention-npu/issues/428) | [Feature]: 补齐 chunk_kda_fwd ATK 精度、内存与确定性测试交付 | FUNC-PRECISION, FUNC-MEMORY, TEST-COVERAGE, FEATURE-DELIVERY |
| [#429](https://github.com/flashserve/flash-linear-attention-npu/issues/429) | [Bug]: CANN 先于 fla_npu 初始化时内嵌 OPP 未注册导致 RGDR 报 561103 | DFX-MESSAGE, ENG-BUILD, FUNC-REGRESSION |
| [#437](https://github.com/flashserve/flash-linear-attention-npu/issues/437) | [Feature]: chunk_gated_delta_rule_fwd 支持 A2/A3/A5 Phase6 架构隔离实现 | ENG-COMPATIBILITY, FEATURE-DELIVERY |
| [#440](https://github.com/flashserve/flash-linear-attention-npu/issues/440) | [Bug]: A3 ChunkKdaFwd varlen 路径输出 bitwise 不确定 | FUNC-PRECISION, FUNC-DETERMINISM, FUNC-VARLEN |
| [#445](https://github.com/flashserve/flash-linear-attention-npu/issues/445) | [Bug]: 4个GDN算子L2_DFX_PHASE_1宏校验缺失参数；4个L2_DFX_PHASE_1宏校验参数顺序和函数入口参数顺序不一致； | DFX-VALIDATION, DFX-MESSAGE, FUNC-REGRESSION |
| [#448](https://github.com/flashserve/flash-linear-attention-npu/issues/448) | [Bug]: A5 chunk_kda_bwd 融合 Gate 的 shared-anchor 路径产生 NaN | FUNC-PRECISION, FUNC-MEMORY, FUNC-STATE |
| [#452](https://github.com/flashserve/flash-linear-attention-npu/issues/452) | [Bug]: ChunkBwdDqkwg不支持<<<>>>调用 | DFX-VALIDATION, FUNC-REGRESSION |
| [#456](https://github.com/flashserve/flash-linear-attention-npu/issues/456) | [Bug]: chunk_bwd_dv_local 精度测试在 T>1 场景全量失败 | FUNC-PRECISION, TEST-COVERAGE |
| [#459](https://github.com/flashserve/flash-linear-attention-npu/issues/459) | [Feature]: 实现 A2/A3/A5 chunk_kda_fwd_prepare 算子 | ENG-COMPATIBILITY, FEATURE-DELIVERY |
| [#462](https://github.com/flashserve/flash-linear-attention-npu/issues/462) | [Bug]: chunk_bwd_dv_local 在特定 GQA head 配置下触发 NPU kernel hang | FUNC-SYNC, FUNC-SHAPE |
| [#471](https://github.com/flashserve/flash-linear-attention-npu/issues/471) | [Feature]: 新增 ChunkGatedDeltaRuleFwdPrepare Ascend 950 融合实现 | FEATURE-DELIVERY |
| [#482](https://github.com/flashserve/flash-linear-attention-npu/issues/482) | [Bug]: FLA_NPU_OPS 指定不支持算子时缺少前置校验和明确报错 | DFX-VALIDATION, FUNC-REGRESSION |
| [#483](https://github.com/flashserve/flash-linear-attention-npu/issues/483) | [Bug]: flash-linear-attention-npu在A5环境执行精度不达标 | FUNC-PRECISION |
| [#486](https://github.com/flashserve/flash-linear-attention-npu/issues/486) | [Bug]: A5 FwdH 缺少 #188/#349 派生场景的独立跨版本回归 | FUNC-REGRESSION |
| [#488](https://github.com/flashserve/flash-linear-attention-npu/issues/488) | [Feature]: 优化 A5 chunk_fwd_h 流式输入的 L2 cache 策略 | PERF-REGRESSION, FEATURE-DELIVERY |
| [#497](https://github.com/flashserve/flash-linear-attention-npu/issues/497) | [Bug]: chunk_bwd_dv_local 变长模式下非法 cu_seqlens 和 chunk_indices 未被拦截 | FUNC-VARLEN, DFX-VALIDATION |
| [#498](https://github.com/flashserve/flash-linear-attention-npu/issues/498) | [Bug]: chunk_bwd_dv_local 变长模式下非法 chunk_indices 导致 AICore 异常 | FUNC-VARLEN, DFX-VALIDATION |
| [#515](https://github.com/flashserve/flash-linear-attention-npu/issues/515) | [Bug] aclnnRecomputeWUFwd使用ATK验证pyaclnn，构造非法输入时，返回错误码与预期不一致 | DFX-VALIDATION, DFX-MESSAGE, TEST-COVERAGE |
| [#519](https://github.com/flashserve/flash-linear-attention-npu/issues/519) | [Bug]: chunk_kda_fwd A2/A3精度不达标 | FUNC-PRECISION, ENG-COMPATIBILITY |
| [#521](https://github.com/flashserve/flash-linear-attention-npu/issues/521) | [Feature]: 为 ChunkGatedDeltaRuleFwd Torch 接口增加分块状态 h 输出 | FUNC-STATE, FEATURE-DELIVERY |
| [#524](https://github.com/flashserve/flash-linear-attention-npu/issues/524) | [Feature]: CI 新增 A5（ascend950）双平台 NPU 验证 | ENG-COMPATIBILITY, CI-WORKFLOW, FEATURE-DELIVERY |
| [#525](https://github.com/flashserve/flash-linear-attention-npu/issues/525) | [Bug]: ascend950 dense backward 在 recompute_w_u_fwd 触发 AICore 超时 | FUNC-SYNC, ENG-COMPATIBILITY |
| [#526](https://github.com/flashserve/flash-linear-attention-npu/issues/526) | [Feature]: A2 GDN 接入分层 FP32 Solve 并完善稳定性验收 | FUNC-DTYPE-FORMAT, FEATURE-DELIVERY |
| [#529](https://github.com/flashserve/flash-linear-attention-npu/issues/529) | [Bug]: chunk_bwd_dv_local A2 A5精度测试/复检失败 | FUNC-PRECISION, TEST-COVERAGE |
| [#531](https://github.com/flashserve/flash-linear-attention-npu/issues/531) | [Feature]: 新增 ChunkGatedDeltaRuleBwd 融合接口 | FEATURE-DELIVERY |
| [#533](https://github.com/flashserve/flash-linear-attention-npu/issues/533) | [Bug]: chunk_fwd_o存在一条小case性能不达标兼部分gpu性能测试未提供结果 | TEST-COVERAGE, PERF-REGRESSION, FUNC-REGRESSION |
| [#539](https://github.com/flashserve/flash-linear-attention-npu/issues/539) | [Feature]: GDN A5 拼接路径支持 use_exp2=false | FEATURE-DELIVERY |
| [#543](https://github.com/flashserve/flash-linear-attention-npu/issues/543) | [Bug]: 【chunk_fwd_o】算子精度特定用例不通过（gpu双标杆） | FUNC-PRECISION, TEST-COVERAGE |
| [#544](https://github.com/flashserve/flash-linear-attention-npu/issues/544) | [Bug]: KDA AscendC 变长序列反向精度问题 | FUNC-PRECISION, FUNC-VARLEN |
| [#547](https://github.com/flashserve/flash-linear-attention-npu/issues/547) | [Feature]: 将 DHU 设计文档迁移到算子目录 | ENG-STRUCTURE, DOC-FORMAT, FEATURE-DELIVERY |
| [#550](https://github.com/flashserve/flash-linear-attention-npu/issues/550) | [Bug]: solve_tril算子问题 | FUNC-REGRESSION |
| [#554](https://github.com/flashserve/flash-linear-attention-npu/issues/554) | [Bug]: solve_tri 在A2/A3/A5 环境执行精度不达标 | FUNC-PRECISION, ENG-COMPATIBILITY |
| [#555](https://github.com/flashserve/flash-linear-attention-npu/issues/555) | [Bug]: solve_tril A2/A3 部分用例确定性计算存在异常 | ENG-COMPATIBILITY, TEST-COVERAGE, FUNC-REGRESSION |
| [#557](https://github.com/flashserve/flash-linear-attention-npu/issues/557) | [Bug]: chunk_scaled_dot_kkt A2环境性能不达标，对比npu triton算子性能劣化 | PERF-REGRESSION, FUNC-REGRESSION |
| [#558](https://github.com/flashserve/flash-linear-attention-npu/issues/558) | [Bug]: chunk_scaled_dot_kkt 对于异常场景拦截缺失，报错信息无符合标准提示 | DFX-VALIDATION, DFX-MESSAGE, FUNC-REGRESSION |
| [#561](https://github.com/flashserve/flash-linear-attention-npu/issues/561) | [Bug]: ChunkBwdDqkwg对于异常场景拦截缺失，报错信息无符合标准提示 | DFX-VALIDATION, DFX-MESSAGE, FUNC-REGRESSION |
| [#565](https://github.com/flashserve/flash-linear-attention-npu/issues/565) | [Feature]: 新增 A2/A3/A5 chunk_kda_fwd_finalize 算子与稳定 Python 入口 | ENG-COMPATIBILITY, FEATURE-DELIVERY |
| [#573](https://github.com/flashserve/flash-linear-attention-npu/issues/573) | [Bug]: causal_conv1d_bwd部分异常场景未做拦截 | DFX-VALIDATION, FUNC-REGRESSION |
| [#574](https://github.com/flashserve/flash-linear-attention-npu/issues/574) | [Bug]: causal_conv1d_bwd资料有误,错误码,编译样例链接404 | DFX-MESSAGE, ENG-BUILD, DOC-FORMAT, FUNC-REGRESSION |
| [#575](https://github.com/flashserve/flash-linear-attention-npu/issues/575) | [Bug]: chunk_kda_fwd内存检测失败 | FUNC-MEMORY |
| [#576](https://github.com/flashserve/flash-linear-attention-npu/issues/576) | [Bug]: chunk_kda_fwd A3性能不达标 | PERF-REGRESSION, FUNC-REGRESSION |
| [#577](https://github.com/flashserve/flash-linear-attention-npu/issues/577) | [Bug]: chunk_kda_fwd 空tensor异常用例，没有返回错误信息，只有错误码 | FUNC-SHAPE, DFX-MESSAGE, TEST-COVERAGE |
| [#578](https://github.com/flashserve/flash-linear-attention-npu/issues/578) | [Bug]: chunk_kda_fwd 资料约束和实际约束不一致 | DOC-FORMAT, DOC-CONSTRAINT, FUNC-REGRESSION |
| [#579](https://github.com/flashserve/flash-linear-attention-npu/issues/579) | 修复 GDN 正向大融合算子 RegBase 芯片路径判断 | ENG-COMPATIBILITY |
| [#581](https://github.com/flashserve/flash-linear-attention-npu/issues/581) | [Bug]: GDN 正向混合精度与六输出接口不一致 | FUNC-PRECISION, FUNC-DTYPE-FORMAT, FUNC-OPTIONAL-OUTPUT |
| [#591](https://github.com/flashserve/flash-linear-attention-npu/issues/591) | [Bug]: chunk_local_cumsum A2性能不达标 | PERF-REGRESSION, FUNC-REGRESSION |
| [#592](https://github.com/flashserve/flash-linear-attention-npu/issues/592) | [Bug]: chunk_local_cumsum A2/5精度不通过 | FUNC-PRECISION |
| [#594](https://github.com/flashserve/flash-linear-attention-npu/issues/594) | [Bug]: chunk_local_cumsum 资料中包含敏感信息 | DOC-FORMAT, FUNC-REGRESSION |
| [#598](https://github.com/flashserve/flash-linear-attention-npu/issues/598) | [Bug]: 26.9.0分支安装测试recompute_wu_fwd报错 | ENG-BUILD, TEST-COVERAGE, FUNC-REGRESSION |
| [#599](https://github.com/flashserve/flash-linear-attention-npu/issues/599) | [Bug]: chunk_gated_delta_rule_bwd_dhu算子的aclnn接口校验不完整 | DFX-VALIDATION, FUNC-REGRESSION |
| [#600](https://github.com/flashserve/flash-linear-attention-npu/issues/600) | [Bug]: NPU CI 的 wheel OPP 安装校验不识别 .data 载荷布局，A5 源码准备会被过期 test merge 阻塞 | FUNC-DTYPE-FORMAT, DFX-VALIDATION, ENG-BUILD, CI-WORKFLOW |
| [#607](https://github.com/flashserve/flash-linear-attention-npu/issues/607) | [Feature]: 优化 A5 GDN 私有 H/VNew 与 H 到 O 流水并泛化支持范围 | DOC-CONSTRAINT, PERF-REGRESSION, FEATURE-DELIVERY |
| [#609](https://github.com/flashserve/flash-linear-attention-npu/issues/609) | [Document]: chunk_gated_delta_rule_bwd_dhu文档补充 | DOC-FORMAT |
| [#612](https://github.com/flashserve/flash-linear-attention-npu/issues/612) | [Bug]: recompute_w_u_fwd算子资料问题 | DOC-FORMAT, FUNC-REGRESSION |
| [#613](https://github.com/flashserve/flash-linear-attention-npu/issues/613) | [Bug]: recompute_w_u_fwd构造K取值非法的异常值用例，执行失败，报507015，没有校验拦截 | DFX-VALIDATION, TEST-COVERAGE, FUNC-REGRESSION |
| [#614](https://github.com/flashserve/flash-linear-attention-npu/issues/614) | [Bug]: recompute_w_u_fwd内存占用测试，有4个用例未达标 | FUNC-MEMORY, TEST-COVERAGE |
| [#615](https://github.com/flashserve/flash-linear-attention-npu/issues/615) | [Bug]: MindSpore 调用 ChunkKdaBwd 时 OriginalShape 展平导致 tiling rank 误判 | FUNC-SHAPE |
| [#616](https://github.com/flashserve/flash-linear-attention-npu/issues/616) | [Bug]: recompute_w_u_fwd异常值用例没有返回错误信息，只有错误码 | DFX-MESSAGE, TEST-COVERAGE, FUNC-REGRESSION |
| [#618](https://github.com/flashserve/flash-linear-attention-npu/issues/618) | [Bug]: FLA_NPU_OPS 一键编包指定单个大融合算子时，依赖算子的 kernel 产物缺失 | ENG-BUILD, FUNC-REGRESSION |
| [#627](https://github.com/flashserve/flash-linear-attention-npu/issues/627) | [Bug]: FLA_NPU编译安装过程太久 | ENG-BUILD, FUNC-REGRESSION |
| [#636](https://github.com/flashserve/flash-linear-attention-npu/issues/636) | [Bug]: causal_conv1d即使不调用也会导致上下文变量异常 | FUNC-REGRESSION |
| [#640](https://github.com/flashserve/flash-linear-attention-npu/issues/640) | [Bug]: recurrent_gated_delta_rule在A3/A5有用例精度测试失败，有infnan校验不过也有复检不过 | FUNC-PRECISION, DFX-VALIDATION, ENG-COMPATIBILITY, TEST-COVERAGE |
| [#641](https://github.com/flashserve/flash-linear-attention-npu/issues/641) | [Bug]: recurrent_gated_delta_rule有异常场景未拦截 | DFX-VALIDATION, FUNC-REGRESSION |
| [#694](https://github.com/flashserve/flash-linear-attention-npu/issues/694) | feat(kda): 前向导出反向 L2Norm 保存值，并暴露 chunk_kda_fwd_prepare 公共入口 | ENG-STRUCTURE |
| [#695](https://github.com/flashserve/flash-linear-attention-npu/issues/695) | [Bug]: 过滤构建缺依赖算子配置，chunk_kda_fwd 的 V2 组合入口运行期报 EZ1013/561103 | DFX-MESSAGE, ENG-BUILD |
| [#720](https://github.com/flashserve/flash-linear-attention-npu/issues/720) | [Feature]: chunk_gated_delta_rule_bwd_dhu在 A5 上，性能提升20% | PERF-REGRESSION, FEATURE-DELIVERY |
| [#723](https://github.com/flashserve/flash-linear-attention-npu/issues/723) | [Feature]: A2 GDN 原生TND输入与O直接写回 | FUNC-VARLEN, FEATURE-DELIVERY |
| [#725](https://github.com/flashserve/flash-linear-attention-npu/issues/725) | [Feature]: 将 GDN A2/A5 megakernel 回迁至 v26.9.0 | FEATURE-DELIVERY |
| [#731](https://github.com/flashserve/flash-linear-attention-npu/issues/731) | GDN正向大融合算子精度复检不通过 | FUNC-PRECISION, TEST-COVERAGE |
| [#733](https://github.com/flashserve/flash-linear-attention-npu/issues/733) | 移除 KDA 反向重计算模式的序列长度对齐限制 | ENG-STRUCTURE |
| [#736](https://github.com/flashserve/flash-linear-attention-npu/issues/736) | [Bug]: chunk_gated_delta_rule_bwd算子泛化用例精度测试，少量用例精度不达标 | FUNC-PRECISION, TEST-COVERAGE |
| [#759](https://github.com/flashserve/flash-linear-attention-npu/issues/759) | [Bug]: GDN正向大融合算子对空tensor处理不符合预期 | FUNC-SHAPE |
| [#760](https://github.com/flashserve/flash-linear-attention-npu/issues/760) | [Bug]: KDA算子支持反向megakernel在A3上精度、确定性、特殊值精度、内存占用不通过，性能采集结果不正确 | FUNC-PRECISION, FUNC-MEMORY, PERF-REGRESSION |
| [#764](https://github.com/flashserve/flash-linear-attention-npu/issues/764) | [Feature]: 新增 CP 反向状态预处理算子pre_process_bwd_kernel_merged | FUNC-STATE, FEATURE-DELIVERY |
| [#766](https://github.com/flashserve/flash-linear-attention-npu/issues/766) | [Bug]: 【GDN main】【算子】【功能】支持 npu_chunk_gated_delta_rule_fwd 融合算子异常值测试时只有错误码看不到具体的信息 | DFX-MESSAGE, TEST-COVERAGE, FUNC-REGRESSION |
| [#771](https://github.com/flashserve/flash-linear-attention-npu/issues/771) | 【GDN main】【算子】【功能】支持 npu_chunk_gated_delta_rule_bwd 融合算子异常值测试时只有错误码看不到具体的信息 | DFX-MESSAGE, TEST-COVERAGE, FUNC-REGRESSION |
| [#772](https://github.com/flashserve/flash-linear-attention-npu/issues/772) | [Bug]: FLA_NPU_BUILD_LEGACY_EXTENSION=1 构建 legacy 扩展链接失败：npu_fast_gelu_custom 已声明并注册但无定义 | ENG-BUILD, FUNC-REGRESSION |
| [#773](https://github.com/flashserve/flash-linear-attention-npu/issues/773) | [Bug]: FLA_NPU_BUILD_LEGACY_EXTENSION=1 构建 legacy 扩展链接失败：npu_fast_gelu_custom 已声明并注册但无定义 | ENG-BUILD, FUNC-REGRESSION |
| [#774](https://github.com/flashserve/flash-linear-attention-npu/issues/774) | [Bug]: 【GDN main】【算子】【功能】支持 npu_chunk_gated_delta_rule_bwd 融合算子测试传空tensor不符合预期 | FUNC-SHAPE, TEST-COVERAGE |
| [#782](https://github.com/flashserve/flash-linear-attention-npu/issues/782) | [Bug]: ChunkGatedDeltaRuleBwd 不支持空 Tensor 输入 | FUNC-SHAPE, DFX-VALIDATION |
| [#783](https://github.com/flashserve/flash-linear-attention-npu/issues/783) | fix: 一键编包安装提示可能误选 dist 中的历史 wheel | ENG-BUILD, FUNC-REGRESSION |
| [#793](https://github.com/flashserve/flash-linear-attention-npu/issues/793) | [Bug]:【GDN main】【算子】【功能】支持 npu_chunk_gated_delta_rule_bwd 融合算子 内存检测有不符合预期ERROR | FUNC-MEMORY |
| [#820](https://github.com/flashserve/flash-linear-attention-npu/issues/820) | [Bug]: A5上KDA算子支持反向megakernel泛化精度不通过，内存占用不通过 | FUNC-PRECISION, FUNC-MEMORY |
| [#821](https://github.com/flashserve/flash-linear-attention-npu/issues/821) | [Feature]: 增加版本标签同步与置顶需求汇总 workflow | FUNC-SYNC, CI-WORKFLOW, FEATURE-DELIVERY |
| [#839](https://github.com/flashserve/flash-linear-attention-npu/issues/839) | [Feature]: CP 家族算子命名统一（反向预处理算子改名为 pre_process_bwd_kernel_merged） | ENG-STRUCTURE, FEATURE-DELIVERY |
| [#859](https://github.com/flashserve/flash-linear-attention-npu/issues/859) | [Bug]: KDA V2 fwd 在多 stream 并发场景下存在 race（fwd vs recompute 输出不一致） | FUNC-SYNC |
| [#867](https://github.com/flashserve/flash-linear-attention-npu/issues/867) | [Bug]: 需求汇总工作流 requirements-index.yml 无法加载，置顶 issue 一直未自动刷新 | CI-WORKFLOW, FUNC-REGRESSION |
| [#870](https://github.com/flashserve/flash-linear-attention-npu/issues/870) | [Bug]: Arch22 chunk_kda_fwd_prepare 在并发下偶发 fwd/recompute 输出不一致（多 Stage 地址复用） | FUNC-SYNC, FUNC-DETERMINISM, FUNC-MEMORY |
| [#874](https://github.com/flashserve/flash-linear-attention-npu/issues/874) | [Bug] npu_chunk_kda_bwd: stale dAkk workspace read corrupts dk/db/dg for odd head counts on 910_93 devices | FUNC-MEMORY, FUNC-SHAPE |
| [#891](https://github.com/flashserve/flash-linear-attention-npu/issues/891) | [Bug]: v26.9.1 wheel 在 A3 上，CI Example/ST 的两个 ChunkKDA 用例 cosine 判据不达标（绝对误差判据全过） | FUNC-PRECISION, ENG-BUILD, TEST-COVERAGE, CI-WORKFLOW |
| [#903](https://github.com/flashserve/flash-linear-attention-npu/issues/903) | [Bug]: v26.9.0 GDN forward 按 SoC 名称选择路径，导致 A3 未启用 DAV_2201 优化 | ENG-COMPATIBILITY, PERF-REGRESSION, FUNC-REGRESSION |
| [#904](https://github.com/flashserve/flash-linear-attention-npu/issues/904) | [Feature]: v26.9.1 移植 A2/A3 DAV_2201 GDN megakernel | ENG-COMPATIBILITY, FEATURE-DELIVERY |
| [#952](https://github.com/flashserve/flash-linear-attention-npu/issues/952) | [Bug]: ascend950 CANN9.1.0安装fla_npu v26.6.0失败 | ENG-COMPATIBILITY, ENG-BUILD, FUNC-REGRESSION |
