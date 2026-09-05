# 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前，请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行CLA协议签署，了解源码仓的贡献流程。

开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。


开发者贡献场景主要包括：

### 算子Bug修复

  如果您在本项目中发现了某些算子Bug，希望对其进行修复，欢迎您新建Issue进行反馈和跟踪处理。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Bug-Report|缺陷反馈` 类Issue对Bug进行描述，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行处理。
  
### 算子优化

  如果您对本项目中某些算子实现有泛化性增强/性能优化思路，希望着手实现这些优化点，欢迎您对算子进行优化贡献。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Requirement|需求建议` 类Issue对优化点进行说明，并提供您的设计方案，
  然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行跟踪优化。

### 贡献新算子

  如果您有全新的算子希望基于 NPU 进行设计与实现，欢迎在 Issue 中提出您的想法与设计方案。完整的贡献流程如下：

  #### 1. 新增 Issue  
  请按照[提交 Issue / 处理 Issue 任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引，新建 `Requirement|需求建议` 类 Issue，并在其中说明新增算子的设计方案。  
  Issue 需包含以下内容：

  - **背景信息**  
  - **价值/作用**  
  - **设计方案**

  同时，请在提交的 Issue 中评论 `/assign` 或 `/assign @yourself` 认领该任务，以便后续完成算子上库。

  #### 2. 需求评审
  Sig成员将对您提交的 Issue 进行评审并给出修改意见。请在完成修改后，于 Issue 中回复：
  > “完成意见修改，申请复审”

  若需求被接纳，sig成员将为您分配合适的算子分类路径（如：`experimental/attention`），以便您将新增算子提交至对应目录。
  如在 Issue 交流中未能达成共识，建议申报 SIG 组双周例会，在会议中进行进一步讨论。

  #### 3. 提交 PR  
  在方案确定后，即可开始开发工作。新增算子的交付内容可基于最小交付件结构调整，请参考以下最小交付件进行检查。其中 `${op_name}` 为新增算子名称。
  ```
    ${op_class}                                          # 算子分类
    ├── ${op_name}                                       # 算子名
    │   ├── op_host                                      # 算子定义、Tiling相关实现
    │   │   ├── ${op_name}_def.cpp                       # 算子定义文件
    │   │   ├── ${op_name}_tiling.cpp                    # 算子Tiling实现文件
    │   │   └── CMakeLists.txt
    │   ├── op_kernel                                    # 算子Kernel目录
    │   │   ├── ${op_name}.cpp                           # Kernel入口文件，包含主函数和调度逻辑
    │   │   ├── ${op_name}.h                             # Kernel实现文件，定义Kernel头文件，包含函数说明、结构定义、逻辑实现
    │   │   ├── ${op_name}_tiling_data.h                 # TilingData文件，存储Tiling策略相关配置信息
    │   │   └── ${op_name}_tiling_key.h                  # TilingKey文件，定义Tiling策略的key，标识不同划分方式
    │   ├── CMakeLists.txt                               # 算子编译配置文件，保留原文件即可
    │   └── README.md                                    # 算子说明文档
    │   └── tests                                        # 算子测试文件
    │   │   ├── ut                                       # 算子ut测试文件
  ```

  开发完成后，请检查以下内容：
  - 代码交付件完整性（含 UT 测试用例代码）  
  - PR 是否已关联对应 Issue  
  - 是否签署 CLA  
  - 通过评论 `compile` 指令触发开源仓门禁，并依据 CI 检测结果进行修改。如涉及codecheck误报，请提交给[sig成员](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-transformer/sig-info.yaml)屏蔽。

  门禁通过后，请在关联的 Issue 中回复：
  > "该 Issue 关联的 PR：XXX，请尽快评审"

  Sig成员检视后将反馈检视意见，请完成所有修改后回复：
  > "该 Issue 关联的 PR：XXX，已完成 PR 问题整改，请尽快评审"

  #### 4. PR 上库  
  Committer 检视通过后，Maintainer 将进行最终审核。确认无误后，将标注 `/lgtm` 和 `/approve` 标签合入PR。

### 本仓特有的贡献要求

本仓在 CANN 算子上层还维护了 `torch_custom/fla_npu` 的 Python runtime 适配，新增算子除上述最小交付件外，还必须满足：

- **提供稳定 Python 入口**：新增算子必须在 `fla_npu.ops.ascendc` 下提供可调用的 Python 接口（经 `_aclnn_ctypes.py` wrapper + `_ASCENDC_OPS` 注册），不得仅以 legacy `torch.ops.npu.*` / `torch_npu.ops.*` 路径交付。
- **交付内容**：至少包含
  - `torch_custom/fla_npu/fla_npu/ops/ascendc/_aclnn_ctypes.py` 中的 `npu_<op>(...)` wrapper；
  - `torch_custom/fla_npu/fla_npu/ops/ascendc/__init__.py` 的 `_ASCENDC_OPS` 注册（需要时同步 `BACKWARD_OPS` 正反向映射与 `MUTATED_ARGUMENTS` mutation 契约）；
  - `torch_custom/fla_npu/test/test_npu_<op>.py` 单算子测试并接入 `test.sh`。
- **默认调用路径**：新增算子与测试默认使用 `fla_npu.ops.ascendc`，新代码不要默认依赖 legacy 路径（legacy 路径仅用于兼容性验证，且需要 `FLA_NPU_BUILD_LEGACY_EXTENSION=1` 额外构建）。
- 具体接入步骤见[开发者指南](docs/开发者指南.md)的场景 3 与 `torch_custom/fla_npu/README.md`。

### 文档纠错

  如果您在本项目中发现某些算子文档描述错误，欢迎您新建Issue进行反馈和修复。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Documentation|文档反馈` 类Issue指出对应文档的问题，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您纠正对应文档描述。
  
### 帮助解决他人Issue

  如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

  如果对应Issue需要进行代码修改，您可以在Issue评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您，跟踪协助解决问题。