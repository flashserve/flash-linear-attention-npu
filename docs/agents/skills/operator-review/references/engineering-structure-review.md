# 工程结构检视 reference

权威规范是
[`工程结构规范`](engineering-example/engineering-structure.md)，逐文件样本是
[`engineering-example/`](engineering-example/README.md)。
本文件只给出检视顺序，不复制规则正文。

## 1. 先确定工程形态

在目录中判断属于哪一类：

| 形态 | 判据 | 必需主体 |
| --- | --- | --- |
| A 独立 Ascend C 算子 | 有自己的 `def`、L0、L2 和 kernel | `op_host` + `op_kernel` + aclnn |
| B 主体独立 + 组合入口 | 主体独立，另有 L2/L0 组合入口 | 主体交付 + 组合入口 |
| C 只有 L2 | 无新 kernel，aclnn 调其他算子的 L0 | `op_host/op_api` + 文档/测试 |

形态判断错误会直接导致“缺少 kernel”或“重复实现”的误报，先看设计文档和
`aclnn_<op>.cpp` 的下发路径。

## 2. 目录与文件

逐项对照工程样例：

1. `CMakeLists.txt`、`README.md`、`docs/api.md`、`docs/design.md` 是否为该层唯一职责；
2. `def`、tiling、L0、L2、kernel、适配层是否各自只有一份事实；
3. `arch22`、`arch35` 是否在规范位置，最终 include 是否只有一个；
4. `op_kernel` 是否薄入口，平台实现是否按 struct/common/cube/vec 等角色拆分；
5. 新增的 `.h` 是否能对应样例角色，是否存在空壳、重复声明或兼容转发文件；
6. 稳定入口、ctypes、legacy 入口是否按仓库边界组织，不在主入口 import `torch_npu`；
7. 测试是否位于仓库既有 `tests/atk/<op>/` 或 `tests/operators/<op>/` 体系，而不是散落在算子目录。

## 3. 文件腐蚀

以下内容默认阻断：

- `build/`、`dist/`、`output(s)/`、报告、日志、性能库、缓存、`__pycache__`、临时脚本；
- 同一平台重复的原始实现、无人引用的旧入口、无构建目标的源码；
- 在多个文件重复维护同一 TilingData、可选输出 mask、tiling key 或 schema；
- 只在某平台存在的入口伪装成公共 export；
- 依赖闭包未在构建侧展开，只靠运行期碰运气命中 OPP；
- 测试数据或分析结果进入源码目录。

存量冗余与本次新增冗余分开报告。PR 检视至少阻止本次 diff 继续增加腐蚀。

## 4. 构建与依赖

核对：

1. `op_host/CMakeLists.txt` 的注册、模块来源和编译选项完整；
2. 组合入口的 `_depends` 同时覆盖构建展开和打包安装；
3. `--cce-auto-sync=off` 未被改为 `on`；
4. 平台编译选项只进入对应 SoC 分支；
5. `def`/schema/aclnn/Python 入口的 ABI 变化同步 CODEOWNERS 要求的检视材料；
6. 示例和测试资产有实际构建入口，不依赖历史目录或外部残留文件。

## 5. 公开接口兼容性

公开接口兼容性按**编码红线问题**处理。只有以下情况需要完整检查：

- 新增、删除或重命名 `fla_npu.ops.ascendc` 的公开算子；
- 修改 `_stable.py` 中公开 wrapper 的参数、默认值、参数种类或返回结构；
- 修改 `kSchema_<op>`、Stable-ABI adapter 与 Python 参数映射；
- 修改稳定的 aclnn 符号、layout/枚举值或可选输出语义；
- 声称保持 legacy `torch.ops.npu`、`torch_npu.ops` 或 ctypes 回退兼容。

### 5.1 先固定比较基线

兼容性基线是改动前的已发布 tag 或 PR base 提交，不是本 PR 修改后的代码、YAML 或文档。优先使用
仓库已提供的公开接口契约文件；当前仓库尚未提供时，至少从基线提交中读取：

1. `fla_npu/ops/ascendc/__init__.py` 的公开名；
2. `fla_npu/ops/ascendc/_stable.py` 的真签名 wrapper；
3. `csrc/src/stable_<op>.cpp` 的 `kSchema_<op>` 和 adapter；
4. `_aclnn_ctypes.py` 中仍发布的回退入口；
5. `docs/api.md`、迁移指南及 `npu_custom.yaml`——仅当 PR 明确继续承诺对应 legacy 入口时使用。

若仓库采用公开接口契约文件，其结构可参考
[`public-api-contract.example.yaml`](public-api-contract.example.yaml)。其中：

- `contract_version` 只表示契约文件格式版本，不能代表接口发布版本；
- `release_tag` 必须填写 `v26.9.1` 这类不可变发布 tag，并记录该 tag 下的公开 Python 接口快照；
- 每个算子记录 Python 入口、Stable-ABI op、位置参数/关键字参数分类、必选或默认值以及返回槽位；
- 发布准备时更新 `release_tag` 后再打 tag；已经发布的 tag 内文件不得回写；
- 检视当前分支时必须读取 `git show <release_tag>:<contract-file>`，与当前契约和当前 wrapper
  三方比较。缺少 `release_tag`、读取不到基线文件或没有历史比较证据时，不能通过兼容性检视。

同一 PR 同时修改代码、schema、YAML 和文档后，这些文件仍然彼此一致是交付完整性检查，不能替代与
基线版本的兼容性比较。

### 5.2 逐项比较调用契约

对每个已发布入口核对：

1. 公开名：算子名、Python 短名和 raw/stable op 名的映射是否保持不变；
2. 参数：参数数量、顺序、名字、positional-only / positional-or-keyword / keyword-only 属性；
3. 默认行为：是否必选、默认值及省略参数时的语义；
4. 类型与语义：Tensor/optional/scalar/int-array/layout 的跨边界形式是否变化；
5. 返回：tuple 数量、顺序、optional 标记、shape/layout 语义；
6. 枚举与开关：字符串 layout、枚举整数码、布尔开关的默认值和含义；
7. 后端一致性：stable、ctypes、legacy 同一次调用不能被解析成不同参数或不同语义。

只允许向后兼容的追加方式：在参数表末尾新增带历史默认值的关键字参数，或按仓库既定规则在返回
tuple 末尾追加可选输出。默认值必须复现旧行为，新增开关不能在默认状态下改变结果。

以下变化默认按编码红线问题阻断：

- 删除或重命名已发布算子或任意参数；
- 删除、插入或重排既有参数；
- 把可选参数改为必选，或把 keyword-only 参数改为可与其他参数混用而改变调用解析；
- 修改既有参数默认值、类型、单位、layout 或语义；
- 修改返回 tuple 的数量、顺序或既有槽位语义；
- 修改 enum/string value 到内部 code 的映射；
- 仅在一个后端改签名、默认值或语义，造成 `FLA_NPU_STABLE_ABI` 切换后调用行为变化。

若不兼容变化确有必要，必须要求以下证据之一：保留旧参数/旧算子作为 deprecated alias 并明确迁移
期限，或发布新的版本化入口。alias 必须有 warning、正向回归和迁移文档；不得静默接受旧关键字后改变
语义。若 PR 声称 legacy 兼容，还需核对对应的 YAML、生成接口和测试，不能只修 stable 入口。

### 5.3 输出与定级

已发布公开接口的参数改名、删除、重排、默认值变化、返回 tuple 变化、枚举语义变化或后端调用形状
不一致，均按 `问题性质：编码红线问题` 输出，不能降级成 cleancode、普通兼容性提示或建议项。
若旧调用会直接失败或得到错误结果，按根规范输出 `问题分类：功能正确性`、`精度级别：lv1`；判据来自
公开接口契约或发布基线。缺少兼容基线、alias、迁移说明或跨后端回归证据时，同样不能把该接口变化
判为可接受。位置必须同时落到公开 wrapper、schema、adapter 或测试的实际行；不能只引用 PR 描述。

## 6. 环境变量与隐式开关预警

新增、删除或改变环境变量、CMake cache 开关、`-D` 编译定义以及会改变 include/构建语义的宏时，
必须在 PR 中单独预警，不能混在其他改动里静默引入。至少写清：

| 项目 | 必须说明 |
| --- | --- |
| 名称与默认值 | 精确名称、未设置时的行为 |
| 生效范围 | 运行期、测试、构建、CI 或仅单 target |
| 消费方 | 哪些 CMake target、脚本、模块或调用链读取 |
| 优先级与隔离 | 与配置文件、命令行参数和其他变量的覆盖关系 |
| 异常值 | 缺失、空值、大小写错误、非法取值时的报错和回退 |
| 生命周期 | 谁设置、谁清理，失败路径是否恢复 |
| 测试矩阵 | A2/A3/A5、构建变体、未设置和非法值 |
| 回退 | 删除变量或恢复默认行为的方法 |

优先用函数参数、CMake target 属性、独立 prelude/target 或显式配置文件表达差异。禁止用新环境变量
或 `-D` 宏隐式选择业务范围、兼容路径、header include 顺序或测试语义。

## 7. 输出

每条结构问题用分类 `cleancode` 或 `兼容性问题`，位置必须落到具体文件。对“缺少文件”写清是在
哪一种工程形态下缺什么职责；对“多余文件”写清谁引用、为什么不满足现状。
