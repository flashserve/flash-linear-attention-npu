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

## 5. 环境变量与隐式开关预警

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

## 6. 输出

每条结构问题用分类 `cleancode` 或 `兼容性问题`，位置必须落到具体文件。对“缺少文件”写清是在
哪一种工程形态下缺什么职责；对“多余文件”写清谁引用、为什么不满足现状。
