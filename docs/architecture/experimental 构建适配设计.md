# experimental 算子严格隔离构建设计

> 状态：目标设计，作为实现与验收依据。
>
> 对应需求：[Issue #931：新增 experimental 算子目录，隔离实验算子与稳定算子构建](https://github.com/flashserve/flash-linear-attention-npu/issues/931)。
>
> 调研、试错、真机验证和被否决方案见 [experimental 构建适配过程记录.md](experimental%20构建适配过程记录.md)。

## 1. 背景与范围

仓库需要接纳尚未完成转测、暂不进入正式商用版本，但仍需持续开发和共同维护的算子。RWKV recurrent 算子是首个使用场景，后续尚未稳定的非模型场景泛化也适用。

本设计解决以下问题：

1. 为实验算子提供独立源码目录。
2. 在稳定模式和 experimental 模式之间严格二选一。
3. 让实验算子具备独立的发现、编译、打包、Python 接入和 CI 验证链路。
4. 保证 experimental 相关改动不改变稳定构建及正式发布产物。

本设计不负责实验算子的转测结论，也不承诺其精度、性能、内存或确定性达到商用要求。进入 `experimental/` 只表示代码可以在仓内继续开发。

## 2. 设计目标

| 目标 | 必须满足的结果 |
| --- | --- |
| 构建范围互斥 | `ENABLE_EXPERIMENTAL=OFF` 只处理稳定树；`ON` 只处理实验树 |
| 依赖严格隔离 | 实验算子的全部算子依赖来自实验树；缺失时明确失败 |
| 公共代码隔离 | experimental 使用独立的算子公共代码；不完整时在构建阶段失败 |
| 产物隔离 | 实验产物只包含所选实验算子及其实验依赖，不包含稳定算子 |
| CI 隔离 | experimental 使用独立 CI job 和独立制品，不进入 Release |
| 稳定路径零回归 | 默认开关关闭；稳定构建的发现范围、内容和发布流程不变 |

以下行为明确禁止：

- experimental 模式回退到稳定算子依赖；
- 依赖不存在时静默跳过；
- 同一次构建或同一产物混合稳定算子与实验算子；
- experimental 修改稳定树的算子公共代码；
- 将实验制品作为正式商用能力发布。

## 3. 主要路线选择

### 3.1 目录组织

评估过三种路线：

| 路线 | 结论 | 原因 |
| --- | --- | --- |
| 稳定目录内增加标签或名单 | 不采用 | 容易被默认 glob、打包或 Release 路径误收录，隔离依赖人工纪律 |
| `experimental/` 后镜像仓库相对路径 | 不采用 | 能减少迁移改动，但保留了对稳定树层级和域分类的耦合，发现与 CI 仍需解析不固定层级 |
| `experimental/<算子名>/` 一级扁平目录 | 采用 | 一级目录名可保证为全局唯一算子名，发现、依赖、PR 识别和产物审计都可直接按名称处理 |

最终目录为：

```text
fla/ops/ascendc/<域>/<算子>/
experimental/<算子名>/
```

实验树不复刻稳定树的域层级，也不是稳定树的增量补丁层。迁移算子时需要显式整理依赖声明和跨目录相对引用，换取简单、稳定且可直接审计的目录契约。

### 3.2 依赖策略

曾验证过“实验树优先、缺失时回退稳定树”的方案。它能降低迁移成本，也能让部分实验算子快速出包，但会把稳定算子加入实验构建图和产物，不符合 Issue #931 的依赖隔离要求，因此不采用。

最终策略是严格依赖闭包：

- experimental 算子的依赖只声明算子名，不声明仓库路径；
- experimental 模式只把依赖名解析为 `experimental/<依赖算子名>`；
- 依赖缺失立即 `FATAL_ERROR`，错误中给出依赖方、依赖名和期望路径；
- 禁止通过 `../`、绝对路径或算子级手写 `add_subdirectory()` 逃逸到稳定树；
- 传递依赖应用同一规则，直到闭包收敛。

### 3.3 公共代码策略

共享稳定公共头虽然迁移成本低，但实验修改会直接影响稳定构建；“稳定头 + 实验头覆盖”的叠加方案仍会让实验构建读取稳定代码，也不满足严格隔离。

最终采用独立副本并整体切换：

```text
fla/ops/ascendc/common/
experimental/_common/
```

experimental 公共代码必须形成完整快照。仓库维护一份必需文件清单，configure 阶段校验实验 common 的完整性；缺文件直接失败，不从稳定目录补齐。

## 4. 目标目录结构

```text
experimental/
├── README.md
├── _common/                       # 保留目录，不是算子
│   ├── manifest.txt
│   └── ...
├── recurrent_gated_delta_rule/    # 一级目录名就是算子名
│   ├── CMakeLists.txt             # 唯一算子入口和依赖声明
│   ├── op_host/
│   │   └── CMakeLists.txt
│   ├── op_kernel/
│   └── tests/                     # 可选
└── <another_operator>/
    ├── CMakeLists.txt
    ├── op_host/
    └── op_kernel/
```

约束如下：

1. 除名称以 `_` 开头的保留目录外，`experimental/` 的每个一级子目录都是一个算子，目录名就是算子名。
2. 算子名在实验树内全局唯一，并满足 `[A-Za-z0-9_][A-Za-z0-9_.-]*`。
3. 每个算子目录必须包含根层 `CMakeLists.txt`，作为唯一发现入口；`op_host/CMakeLists.txt` 仍是 host 构建入口。
4. 根层 CMake 通过统一 helper 声明 `NAME` 和 `DEPENDS`，其中 `NAME` 必须等于目录名，`DEPENDS` 只能填写其他实验算子名。
5. 一个实验算子的直接依赖和传递依赖必须全部存在于实验树。
6. 实验算子需要的公共头必须存在于 `_common/`。
7. 工具、公共代码和清单目录统一使用 `_` 前缀，并在发现逻辑中显式排除。

算子根层 CMake 的目标形态为：

```cmake
add_subdirectory(op_host)
```

`op_host/CMakeLists.txt` 负责调用 `add_op_to_compiled_list()` 并声明 `<operator>_depends`。依赖值只填写其他实验算子名；构建框架负责校验依赖、挂载依赖根 CMake，并迭代到闭包收敛。算子不得绕过统一解析器直接挂载仓库其他位置。

## 5. 构建设计

### 5.1 开关与入口

`ENABLE_EXPERIMENTAL` 默认值为 `OFF`，`build.sh --experimental` 将其置为 `ON`。每次配置都显式下发 `TRUE/FALSE`，避免任何调用方式复用旧 CMake 状态。

两种模式共享构建框架，但算子树根、公共代码根、算子注册清单和产物身份必须同时切换。

| 配置项 | OFF | ON |
| --- | --- | --- |
| 算子树根 | `fla/ops/ascendc` | `experimental` |
| kernel common | `fla/ops/ascendc/common` | `experimental/_common` |
| 算子名校验 | 稳定树清单 | 实验树清单 |
| PR 变更解析 | 稳定流程 | experimental 专用流程 |
| 制品 | 稳定制品 | 带 experimental 标识的独立制品 |

### 5.2 算子发现

稳定模式保留现有发现逻辑。experimental 模式只枚举 `experimental/` 的一级子目录，以根层 `CMakeLists.txt` 作为算子标记：

```cmake
file(GLOB _experimental_cmake
     LIST_DIRECTORIES false
     "${CMAKE_SOURCE_DIR}/experimental/*/CMakeLists.txt")
foreach(_cmake ${_experimental_cmake})
    get_filename_component(_operator_dir "${_cmake}" DIRECTORY)
    get_filename_component(_operator_name "${_operator_dir}" NAME)
    if(NOT _operator_name MATCHES "^_")
        list(APPEND OP_HOST_CMAKE_FILES "${_cmake}")
    endif()
endforeach()
```

发现逻辑不递归猜测算子边界，也不维护域白名单。算子内部可以继续使用任意深度的 `op_host`、`op_kernel` 和 `tests` 子目录，但它们都归属于第一个路径段表示的算子。

`OP_DIR_LIST` 中的每条路径都必须位于当前 `_ops_root` 下；挂载前执行规范化路径检查，阻止符号链接或 `..` 越界。

### 5.3 `--ops` 校验

`build.sh` 的前置校验器必须接收构建模式，并从对应树生成可用算子名：

```text
OFF -> 按现有稳定树规则生成
ON  -> experimental/<算子名>/CMakeLists.txt
```

指定名称不在当前树时立即报错，不进入 CMake。`--list-ops` 同样支持 `--experimental`，避免用户看到稳定树清单后误选。

### 5.4 依赖闭包

依赖解析只能有一个实现，并被所有消费者复用。至少包括：

1. 依赖 `op_host` 补挂；
2. kernel 源复制；
3. 入包的 `ascendc/<dep>` 源码；
4. proto、动态适配文件和 kernel 配置生成；
5. 算子目录内的定制依赖挂载。

experimental 依赖声明先校验为单个合法算子名，禁止 `/`、`\\`、`..` 和绝对路径。解析规则为：

```cmake
set(_candidate "${CMAKE_SOURCE_DIR}/experimental/${depend_name}")
if(NOT EXISTS "${_candidate}/CMakeLists.txt")
    message(FATAL_ERROR
        "experimental dependency is missing: owner=${owner}, "
        "dependency=${depend_name}, expected=${_candidate}")
endif()
```

解析后的绝对路径还需断言位于 experimental 树根下。依赖闭包按“发现新依赖、校验、加入队列”迭代，直到没有新增节点；循环依赖和同名路径冲突必须明确失败。

### 5.5 公共代码

`OPS_ADV_UTILS_KERNEL_INC` 随模式切换。experimental 模式下必须先完成：

1. `experimental/_common/manifest.txt` 存在；
2. manifest 中的所有文件都存在于实验 common；
3. 路径中没有指向稳定树的符号链接；
4. staging 和 install 的来源都是实验 common。

Catlass、CANN SDK 头和纯构建框架可以继续作为两种模式的外部共享基础设施，但不得包含本仓稳定算子的实现或注册信息。

仓根 `common/` 当前无条件参与构建，也必须完成内容分类：如果其中包含算子实现、注册或入包文件，则需要提供 experimental 独立副本并随模式切换；只有经审计确认与算子无关的构建基础设施才允许共享。此目录的变更必须同时触发 experimental 与稳定回归。

### 5.6 打包与产物审计

实验制品使用独立名称，例如：

```text
fla_npu_experimental_linux-<arch>.run
```

打包前生成算子清单和源码来源清单，并执行以下门禁：

- 每个算子来源路径都在 `experimental/` 下；
- 算子集合等于“用户选择 + 实验依赖闭包”；
- kernel、config、dynamic、proto 和 aclnn 接口清单相互一致；
- 不存在稳定算子路径或稳定算子名；
- common 来源为实验 common；
- 包元数据标记 `experimental` 和源码 commit。

实验制品不得进入正式 Release，也不得覆盖同一流水线中的稳定制品。

### 5.7 Python 与 wheel

Python/wheel 构建必须感知同一个 experimental 开关，不能继续使用完整稳定算子注册表。实现要求是：

- 从本次实验算子清单生成或筛选 Stable-ABI 注册；
- 只打入实验算子 wrapper 和本次实验 OPP；
- 使用独立的 experimental wheel/制品标识；
- 安装检查能够列出实验算子，并确认没有稳定算子接口被意外带入；
- RWKV recurrent 的 Python 调用链能加载实验 OPP 并完成最小运行验证。

## 6. PR 识别与 CI

### 6.1 变更识别

experimental PR 解析器以 `experimental/` 为根。第一个路径段若不是 `_` 前缀保留名，就直接作为算子名；`op_host`、`op_kernel`、`tests` 等后续层级不再参与算子边界推断。

以下变更触发 experimental 全量构建，而不是返回“无算子”：

- `cmake/`、`build.sh`、打包脚本；
- experimental common 或 manifest；
- Python/wheel 注册与生成逻辑；
- experimental CI 自身。

删除整个算子目录时，解析器使用 base/head 差异信息保留被删除算子的身份，供产物和注册清理检查使用。

### 6.2 独立 CI job

新增 experimental 专用 job，至少执行：

1. 路径识别单元测试；
2. 依赖完整性和跨树路径静态门禁；
3. 真实 RWKV experimental 算子 configure、编译和打包；
4. 包内算子清单及来源审计；
5. Python/wheel 最小安装与调用验证；
6. A2/A3/A5 稳定构建回归。

experimental job 的产物单独命名、单独上传，不进入 Release job。

## 7. 验收矩阵

| 场景 | 预期结果 |
| --- | --- |
| OFF，实验树含同名或非法文件 | 稳定构建与改造前一致，产物无实验内容 |
| ON，稳定树含同名算子 | 只选择实验算子，产物无稳定版本 |
| ON，实验直接依赖缺失 | configure 明确失败，不回退、不静默跳过 |
| ON，实验传递依赖缺失 | configure 明确指出依赖链和缺失路径 |
| ON，实验 common 缺文件 | 构建开始编译前失败 |
| ON，新增 experimental-only 算子 | 一级目录和根 CMake 存在即可识别，无需修改域名单 |
| `--experimental --ops=<实验独有算子>` | 前置校验通过并只构建该算子闭包 |
| experimental PR 修改算子深层文件 | CI 按第一个路径段精确识别算子并完成独立打包 |
| 修改构建框架或实验 common | experimental CI 全量构建 |
| RWKV recurrent | `recurrent_gated_delta_rule` 位于实验树，生成独立实验包并完成最小 Python 调用验证 |
| A2/A3/A5 稳定构建 | 发现范围和产物内容不变 |

## 8. 实现边界

实现至少涉及以下模块：

| 模块 | 责任 |
| --- | --- |
| `cmake/func.cmake` | 树根选择、严格依赖解析、闭包与路径断言 |
| `cmake/custom_build.cmake` | 统一使用解析后路径、产物来源审计 |
| `cmake/config.cmake` | common 根随模式切换 |
| `build.sh` | 开关透传、实验算子名校验、独立制品参数 |
| `scripts/check_build_ops.py` | 按模式生成算子清单 |
| `scripts/ci/parse_changed_ops.py` | 一级算子名识别、保留目录处理及全量触发规则 |
| `.github/workflows/*` / `ci/*` | experimental 独立 job 和包检查 |
| `torch_custom/fla_npu/` | experimental 注册、wheel 和加载验证 |
| `experimental/README.md` | 使用范围、命令和非商用声明 |

## 9. 完成定义

只有当以下条件全部满足时，Issue #931 才可关闭：

1. 构建代码中不存在 experimental 到稳定算子目录的回退路径。
2. 所有依赖消费者使用同一严格解析结果。
3. 公共代码、算子注册、Python 接口和产物均完成模式切换。
4. 真实 RWKV 算子在 experimental 树中完成独立出包。
5. experimental CI 和包内容审计通过。
6. A2/A3/A5 稳定构建及原有共享流程回归通过。
7. 用户文档明确 experimental 非商用，且不给出混合构建方式。
