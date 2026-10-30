# ChunkKdaBwdFinalize host tiling 测试

## 用途与边界

本目录是独立的 host tiling 测试工程。测试直接编译并调用当前算子的
`Tiling4ChunkKdaBwdFinalize`，通过真实 `TilingContext` 检查以下契约，不构造任何新增属性：

| 平台信息 | db/dg 输出 shape | 检查点 |
| --- | --- | --- |
| Ascend910B、Ascend910_93 | head-major | 返回 `ge::GRAPH_FAILED`；实际 OP 日志包含 `dBetaOut/dGOut`、`head-major`、`A2/A3` 和 `token-major` 平台原因 |
| Ascend910B、Ascend910_93 | token-major | 原 Arch22 校验和转发路径成功，tiling key 为 101 |
| Ascend950 | head-major、token-major | 两种输出 shape 均接受，并由现有 `betaMode` 的保留高位传递布局，不新增 attr 或 TilingData 字段 |

测试同时用 `static_assert` 检查 `TilingData` 为基线的 168 字节，并检查 Arch22 的
`betaMode` 不携带布局高位，避免布局优化改变原有 ABI 或 beta 语义。
这些平台由 host 上下文模拟。测试不启动设备 kernel，不构成 A2/A3/A5 实机、aclnn 或精度结论。

## 环境变量/配置变量契约

本工程新增的是两个 **CMake 配置变量**，不是仓库级环境变量或生产构建开关。它们仅由本目录的独立
CMake target 消费，不被算子、wheel、OPP 或安装流程读取。命令行 `-D` 值写入对应构建目录的 CMake cache；
再次配置时，新的 `-D` 值优先，未重新指定的值沿用该 cache。使用新构建目录可消除旧 cache 的影响。

- `ASCEND_CANN_PACKAGE_PATH`
  - 默认值：`/usr/local/Ascend/latest`。
  - 作用域和消费方：仅供本测试确定 CANN 头文件、库目录及运行时 rpath；架构子目录由
    `${CMAKE_SYSTEM_PROCESSOR}-linux` 推导。
  - 优先级：显式 `-DASCEND_CANN_PACKAGE_PATH=...` 覆盖默认值；不读取生产构建的安装配置。
- `UT_COMMON_DIR`
  - 默认值：空。
  - 作用域和消费方：仅供本测试编译、包含现有 host UT common 的 `tiling_context_faker.cpp/.h` 及其同目录依赖。
  - 优先级：必须通过 `-DUT_COMMON_DIR=...` 指向一个已经存在的 UT common 目录；没有隐式下载或仓库位置猜测。

运行命令中的 `LD_LIBRARY_PATH` 只是动态加载器需要时的进程级环境变量，不是新增 CMake 开关，也不会写入生产环境。
测试二进制仅在调用方未设置时使用 `ASCEND_SLOG_PRINT_TO_STDOUT=1` 和 `ASCEND_GLOBAL_LOG_LEVEL=3`，用于捕获并检查真实 OP 日志。

## 非法值和失败行为

- `UT_COMMON_DIR` 保持默认空值时，CMake 配置以 `FATAL_ERROR` 终止，不生成测试目标，也不回退到伪造路径。
- `UT_COMMON_DIR` 指向不存在目录或缺少所需文件时，配置或编译失败；不会下载替代实现。
- `ASCEND_CANN_PACKAGE_PATH` 指向不存在或内容不兼容的 CANN 包时，头文件查找、编译或链接失败；不会切换到另一套 CANN。
- A2/A3 的 head-major 输出 shape 用例中，`GRAPH_FAILED` 是预期结果，不是测试框架故障。若返回码不符，或实际报告缺少上述任一文本，测试二进制返回非零。
- 任一正常路径、布局字段或报告检查失败时，二进制打印 `FAIL`、返回非零，CTest 随之失败。只有内部 6/6 用例通过时二进制才返回零。

这些变量只在测试配置和测试进程生命周期内有效。工程不安装文件、不修改 CANN 或 UT common，也不生成生产包；删除其独立构建目录即可清除对应 cache 和构建产物。

## 复现

在仓库根目录执行。先将 `UT_COMMON` 替换为本机实际存在且包含 `tiling_context_faker.cpp` 的 UT common 目录；
`CANN_ROOT` 可按实际安装版本修改，`BUILD_DIR` 应使用独立目录。

```bash
CANN_ROOT=/usr/local/Ascend/cann-9.1.0
UT_COMMON=/absolute/path/to/existing/ut/common
BUILD_DIR="$PWD/build_tests_operators"

cmake -S tests/operators/chunk_kda_bwd_finalize -B "$BUILD_DIR" \
  -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" \
  -DUT_COMMON_DIR="$UT_COMMON"
cmake --build "$BUILD_DIR" -j4

export LD_LIBRARY_PATH="$CANN_ROOT/$(uname -m)-linux/lib64:$CANN_ROOT/lib64:${LD_LIBRARY_PATH:-}"
ctest --test-dir "$BUILD_DIR" --output-on-failure
"$BUILD_DIR/chunk_kda_bwd_finalize_tiling_test"
```

预期 CTest 为 1/1 通过，直接运行二进制为 6/6 通过，并打印两个 A2/A3 负向用例的完整输出 shape 报告。

## 依赖与回退

测试仅复用现有 CANN 开发包和调用方指定的 host UT common，**没有新增生产依赖**，也不改变生产算子的链接、安装或运行方式。
不配置本独立工程即可完全禁用这些测试。需要回退测试改动时，移除 `tests/operators/chunk_kda_bwd_finalize/` 即可，无生产代码、OPP 或设备状态回退步骤。
若只需回退配置，使用新的构建目录重新指定有效值；`ASCEND_CANN_PACKAGE_PATH` 可恢复为默认值，`UT_COMMON_DIR` 没有可用的空值回退，必须提供有效目录。
