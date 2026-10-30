# ChunkKdaBwdFinalize host tiling 测试

## 用途与边界

本目录是独立的 host tiling 测试工程。测试直接编译并调用当前算子的
`Tiling4ChunkKdaBwdFinalize`，通过真实 `TilingContext` 构造 11 项属性，检查
`output_layout` 与末尾新增的内部可选属性 `model_layout`。`model_layout=""`
表示独立 Finalize；非空时仅接受 `BSND/BNSD/TND/NTD`，表示总接口的混合布局路径。

| 平台信息 | 场景 | 检查点 |
| --- | --- | --- |
| Ascend910B、Ascend910_93、Ascend950 | 独立调用，覆盖 NTD/TND 输出及 `T == H` | `output_layout` 显式控制全部五个梯度；A5 tiling key=1，Arch22 key=101，betaMode=0，不从方形 shape 猜测布局 |
| Ascend950 | 总接口 BSND/BNSD/TND/NTD，覆盖 rstd、BF16 raw_g 及 `T == H` | q/k/v 按 `model_layout` 寻址，有 rstd 时 q/k 固定 head-major；raw_g 与公开 db/dg 为 token-major；无 rstd 时 key=1，有 rstd 时 key=3 |
| Ascend910B、Ascend910_93 | 总接口 BSND/TND，覆盖 rstd、BF16 raw_g 及 `T == H` | 保持 Arch22 混合布局校验与 tiling key=101/102，不借用 A5 的结构或 raw tiling 大小 |
| Ascend950 | 非法 `model_layout="BTHD"` | 返回 `ge::GRAPH_FAILED`，实际 OP 日志包含参数名 `model_layout`、实际值 `BTHD` 和合法集合 `BSND/BNSD/TND/NTD` |
| Ascend950 | 独立调用传入不匹配的 token-major raw_g shape | 返回 `ge::GRAPH_FAILED`，不根据输出 shape 或同形维度推断 raw_g 布局 |

测试用 `static_assert` 检查 A5 `TilingData` 的逻辑大小为 **184 字节**，并检查 host 设置的
raw tiling size 为 `sizeof(TilingData)+8`，即 **192 字节**。读取顺序也是检查的一部分：
typed getter `GetTilingData<T>()` 会重置 raw size，因此测试先调用
`GetRawTilingData()->GetDataSize()` 保存 192，再通过 typed getter 检查字段；不能交换这两步。
`betaMode` 只表示 beta 计算模式，不承担布局编码。

这些平台由 host 上下文模拟。测试不启动设备 kernel，不构成 A2/A3/A5 实机、aclnn 或精度结论。
当前基线已实测二进制 **15/15**、CTest **1/1** 通过；该结果不替代设备验证。

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
- 非法 `model_layout` 和独立调用的 raw_g shape 负向用例中，`GRAPH_FAILED` 是预期结果，不是测试框架故障。非法布局用例还必须完整匹配参数名、实际值和合法集合；任一文本缺失都判定失败。
- 任一正常路径、布局字段、raw size 或报告检查失败时，二进制打印 `FAIL`、返回非零，CTest 随之失败。只有内部 15/15 用例通过时二进制才返回零。

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

复现预期为 CTest 1/1、直接运行二进制 15/15 通过，并打印非法 `model_layout`、raw_g 负向用例及
`T == H` 方形用例的完整报告。

## 依赖与回退

测试仅复用现有 CANN 开发包和调用方指定的 host UT common，**没有新增生产依赖**，也不改变生产算子的链接、安装或运行方式。
不配置本独立工程即可完全禁用这些测试。需要回退测试改动时，移除 `tests/operators/chunk_kda_bwd_finalize/` 即可，无生产代码、OPP 或设备状态回退步骤。
若只需回退配置，使用新的构建目录重新指定有效值；`ASCEND_CANN_PACKAGE_PATH` 可恢复为默认值，`UT_COMMON_DIR` 没有可用的空值回退，必须提供有效目录。
