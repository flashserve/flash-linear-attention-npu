# CATLASS C++ / DSL：同一个矩阵乘法的编译实验

先读 [编译入门教程](../../docs/catlass-cpp-dsl-compilation.md)，再按本页运行。
这是独立教学实验，不注册 FLA 算子，不参与 OPP/wheel 的构建或安装。

两份实现计算同一公式：

```text
C[i,j] = sum(A[i,k] * B[k,j], k=0..15)
A、B：16×16、FP16、连续行主序
C：16×16、FP32、连续行主序；覆盖写入，无 bias、无累加到旧 C
```

- [C++ 实现](cpp/matmul.cpp)：实际使用 CATLASS 的 BlockMmad、BasicMatmul、DeviceGemm；本实验面向 A2。
- [DSL 实现](dsl/matmul.py)：实际使用 CATLASS 的 tla.copy、tla.mmad；面向 Ascend 950。
- [输入和 CPU 标杆](fixtures.py)：三组完全确定的数据，两边输入字节相同，CPU 使用 FP64 独立计算。
- [实测证据](evidence/README.md)：环境、结果、真实编译产物；不据此比较两代硬件性能。

这里的固定 shape 是教学范围，不是一个支持任意 shape 的 matmul API。
C++ 入口对输入文件严格检查大小；DSL 从同一组文件读取固定 shape。
三个用例覆盖零输入、单位矩阵和带正负值的非对称矩阵。输入为 1/8 的整数倍，
其 16 项内积可用 FP32 精确表示，因此本实验要求逐元素完全相等。
此精度规则不推广到随机浮点输入或其它 GEMM。

## A2：编译和运行 CATLASS C++

需要 Linux、A2、CANN 9.2.0、G++、CMake、Python 和 numpy。
先启用 CANN 和 Python 环境，再指定 CATLASS 源码：

```bash
source /path/to/Ascend/ascend-toolkit/set_env.sh
conda activate zzy

# 建议使用 FLA cmake/third_party/catlass.cmake 固定的版本：
# 41bf90da655bba3c66d0acd7e00abe33960ecfd6
# 已有 third_party/catlass 时可直接使用，无需重新下载。
export CATLASS_ROOT=/path/to/catlass
cd examples/catlass_compile_walkthrough
bash run_cpp.sh 0
```

脚本依次生成输入、构建、启动三次 NPU kernel、比较 CPU 标杆。
参数 `0` 是设备号。输出 `build/fixtures/cpp_results.json`，
每项应为 `passed: true`、`max_abs_error: 0.0`。

源码版本准备命令（在新的依赖目录执行）：

```bash
git clone https://gitcode.com/cann/catlass.git /path/to/catlass
git -C /path/to/catlass checkout 41bf90da655bba3c66d0acd7e00abe33960ecfd6
```

`run_cpp.sh` 把当前 G++ 的标准库搜索目录传给 Bisheng 及其内部链接子进程，
用于处理系统同时存在完整 GCC 11 与不完整 GCC 12 的情况；环境变量仅作用于本次脚本。
编译器仍为 CANN 的 Bisheng。构建日志保存在 `build/cpp-build.log`。

## A5：编译和运行 CATLASS DSL

需要 Ascend 950PR/950DT、CANN 9.2.0、已可用的 torch/torch_npu。
本实验固定 `ascend-catlass-dsl==2.2.0`。可在已有 conda 环境之上建立独立 venv，
复用 torch/torch_npu，将新增的 DSL 包安装在 venv 内：

```bash
source /path/to/Ascend/ascend-toolkit/set_env.sh
conda activate zzy
python -m venv --system-site-packages /path/to/catlass-tutorial-venv
source /path/to/catlass-tutorial-venv/bin/activate
python -m pip install ascend-catlass-dsl==2.2.0

cd examples/catlass_compile_walkthrough
bash run_dsl.sh 0
python dsl/dump_backend.py build/dsl/artifacts.json
```

第一条脚本完成 BC 依赖准备、DSL 编译、三组 NPU 执行和 CPU 比较；
第二条命令保留后端产生的实际 LLVM IR，并检查诊断编译的二进制是否与已经运行的版本相同。
保存产物的额外编译有 60 秒超时；相等性检查失败时返回非零。
所有缓存和中间文件默认位于本例 `build/` 下。

重点打开：

```text
build/dsl/01-tla.mlir                    Python DSL 前端输出
build/dsl/02-hivm.mlir                   TLA 转换后的 MLIR，包含 HIVM 等操作
build/dsl/backend-temps/recompiled.ll    hivmc-a5 保存的 LLVM IR
build/dsl/backend-command.txt           实际执行的后端命令
build/dsl/artifacts.json                当前缓存、编译器和 kernel.o 的路径
build/dsl/backend-check.json            两个 kernel.o 的 SHA-256 比较
build/fixtures/dsl_results.json         三组 NPU 精度结果
```

本例使用 CATLASS 2.2.0 的产物接口；该版本 `artifacts.LLVM` 字段实际保存
lowered MLIR，不能仅凭属性名把它当作 `.ll`。源码把它正确保存为 `02-hivm.mlir`。
`dump_backend.py` 生成的 `.ll` 才是 LLVM IR 文本。

## 返回码和验证范围

- 缺少依赖、输入文件不完整、编译失败、执行失败、超时、精度失败都会返回非零。
- A2 上运行 DSL 会被产品检查拦截。更改 `--npu-arch` 不能让当前 DSL 自动支持 A2。
- 本例未接入公开 FLA 接口，因此不需要安装本仓 wheel 或更新已有 OPP。
- 文件中的 CANN 许可声明适用于参考 CATLASS 改写的两份 kernel，许可证在仓库
  [LICENSES](../../LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt)。
