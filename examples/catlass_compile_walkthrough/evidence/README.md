# 验证记录：2026-11-03

此目录记录本 PR 独立教学示例的实际验证，不代表 FLA 全仓算子或双平台 CI 已通过。

## 版本和范围

| 项目 | C++ 实验 | DSL 实验 |
| --- | --- | --- |
| 产品 | A2，910B2，device 1 | A5，Ascend950PR，device 0 |
| Host | Linux aarch64 | Linux x86_64 |
| CANN | 9.2.0 | 9.2.0 |
| Python | 3.12.16，conda zzy | 3.12.15，基于 conda zzy 的独立 venv |
| CATLASS | FLA 固定的 C++ 源码 41bf90da655bba3c66d0acd7e00abe33960ecfd6 | ascend-catlass-dsl 2.2.0 wheel |
| 主要编译工具 | Bisheng，报告 clang 15.0.5；host GCC 11.4.0 | hivmc-a5 2.0.0；报告 AscendNPU-IR efb7141581c0、LLVM 19.1.7 |
| Torch | 独立 ACL 程序，不使用 Torch | torch 2.13.0+cpu、torch_npu 2.13.0 |

参考 FLA 基线：`8f6c77f4c67b1b8af8ea9976d7dbeeb2949e5c18`。
实现文件内容的 SHA-256 见 [source-sha256.json](source-sha256.json)。
C++ 参考
[00_basic_matmul 固定版本](https://gitcode.com/cann/catlass/blob/41bf90da655bba3c66d0acd7e00abe33960ecfd6/examples/00_basic_matmul/basic_matmul.cpp)；
DSL 参考
[官方快速开始](https://catlass.readthedocs.io/zh-cn/latest/4_CATLASS_DSL/quick_start/)，
实际 API 和产物以安装的 2.2.0 wheel 为准。

## 实际运行

按 [示例 README](../README.md) 启用环境后，执行：

```bash
# A2
bash run_cpp.sh 1

# A5
bash run_dsl.sh 0
python dsl/dump_backend.py build/dsl/artifacts.json
```

| 用例 | C++ 最大绝对误差 | DSL 最大绝对误差 | 输入/输出 SHA-256 |
| --- | --- | --- | --- |
| zero | 0 | 0 | 两边分别一致 |
| identity | 0 | 0 | 两边分别一致 |
| pattern | 0 | 0 | 两边分别一致 |

完整记录：[cpp_results.json](cpp_results.json)、[dsl_results.json](dsl_results.json)。
每组输出都与独立 FP64 CPU matmul 转 FP32 的标杆逐元素完全相等。
pattern 的左上角 2×2 输出为：

```text
-0.296875  -0.078125
 1.218750  -1.265625
```

DSL 只编译一次，再复用同一 compiled 对象运行三个不同输入。

## 中间产物

- [cpp-build.txt](cpp-build.txt)：成功的 C++ 编译和链接日志。
- [dsl-01-tla.mlir](dsl-01-tla.mlir)：DSL 前端实际生成的 TLA IR。
- [dsl-02-hivm.mlir](dsl-02-hivm.mlir)：CATLASS 转换后交给后端的 MLIR。
- [dsl-03-llvm.ll](dsl-03-llvm.ll)：后端通过 `--save-temps` 保存的 LLVM IR。
- [backend-command.txt](backend-command.txt)：保留 LLVM IR 时执行的实际命令。
- [backend-check.json](backend-check.json)：诊断重编译产物与已经运行的 kernel 的 SHA-256。

两个 DSL kernel 二进制的 SHA-256 均为：

```text
3b47e867bad2c8fc86460fee0a312f1c2f383fea6d4c462814dc055e11ef0a18
```

归档整理说明：文件保留实际 IR 和日志，清理行尾空白、末尾空行，并将本次工作目录改成
`examples/catlass_compile_walkthrough`，
CANN、C++ CATLASS、conda 路径改成 `<CANN>`、`<CATLASS_ROOT>`、`<CONDA_ENV>`。
`cpp-build.txt` 和 `backend-command.txt` 用于查看编译过程；
重新运行请使用脚本，脚本会为当前环境生成有效绝对路径。
IR 中的文件名归一化只修改源码位置注记。二进制未改写，也未提交到仓库。

C++ 构建实际保留以下文件（大小为本次结果）：

| 文件 | 字节 | 类型 |
| --- | ---: | --- |
| matmul-host-aarch64-unknown-linux-gnu.o | 56208 | host 目标文件 |
| matmul-cce-hiipu64-hisilicon-cce-dav-c220-cube.o | 22560 | Cube 设备目标文件 |
| matmul-cce-hiipu64-hisilicon-cce-dav-c220-vec.o | 3416 | 配套 Vector 目标文件 |
| matmul.cpp.o | 87936 | ASC 驱动组合后的 host 目标文件 |
| DSL kernel.o | 2328 | DSL 的设备目标文件 |

本次没有导出 C++ 路线的 LLVM IR；普通 Clang 的 `-emit-llvm` 用法不能直接套在该 ASC 驱动上。
DSL 后端重编译有工具链提示：找不到 module attribute 中的 BC，以及旧参数被忽略。
本实验通过显式 `--link-aicore-bitcode` 提供依赖，命令返回 0，
且二进制相等性检查通过。未把提示隐藏成“无 warning”。

## 错误路径与静态检查

[negative-checks.json](negative-checks.json) 记录：

1. 输入 A 被截断为 2 字节，C++ 程序在启动前报“expected exactly 512 bytes”，退出 1。
2. 在测试副本中将 pattern 输出第一个元素加 1，CPU 检查器报精度错误，退出 1。

`python -m py_compile fixtures.py dsl/matmul.py dsl/dump_backend.py`
和 `bash -n run_cpp.sh run_dsl.sh` 均通过。

不含性能测试、任意 shape/dtype 支持、OPP/wheel 安装、FLA 公共算子注册或全仓 Example/ST。
A2 C++ 和 A5 DSL 分别验证，不能把结果写成同一硬件上的语言性能对比。
