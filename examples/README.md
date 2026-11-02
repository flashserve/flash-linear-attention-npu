# Examples 说明

本目录提供 flash-linear-attention-npu 的端到端示例与算子开发参考模板：

- `flash_gated_delta_rule.py`：GDN 模块端到端接入示例，组装 GDN 相关前向/反向算子，覆盖 AscendC 与 Triton 调用链。
- `add_example/`：一个完整的 Ascend C 算子工程模板（add 算子），包含 `op_host`（def / tiling / infershape）、`op_kernel`、`op_graph` 与测试工程，可作为新增 Ascend C 算子的目录结构参考。
- `fast_kernel_launch_example/`：演示如何使用 Ascend C + PyTorch Extension 能力开发自定义 NPU 算子，单个交付件完成算子开发与 PyTorch 框架适配，并支持 `<<<>>>` 语法启动核函数。

## 目录说明

```
├── flash_gated_delta_rule.py        # GDN 端到端调用示例
├── add_example/                     # Ascend C 算子工程模板（add 算子）
│   ├── CMakeLists.txt
│   ├── examples/                    # 算子使用示例（test_aclnn_add_example.cpp）
│   ├── op_graph/                    # 算子构图相关目录
│   ├── op_host/                     # def / tiling / infershape 实现
│   ├── op_kernel/                   # AI Core kernel
│   ├── op_kernel_aicpu/             # AICPU kernel（含 fallback 场景）
│   ├── tests/                       # 测试工程
│   └── README.md                    # 算子说明文档
├── fast_kernel_launch_example/      # Ascend C + PyTorch Extension 快速开发示例
│   ├── ascend_ops/                  # 算子源码与 PyTorch 适配
│   ├── csrc/                        # C++ 侧封装
│   ├── tests/                       # 测试用例
│   ├── setup.py / build_and_test.sh # 构建与测试脚本
│   └── README.md                    # 示例说明与安装步骤
└── CMakeLists.txt                   # 编译配置（遍历含 CMakeLists.txt 的子目录）
```

## 快速运行

### GDN 端到端示例

完成 README 的构建与安装后，直接运行：

```sh
python examples/flash_gated_delta_rule.py
```

该脚本会组装 GDN 相关前向/反向算子，覆盖 AscendC 与 Triton 调用链，输出各算子的运行结果与精度对比。

### ChunkKDA 精度示例

```sh
python examples/chunk_kda.py --accuracy-check
```

CPU 前向参考和五个参考梯度在首次运行时生成缓存，相同输入和参考实现再次运行时直接复用。
NPU 前向、反向和精度比较仍会每次执行。输入内容、shape、dtype、参考参数、参考实现或
PyTorch 版本变化时会生成新的缓存；损坏的缓存会重新生成。

默认目录为 `third_party/gdr_accuracy_golden/chunk_kda`；设置 `GDR_ACCURACY_CACHE_DIR` 时，
使用其下的 `chunk_kda` 子目录，因此 CI 直接复用已有持久缓存挂载。也可以通过
`--accuracy-cache-dir <目录>` 指定位置，或用 `--accuracy-force-regenerate` 强制重新生成。
日志中的 `accuracy golden: generated` / `accuracy golden: reused` 表示本次生成或复用缓存。
缓存无法写入时会提示警告，并使用本次计算的参考结果继续验证。

### add_example

参考 `add_example/README.md` 与 `add_example/examples/test_aclnn_add_example.cpp`，以 `add_example` 为模板实现自定义 Ascend C 算子，并接入本仓 `torch_custom/fla_npu` 的 Python 接口（新增接口步骤见[开发者指南](../docs/开发者指南.md)的场景 3）。

### fast_kernel_launch_example

进入 `fast_kernel_launch_example/`，按该目录 `README.md` 安装依赖、构建 wheel 并运行测试，即可体验单文件交付的算子开发流程。

## 新增示例要求

- 新增示例必须可独立运行，避免依赖其他示例的中间产物。
- 涉及新算子的示例，需同时提供该算子的单算子测试（`torch_custom/fla_npu/test/test_npu_<op>.py`）并接入 `test.sh`。
- 示例代码优先使用稳定 Python 入口 `fla_npu.ops.ascendc` / `fla_npu.ops.triton`，不要默认走 legacy `torch.ops.npu.*` 路径。
- 在 `examples/` 新增子目录时，如需参与统一编译，请提供对应 `CMakeLists.txt`。
