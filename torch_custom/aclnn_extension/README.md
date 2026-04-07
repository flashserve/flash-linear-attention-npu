# aclnn_extension 样例说明

本样例是 **「自定义算子接入 - 结构化代码生成」** 的示例工程，演示如何通过 YAML 描述 + 代码生成工具，将自定义 aclnn 算子接入 PyTorch NPU 生态，并生成可参与 torch_npu 编译的适配代码。

## 适用场景

本样例适用于以下情况：

- **需要接入自定义 aclnn 算子**：CANN 已提供 aclnn 接口（如 `aclnnFastGelu`），希望以 PyTorch 自定义算子的形式在 `torch_npu` 中暴露给用户。
- **算子满足「结构化适配」条件**：  
  aclnn 算子与目标 ATen IR 的语义一致，适配层**除申请 output tensor（shape/dtype 由输入推导）外，无其他复杂逻辑**。  
  满足时即可用 YAML 做结构化描述，由工具自动生成 C++ 适配代码，无需手写适配层。


不适用或需手写适配的情况：算子语义与 ATen 不一致、需要复杂 shape/type 推导、或需在适配层写额外逻辑时，应使用非结构化方式（如 opapi）。

## 环境与依赖

- **运行/编译依赖**：PyTorch、torch_npu、CANN。  
  安装与版本要求见 [torch_npu 安装说明](https://gitcode.com/ascend/pytorch#%E5%AE%89%E8%A3%85)。


## 目录与文件结构

### 运行前
```
aclnn_extension/
├── aclnn_extension/
│   └── __init__.py                   # 构建用init文件
├── deprecated.yaml                   # 废弃api配置
├── gen.sh                            # 一键生成脚本：调用 torchnpugen 生成算子适配代码
├── setup.py                          # 项目构建脚本，用于编译生成 whl 包
├── npu_custom.yaml                   # 自定义算子 YAML（含前向/反向 ATen IR 与 aclnn 映射）
├── npu_custom_derivatives.yaml       # 前向/反向绑定配置
├── test_native_functions.yaml        # NPU backend 声明（生成 stub 等时会使用）
├── test/
│   └── test_npu_fast_gelu_custom.py  # 自定义算子测试脚本
└── README.md
```



### 运行后

先执行 `gen.sh` 生成适配代码，再执行 `setup.py` 构建后，在 `./dist/`下得到 **`aclnn_extension*.whl`**：

```
aclnn_extension/
├── ...         # 运行前已有文件保持不变
├── build/      # gen.sh生成的中间产物
├── op_plugin/  # gen.sh生成的中间产物
├── torch_npu/  # gen.sh生成的中间产物
├── dist/
│   └── aclnn_extension*.whl # 最终生成的whl包
└── ...
```

使用方式：`pip install aclnn_extension*.whl` 安装后，即可在 Python 中调用本样例接入的自定义算子（如 `torch.ops.npu.npu_fast_gelu_custom`）。


## 一键运行样例

若仅想快速跑通本样例（不修改算子），在 `aclnn_extension` 目录下**按顺序**执行：

```bash
# 1. 先执行 gen.sh，生成适配代码
bash gen.sh npu_custom.yaml

# 2. 再执行 setup.py 构建 whl 包并安装
python setup.py bdist_wheel
cd dist
pip install aclnn_extension*.whl --force-reinstall --no-deps

# 3. 运行测试验证
cd ..
cd test && python test_npu_fast_gelu_custom.py
```

测试通过即表示样例已跑通，可直接调用 `torch_npu.npu_fast_gelu_custom` 算子。

---

**改成接入自己的算子时**：只需把 YAML 里的内容换成自己算子的定义（见下方「使用步骤」），然后同样先跑 `gen.sh`（传入你的 YAML 文件名），再跑 `setup.py` 即可。

## 使用步骤

### 1. 编写自定义算子 YAML（结构化描述）

在 `npu_fast_gelu.yaml` 的 `custom` 段中，按「ATen IR ↔ aclnn」一一对应的方式填写。  
是否可结构化的判断标准：**opapi 对应的 aclnn 与 ATen IR 语义一致，适配层除申请 output tensor 外无其他逻辑**。

示例（前向 + 反向）：

```yaml
custom:
  - func: npu_fast_gelu_custom(Tensor self) -> Tensor
    op_api: all_version
    gen_opapi:
      out:
        size: self
        dtype: self
      exec: aclnnFastGelu
  - func: npu_fast_gelu_custom_backward(Tensor grad, Tensor self) -> Tensor
    op_api: all_version
    gen_opapi:
      out:
        size: grad
        dtype: grad
      exec: aclnnFastGeluBackward
```

- `func`：PyTorch 侧暴露的算子签名（ATen IR 形式）。
- `gen_opapi.out`：输出 tensor 的 shape/dtype 由哪个输入推导（如 `self` / `grad`）。
- `exec`：实际调用的 aclnn 接口名。

### 2. 执行代码生成

在 `aclnn_extension` 目录下执行：

```bash
bash gen.sh npu_custom.yaml
```

脚本会根据当前环境的 PyTorch 版本生成各类适配代码，用户无需关注。

### 4. 构建 whl 包并运行测试

在步骤 3 完成（gen.sh 已执行）后，再执行 `setup.py` 构建，在 `./dist/`下得到 **`aclnn_extension*.whl`**：

```bash
python setup.py bdist_wheel
cd dist
pip install aclnn_extension*.whl
```

安装完成后即可在代码中调用接入的自定义算子（如 `torch_npu.npu_fast_gelu_custom`）。例如用样例自带的测试用例验证：

```bash
cd test
python test_npu_fast_gelu_custom.py
```

测试通过即说明算子已正确接入、结果与参考实现一致。

注意运行测试脚本时不能在`aclnn_extension` 目录下执行，会受init文件所在同名路径影响。

## 自用时的替换与扩展

改成接入自己的算子时，**核心是把 YAML 里的内容换成自己算子的定义**：

- **算子 YAML**（如 `npu_custom.yaml` 或自建 `my_op.yaml`）：在 `custom` 段中写上自己的 `func`、`gen_opapi.out`、`exec`（aclnn 接口名）等，格式参考本样例。
- **gen.sh 参数**：若使用新文件名（如 `my_op.yaml`），则执行 `bash gen.sh my_op.yaml` 或 `bash gen.sh my_op.yaml`。
- **测试脚本**：在 `test/` 下修改或新增测试，调用你暴露的算子名做数值验证。

流程不变：先执行 `gen.sh`（传入你的 YAML），再执行 `setup.py` 构建得到 `aclnn_extension*.whl`，`pip install` 后即可调用。

## gen.sh 结构与代码生成指令说明

本节说明当前目录下 `gen.sh` 的脚本结构及其中调用的代码生成指令，便于有更个性化需求的用户自行调整（如修改输出路径、增删步骤、更换 YAML 等）。

### 脚本参数与环境

- **入参**：`gen.sh` 接收两个参数（第二个可选）。
  - `$1`：算子 YAML 文件（必填），如 `npu_custom.yaml`。
- **工作目录**：脚本会 `cd` 到自身所在目录（`aclnn_extension/`），后续路径均相对该目录。
- **版本与目录名**：从当前环境的 `torch.__version__` 解析出 `PYTORCH_VERSION`（如 `2.7.0`），并得到目录后缀 `PYTORCH_VERSION_DIR`（如 `v2r7`），用于 `op_plugin/config/v2r7/` 等路径。
- **环境变量**：脚本会设置并 `export`：
  - `PYTORCH_VERSION`：PyTorch 版本号。
  - `PYTORCH_CUSTOM_DERIVATIVES_PATH`：生成的 derivatives 文件路径，供后续 autograd 等使用。
  - `ACLNN_EXTENSION_PATH`：当前样例根目录。
  - `ACLNN_EXTENSION_SWITCH="TRUE"`：标识走 aclnn extension 逻辑（部分 torchnpugen 模块会据此分支）。
- **创建的目录**：若不存在则会创建 `build`、`op_plugin`、`torch_npu`。

### 代码生成指令（执行顺序）

脚本按顺序调用以下 **torchnpugen** 模块，对应不同的代码生成步骤。可根据需要增删或改参数。

| 顺序 | 模块 | 作用 | 主要参数 |
|------|------|------|----------|
| 1 | `torchnpugen.gen_op_plugin_functions` | 根据算子 YAML 生成并清洗 `op_plugin_functions.yaml`（ATen IR 与版本信息等） | `--version`、`--output_dir`（如 `op_plugin/config/v2r7/`）、`--source_yaml`（传入的算子 YAML） |
| 2 | `torchnpugen.gen_derivatives` | 仅当传入 `$2` 时执行；根据 derivatives YAML 生成前反向绑定文件 `derivatives.yaml` | `--version`、`--output_dir`、`--source_yaml`（传入的 derivatives YAML） |
| 3 | `torchnpugen.gen_op_backend` | 根据 `op_plugin_functions.yaml` 生成 op_plugin 对外接口与路由（如 OpInterface、OpApiInterface 等） | `--version`、`--output_dir`（`op_plugin/`）、`--source_yaml`（上一步生成的 `op_plugin_functions.yaml`）、`--deprecate_yaml` |
| 4 | `torchnpugen.struct.gen_struct_opapi` | 根据算子 YAML 与 `op_plugin_functions.yaml` 生成 aclnn 结构化适配实现（如 `StructKernelNpuOpApi.cpp`） | `--output_dir`（`op_plugin/ops/opapi/`）、`--native_yaml`（`op_plugin_functions.yaml`）、`--struct_yaml`（传入的算子 YAML） |
| 5 | `torchnpugen.gen_backend_stubs` | 根据 `test_native_functions.yaml` 等生成 torch_npu 侧 backend stub 代码到 `csrc/aten` | `--output_dir`、`--source_yaml`（当前为 `./test_native_functions.yaml`）、`--impl_path`、`--op_plugin_impl_path`、`--op_plugin_yaml_path` |
| 6 | `torchnpugen.autograd.gen_autograd` | 根据 `test_native_functions.yaml` 生成 autograd 相关代码到 `csrc/aten` 与 `autograd` | `--out_dir`、`--autograd_dir`、`--npu_native_function_dir`（当前为 `./test_native_functions.yaml`） |

- **步骤 1～4** 使用算子 YAML 和生成的 `op_plugin_functions.yaml`，产出在 `op_plugin/` 下，是 aclnn 扩展适配的核心。
- **步骤 5～6** 使用 `test_native_functions.yaml`，产出在 `csrc/aten`、`autograd`，用于与 torch_npu 侧的注册与求导衔接。

### 自定义时可调整的内容

- **更换输入 YAML**：修改脚本入参或内部 `$YAML_FILE` / `$DERIVATIVES_YAML_FILE`，或增加新的 YAML 变量并传给对应模块的 `--source_yaml`、`--struct_yaml` 等。
- **输出路径**：修改 `OUTPUT_DIR`、`OPAPI_OUTPUT_DIR`、`--output_dir`、`--out_dir`、`--autograd_dir` 等，使生成结果落到你希望的目录（需与 `setup.py` 的编译路径一致）。
- **PyTorch 版本**：脚本已按当前环境自动解析版本；若需写死某版本，可改 `PYTORCH_VERSION` / `PYTORCH_VERSION_DIR` 的赋值。

按上述说明即可在保留主流程的前提下，按需裁剪或扩展 `gen.sh` 中的指令与参数。

## 小结

- **适用**：自定义 aclnn 算子、语义与 ATen 对齐、适配层仅做 output 申请的结构化场景。  
- **执行流程**：先执行 `gen.sh npu_custom.yaml` 生成适配代码，再执行 `setup.py` 构建得到 `aclnn_extension*.whl`，`pip install` 后即可调用接入的算子。  
- **自用**：把 YAML 里的内容换成自己算子的定义，gen.sh 传入对应 YAML 文件名，同样先 gen.sh 再 setup.py 即可。
