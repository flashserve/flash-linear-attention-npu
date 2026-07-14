# fla_npu Python 适配说明

`torch_custom/fla_npu` 是 FLA NPU 的 Python runtime 与可选 legacy PyTorch dispatcher 适配工程。当前默认交付目标是：

```python
from fla_npu.ops import ascendc as ascendc_ops

out = ascendc_ops.npu_chunk_fwd_o(...)
```

也可以按公开短名导入：

```python
from fla_npu.ops.ascendc import chunk_fwd_o
```

默认路径通过 Python `ctypes` 直调已安装 OPP 里的 `libcust_opapi.so`，不依赖 PyTorch dispatcher 注册，也不会默认编译或加载 `torch_npu` 自定义扩展。旧的 `torch.ops.npu.*` / `torch_npu.ops.*` 兼容路径仍可做，但只作为迁移期可选能力，不推荐新增代码使用，也不会默认使能。

## 默认交付件

### Python runtime wheel

默认执行：

```bash
python3 setup.py bdist_wheel
```

会生成纯 Python wheel，核心内容包括：

- `fla_npu/__init__.py`：定位并加载内嵌或外部 OPP。
- `fla_npu/ops/ascendc/__init__.py`：稳定 Python 调用入口、短名导出和正反向自动绑定。
- `fla_npu/ops/ascendc/_aclnn_ctypes.py`：具体算子的 Python wrapper，只描述输入输出、标量转换和算子级 ABI 特例。
- `fla_npu/ops/ascendc/_runtime.py`：公共 ctypes runtime，封装 aclTensor/aclIntArray 描述符、workspace、stream 和异步 launch 生命周期。
- `fla_npu/opp/vendors/fla_npu_transformer/...`：一键 wheel 打包时内嵌的 OPP 产物。

### Ascend C OPP 产物

一键 wheel 或 run 包安装后，OPP vendor 目录为：

```text
fla_npu/opp/vendors/fla_npu_transformer
```

关键产物包括：

- `op_api/lib/libcust_opapi.so`：自定义 aclnn op_api 动态库。
- `op_api/lib/libopapi.so`：wheel 内优先解析用的同名兼容副本。
- `op_api/include/aclnnop/aclnn_*.h`：Python ctypes ABI 对齐依据。
- `op_impl/ai_core/tbe/op_host/...`、`op_tiling/...`、`op_proto/...`：host、tiling、proto 动态库。
- `op_impl/ai_core/tbe/kernel/...`：AI Core kernel `.o` 和 config。

## 推荐调用方式

新增或修改测试、example、上层业务时，默认使用：

```python
from fla_npu.ops import ascendc as ascendc_ops

result = ascendc_ops.npu_recompute_w_u_fwd(...)
```

或者：

```python
from fla_npu.ops.ascendc import recompute_w_u_fwd

result = recompute_w_u_fwd(...)
```

测试中不要默认调用：

```python
fla_npu.load_legacy_torch_ops()
torch.ops.npu.npu_xxx(...)
```

这样可以避免把测试结果绑定到 PyTorch/torch_npu dispatcher ABI。

## 新算子如何接入默认 runtime

新增 Ascend C 算子后，Python 默认路径需要同步做以下适配：

1. 在 `_aclnn_ctypes.py` 增加 `npu_xxx(...)` wrapper。
2. wrapper 内按 `aclnn_xxx.h` 的函数签名顺序构造参数，常用转换如下：
   - Tensor 输入/输出：`ctx.tensor(tensor, "name")`
   - `Optional[list[int]]` / `Sequence[int]`：`ctx.int_array(values)`
   - aclnn 签名中要求 Tensor 形式的索引：`ctx.int_tensor(values, device)`
   - 标量：显式使用 `ctypes.c_int64`、`ctypes.c_double`、`ctypes.c_bool` 等。
3. 在 wrapper 内申请输出 tensor，并把输出传给 `_call_aclnn(...)`。
4. 如果 aclnn 参数里有 `char *`、字符串、或 ctypes 不能安全自动转换的参数，在 `_GET_WORKSPACE_ARGTYPES` 中补充 `GetWorkspaceSize` 的 `argtypes`。
5. 在 `fla_npu/ops/ascendc/__init__.py` 的 `_ASCENDC_OPS` 中加入 `npu_xxx`，这样会自动导出 `npu_xxx` 和去掉 `npu_` 前缀后的短名。
6. 如果存在明确的正反向关系，在 `BACKWARD_OPS` 中补充映射；需要 autograd 自动绑定时，在 `__init__.py` 中增加对应 `torch.autograd.Function` 包装。
7. 新增或更新测试，默认调用 `fla_npu.ops.ascendc` 路径。

新增算子通常不需要修改 `_runtime.py`，也不需要感知 `_AclTensor`、`_AclIntArray`、workspace 申请或 stream launch 细节。只有公共 ctypes 调发框架本身需要演进时，才修改 `_runtime.py`。

示例骨架：

```python
def npu_my_op(x, weight, *, scale=1.0, indices=None):
    out = _empty_like(x)
    return _call_aclnn(
        "aclnnMyOp",
        lambda ctx: [
            ctx.tensor(x, "x"),
            ctx.tensor(weight, "weight"),
            ctx.int_array(indices),
            ctypes.c_double(float(scale)),
            ctx.tensor(out, "out"),
        ],
        out,
    )
```

## 构建和验证默认 runtime

只构建 Python runtime wheel：

```bash
python3 setup.py bdist_wheel
python3 -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
```

这个 standalone wheel 会先安装 Python runtime 和空的 OPP vendor 骨架：

```text
site-packages/fla_npu/opp/vendors/config.ini
site-packages/fla_npu/opp/vendors/fla_npu_transformer/
```

随后安装算子 run 包即可把真实 OPP 产物合并到同一个位置：

```bash
bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu
./build_out/fla-npu-*.run --full
```

安装完成后，`site-packages/fla_npu/opp/vendors/fla_npu_transformer` 下会包含 `libcust_opapi.so`、`libopapi.so`、aclnn 头文件、host/tiling/proto 动态库和 kernel 产物，`from fla_npu.ops.ascendc import 算子名` 的运行时布局与一键 wheel 保持一致。standalone wheel 的分发包名与一键 wheel 保持一致，均为 `flash-linear-attention-npu`；Python 导入名仍为 `fla_npu`。

如果使用仓库根目录的一键 wheel，OPP 会内嵌到 `site-packages/fla_npu/opp`：

```bash
FLA_NPU_SOC=ascend910b python3 -m pip wheel --no-build-isolation --no-deps . -w dist
python3 -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl
python3 scripts/check_packaged_wheel_api.py
```

单算子 run 包覆盖已安装 wheel 内嵌 OPP 或 standalone wheel 已安装 OPP 时：

```bash
bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu --ops=chunk_fwd_o
./build_out/fla-npu-*.run --full
```

安装器会列出 scoped run 包覆盖后的算子状态。`WARNING` 表示安装后不可用，`NOTICE` 表示需要人工关注，`OK` 表示 ABI 一致并继续可用。

## legacy torch_npu / torch.ops.npu 路径

如果确实需要兼容 `torch_npu.ops.xxx`，可以显式安装 Python wrapper：

```python
import torch_npu
from fla_npu.ops import ascendc

ascendc.install_torch_npu_ops_compat()
torch_npu.ops.npu_xxx(...)
```

如果确实需要兼容更旧的 `torch.ops.npu.xxx` 调用：

```python
import fla_npu

fla_npu.load_legacy_torch_ops()
torch.ops.npu.npu_xxx(...)
```

则需要显式生成并构建 legacy extension：

```bash
bash gen.sh npu_custom.yaml
FLA_NPU_BUILD_LEGACY_EXTENSION=1 python3 setup.py bdist_wheel
```

legacy 路径会生成或使用：

- `op_plugin/`
- `torch_npu/csrc/`
- `custom_aclnn_extension_lib*.so`
- `npu_custom.yaml`、`test_native_functions.yaml`、`deprecated.yaml`

这些兼容路径不会默认使能。`torch.ops.npu` legacy extension 会重新绑定 PyTorch、Python、C++ ABI 和 torch_npu dispatcher 行为，因此只用于历史接口兼容或专项验证。新增算子默认不要以 legacy extension 作为唯一调用方式。

## 测试要求

- 新测试默认使用 `from fla_npu.ops import ascendc as ascendc_ops`。
- 不要在默认测试里调用 `fla_npu.load_legacy_torch_ops()`。
- 不要把 `torch.ops.npu.*` 作为默认正确性路径。
- 如果 legacy 路径确实需要覆盖，应单独写清楚测试目的，并显式打开 `FLA_NPU_BUILD_LEGACY_EXTENSION=1`。
