# Thin Launcher 发布矩阵说明

> 面向 PyPI 公网分发的完整差异对比（与 ctypes 版 wheel 的 tag/体积/依赖/
> 打包矩阵差异，以及 SOC 维度的发布路线）见
> [pypi-wheel-vs-ctypes.md](pypi-wheel-vs-ctypes.md)。

## 1. 为什么 wheel 不再是 `py3-none-any`

开启 thin 薄层后，`fla_npu._C_thin` 是编译进 wheel 的原生扩展
(`.so`，通过 pybind11 绑定)。wheel 因此带上平台/ABI 标识，例如
910b 一键编包产物：

```
flash_linear_attention_npu-26.7.0.dev0-910b.aarch64-cp311-cp311-linux_aarch64.whl
```

其中：

| 段 | 含义 |
| --- | --- |
| `910b` | 编译产物内嵌的 OPP 面向 ascend910b（`FLA_NPU_SOC`） |
| `aarch64` | CPU 架构（本机为 ARM64） |
| `cp311` | CPython ABI 版本（3.11） |

## 2. 发布矩阵（预期形态）

thin 扩展基于 torch C++ extension，ABI 绑定 CPython 版本与
`linux_aarch64`/`linux_x86_64`，因此发布需要按矩阵产包：

| Python | linux x86_64 | linux aarch64 |
| --- | --- | --- |
| 3.10 | `cp310-cp310-linux_x86_64` | `cp310-cp310-linux_aarch64` |
| 3.11 | `cp311-cp311-linux_x86_64` | `cp311-cp311-linux_aarch64` |
| 3.12 | `cp312-cp312-linux_x86_64` | `cp312-cp312-linux_aarch64` |

同一源码/规格在对应机器上以对应 Python 一键编包即可（OPP 内核产物
面向 Soc，如 ascend910b/910_93/950，通过 `FLA_NPU_SOC` 选择；如需
多 Soc 共存需分别产包或按厂商目录合并）。

## 3. 依赖与解耦说明

- thin 路径编译期只依赖 torch（C++ extension / ATen），不 include
  torch_npu 头、不链接 torch_npu 库；CANN aclnn/opapi 符号在运行时
  通过 `dlopen`（custom `libcust_opapi.so` + `libopapi.so`）解析。
- 运行时仍需：CANN set_env（`libopapi.so` 等）、`fla_npu` 内嵌 OPP
  （自动设置 `ASCEND_CUSTOM_OPP_PATH`/`FLA_NPU_OP_API_LIB`）、
  torch_npu（当前用户侧 tensor/stream 语义仍依赖，属于运行期依赖，
  不是编译期依赖）。
- `FLA_NPU_THIN_LAUNCHER=0/off/false/no` 可整进程回退 ctypes；
  未设/`1` 时已适配算子走 thin，未适配算子自动走 ctypes（动态白名单
  由 `_thin` 模块函数推导）。

## 4. 回归方式

- `test_wheel_install_smoke.py`：安装态（`pip install --target` 或
  site-packages）下的扩展/OPP 加载与 dispatch 冒烟。
- `tests/regression_thin_ops.py`：安装态数值回归，覆盖已迁移算子的
  ctypes-vs-thin parity（逐元素差 0）。HEAD 上为 20 个场景（37 组 PASS
  输出）：fast_gelu、recurrent GDR、recompute、pwy bwd(full/da)、
  dv_local、gated fwd_h、chunk_fwd_h/o、bwd_dhu、conv1d_bwd、KDA
  fwd/bwd/intra、dqkwg、chunk_local_cumsum、scaled_dot_kkt、solve_tri(dense)、
  kda_gate_cumsum。
- 每个算子的数值 parity/benchmark 以 [thin-migration-inventory.md](thin-migration-inventory.md)
  验证状态表为准（ctypes vs thin 输出差 0.0，host P50 已记录）。

## 5. 实测记录（截至 2026-09-09/10）

- 221（910B3，w16 wheel，HEAD d03fcdc5）：regression_thin_ops 20 场景/37 组
  PASS；安装态 smoke 3/3、customer-compat + multi-stream 9 passed/13 subtests
  全绿。
- 950（Ascend950PR 共享机，950d wheel，对应 HEAD 代码）：regression_thin_ops
  20 场景/37 组 PASS（含 chunk_local_cumsum 与 solve_tri dense）；950-only 3
  场景（fwd_prepare/bwd_finalize/recurrent_kda）PASS；安装态 smoke 3/3 OK；
  solve_tri host P50 0.033 → 0.009 ms。
- 说明：950 整 wheel 早期多次被共享机 /home 空间耗尽打断（可用 <2G，kernel
  编译期 ENOSPC）；清理本机可再生成缓存（vscode-cpptools/pip/uv/ccache/
  catlass/Trash 等约 69G）后以 -j2 重编成功，不再受环境限制。
