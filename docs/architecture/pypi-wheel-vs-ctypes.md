# 发布到 PyPI：thin launcher wheel 与 ctypes wheel 的差异

> 面向“把 wheel 发到 PyPI 给客户安装”这一场景，对比当前薄层（thin
> launcher）与之前纯 ctypes 版本的差异，并给出发布矩阵与命名建议。
> 实测对象：910b 环境（CANN 9.1.0，Python 3.11 + torch 2.9 + torch_npu
> 2.9.0.post2），950 环境（Python 3.12 + torch 2.9 + torch_npu 2.9.0.post2）。

## 1. 结论摘要

| 维度 | ctypes 版 wheel | thin 版 wheel |
| --- | --- | --- |
| wheel tag | `py3-none-any`（纯 Python） | `cp3XX-cp3XX-linux_{aarch64,x86_64}` |
| 单文件体积（910b，全算子 OPP） | ~2.4 MB（压缩） / 8.6 MB（解包） | 26.4 MB（压缩） / 79.5 MB（解包） |
| 一份 wheel 覆盖范围 | 任意 Python、任意 CPU 架构 | 一个 Python 版本 × 一个 CPU 架构 × 一个 SOC |
| PyPI 发布矩阵 | 1 个/SOC | Python 版本 × 架构 × SOC |
| 编译期依赖 | 无（纯 Python） | 仅 torch（C++ extension）；**不依赖 torch_npu** |
| 运行期依赖 | CANN + 内嵌 OPP（无需 torch_npu 参与调度） | CANN + 内嵌 OPP + torch_npu（tensor/stream 语义） |
| 热路径 host 开销 | 高（每调用 Python/ctypes descriptor 建销） | 低（C++ 适配 + 每调用 raw stream 查询） |
| 非法输入报错 | Python 校验，多为 `TypeError`/`ValueError` | 大多直接由 aclnn 报错（类型不保证一致） |
| 回退能力 | 无（就是唯一路径） | `FLA_NPU_THIN_LAUNCHER=0` 整进程回退 ctypes；未适配算子自动回退 |

结论：**thin 版不能再用一份 wheel 覆盖所有 Python/架构**，必须按矩阵产包；
同时因为 SOC（910b/910_93/950）目前靠 build tag 编码，直接往同一个 PyPI
项目上传多个 SOC 的 wheel 会有“装错 SOC”的风险（见 §3.2）。

## 2. 实测差异（同一 910b 环境）

老 wheel（thin 关闭，纯 Python + 内嵌 OPP）：

```
flash_linear_attention_npu-26.7.0.dev0-910b.aarch64-py3-none-any.whl
  356 files, 8.56 MB (unpacked)
  WHEEL: Root-Is-Purelib: true | Build: 910b.aarch64 | Tag: py3-none-any
  has _C_thin: False | opp vendor: True
```

新 wheel（thin 默认编译，512 验证分支，只内嵌 2 个算子 OPP）：

```
flash_linear_attention_npu-26.7.0.dev0-910b.aarch64-cp311-cp311-linux_aarch64.whl
  373 files, 9.33 MB (unpacked)
  WHEEL: Root-Is-Purelib: false | Build: 910b.aarch64 | Tag: cp311-cp311-linux_aarch64
  has _C_thin: True | opp vendor: True
```

主分支全算子 OPP 的 thin wheel：压缩 26.4 MB、解包 79.5 MB，其中 OPP
run package 占 78.4 MB（各算子内核 `.o`/`.json`），`_C_thin.so` 只有
0.51 MB。按需只编目标算子时体积会显著下降（512 验证分支只编
`causal_conv1d` + `recurrent_gated_delta_rule`，wheel 仅 2.66 MB 压缩 /
9.33 MB 解包），这也是“只编用到的算子更快更小”的另一个好处。

两个 wheel 的 `METADATA` 依赖列表完全一致（`pyyaml/numpy/decorator/sympy/
scipy/attrs/protobuf/psutil/expecttest/packaging`），**都不声明
torch/torch_npu**，`Requires-Python` 都是 `>=3.9`——thin wheel 实际只支持
`cp310/cp311/cp312`，这条元数据需要修正（见 §3.4）。

## 3. 对 PyPI 发布的影响

### 3.1 新增两个维度：Python 版本 × CPU 架构

thin 扩展是 torch C++ extension，ABI 绑定 Python 版本与 CPU 架构。因此同一
份代码要覆盖客户环境，需要：

| Python | linux x86_64 | linux aarch64 |
| --- | --- | --- |
| 3.10 | `cp310-cp310-linux_x86_64` | `cp310-cp310-linux_aarch64` |
| 3.11 | `cp311-cp311-linux_x86_64` | `cp311-cp311-linux_aarch64` |
| 3.12 | `cp312-cp312-linux_x86_64` | `cp312-cp312-linux_aarch64` |

再加上 SOC（910b / 910_93 / 950）与 CANN 兼容性，单次发布的产物数量从
“每 SOC 1 个”变成“Python × 架构 × SOC”。

### 3.2 SOC 目前编码在 build tag，PyPI 上会互相覆盖

两种 wheel 的文件名里 SOC 都在 build tag 位（`-910b.aarch64-`），而
`Version` 相同（`26.7.0.dev0`）、tag 组合也相同。pip 选包时在同一版本/同一
tag 组合下会按 build tag 取较大者，因此：

- 把 910b、910_93、950 的 wheel 都传到同一个 PyPI 项目，客户 `pip install`
  很可能装到 build tag 最大的那个 SOC 版本，与实机不匹配；
- 这个问题在 ctypes 时代就存在（当时也是 `Build: 910b.aarch64`），只是当时
  只有 1 个 tag 维度、更容易人工规避；thin 之后把 Python/架构维度也加进来，
  冲突面变大。

不能靠 PEP 440 local version（`26.7.0+910b`）解决：PyPI 不接受 local version，
且 wheel 文件名会把 `+` 归一化成 `_`。

### 3.3 可选的三条路线

| 方案 | 做法 | 优点 | 代价 |
| --- | --- | --- | --- |
| A. 分发包名（推荐） | 按 SOC 拆项目：`flash-linear-attention-npu-910b` / `-950`…，包内 `import fla_npu` 不变 | PyPI 语义清晰，pip 不会装错 SOC | 需要维护多个 PyPI 项目与各自 CI |
| B. 主包 + OPP 插件包 | 主包只含 `_C_thin` + Python（按 Python×架构发），OPP 单独发 `fla-npu-opp-<soc>`；安装时按 SOC 选择 | 主包矩阵小、内核可独立升级 | 需要新增依赖声明与加载路径约定（当前 `ASCEND_CUSTOM_OPP_PATH`/`.pth` 机制可复用） |
| C. 继续单项目多 build tag | 保持现状，靠文档/脚本让用户指定 `==26.7.0.dev0+...`/直链下载 | 改动最小 | pip 选择不可控，容易装错；不建议对外发布 |
| D. wheelnext variant wheel（vllm-ascend 的做法） | 仍发一个 PyPI 项目，另用 `variantlib` 生成带 SOC 后缀的 variant wheel 放到 variant 索引，用户用 `uv-wheelnext` 按硬件属性选择 | 单项目 + 硬件维度可选，pip/uv 能按 variant 精确匹配 | 依赖 wheelnext/uv 生态；需要在 CI 里按 SOC 各编一次并生成 variants |

内网/离线交付（当前 `build_wheel.py` + `pip install <本地 whl>`）不受以上限制，
按 SOC 各编各装即可；上面的分歧只影响“上 PyPI 公网分发”。

### 3.6 参考：vllm-ascend 的 wheel 与 SOC 处理方式

vllm-ascend 同时提供 **pre-built wheel** 和源码安装（其安装文档
`docs/source/installation.md` 明确给出 `pip install vllm-ascend==...` 与
`uv-wheelnext` 两条路径）：

- PyPI 项目 `vllm-ascend`（示例 0.23.0）：6 个 wheel
  （`cp310/cp311/cp312` × `manylinux_2_34_{aarch64,x86_64}`，约 27 MB/个）
  + 一个 sdist；文件名里**不带 SOC**；
- SOC 维度放到 **wheelnext variant wheel**：发布流水线
  `.github/workflows/schedule_release_code_and_wheel.yml` 按 SOC 分别用
  `Dockerfile.buildwheel.a2 / .a3 / .310p`（A2 里 `SOC_VERSION: ascend910b1`）
  构建 → `auditwheel repair` → `wheelnext/variantlib` 生成 variant，variant
  标签定义在其 `scripts/wheel/config.json`（`310p` / `a2` / `a3`，
  properties 形如 `ascend :: npu_type :: a2`）；
- 华为 variant 索引上的文件名因此带 SOC 后缀，例如
  `vllm_ascend-0.17.0rc1-cp310-cp310-manylinux_2_24_aarch64-910b.whl`、
  `...-a3.whl`、`...-310p.whl`、`...-a5.whl`，用户用
  `uv pip install --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant vllm-ascend==<ver>`
  安装；
- 他们的构建期依赖明显更重：`pyproject.toml` 的 build-system 需要
  `torch==2.10.0` + `torch-npu==2.10.0` + cmake/pybind11，并把 CANN 自定义算子
  编进 `vllm_ascend/_cann_ops_custom`；因此也建议用户优先装 wheel，源码安装
  需要 CANN + 编译器。

对我们的启发：**方案 D 就是“方案 A 的单项目版本”**，既保留一个 PyPI 项目，
又让 SOC 维度由 variant 表达；如果将来我们也要公网分发，建议直接对齐
vllm-ascend 的 variant 机制（我们的 SOC 标签可定为 `910b` / `a3` / `950`），
并考虑用 `auditwheel` 把 `linux_aarch64` 升级为 `manylinux_*`。

### 3.7 当前打包的体积/构建成本，以及两点澄清

当前实现（HEAD `ef285a98`，910b 全算子 wheel）实测：

| 项 | 数值 |
| --- | --- |
| wheel 解包总大小 | 79.45 MB |
| 其中 `fla_npu/opp/vendors` | 78.41 MB（977 个文件：`.h/.hpp/.json/.o/.cpp/.py/.so`） |
| 其中 `_C_thin.cpython-311-*.so` | 0.51 MB |
| wheel 压缩后 | 26.4 MB |
| 对比：512 分支只编 2 个算子的 wheel | 9.33 MB 解包 / 2.66 MB 压缩 |

需要澄清两点：

1. **“整份 OPP / 全算子”是当前一键默认，不是必然**。`setup.py` 里
   `FLA_NPU_OPS` 默认为空，`build.sh --pkg` 因此编译仓库里全部算子；设成
   `FLA_NPU_OPS="recurrent_gated_delta_rule,causal_conv1d"` 时 OPP 体积随算子数
   线性下降（2 算子时整个 wheel 只有 2.66 MB）。
2. **OPP 与 Python 版本无关**，只与 SOC 和 host 架构有关：977 个 OPP 文件里
   没有任何 `cpython-3XX` 命名，5 个 `.so`（`libcust_opapi.so`/`liboptiling.so`/
   `libcust_opmaster_rt2.0.so`/`libes_transformer_cust.so`/`libcust_opsproto_rt2.0.so`）
   都是 CANN host 库、不链接 libpython。因此“每个 Python 版本各带一份 78 MB
   OPP”是**当前打包实现的重复**，不是本质需求。

代价估算（当前实现，Python 3.10/3.11/3.12 × {aarch64, x86_64} × SOC
{910b, 910_93, 950} = 18 份）：上传量约 18 × 26.4 MB ≈ **475 MB**，
构建则要重复 18 次 OPP 编译（910b 约 20-40 min/次，950 更久）。

优化路线（按收益排序）。**注意：下面是提案，当前实现仍是“OPP 内嵌在
每个 fla_npu wheel 里”（`package_data` 含 `fla_npu/opp/**/*`，由
`fla_npu_opp_env.pth` 在解释器启动时把 `ASCEND_CUSTOM_OPP_PATH` 指向
wheel 内的 `fla_npu/opp/vendors/fla_npu_transformer`），尚未拆分**：

| 路线 | 做法 | 结果 |
| --- | --- | --- |
| OPP 拆独立包（推荐） | 主 wheel 只含 Python + `_C_thin.so`（~1 MB），OPP 按 `SOC × host 架构` 发 `fla-npu-opp-<soc>`；沿用现有 `ASCEND_CUSTOM_OPP_PATH` / `fla_npu_opp_env.pth` 加载 | 上传量 ≈ 6 × 1 MB（thin wheel）+ ≤6 × 26 MB（OPP 包）≈ 160 MB；OPP 编译次数从 18 降到 ≤6 |
| 只编需要的算子 | `FLA_NPU_OPS=<用到的算子>` | OPP 体积/构建时间按算子数下降（2 算子 ≈ 2.7 MB wheel） |
| OPP 复用开关（最小改动） | 给 `setup.py` 增加“复用已有 run package”的入口，避免 `_build_run_package()` 每次 `rm -rf build_out` 后重跑 `build.sh` | 各 Python 版本共用同一份 OPP，构建时间只花 1 次；wheel 体积仍是每份 78 MB |

补充：当前 `setup.py::_build_run_package()` 每次构建都会
`shutil.rmtree(build_out)` 再执行 `bash build.sh --pkg`，没有任何跨构建缓存，
所以现状下每换一个 Python 版本都要重编一遍全部算子——这是后续做发布流水线
时最值得先改造的一点。

### 3.8 “上传体积”与“客户下载体积”是两件事

- **OPP 内嵌（现状）**：发布侧要上传 `Python × 架构 × SOC` 份、每份约 26 MB
  （全算子），但**客户只下载一份**——`pip install flash-linear-attention-npu`
  只会选中匹配自己 Python 版本/架构/SOC 的那个 wheel（外加声明的第三方依赖）。
  若接受上传体积，这是客户体验最好的形态，不需要拆包。
- **OPP 拆包（方案 A）**：上传量下降，但客户要装**两个**包（主 wheel ~1 MB +
  `fla-npu-opp-<soc>` ~26 MB），总下载量相近，安装步骤变多。

因此“接受上传体积大”时，建议保持内嵌，只需解决 SOC 维度的选择问题
（独立包名或 wheelnext variant），而不是为了省上传量去拆包。

### 3.9 方案 C 的接口与 CI 用法（提案）

C 不是“上传时才开的宏”，而是**构建期可选环境变量**，默认不设置时行为与现在
完全一致（每次重新跑 `build.sh --pkg`），因此对正常/一键编译没有影响：

```bash
# 1) 每个 SOC/host 架构只编一次 OPP，产出 run 包
bash build.sh --soc=ascend910b --pkg --vendor_name=fla_npu --ops=<需要的算子>
#   -> build_out/fla_npu_linux-<arch>.run   （作为 CI artifact 传给后续 job）

# 2) 各 Python 版本复用它，只编薄层扩展
FLA_NPU_SOC=ascend910b \
FLA_NPU_REUSE_OPP=/path/to/fla_npu_linux-aarch64.run \
python scripts/build_wheel.py --wheel-dir dist
```

落地要点（实现时）：

- `setup.py::_build_run_package()` 增加 `FLA_NPU_REUSE_OPP`（run 文件）与
  `FLA_NPU_OPP_DIR`（已解包目录）两个入口，命中时跳过 `rmtree(build_out)` 与
  `build.sh`，直接交给现有 `_stage_run_package()`；
- **校验**：run 包内 `version.info` 的 CANN 版本与 `op_impl/.../kernel/<soc>`
  目录名必须匹配 `FLA_NPU_SOC`，不匹配直接报错；日志打印复用来源与哈希，
  避免误用过期 run 包；
- 效果：OPP 编译次数从 `Python × 架构 × SOC` 降到 `架构 × SOC`，薄层扩展仍是
  每 Python 版本编一次（约 1 min）；**wheel 内容与体积不变**（每份仍内嵌
  同一份 OPP）。

### 3.4 元数据需要修正的点

1. `Requires-Python`：thin wheel 应至少是 `>=3.10`（当前写成 `>=3.9` 与实际
   tag 不符），最好与产包矩阵一致（如 3.10–3.12）。
2. `torch`/`torch_npu`：故意不写进 `Requires-Dist`（避免 pip 拉错 CPU 版
   torch 或与 CANN 不匹配的 torch_npu），但应在 README/描述里明确
   “先按昇腾官方指引安装 torch + torch_npu + CANN”。
3. 平台 tag：`linux_aarch64`/`linux_x86_64` 是合法可上传的 tag，pip 会安装，
   但不携带 glibc 基线；若后续希望放宽兼容性声明，可用
   `auditwheel repair` 产出 `manylinux_2_28_*`。
4. sdist：thin 的 OPP 需要 CANN 工具链（`build.sh`）才能产出，从 sdist
   源码安装不现实；建议**只发二进制 wheel**，或允许 sdist 走
   `FLA_NPU_BUILD_THIN=0` 编出纯 Python wheel（性能退化）。
5. **版本与 build tag**：`scripts/fla_npu_artifacts.py` 默认在 `main` 分支上会
   生成 local version（`26.7.0.dev0+main.<sha>`）并带 SOC build tag
   （`910b.aarch64` / `950.x86_64`）。**PyPI 不接受 local version**；发布构建
   必须设 `FLA_NPU_DISABLE_LOCAL_VERSION=1`（同时会去掉 build tag，得到干净的
   `...-cp311-cp311-linux_aarch64.whl`），SOC 维度改用独立包名或 wheelnext
   variant 表达。也可用 `FLA_NPU_LOCAL_VERSION` / `FLA_NPU_WHEEL_BUILD_TAG`
   显式控制。

### 3.5 为什么 torch_npu 版本不进入编译期约束（但会影响运行期）

`_C_thin.so` 的 NEEDED 只有 torch 与系统库，没有 `libtorch_npu`：

```
NEEDED: libc10.so, libtorch_cpu.so, libtorch_python.so,
        libstdc++.so.6, libgcc_s.so.1, libc.so.6, ld-linux-aarch64.so.1
```

内嵌 OPP 的 host 库 `libcust_opapi.so` 依赖的是 CANN 库（`libnnopbase.so`、
`libprofapi.so`、`libopapi_math.so`）。

因此：

- **编译期**：thin wheel 与 torch_npu 版本无关；换 torch_npu 重新编出的
  wheel，只要 Python/torch ABI/架构相同，`_C_thin.so` 是同一份；
- **运行期**：`_thin.py` 通过 `torch_npu._C._npu_getCurrentRawStream` 读当前
  线程 stream，老版本 torch_npu 没有该接口时回退
  `torch.npu.current_stream().npu_stream`，不构成硬版本依赖；
- **真正要匹配的是 torch 与 CANN**：torch extension ABI 与 torch 版本绑定，
  内嵌 OPP 内核与 CANN 版本绑定。所以“跨 torch_npu 复用同一 wheel”在
  Python/torch/CANN 一致时成立，反之不成立。

## 4. 客户使用视角的行为差异（与发布相关的部分）

1. **回退开关**：`FLA_NPU_THIN_LAUNCHER=0/off/false/no` 整进程回到 ctypes；
   缺失/不匹配时不设该变量也可用。
2. **ND-only**：thin 只覆盖 ND（连续/普通视图）输入域，非 ND 或未适配的
   子域由 wrapper 自动回退 ctypes（例如 `chunk_kda_bwd*`、`solve_tri` 的
   varlen）。
3. **非法输入**：thin 是性能路径，跳过部分 Python 校验，报错类型可能与
   ctypes 不同（但都会失败）。
4. **in-place 语义不变**：`state`/`conv_states` 等仍由 mutation 契约维护
   （grad 限制 + version counter），客户无需改代码。
5. **多线程/多 stream**：修复后每调用读取本线程当前 stream（与 vLLM 的
   `c10_npu::getCurrentNPUStream().stream()` 语义一致），可安全混用；
   早期全局缓存的版本会导致跨线程串流，已修复并加了回归用例。

## 5. 性能差异（910b 实测，P50）

| 指标 | ctypes | thin |
| --- | --- | --- |
| Recurrent host P50（同一 probe，batch 32） | 0.6341 ms | 0.1294 ms（≈4.9×） |
| stream 查询开销 | 0.0233 ms（`current_stream()` 对象路径） | 0.0012 ms（raw accessor） |
| 每 decode step（~30 次调用）host 开销 | 基线 | 约 -15 ms/step |

说明：修复 stream 缓存问题采用了“每调用 raw 查询”。专项 microbench
（见 `cpp-thin-launcher-measurement-report.md`）里旧缓存版 0.1103 ms →
raw 0.1167 ms，只增加约 0.006 ms/次；若改用简单的
`torch.npu.current_stream()` 每次调用，按 0.0233 ms/次估算，每步会多约
0.7 ms。

## 6. 发布前 checklist（thin wheel）

1. 按矩阵产包：Python 3.10/3.11/3.12 × {x86_64, aarch64} × 目标 SOC；
2. 每个产物跑安装态验证：`tests/regression_thin_ops.py`（当前 20 场景/37 组，
   ctypes vs thin 逐位一致）、`test_wheel_install_smoke.py`（dispatch 门控 +
   `=0` 回退）、多线程/多 stream 用例
   （`test_thin_stream_interleaving.py`、512 分支的
   `test_thin_stream_vllm_pattern.py`）；
3. 核对 wheel 元数据：`Requires-Python`、`Root-Is-Purelib: false`、SOC build
   tag、`.so` 的 NEEDED 不含 `libtorch_npu`；
4. 记录 host P50 与 parity 结果到 thin-migration-inventory.md；
5. 明确 torch/torch_npu/CANN 的安装前提与 SOC 选择方式（避免 §3.2 的装错风险）。

## 7. 相关文档

- `docs/architecture/thin-launcher-release-matrix.md`：发布矩阵与依赖说明；
- `docs/architecture/thin-migration-inventory.md`：逐算子 parity/host 记录；
- `docs/architecture/cpp-thin-launcher-measurement-report.md`：host/device
  实测与 vLLM stream 修复 A/B；
- `docs/兼容与迁移指南.md`：客户迁移视角的整体差异说明。
