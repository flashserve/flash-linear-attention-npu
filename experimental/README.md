# experimental 实验算子

用于仓内共同开发、尚未进入正式算子集合的实现。当前只包含从 [PR #446](https://github.com/flashserve/flash-linear-attention-npu/pull/446) 迁入的 `fused_recurrent_rwkv8`。

- [构建、安装与装载设计](docs/architecture/experimental%20构建适配设计.md)
- [当前验证记录](validation.md)
- [RWKV8 接口和实现](ops/fused_recurrent_rwkv8/README.md)

## 方案状态

2026-11-13 已同步 main `7065b0cd`，包含 #1177 和 #1148。实验专属构建、安装、测试及文档已收敛到
`experimental/`；公共入口只做必要分发。增量安装复用主干管理器，开发者须自行选对实验 base。
转正式按设计中的文件映射迁移，不新增 `promote.py`。标准编包沿用主干单 SoC 边界。

## 当前实现边界

1. 正式构建只包含正式算子，`--experimental` 只包含实验算子及其实验依赖。
2. 实验 `.run`、wheel、vendor、Python 包、launcher 和 dispatcher 使用独立身份；不得覆盖正式 wheel 或 OPP。
3. experimental 不跑 CI；构建、精度和隔离用例仅供手动验证。

`build.sh` 未传 `--experimental` 时，省略 `--ops` 或使用 `--ops=all` 都只编正式算子。
显式传入 `--experimental` 后，全选也只编实验算子。当前没有独立的 `--all` 参数。

实验支持通用增量安装；不支持运行中替换 OPP 或正式/实验同进程混用。使用独立 Python 环境和新进程。增量 base 的归属由开发者保证，不提供跨集合误装的专属检查或恢复。

## 构建与安装

先初始化 CANN 和匹配的 Python 环境：

```bash
# 关闭正式包启动注入；一键编译全部实验算子并生成独立 wheel。
export FLA_NPU_DISABLE_PTH=1
FLA_NPU_SOC=ascend910b python scripts/build_wheel.py --experimental
# 只编指定实验算子，使用 8 个编译任务，将 wheel 输出到指定目录。
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=fused_recurrent_rwkv8 \
    python scripts/build_wheel.py --experimental --build-args='-j8' --wheel-dir dist/experimental
```

`FLA_NPU_OPS` 支持逗号分隔多个算子；未设置或 `all/ALL` 全编实验集合。所选算子须具备 adapter 和
已登记的 Python wrapper。`--debug`、`--sanitizer`、`--oom`、`--no-progress` 及 `--build-args`
沿用一键入口；透传参数中的同一选项重复指定时，以最后一次为准。SoC 使用 `--soc`；若同时传入
历史参数 `-c/--compute-unit`，二者值必须一致，冲突组合不受支持，不新增兼容处理或冲突校验。
实验日志为输出目录下的
`fla-npu-experimental-build.log`。输出目录不能位于会被清理的 `build/`、`build_out/` 或 `output/` 内。
必须在一键入口显式传 `--experimental`，仅在 `--build-args` 或 `FLA_NPU_BUILD_ARGS` 中传入会提前报错。
编包自动使用临时 OPP 安装目录，不安装到当前 Python 环境；完成后打印生成 wheel 的准确安装命令。

需要分别执行 run 构建、安装和 wheel 打包时，仍可使用下面的原有入口：

```bash
# 关闭正式包的启动注入。
export FLA_NPU_DISABLE_PTH=1
# 编译指定实验算子；多选用 --ops=op1,op2，省略或 all 编全部实验算子。
bash build.sh --pkg --experimental --soc=ascend910b --ops=fused_recurrent_rwkv8 -j4
# 将实验 run 安装到独立空目录。
bash build_out/fla_npu_experimental_linux-$(uname -m).run \
    --install-path=/absolute/empty-experimental-opp --quiet
# 将匹配的实验 launcher 和已安装 OPP 打成独立 wheel。
python experimental/_framework/build.py wheel \
    --manifest build/experimental-build-manifest.json \
    --install-root /absolute/empty-experimental-opp --output /absolute/development-wheels
# 把开发 wheel 安装到当前实验 Python 环境。
python -m pip install --no-deps /absolute/development-wheels/*.whl
```

`.run` 仅接受显式绝对空目录或父目录存在的新目录；普通安装拒绝写入正式 wheel/OPP 或默认 CANN 目录。增量安装使用下文通用管理入口。wheel 包名为 `flash-linear-attention-npu-experimental-a2/a3/a5`，Python 包为 `fla_npu_experimental`，不写正式 `fla_npu`，不安装自动启动 `.pth`。开发 wheel 不进入正式发布流程。

实验 wheel 版本根据已组装的 Python、launcher、OPP、清单和管理入口的内容生成，独立于 launcher ABI 指纹。
只改 kernel 也会生成新版本，普通 `pip install <新wheel>` 会安装更新；内容完全相同的包仍可能被 pip 跳过。
管理器 `--source` 不按相同版本跳过安装；`pip install --force-reinstall` 继续用于强制重装。

算子放在 `experimental/ops/<op>/`，必须有 `CMakeLists.txt`、`op_host/`、`op_kernel/` 和 wheel 所需 adapter。只扫描 `experimental/ops/` 的直接子目录，其他实验目录不参与算子发现；`ops/` 内仍保留原有的名称限制。实验依赖必须闭合在实验树内；多算子发现和依赖规则有临时夹具测试。当前真实算子只有 RWKV8；新增算子需在 `_framework/build.py` 登记 wrapper，补充手动精度用例。

构建参数的支持范围见[设计 §3.1](docs/architecture/experimental%20构建适配设计.md#31-buildsh-参数兼容边界)。不把已有构建选项全部视为已实测。

## 运行

在安装实验 wheel 的环境中，先初始化 CANN，再启用 wheel 内的实验 vendor：

```bash
export FLA_NPU_DISABLE_PTH=1
source /absolute/site-packages/fla_npu_experimental/opp/vendors/fla_npu_experimental_transformer/bin/set_env.bash
python - <<'PY'
from fla_npu_experimental import fused_recurrent_rwkv8  # 先于 torch_npu/acl
import torch
import torch_npu
# 构造目标设备上的输入，再调用 wrapper。
PY
```

默认装入实验 base 的 `libfla_npu_experimental_stable.so` 和 `opp/vendors/fla_npu_experimental_transformer/op_api/lib/libcust_opapi.so`，注册 `torch.ops.fla_npu_experimental_stable`。
`FLA_NPU_EXPERIMENTAL_OPP_PATH` 表示显式 OPP root，默认使用内嵌 `opp`；合法增量由通用管理器选择其 OPP 组件。
首次装载检查 ABI、launcher 指纹及符号来源；每次调用只检查环境和正式包混用。文件更新后必须重启，不监控运行中的文件变更。
RWKV8 返回 `(o, s, sa)`，`chunk_len` 范围为 `[1,4294967295]`。

## 增量安装、卸载与恢复

先确认当前环境的 base 是实验 wheel。管理命令不导入算子包或 NPU，直接复用主干的 layer 协议和安装实现。

```bash
# 安装实验 wheel 增量，同时更新适配层和它携带的 OPP。
python -m fla_npu_experimental_manage --source /absolute/update.whl
# 只更新 OPP 组件，保留当前适配层。
python -m fla_npu_experimental_manage --source /absolute/update.run
# 查看 active/pending 组件及兼容性问题。
python -m fla_npu_experimental_manage --status
# 重新设置选中组件的 OPP 顺序，然后启动新的 Python 进程。
source /absolute/site-packages/fla_npu_experimental/opp/vendors/fla_npu_experimental_transformer/bin/set_env.bash
# 临时禁用增量；重新 source 后，新进程只使用 base。
export FLA_NPU_OVERLAY_VENDOR=''
# 恢复默认启用策略；重新 source 后，新进程使用 active 增量。
unset FLA_NPU_OVERLAY_VENDOR
# 移除增量并恢复 base；执行前退出使用该增量的进程，之后重新 source。
python -m fla_npu_experimental_manage --remove-overlay
```

管理器支持 `--package-dir` 显式指定目标包目录；`--source update.whl --opp-source update.run`
沿用主干配套安装机制。不修改 base 文件；ABI 不匹配时使用通用 pending 行为。

```text
site-packages/
├── fla_npu_experimental_manage.py                # 实验管理入口
└── fla_npu_experimental/                         # 以下均属实验包
    ├── __init__.py / _rwkv8.py / _stable.py
    ├── _layers.py / _wheel_overlay.py            # 布局配置 / 公共管理器副本
    ├── layer_manifest.json / _stable_hash.py / _build.json
    ├── libfla_npu_experimental_stable.so         # base 适配层
    └── opp/
        ├── vendors/fla_npu_experimental_transformer/  # base OPP
        │   ├── bin/set_env.bash
        │   ├── op_api/lib/libcust_opapi.so
        │   └── op_impl/...
        └── wheel_layers/                       # 通用增量存储
            ├── state.json
            └── <component-id>/
                ├── components.json / installation.json
                └── fla_npu_experimental/       # 增量适配层和/或 OPP
```

有增量 wheel 时，实际 wrapper 和 launcher 来自选中的组件目录，注册
`fla_npu_experimental_overlay_stable`；OPP 按选中增量 → base 顺序装载。
只有增量 run 时，launcher 保持 base，OPP 使用同一顺序。完整选择在首次导入时固定。

## 手动验证

experimental 不接入 CI，不提供评论触发、独立 job 或必需检查状态。以下用例仅在开发者显式执行时运行。

在仓库根目录初始化 CANN 和 Python 环境后执行；构建依赖通过正常构建流程准备，离线环境需预先准备 `third_party/`：

```bash
# 关闭正式包启动注入，避免污染实验进程。
export FLA_NPU_DISABLE_PTH=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
# A2：编译并独立安装 run/wheel，检查装载隔离，再用默认/inline 各 3 条对拍 CPU reference。
python3 experimental/_tests/validate.py --soc ascend910b --mode quick --device 0 -j4 --output /absolute/new-results
# A5：只检查构建、独立安装、ABI 和 host 装载，不执行 NPU 精度。
python3 experimental/_tests/validate.py --soc ascend950 --build-only -j4 --output /absolute/new-build-results
# 运行构建/安装规则和验证工具的本地单元测试，不启动 NPU 构建或精度任务。
python -m unittest discover -s experimental/_tests -p 'test_*.py' -b
# 运行 Python 装载隔离的本地单元测试。
python -m unittest discover -s experimental/_python/tests -b
```

输出目录必须尚不存在。`--mode full` 使用全部 209 条精度用例，默认队列与 inline 各运行一次。
统一入口在构建前检查依赖；缺必要依赖立即失败，缺 ATK 只 warning。A2 包含 3 项安装拒绝和
17 项装载隔离。结果写入指定目录的 `summary.json`，同时保留阶段日志、数值明细和库来源记录。
增量安装手动验证（在已安装实验 wheel 的 Python 环境执行）：脚本自动定位当前环境管理器对应的
实验 base，安装、执行和清理统一使用该目标，不再提供 `--package` 选择其他目录。

```bash
# 依次验证 wheel、wheel+run、禁用、恢复和仅 run；每种方式启动新进程完成 3 条 NPU 对拍。
python experimental/_tests/incremental.py --wheel /absolute/update.whl \
    --run /absolute/update.run --device 0 --output /absolute/new-incremental-results
```

该入口不读取 CI 环境参数，不调用 CI 容器或缓存脚本，也不上传报告或发布 GitHub 状态。

已安装匹配实验 wheel 时，可按 [ATK 工程说明](tests/atk/fused_recurrent_rwkv8/README.md#手动独立对拍)
直接运行 `experimental/_tests/accuracy.py`。ATK 保留为手工复验备选；独立对拍不依赖 ATK。
PTA/dump 与独立对拍共用 CPU reference；独立对拍沿用 FP64→FP32 和原混合容差。
reverse、并发、确定性、MSS、性能以及 A5 数值不在当前已验证范围内。
