# experimental 实验算子目录

本目录存放尚未完成转测、暂不进入正式商用版本，但需要在仓内持续开发和共同维护的算子。进入本目录不表示算子已经达到正式发布要求。

- 目标设计：[experimental 构建适配设计.md](../docs/architecture/experimental%20构建适配设计.md)
- 验证记录：[A5 main 构建适配](validation-20261103.md)、[A2 功能与隔离](validation-a2-20261103.md)
- 当前主干同步：[2026-11-04 最小适配与回归](validation-main-20261104.md)
- 当前算子范围：[2026-11-06 仅保留 RWKV8](validation-rwkv-only-20261106.md)
- 用例与脚本：[A2 功能与隔离验证方案](_tests/a2_isolation/README.md)
- 验收要求：[Issue #931](https://github.com/flashserve/flash-linear-attention-npu/issues/931)

## 目录约定

`experimental/` 后的第一个子目录名就是算子名，算子目录必须包含根层 `CMakeLists.txt`：

```text
experimental/
├── _common/
│   └── manifest.txt
├── _framework/  # adapter 注册生成、独立 launcher / wheel 构建
├── _python/
│   └── fla_npu_experimental/
├── fused_recurrent_rwkv8/
│   ├── CMakeLists.txt
│   ├── op_host/
│   └── op_kernel/
└── <another_operator>/
    ├── CMakeLists.txt
    ├── op_host/
    └── op_kernel/
```

名称以 `_` 开头的一级目录是框架保留目录，不作为算子发现。算子名在实验树内必须全局唯一；根层 CMake 中的依赖只填写其他实验算子名，不填写仓库路径。实验算子的直接依赖、传递依赖和公共代码必须全部位于 `experimental/` 下，形成可独立构建的完整闭包。

## 严格隔离约束

1. 未启用 experimental 时只能发现、编译和打包稳定算子。
2. 启用 experimental 后只能发现、编译和打包实验算子。
3. 实验依赖缺失时构建必须明确失败，不得回退到稳定树或静默跳过。
4. `experimental/_common` 不完整时构建必须明确失败，不得从稳定 common 补齐。
5. 实验产物只包含所选实验算子及其实验依赖，并使用独立制品标识。
6. 实验制品不得进入正式 Release，也不能作为正式商用能力使用。

## 构建方式

支持 custom OPP `.run` 与独立开发 wheel，默认关闭 experimental：

```bash
# 构建一个实验算子及其完整实验依赖闭包
bash build.sh --pkg --experimental --soc=ascend950 --ops=fused_recurrent_rwkv8 -j4

# 构建实验树中的全部算子
bash build.sh --pkg --experimental --soc=ascend950 -j4
```

当前仅接入从 [PR #446](https://github.com/flashserve/flash-linear-attention-npu/pull/446) 迁入的 `fused_recurrent_rwkv8`，直接用于框架功能和隔离验证。多算子规则使用临时测试夹具覆盖。RWKV8 保留 `ascend910b` 配置并新增 `ascend950` 编译配置；配置存在不代表该平台已经通过运行与精度验收。

实验模式使用 `fla_npu_experimental_linux-<arch>.run` 和独立 vendor `fla_npu_experimental_transformer`。同一 CMake 构建目录不能切换稳定/实验模式；普通 `build.sh` 会重新创建构建目录。构建检查完整依赖闭包、循环、越界符号链接、显式稳定源/头文件路径与实验 common 必需文件，并记录所选算子、依赖、SoC、源码身份和文件摘要。可执行 CMake 仍属于受信任源码，检查器不是任意 CMake 的安全沙箱。

## 独立安装与 Python

只接受显式、绝对、空或尚未创建的安装目录，且父目录必须存在：

```bash
bash build_out/fla_npu_experimental_linux-$(uname -m).run \
    --install-path=/absolute/empty-experimental-opp --quiet
```

安装器拒绝默认安装、稳定 wheel/OPP、非空目录、overlay 参数以及越界链接。当前稳定 wheel 入口拒绝 `--experimental`；实验 Python 使用独立命名空间 `fla_npu_experimental`，不写入稳定 `fla_npu` 或 `torch.ops.npu`。新进程必须显式启用实验 OPP，使用方式见 [_python/README.md](_python/README.md)。独立开发 wheel 通过 `experimental/_framework/build.py wheel` 构建，不进入正式发布流程。

## 当前验证范围

初始化 CANN/conda 后执行：

```bash
CI_SOC=ascend950 CI_EXPERIMENTAL_OPS=fused_recurrent_rwkv8 CI_JOBS=4 \
    bash ci/run_experimental_validation.sh
```

该入口执行隔离契约测试、真实编译打包、空目录安装、双清单审计、独立 wheel 构建及 host-only 加载检查，输出 `build_out/experimental-package-audit.json` 和 `experimental-load-audit.json`。它不执行 NPU；历史 A2 功能与隔离结果见 [A2 验证记录](validation-a2-20261103.md)，其中 RWKV8 209 条精度通过，28 项装载隔离通过。历史制品包含现已移除的样例，不能代替清理后的制品验证。

实验代码及共享构建设施变更保守触发实验树全量重建，覆盖反向依赖，并保留删除/重命名身份。仓内入口尚未接入正式 GitHub 独立实验 job；双平台正式 CI、正式算子完整回归和实验完整验收仍需补齐，当前不能据此关闭 Issue #931。
