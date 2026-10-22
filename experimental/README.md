# experimental 实验算子目录

本目录存放尚未完成转测、暂不进入正式商用版本，但需要在仓内持续开发和共同维护的算子。进入本目录不表示算子已经达到正式发布要求。

- 目标设计：[experimental 构建适配设计.md](../docs/architecture/experimental%20构建适配设计.md)
- 调研与验证记录：[experimental 构建适配过程记录.md](../docs/architecture/experimental%20构建适配过程记录.md)
- 验收要求：[Issue #931](https://github.com/flashserve/flash-linear-attention-npu/issues/931)

## 目录约定

`experimental/` 后的第一个子目录名就是算子名，算子目录必须包含根层 `CMakeLists.txt`：

```text
experimental/
├── _common/
│   └── manifest.txt
├── recompute_w_u_fwd_experimental/
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

目标命令如下：

```bash
# 构建一个实验算子及其完整实验依赖闭包
bash build.sh --pkg --experimental --soc=ascend910b --ops=<实验算子名>

# 构建实验树中的全部算子
bash build.sh --pkg --experimental --soc=ascend910b
```

`recompute_w_u_fwd_experimental` 是从稳定算子复制并重命名的构建验证用例；稳定算子仍留在原目录。A5 验证只编译、打包并安装到隔离目录，检查算子文件清单，不运行算子。缺失 `_common/manifest.txt` 的负向用例只检查构建期失败。

当前仍需完成独立 experimental CI、独立制品命名和完整产物门禁，Issue #931 才能验收。
