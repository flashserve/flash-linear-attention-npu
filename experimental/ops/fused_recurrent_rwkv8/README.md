# FusedRecurrentRwkv8 experimental

真实 RWKV8 算子从 [PR #446](https://github.com/flashserve/flash-linear-attention-npu/pull/446)
提交 `049eb0d` 迁入。实验构建框架来自 PR #954；此算子没有其他仓内算子依赖，
kernel 只引用 CANN SDK 和本目录私有头文件。host 使用仓库公共 tiling/log/err 基础设施。

- [接口与支持范围](docs/api.md)
- [实现迁移设计](docs/design.md)
- [本轮 main 适配验证](../../validation.md)
- [aclnn ABI](docs/aclnnFusedRecurrentRwkv8.md)

手动验证入口在 A2 检查构建、安装/装载隔离和 NPU / CPU reference 精度；A5 只验证构建与 host 装载。
各版本的实际结果与未覆盖范围见验证记录，不以编译通过代替运行或正式验收。

## 编译

在仓库根目录初始化 CANN 9.2.0 和 conda 环境后执行：

```bash
bash build.sh --pkg --experimental --soc=ascend950 --ops=fused_recurrent_rwkv8 -j8
```

实验产物使用独立 vendor `fla_npu_experimental_transformer`。本轮验证编译、独立安装与 host 加载，
不执行 NPU 算子。构建匹配的独立开发 wheel 后，在新 Python 进程中使用
[实验 Python 入口](../../README.md)。不要从稳定 `fla_npu` 调用此算子。

## 已保留的回归资产

`tests/cases.py` 保留 13 条原 PTA 规格；`../../_tests/rwkv8_reference.py` 保留原独立 PyTorch CPU
标杆，数学内容未修改。`tests/pta/test_accuracy.py` 已改成实验入口，`examples/` 保留
原 aclnn C++ 对拍与 Python dump 对拍脚本，并改为相对算子目录查找 golden。

原 PR README 中的结果属于来源历史记录；迁移后的 A2 手动实测及未覆盖范围见
[验证记录](../../validation.md)。本算子不接入 CI，不把已有编译或精度结果等同于完整正式验收。
