# FusedRecurrentRwkv8 experimental

真实 RWKV8 算子从 [PR #446](https://github.com/flashserve/flash-linear-attention-npu/pull/446)
提交 `049eb0d` 迁入。实验构建框架来自 PR #954；此算子没有其他仓内算子依赖，
kernel 只引用 CANN SDK 和本目录私有头文件。host 使用仓库公共 tiling/log/err 基础设施。

- [接口与支持范围](docs/api.md)
- [实现迁移设计](docs/design.md)
- [本轮 main 适配验证](../validation-20261103.md)
- [旧 PR #954 基线验证历史](docs/validation.md)
- [aclnn ABI](docs/aclnnFusedRecurrentRwkv8.md)

本轮范围是 A5 编译验证。A2 原实现保留，待用户提供 A2 环境后回归；编译通过不能替代
A5 精度、性能、确定性或内存检查。

## 编译

在仓库根目录初始化 CANN 9.2.0 和 conda 环境后执行：

```bash
bash build.sh --pkg --experimental --soc=ascend950 --ops=fused_recurrent_rwkv8 -j8
```

实验产物使用独立 vendor `fla_npu_experimental_transformer`。本轮验证编译、独立安装与 host 加载，
不执行 NPU 算子。构建匹配的独立开发 wheel 后，在新 Python 进程中使用
[实验 Python 入口](../_python/README.md)。不要从稳定 `fla_npu` 调用此算子。

## 已保留的回归资产

`tests/cases.py` 保留 13 条原 PTA 规格；`tests/pta/golden.py` 保留原独立 PyTorch CPU
标杆，数学内容未修改。`tests/pta/test_accuracy.py` 已改成实验入口，`examples/` 保留
原 aclnn C++ 对拍与 Python dump 对拍脚本，并改为相对算子目录查找 golden。

原 PR README 记录的 PTA 13/13、ATK 200+9 条 A2 PASS 是来源历史记录。本迁移版本尚未
重跑这些 NPU 用例；ATK 全量用例包和原 GPU fixture 需在 A2 验证阶段补齐，不能把本目录
的 PTA 或编译记录当成 ATK 正式验收。
