# 独立实验 Python 与 Stable ABI wheel

`fla_npu_experimental` 通过独立 `libfla_npu_experimental_stable.so` 调用实验 ACLNN，
只注册 `torch.ops.fla_npu_experimental_stable`。旧 ctypes tensor 描述符执行后端已移除；
ctypes 仅用于读取 launcher 指纹和队列能力。稳定 `fla_npu` 不作为依赖。

当前公开包装函数只有 `fused_recurrent_rwkv8`。
launcher 仅注册本次所选实验闭包；调用未构建的包装函数会明确报错。
RWKV8 `chunk_len` 范围为 `[1,4294967295]`，与 adapter/aclnn/tiling 一致。

## 构建开发 wheel

初始化 CANN/conda，设置 `FLA_NPU_DISABLE_PTH=1`，先编译并独立安装 `.run`：

```bash
export FLA_NPU_DISABLE_PTH=1
bash build.sh --pkg --experimental --soc=ascend950 --ops=fused_recurrent_rwkv8 -j4
bash build_out/fla_npu_experimental_linux-$(uname -m).run \
    --install-path=/absolute/empty-experimental-opp --quiet
python experimental/_framework/build.py wheel \
    --manifest build/experimental-build-manifest.json \
    --install-root /absolute/empty-experimental-opp \
    --output /absolute/development-wheels
```

wheel 名为 `flash_linear_attention_npu_experimental_a5-0.1.0.dev0+<fingerprint>-py3-none-linux_<arch>.whl`。
SoC 分档复用 main 的 a2/a3/a5 映射；当前为本机开发包，不声明 manylinux 发布兼容性。
打包强制执行 main 的 ACLNN 和 ELF 审计，保留 GLIBCXX 3.4.21 上限。

## 安装与加载

可以安装至独立 Python 环境，或用 `pip --target` 避免改变当前环境：

```bash
python -m pip install --no-deps --target /absolute/experimental-python /absolute/development-wheels/*.whl
export PYTHONPATH=/absolute/experimental-python
source /absolute/experimental-python/fla_npu_experimental/opp/vendors/fla_npu_experimental_transformer/bin/set_env.bash
python ci/experimental_load_smoke.py
```

需已有 torch / torch_npu >= 2.7.1；由运行环境提供 CANN。必须先 source CANN，再 source
wheel 内实验环境脚本，最后启动新的 Python 进程。包必须早于 torch_npu/ACL 导入：

```python
from fla_npu_experimental import fused_recurrent_rwkv8
import torch
import torch_npu
```

wheel 只接受其内嵌 OPP，拒绝与外部 `.run` 拼装。源码目录直接加入 `PYTHONPATH` 不能执行
算子，因为尚无匹配的 launcher 和构建清单。切换算子或修改 Python/adapter/OPP 后应重建 wheel
并启动新进程。

加载器核验实验来源清单、main ABI 清单、Python 文件/launcher 摘要及编译指纹。
实验 ACLNN 符号缺失时立即失败；只有 SDK 辅助符号可以从 CANN 解析。
稳定包、稳定 launcher、其他 vendor 或已经初始化的 CANN 与实验入口混用会被拒绝。
环境和文件在加载后改变也需要重启进程。

`ci/experimental_load_smoke.py` 只做 host 加载与注册核验，检查 NPU 尚未初始化。
它不证明算子精度、性能或 NPU 运行成功。独立的 A2 数值回归见
[RWKV8 验证记录](../validation-rwkv-only-20261106.md)；A5 未获得数值运行证据。

CPU 契约检查：

```bash
python -m unittest discover -s experimental/_python/tests -v
python -m unittest discover -s ci/tests -p 'test_experimental_*.py' -v
```
