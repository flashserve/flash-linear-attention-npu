# VDN 视频线性注意力前向

`vdn_forward` 计算视频局部 softmax 窗口**以外**的线性注意力贡献。当帧窗口覆盖整段视频且没有文本初态时，输出为零。该接口供推理使用，返回 RMSNorm、输出门控和输出投影之前的张量。Anchor 提取、模型投影和 TP 通信由调用方完成。

```python
import torch
import torch_npu
from fla_npu.ops.vdn import vdn_forward

torch.npu.matmul.allow_hf32 = False
# q、k、v 为 [帧, 本卡头数, 每帧 token 数, 128]。bounds 来自模型的窗口划分。
output = vdn_forward(
    q, k, v, beta, alpha, bounds, text_state,
    delta_rule="newton_schulz", fuse_silu=True, fuse_qk_l2norm=True,
    frame_tile=64, value_first=True,
)
```

接入发布模型的 Cholesky 路径时使用 `delta_rule="vdn_solve"`（默认）。已在上游完成激活或归一化时，关闭对应开关可避免重复计算。以同形状合成输入测试时，应让相邻窗口、不同帧和两个递推方向都能影响输出。

- [输入、窗口语义与选项](docs/api.md)
- [数据流、内存与性能复现条件](docs/design.md)


## 在 910B 上启用原生加速

先按仓库根目录的说明准备 CANN、PyTorch、torch_npu 和 Triton Ascend。在仓库根目录构建 A2 wheel：

```sh
FLA_NPU_SOC=ascend910b FLA_NPU_VDN_NATIVE=1 python scripts/build_wheel.py
```

安装构建出的 wheel 后，在新进程中检查插件。启用原生后端时，双向递推、窗口合并和符合条件的 Newton 对角校正通过独立的 PyTorch Stable ABI dispatcher 启动。标准组合路径也能在未编入插件时使用 PyTorch 和 Triton 完成推理。

```python
import torch
import torch_npu
from fla_npu.ops.vdn import vdn_native

device = torch.device("npu", torch.npu.current_device())
print(torch.npu.get_device_name(device), vdn_native.library(device) is not None)
```

完整测试矩阵、精度阈值和基准命令见[测试说明](../../../../../tests/atk/vdn_forward/README.md)。从仓库根目录运行单算子 smoke 的方式如下：

```sh
bash torch_custom/fla_npu/test/test.sh --device 0 --op vdn_forward
```
