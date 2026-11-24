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

## 构建与测试

在仓库根目录构建并安装 Ascend 910B wheel。双向扫描、窗口状态合并和 Newton 对角校正随 A2 OPP 安装，通过仓库的 Stable ABI 适配层调用。
安装方法见[开发者指南](../../../../../docs/开发者指南.md)，接口分层见[架构说明](../../../../../docs/architecture/torch-npu-decoupled-architecture.md)。

```sh
FLA_NPU_SOC=ascend910b python -m pip wheel --no-build-isolation --no-deps . -w dist
python -m pip install --no-deps dist/flash_linear_attention_npu*.whl
bash torch_custom/fla_npu/test/test.sh --device 0 --op vdn_forward
```

ATK 用例、独立 CPU 标杆、求逆残差和端到端基准的运行方法见[测试说明](../../../../../tests/atk/vdn_forward/README.md)。
