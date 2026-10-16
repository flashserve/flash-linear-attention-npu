# GDN / KDA CP 示例

[GDN](chunk_gdn_cp.py) 和 [KDA](chunk_kda_cp.py) 分别演示 CP 切分后的正反向调用。脚本自动生成输入，可与完整序列 CPU 参考对比。

## 环境

在 Linux 昇腾服务器上加载 CANN，激活已安装 `torch`、`torch_npu`、`fla_npu` 的环境。算子包需包含对应平台的 GDN/KDA 小算子和 CP 算子，安装见[开发者指南](../../docs/开发者指南.md)。

```bash
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export FLA_NPU_STABLE_ABI=ctypes
export HCCL_EXEC_TIMEOUT=120
```

包需提供 `pre_process_fwd_kernel_merged`、`chunk_delta_h_bwd_preprocess`（反向预处理旧名）、`merge_fwd_bwd_kernel`。A5 KDA 还需独立的 `aclnnChunkKdaBwdPrepare` 和 `aclnnChunkKdaBwdFinalize`。

## 运行

在仓库根目录执行，将设备 `0,1` 替换为分配到的卡。脚本自动识别平台；双卡时全局 T=256，每卡处理 128 个 token。

```bash
# GDN 双卡
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  examples/cp/chunk_gdn_cp.py --tokens 256 --heads 2 --accuracy-check

# GDN 双卡 GVA：Hk=2，Hv=6
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  examples/cp/chunk_gdn_cp.py --tokens 256 --heads 2 --value-heads 6 --accuracy-check

# KDA 双卡
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  examples/cp/chunk_kda_cp.py --tokens 256 --heads 2 --accuracy-check

# 单卡；GDN 可替换为 chunk_gdn_cp.py
ASCEND_RT_VISIBLE_DEVICES=0 python examples/cp/chunk_kda_cp.py \
  --device 0 --tokens 128 --heads 2 --accuracy-check

# 仅运行 CPU 参考；GDN 可替换为 chunk_gdn_cp.py
python examples/cp/chunk_kda_cp.py --tokens 128 --heads 2 --cpu-reference-only
```

双卡成功时打印 `gdn/kda CP world=2 accuracy: PASS`，退出码为 0。`--accuracy-check` 检查输出、梯度及 h0/dht；不加该参数只检查结果有限。

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--tokens` | `256` | 全局长度，须为 `64 × WORLD_SIZE` 的正整数倍 |
| `--heads` | `2` | Q/K head 数，范围 `[1,128]` |
| `--value-heads` | 等于 heads | 仅 GDN；Hv 在 `[1,128]`，Hv/Hk 为 1、2、3 或 4 |
| `--device` | `0` | 仅单进程使用；多进程由 LOCAL_RANK 选卡 |
| `--seed` | `20261010` | 随机种子 |
| `--accuracy-check` | 关闭 | 与完整序列 CPU 参考比较 |
| `--cpu-reference-only` | 关闭 | 只运行 CPU 参考，不使用 torchrun |
| `--output-tol` | `0.005` | 输出 atol/rtol |
| `--grad-tol` | `0.008` | dq/dk/dv atol/rtol |
| `--gate-tol` | `0.02` | 门控及其参数梯度 atol/rtol |
| `--boundary-tol` | `0.02` | h0/dht atol/rtol |

精度检查同时要求结果有限、普通量余弦相似度 ≥0.999、门控梯度 ≥0.99；`rel_l2` 仅供参考。

## 模型调用入参

内部入口为 `run_gdn(x, *, scale, rank, world, architecture="a5")` 和 `run_kda(...)`，均同时执行前向和显式反向。`x` 为下表字典，`L=T/world`。

四维输入使用连续 **BNSD `[B,H,T,D]`** 布局；BSND 输入先做 `permute(0,2,1,3).contiguous()`。固定 `B=1`、`K=V=128`，Q/K 已做 L2 归一化。

### GDN

| 键 | dtype | 本地 shape | 说明 |
| --- | --- | --- | --- |
| `q`、`k` | BF16 | `[1,Hk,L,128]` | Q/K |
| `v`、`do` | BF16 | `[1,Hv,L,128]` | value、上游输出梯度 |
| `g` | FP32 | `[1,Hv,L]` | 自然对数域逐 token gate，未做 cumsum |
| `beta` | FP32 | `[1,Hv,L]` | 已激活的更新系数，非 logits |

返回 `o/dq/dk/dv/dbeta/dg/h0/dht`。`o/dv` 使用 Hv，`dq/dk` 使用 Hk；GVA 的共享 Q/K 梯度求和。`h0/dht` 为 `[1,Hv,128,128]`。

### KDA

| 键 | dtype | 本地 shape | 说明 |
| --- | --- | --- | --- |
| `q`、`k`、`v`、`do` | BF16 | `[1,H,L,128]` | Q/K/V、上游输出梯度 |
| `beta` | FP32 | `[1,H,L]` | 已激活的更新系数 |
| `g` | FP32 | `[1,H,L,128]` | 原始 gate，未激活、未做 cumsum |
| `A_log` | FP32 | `[H]` | 各 rank 共享 |
| `dt_bias` | FP32 | `[H,128]` | 各 rank 共享 |

固定 `safe_gate=True`，激活公式为 `gate = -5 × sigmoid(exp(A_log) × (g + dt_bias))`。

返回 `o/dq/dk/dv/dbeta/dg/dA_log/ddt_bias/h0/dht`，`h0/dht` 为 `[1,H,128,128]`。`run_kda()` 返回本 rank 的参数梯度，`main()` 再做 AllReduce SUM；模型接入时只归约一次。

### 调用约定

- `scale=128**-0.5`；`architecture` 为 `"a5"` 或 `"a2_a3"`。
- 直接调用前加载 `fla_npu`、初始化 NPU，并将输入放到本 rank 设备；多卡先初始化 HCCL，`rank/world` 与默认进程组一致。有其他并行组时，需改为使用模型的 CP 组。
- 仅沿 token 维连续等长切分；KDA 的 `A_log/dt_bias` 不切分。
- 示例未封装 autograd。训练接入需拆开前后向并保存中间量；`dq/dk` 不含 L2Norm backward。

## 支持范围

- `chunk_size=64`，保存中间量、不开重计算。GDN 支持 GVA；KDA 示例要求 Hk=Hv。
- A5 使用融合阶段小算子；A2/A3 使用小算子链，KDA 部分反向计算使用 PyTorch NPU 运算。
- A2/A5 已通过 GDN/KDA 单卡、双卡及 GDN GVA 精度验证；A3 共用分支由 A2 验证，未做 A3 实机验证。
- 不覆盖 varlen/packed、zigzag、不等长切分、外部非零初始状态、末状态损失和多机验证。
- 当前 Dhu 的 dht 接入通过追加两个虚拟 chunk 后裁剪实现，有额外开销；未做性能验收。
