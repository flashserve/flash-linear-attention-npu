# 验证记录

- CPU reference：覆盖 int32/int64 metadata、align on/off、null/zero 行和 source/boundary 同页。
- NPU：950DT 上 6 项通过，覆盖 BF16/FP32、127/2048/2560/5120 channels、两种物理布局、共享和逐层 dim stride。
- 精度标准：完整 backing storage 与 PyTorch slice-copy reference 零误差一致。
- 性能口径：使用 `msprof` 的 `Task Duration(us)`。
