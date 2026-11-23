# MegaRecurrentKda ATK

安装对应产品的 wheel 后从仓库根目录运行。A2 使用 `ascend910b`，A3 使用 `ascend910_93`，Ascend 950 使用 `ascend950`。

```bash
bash tests/atk/run_test_cpu.sh -op=mega_recurrent_kda -soc=ascend950 -npu_device_id=0 -scope=accuracy
```

九个用例覆盖 BSH、TH、不等长序列跨越 64 行、卷积权重的三种布局、物理状态槽间隙、缓冲区复用及 4096 token。CPU 与设备由相同种子生成输入，CPU 执行独立的 PyTorch 参考。设备调用公开 Stable ABI，检查四个状态输出与输入共用地址，以及 checkpoint 保持原值。

| 文件与分支 | case ID | 覆盖 |
|---|---|---|
| 精度，T01 | 0、1、4、5、6、7 | 单 token、不等长短序列、状态槽间隙、两种主布局、五个 64 行块 |
| 精度，T02 | 2、3、8 | 不等长序列跨界、129 token、多次状态交接至 4096 token |
| 性能 | 5、6 | B40/N8/S4，分别为 BSH、TH |
| 确定性及内存 | 0、4 为 TilingKey 1，2 为 TilingKey 2 | 基本路径、带槽间隙路径和跨块路径 |

在 Ascend 950、CANN 9.2.0-beta.2 上用相同的 CPU 标杆执行完整 ATK。精度 9/9、确定性 3/3、内存检查 3/3 均通过。两项性能任务成功采集数据。另通过公开 Python 入口逐项验证 out、conv_state 和三项 replay，重复运行结果一致，checkpoint 和未写区域保持原值。非法参数覆盖见 [10 个反例](test_invalid_inputs.py)。

B40/N8/S4 的交付构建预热 3 次，随后 5 次融合 kernel 的 Task Duration 中位数为 BSH 99.946 µs、TH 97.624 µs；同批形状在另一比较构建上为 94.160/93.781 µs。目标均为 39.96 µs，两项均未达标。CPU 计时覆盖 Python 调用至设备同步完成，交付构建测得 BSH 186.170 µs、TH 270.777 µs。这个口径包含元数据传输、分发和同步。

在 Atlas A2 上，公开入口通过 ATK 精度 9/9 和非法输入 10/10。另用 104 个形状和布局组合对比五项输出，检查重复运行、状态别名、checkpoint 及未写区域，并单测 N=129 的两种布局。Atlas A3 已交叉编译。
