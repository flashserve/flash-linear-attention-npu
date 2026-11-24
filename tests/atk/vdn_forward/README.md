# VDN 前向测试

DUT 调用公开入口 `fla_npu.ops.vdn.vdn_forward`。输入格式、预处理开关和原生子算子接口见[接口文档](../../../fla/ops/ascendc/vdn/vdn_forward/docs/api.md)。CPU FP32 标杆独立计算加权 Gram，对每帧每头直接解线性系统，两端按帧推进并分别计算窗口衰减乘积。BF16 标杆保留加权 V、value Gram、状态读出与输出的模型舍入点。

## 精度与逻辑分支

固定精度集包含 24 个 case，每行配置各运行两个求解器，每个求解器用种子 431、433、439。ATK 对 CPU 与 NPU 使用同一输入和仓库的 `mixed_tolerance_bm`。HF32 关闭，NPU 正向完成后比较完整输出。测试矩阵的生成、合法值域和独立回归见[用例设计](test_case_design_report.md)。

| 精度 case id | F × H × S | dtype | 文本初态 | SiLU / QK L2 | frame_tile / pad_tail / value_first |
| --- | --- | --- | --- | --- | --- |
| 0–5 | 1 × 1 × 1 | FP32 | 有 | 关 / 关 | 1 / 开 / 关 |
| 6–11 | 7 × 2 × 13 | FP32 | 无 | 开 / 关 | 4 / 关 / 开 |
| 12–17 | 8 × 2 × 13 | BF16 | 有 | 关 / 开 | 4 / 开 / 关 |
| 18–23 | 11 × 3 × 33 | BF16 | 无 | 开 / 开 | 8 / 开 / 开 |

每组前 3 个 id 用 Cholesky，后 3 个用 Newton–Schulz。辅助回归覆盖空间与时间短卷积跨越帧块边界、转置视图、非法窗口/求解器/分块、原生子算子的写回，以及 FP32 求逆残差。独立端到端测试要求相对 L2 误差 < 2e-4（FP32）或 < 0.007（BF16），相对参考输出峰值的最大绝对误差 < 4e-4 或 < 0.04。对单位阵、最大特征值 405 的秩一 Gram 与稠密旋转谱，要求 `||MX−I||F / ||I||F < 2e-5`。

## 性能与模板覆盖

| 性能 case id | F × H × S × D | 路径 |
| --- | --- | --- |
| 0、1 | 85 × 7 × 405 × 128 | Cholesky、Newton–Schulz |
| 2、3 | 87 × 7 × 405 × 128 | Cholesky、Newton–Schulz |

模型用例为 BF16、TP8 后每卡 7 头、带文本状态、SiLU 与 QK L2 开启、帧块 64、尾块补齐、value Gram 优先。独立基准在计时前将输出与 CPU FP32 标杆逐元素比较，8 次预热后测量 31 次同步前向。报告 P10、P50、P90、峰值 allocated、增量 allocated 以及 reserved，输入准备和 CPU 比对位于计时区间外。

| 原生子算子模板 | MSS / 确定性 case id | 选择条件 |
| --- | --- | --- |
| VdnScan，D=128 | 0、1、2 | FP32 transition 与前后状态 |
| VdnBridge，OUT_BF16=0 | 0 | CPU 窗口边界，FP32 输出 |
| VdnBridge，OUT_BF16=1 | 1、2 | CPU 窗口边界，BF16 输出 |
| VdnAddDiagonal，64 列分块 | 2 | 32 帧 × 7 头 = 224 个矩阵，超过原生 Newton 对角校正阈值 |

## 运行

在已安装 A2 wheel/OPP 并初始化 CANN 的环境中，从仓库根目录执行。ATK 环境配置见[公共说明](../README.md)。

```bash
bash tests/atk/run_test_cpu.sh -op=vdn_forward -scope=gen_cases -soc=ascend910b
bash tests/atk/run_test_cpu.sh -op=vdn_forward -scope=accuracy -soc=ascend910b -npu_device_id=0
bash tests/atk/run_test_cpu.sh -op=vdn_forward -scope=determinism -soc=ascend910b -npu_device_id=0
bash tests/atk/run_test_cpu.sh -op=vdn_forward -scope=performance -soc=ascend910b -npu_device_id=0
VDN_TEST_DEVICE=0 python -m unittest discover -s tests/atk/vdn_forward -v
python tests/atk/vdn_forward/scripts/bench_forward.py --device 0 --frames 85
python tests/atk/vdn_forward/scripts/bench_forward.py --device 0 --frames 87 --rule vdn_solve
```

CPU 标杆自检可独立运行 `python -m unittest discover -s tests/atk/vdn_forward -p test_reference.py -v`。

内存检查需同时读取 ATK 的用例报告和 mssanitizer 原始输出。部分 ATK/CANN 组合将设备访存诊断写到 stdout，而 `-msl` 指向的文件可能为空。以下 Bash 命令保留完整日志，并在工具命令失败或检测到原始错误时返回非零值。

```bash
set -o pipefail
failed=0
log=vdn_memcheck.log
bash tests/atk/run_test_cpu.sh -op=vdn_forward -scope=mssanitizer -soc=ascend910b -npu_device_id=0 \
    2>&1 | tee "$log" || failed=1
if grep -aF '====== ERROR:' "$log"; then failed=1; fi
test "$failed" -eq 0
```

## A2 验证记录

Ascend 910B2C、CANN 8.5.0、PyTorch/torch_npu 2.9.0。标准 A2 OPP 随 wheel 安装，NPU 子算子经 Stable ABI 调用。CPU 标杆自检 3/3 通过；完整单测 13/13 通过，覆盖正反向扫描、FP32/BF16 窗口合并、对角更新和共享存储的合法/非法区间。ATK 26.9.8 的 GM 初始化精度套在最终 wheel 上 24/24 通过。确定性套在同一内核实现上 3/3 通过，每例 50 次。设备的原始内存检查日志实际记录了 VdnScan、VdnBridge 两种输出模板和 VdnAddDiagonal 四个二进制入口。

下表为同一张 910B2C 上的端到端同步前向。两个帧数共用同一 87 帧输入前缀，先验全输出，然后交错帧数采样 3 轮，每配置各 96 次。两版都使用 H=7、S=405、D=128、BF16、显式 frame_tile=64。旧版与当前版的中位数可直接比较；Cholesky 旧版的 P90 曾超过 40 ms，尾部波动较大。

| 求解器 | 帧数 | 相对 L2 | 峰值归一化误差 | 旧版 P50 ms | 当前 P10 / P50 / P90 ms | 增量峰值 MiB |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Newton–Schulz | 85 | 2.524e−4 | 4.566e−3 | 4.2130 | 4.1199 / 4.2101 / 4.4163 | 382.22 |
| Newton–Schulz | 87 | 2.514e−4 | 4.566e−3 | 4.2402 | 4.1649 / 4.2388 / 4.3744 | 383.97 |
| Cholesky | 85 | 2.512e−4 | 4.566e−3 | 8.3317 | 8.0956 / 8.1872 / 9.5923 | 382.22 |
| Cholesky | 87 | 2.499e−4 | 4.566e−3 | 8.3394 | 8.1369 / 8.2142 / 9.7381 | 383.97 |
