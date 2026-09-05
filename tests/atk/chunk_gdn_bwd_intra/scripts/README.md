# 辅助脚本

逐 Stage 验收统一使用 `tests/atk/run_test_cpu.sh`。`smoke_stage.py` 用于在进入 ATK
前直接调用开发期接口，快速确认 ABI、kernel 启动和核间同步是否正常。

## smoke_stage.py

示例：

```bash
python tests/atk/chunk_gdn_bwd_intra/scripts/smoke_stage.py \
  --stage 2 --dtype bf16 --g-dtype fp32 --beta-dtype fp32 \
  --group 2 --hk 3 --hv 6 --tokens 65 --no-use-exp2 --compare
```

主要参数：

- `--stage 0/1/2`：选择开发期 Stage，默认 0。
- `--dtype`、`--g-dtype`、`--beta-dtype`：选择主输入和辅助输入 dtype。
- `--group`、`--hk`、`--hv`：设置 `G=HV/HK`；显式传入时必须满足
  `HV=HK*G`。
- `--batch`、`--tokens`、`--seed`：设置输入规模和确定性 seed。
- `--use-exp2/--no-use-exp2`：选择 gate 计算，默认使用 exp2。
- `--compare`：生成 FP64 CPU 结果，并打印逐输出误差；这是定位信息，不是正式 ATK
  验收结果。

脚本依次打印 `operator_call_begin`、`operator_call_returned` 和
`operator_synchronize_done`，用于区分调用前阻塞、kernel 调用和设备同步。60 秒 watchdog
从实际算子调用前启动，到紧随其后的 `torch.npu.synchronize()` 完成后结束；输入生成、
编译、安装、环境初始化和 CPU 比较不计入这 60 秒。
