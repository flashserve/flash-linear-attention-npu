# 详设模型性能报告（2026-11-25）

硬件：179.183.2.2，Ascend910B3（A2），设备0。CANN9.3.0，ATK26.9.24。
使用此前精度200/200、确定性20/20、MSS20/20的同一正常kernel，公开Stable入口。
本次仅更正模型性能资产并重新测量，未修改算子数学实现。

M1：B=1、HK=16、HV=128、R=8、T=11264，定长BNSD。
M2：B=1、HK=16、HV=128、R=8、T=16384，变长NTD；用户明确授权的16×1024开发探针。
M2的16条序列长度均为1024，cu_seqlens=[0,1024,...,16384]，共256个chunk。
M2不能等同于未提供实际cu_seqlens的原始模型数据；原始模型M2仍未验收。
两者BF16，C=64、K=V=128，scale=1/sqrt(128)，state_v_first=False；seed分别20261125、20261126。

每条5轮，各预热10次、有效采样20次，合计100个有效样本。
ATK performance_device采集PROF，随后msprof --export=on导出op_summary。
以单个主kernel的Task Duration(us)统计，不含输入构造、元数据转换或API总耗时。

| 模型 | 中位数 us | 均值 us | p95 us | min/max us |
|---|---:|---:|---:|---:|
| M1 | 2473.7195 | 2474.28166 | 2483.270 | 2461.409/2488.490 |
| M2（16×1024探针） | 3610.0920 | 3610.51129 | 3624.313 | 3592.931/3630.492 |

此前H96/T8192和H96/T16384结果不对应详设M1/M2，只保留为其它shape的历史测量。
没有原生算子对比基线，不宣称0.4x目标达标，也不外推A3/A5。

全部命令、输入及原始数据：/home/yqw/atk/prepare_models_design。
- run_M1.sh / run_M2.sh：可复现采集命令。
- performance_profiles.json / M2_metadata.json：属性、seed和精确变长元数据。
- msprof_model_summary.json：逐轮原始100样本、op_summary绝对路径及统计。
- model_performance_manifest.json：输入与命令哈希。
- results_M1_round0..4 / results_M2_round0..4：原始ATK与PROF数据。

## 同shape精度复核（2026-11-25）

M1与M2授权16×1024探针均使用性能测试同一seed和属性完成CPU FP32/FP64双标杆精度验证：2/2用例、6/6输出通过，失败0、未执行0，阈值未变。详见 /home/yqw/atk/prepare_models_design/accuracy_models/model_accuracy_report.md。
