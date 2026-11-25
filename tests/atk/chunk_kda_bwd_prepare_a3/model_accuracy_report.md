# 两条模型 shape 精度验证（2026-11-25）

设备：Ascend910B3（A2），物理设备1；沿用性能测试的正常kernel与公开Stable入口。
两条用例均使用性能测试的相同输入生成方式、seed、scale、状态方向和元数据，每条一个seed。
NPU为首个Actual节点，普通CPU节点为FP32 Bench，自动CPU benchmark为FP64 Golden。
原生cv_fused_double_benchmark：max/avg/RMS比值5/1.5/1.5，小值域错误数比值2；未修改阈值。

| 用例 | B/HK/HV/T | 布局/序列 | seed | d_aqk FP32 | dv BF16 | dq_raw FP32 |
|---|---|---|---|---|---|---|
| M1 | 1/16/128/11264 | 定长BNSD | 20261125 | Pass | Pass | Pass |
| M2开发探针 | 1/16/128/16384 | 变长NTD，16×1024 | 20261126 | Pass | Pass | Pass |

结果：2/2通过，6/6输出通过，失败0、未执行0。比较完整输出，包含上三角和尾列。
BF16输入，C64、K128、V128、R8、scale=1/sqrt(128)、state_v_first=False。
M2仍明确为用户授权的16×1024探针，不代表尚未提供真实cu_seqlens的原始模型数据。

运行目录：/home/yqw/atk/prepare_models_design/accuracy_models。
复现：bash run_accuracy_models.sh。manifest.json保存输入规格与文件哈希。
accuracy_models_summary.json包含三路逐输出误差及原始ATK报告路径。
