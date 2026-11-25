# Dhu 接口补充：A2/A3 FP32 缓存

既有公开接口与低精度分支见 [README](../README.md)。本轮不更改参数顺序。

KDA C64 通用分支使用 FP32 qg/kg/w/dv/gk，dO 为 FP16/BF16；相同 head，支持 K/V=60/60、64/64、100/100、128/128、256/256、64/32。
dv2 为 FP32；dh/dh0 为 FP32。通用 dh 随 stateVFirst 为 `[B,Nc,H,K,V]` 或 `[B,Nc,H,V,K]`，每 chunk 写入更新前 Dnext。
旧低精度分支的 dh 仍采用原 K-first 缓存；stateVFirst 对其端点状态的含义不变。

qg 不含 scale。完整计算 termQ=qg.T@dO 后乘 scale，再与 decay*Dnext-W.T@dvScan 合成 Dprev。
独立 h0/dht 可选；组合 KDA 本轮不开放 h0/dht/dh0。
独立阶段两种布局已由 KDA 反向公开阶段用例验证；本次不宣称 GDN 端到端验证。
