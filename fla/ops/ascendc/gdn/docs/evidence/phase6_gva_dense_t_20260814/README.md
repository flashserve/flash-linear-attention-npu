# Phase6 原生 GVA 与任意 dense T：A2 最小闭环

日期：2026-08-14（Asia/Shanghai）

## 结论

状态：`PHASE6_GVA_DENSE_T_MINIMAL_PASS`

Phase6 已在 A2（Ascend 910B3）跑通原生 GVA 最小用例：`BF16, B=1, Hk=2, Hv=8, T=130, K=V=128, chunk_size=64/128`。生产调用没有展开 q/k；测试中仅额外构造等价的 q/k 展开路径作为参考。两个 chunk_size 下，原生路径与展开参考的 `output`、`final_state`、`g_cumsum`、`A` 均逐位一致，并且全部有限。

本结论证明原生 `Hk < Hv` 和非白名单 dense `T=130` 已越过 host、tiling、kernel 与 ACLNN 调用链。它不是全规格矩阵、性能验收、A5 适配或 `T=8192` 的替代结论。

## 实现门禁

| 门禁 | 证据 | 结果 |
| --- | --- | --- |
| 生产路径不扩 head | `_aclnn_ctypes.py` 直接把物理 Hk 的 q/k 传入 Phase6；`repeat_interleave` 只存在于 smoke 参考路径 | 通过 |
| 原生 GVA kernel entry | `chunk_gdn_core_fwd` 仍为单 kernel entry，tiling trailer 携带 `Hk/Hv/hvPerHk` | 通过 |
| Hk/Hv/m 元数据贯通 | `ChunkGdnCoreFwdAbcTiling` 与 host tiling 分别记录 `Hk`、`Hv`、`Hv/Hk` | 通过 |
| workspace/output/state 按 Hv | ABC task、A/cumsum workspace、output、final_state 与 A 均按 Hv 分配 | 通过 |
| q/k 的 Hv→Hk 映射 | KKT cube 使用 `taskHead / hvPerHk`；FwdH/FwdO scheduler 使用 `vHeadIdx / headGroups` | 通过 |
| ACLNN 参数校验 | Phase6 要求 `Hq=Hk`、`Hv % Hk == 0`；早期 phase 仍要求三者相等 | 通过 |
| 任意正 dense T | Phase6 tiling 移除旧的 `128/1024/1025` 白名单，只保留正数 T；`T=130` 实机通过 | 通过 |
| A/output/state 头维 | `a_storage=[B,Hv,T,C]`，`o=[B,Hv,T,V]`，state=`[B,Hv,K,V]` | 通过 |

## 构建与部署

- 隔离根：`/opt/chw/gdn-phase6-gva-dense-t-20260814-r1`
- CANN：`/opt/chw/zhengbao/9.1.0.beta1/ascend-toolkit/set_env.sh`
- Conda：`chw-py11`
- SoC：`ascend910b`
- 完整 kernel/package 构建日志：`build_evidence/build.log`，SHA256=`58c13dca360d43280c389b48eb1515d892cbaffa5c30a865dd613a411b1345ff`
- 完整 Phase ACLNN opapi 构建日志：`build_evidence/opapi_full_build.log`，SHA256=`4f351a2caded0f0b14507c9ce25affd19043cd745612e1192a94a979ea316531`
- 最终 host 增量构建日志：`build_evidence/opmaster_rebuild_a_shape.log`，SHA256=`ed29cc732a5e4dfe65266e0f7023684d84f79eb490c7fd2e749bf6ccc3a0122f`
- overlay opapi：SHA256=`1366477f0b159aa5ff003ffb5fc126aaa985cf420adb99d0ed7ddb6d992bdf62`
- overlay opmaster：SHA256=`0d3cdee51c41314987ab8abc2bd60bae4e3ee0f4a030e8bef47cc8b500b041f5`

完整构建日志包含 `Built target cust_opmaster`、`Build libs opapi_transformer success` 和最终 `.run generated`。第一次 smoke 还帮助识别并排除了两个部署问题：旧 opmaster 的 dense-T 白名单，以及只打包 core 时遗漏 `Tiling4ChunkRecomputeWUFwdHO` 依赖；最终 overlay 使用闭合的两算子 host 与完整 Phase opapi。

## A2 smoke

- 脚本：`torch_custom/fla_npu/test/validate_gdn_phase6_gva_dense_t.py`
- 设备：A2 device 0
- 输入：`BF16, B=1, Hk=2, Hv=8, T=130, K=V=128, C=64/128`
- 原生输出 shape：
  - `output=[1,8,130,128]`
  - `final_state=[1,8,128,128]`
  - `g_cumsum=[1,130,8]`
  - `A=[1,8,130,64]`
- 原生对展开参考：四项 `torch.equal=true`
- 有限值：四项全部 `true`
- 原始输出：[smoke_final_state.log](smoke_final_state.log)，SHA256=`30244cb5a5b2dfd935e3ee4c66584ac1d5c6ee670dc85c0f11230b563a8e08bd`
- C128 原始输出：[smoke_final_state_c128.log](smoke_final_state_c128.log)，SHA256=`533f241ba6c3573304d719997357981d9b5447ac39f721fac16f094ffca768d4`
- 结束后没有残留 smoke 进程，device 0 无运行进程。

## 范围边界

当前只收口 A2 最小功能/精度门禁。尚未执行：GVA 全 head-ratio/全 B/全 T/双 chunk_size 矩阵、性能 AB/BA、`T=8192` 正式验证、A5 构建与实机验证。后续扩大测试时应继续小步跑，先做少量代表点，再进入泛化矩阵。
