# ChunkKdaBwdFinalize 输出约定

沿用现有 Python/ACLNN 签名及七个返回值顺序 `(dq, dk, dv, db, dg, dA, dbias)`。
输入 dtype、shape、可选属性和数学语义不变。

`output_layout` 在 A2/A3/A5 统一控制 dq/dk/dv/dg/db；db 省略末尾 D 维。

| output_layout | dq/dk/dv/dg | db |
| --- | --- | --- |
| BNSD | [B,H,T,D] | [B,H,T] |
| BSND | [B,T,H,D] | [B,T,H] |
| NTD | [H,T,D] | [H,T] |
| TND | [T,H,D] | [T,H] |

Python 不传 output_layout 时，rank-4 输入默认 BNSD，rank-3 默认 NTD。
输出 dtype：dq/dk/dv 为 BF16，db 跟随 beta，dg 跟随 raw_g，dA 跟随 a_log，dbias 为 FP32。
dA/dbias 不受 output_layout 控制。非法布局、rank 或输出 shape 在 Host 拒绝。

相对 #962，小接口的 db/dg 不再固定 token-major。直接使用该接口的调用方必须
按 output_layout 接收输出；旧反向总接口客户仍保持原有布局处理。

## Gradient dtype change (2026-10-28)

The public total backward API returns dg/db/dA in the dtype of raw_g/beta/A_log
on A2/A3 and A5. This is an intentional breaking dtype change; output_layout
and return ordering are unchanged. dt_bias and its gradient remain FP32.
For direct Finalize calls, its existing input restrictions remain: raw_g is
FP32, A2/A3 a_log is FP32, and A5 a_log can be BF16/FP32. Outputs follow those
actual inputs. In sigmoid mode beta means the supplied saved FP32 beta_eff;
the API cannot infer the dtype of an earlier model projection.

The private L0 permits BF16/FP32 dg and dA descriptors independently of its
working inputs, so total L2 can preserve the original input dtype after its
internal FP32 conversions. Public ACLNN validates the matching dtype.
