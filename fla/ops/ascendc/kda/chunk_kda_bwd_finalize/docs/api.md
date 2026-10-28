# ChunkKdaBwdFinalize 输出约定

沿用现有 Python/ACLNN 签名及七个返回值顺序 `(dq, dk, dv, db, dg, dA, dbias)`。
输入 shape、可选属性和数学语义不变；参数 dtype 支持范围见下文。

`output_layout` 在 A2/A3/A5 统一控制 dq/dk/dv/dg/db；db 省略末尾 D 维。

| output_layout | dq/dk/dv/dg | db |
| --- | --- | --- |
| BNSD | [B,H,T,D] | [B,H,T] |
| BSND | [B,T,H,D] | [B,T,H] |
| NTD | [H,T,D] | [H,T] |
| TND | [T,H,D] | [T,H] |

Python 不传 output_layout 时，rank-4 输入默认 BNSD，rank-3 默认 NTD。
输出 dtype：dq/dk/dv 为 BF16，db 跟随 beta，dg 跟随 raw_g，dA 跟随 a_log，dbias 跟随 dt_bias。
dA/dbias 不受 output_layout 控制。非法布局、rank 或输出 shape 在 Host 拒绝。

相对 #962，小接口的 db/dg 不再固定 token-major。直接使用该接口的调用方必须
按 output_layout 接收输出；旧反向总接口客户仍保持原有布局处理。

## Gate parameter dtype contract

On A2/A3 and A5 V2/direct Finalize, raw_g, beta, A_log and dt_bias independently
accept BF16 or FP32. Their gradients follow their respective input dtypes.
This is an intentional gradient dtype change; layouts and return ordering stay
unchanged. In sigmoid mode beta is the saved FP32 beta_eff, not the original
projection; this input restriction remains. q/k/v and their gradients are BF16.

FP32 gate math and parameter-gradient partials are retained. BF16 inputs are
converted on-chip in Stage11; Stage12 rounds to BF16 only after FP32 reduction.
No GM cast intermediate is allocated for Finalize. L0 and public ACLNN both
require gradient descriptors to match their respective inputs. The total
recompute branch retains its own FP32 conversions and passes original inputs
to Finalize. A5 legacy keeps its existing dt_bias restrictions and implementation.
