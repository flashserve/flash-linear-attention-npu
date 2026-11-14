"""RWKV8 public signature preserved; execution uses the experimental Stable dispatcher."""
from ._loader import validate_active_environment
from ._contract import validate_chunk_len

def _shape(tensor):
    return tuple(tensor.shape)

def _optional_bool(value, default):
    return default if value is None else bool(value)

def ensure_npu_tensor(tensor, name):
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "npu":
        raise RuntimeError(f"{name} must be an NPU tensor")

def fused_recurrent_rwkv8(
    q,
    w,
    k,
    v,
    z,
    b,
    *,
    scale=1.0,
    initial_state=None,
    reverse=False,
    output_chunk_state=False,
    output_sa=False,
    chunk_len=16,
):
    """RWKV-v8（WKV7）fused recurrent 前向，逐 token 递推。

    布局为 BHTC：q/w/k/z/b 是 (B,H,T,K)，v 是 (B,H,T,V)（K 与 V 独立，
    K==V 为特例）。与 fla 上游的 (B,T,H,C) 口径不同，对接时需
    transpose(1,2)——transpose 只改视图，建议随后 .contiguous()，否则
    aclnn 层会对每个非连续输入静默补一次 GM 拷贝。initial_state 朝向
    (B,H,K,V)（= 内核 Sᵀ 原样，与 s 快照一致），fp32，缺省零初态。

    返回 (o, s, sa)：o 为 (B,H,T,V)、dtype 跟随 q；s 为 chunk 快照
    (B,H,T//chunk_len,K,V) fp32（output_chunk_state=True 时产出，
    否则为 None）；sa 为每 token 更新前的 state@z，(B,H,T,V) fp32
    （output_sa=True 时产出，否则为 None）。
    """
    validate_active_environment()
    import torch

    allowed_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    tensors = {"q": q, "w": w, "k": k, "v": v, "z": z, "b": b}
    for name, tensor in tensors.items():
        ensure_npu_tensor(tensor, name)
    if initial_state is not None:
        ensure_npu_tensor(initial_state, "initial_state")
    shapes = {name: _shape(tensor) for name, tensor in tensors.items()}
    for name, tensor in tensors.items():
        if len(shapes[name]) != 4:
            raise RuntimeError(f"npu_fused_recurrent_rwkv8: {name} must be rank-4 (B,H,T,C).")
        if tensor.dtype not in allowed_dtypes:
            raise RuntimeError(f"npu_fused_recurrent_rwkv8: {name} dtype must be float16/bfloat16/float32.")
        if tensor.dtype != q.dtype:
            raise RuntimeError("npu_fused_recurrent_rwkv8: q/w/k/v/z/b must share the same dtype.")
    q_shape = shapes["q"]
    for name in ("w", "k", "z", "b"):
        if shapes[name] != q_shape:
            raise RuntimeError(f"npu_fused_recurrent_rwkv8: {name} shape must equal q shape (B,H,T,K).")
    batch, heads, seqlen, k_dim = q_shape
    v_shape = shapes["v"]
    if v_shape[:3] != (batch, heads, seqlen):
        raise RuntimeError("npu_fused_recurrent_rwkv8: v must be (B,H,T,V) with the same B/H/T as q.")
    v_dim = v_shape[3]
    if k_dim <= 0 or v_dim <= 0 or k_dim % 8 or k_dim > 128 or v_dim % 8 or v_dim > 128:
        raise RuntimeError("npu_fused_recurrent_rwkv8: K/V must be multiples of 8 and <= 128.")

    reverse = _optional_bool(reverse, False)
    output_chunk_state = _optional_bool(output_chunk_state, False)
    output_sa = _optional_bool(output_sa, False)
    chunk_len = validate_chunk_len(chunk_len)

    if initial_state is not None:
        if _shape(initial_state) != (batch, heads, k_dim, v_dim):
            raise RuntimeError("npu_fused_recurrent_rwkv8: initial_state must be (B,H,K,V).")
        if initial_state.dtype != torch.float32:
            raise RuntimeError("npu_fused_recurrent_rwkv8: initial_state must be float32.")

    from ._stable import invoke
    return invoke("npu_fused_recurrent_rwkv8", q,
                  (q, w, k, v, z, b, initial_state, float(scale), reverse,
                   output_chunk_state, output_sa, chunk_len))
