# fused_recurrent_rwkv8 (WKV7) 测试用例清单。
#
# 由旧 tests/op_cases/fused_recurrent_rwkv8.json（schema_version 1）迁移而来，
# 是 pta 精度脚本与 tests/atk 用例生成器的唯一 case 源。
#
# 上游语义锚点：
#   semantics:      BlinkDL/RWKV-LM @ 9521024
#                   RWKV-v8/cuda/wkv7_cuda.cu forward_kernel (lines 10-52)
#                   decay = exp(-exp(w))；z = -kk，b = kk * a_inctx
#   precision_peer: fla-org/flash-linear-attention @ a4a2624b
#                   fla/ops/generalized_delta_rule/dplr（state 朝向互为转置）
#
# 每条 case 字段：
#   id / tags / seed
#   B, H, T, K, V            形状（io 布局 BHTC；K、V 独立，K==V 是特例）
#   dtype                    q/w/k/v/z/b 的 dtype（state 恒为 fp32）
#   scale                    q 读出缩放
#   chunk_len                s 快照间隔（默认 16 对齐官方 backward 重建粒度）
#   initial_state            是否给非零初态（fp32 (B,H,K,V)）
#   output_chunk_state/sa    是否产出 s / sa
#   compare_outputs          需要对拍的输出
#   negative                 None 为正例；否则为错误信息关键词（应抛异常）

CASES = [
    {
        "id": "main_accuracy",
        "tags": ["accuracy", "example", "route"],
        "B": 2, "H": 4, "T": 64, "K": 64, "V": 64,
        "dtype": "float32",
        "scale": 1.0, "chunk_len": 16,
        "initial_state": False,
        "output_chunk_state": False, "output_sa": False,
        "compare_outputs": ["o"],
        "negative": None,
        "seed": 42,
    },
    {
        "id": "init_and_scale",
        "tags": ["accuracy", "generalization"],
        "B": 1, "H": 2, "T": 16, "K": 64, "V": 64,
        "dtype": "float32",
        "scale": 0.125, "chunk_len": 16,
        "initial_state": True,
        "output_chunk_state": False, "output_sa": False,
        "compare_outputs": ["o"],
        "negative": None,
        "seed": 43,
    },
    {
        "id": "t33_tail",
        "tags": ["accuracy", "boundary", "tail"],
        "B": 2, "H": 4, "T": 33, "K": 64, "V": 64,
        "dtype": "float32",
        "scale": 1.0, "chunk_len": 16,
        "initial_state": True,
        "output_chunk_state": False, "output_sa": False,
        "compare_outputs": ["o"],
        "negative": None,
        "seed": 44,
        "note": "T 非 16 倍数（区别于官方 CUDA kernel 的 T%16 约束）",
    },
    {
        "id": "decode_t1",
        "tags": ["accuracy", "boundary"],
        "B": 1, "H": 1, "T": 1, "K": 64, "V": 64,
        "dtype": "float32",
        "scale": 1.0, "chunk_len": 16,
        "initial_state": False,
        "output_chunk_state": False, "output_sa": False,
        "compare_outputs": ["o"],
        "negative": None,
        "seed": 45,
    },
    {
        "id": "long_seq_performance",
        "tags": ["generalization", "performance"],
        "B": 1, "H": 2, "T": 2048, "K": 64, "V": 64,
        "dtype": "float32",
        "scale": 1.0, "chunk_len": 16,
        "initial_state": False,
        "output_chunk_state": False, "output_sa": False,
        "compare_outputs": ["o"],
        "negative": None,
        "seed": 46,
    },
    {
        "id": "multi_core_parallel",
        "tags": ["accuracy", "generalization"],
        "B": 4, "H": 8, "T": 128, "K": 64, "V": 64,
        "dtype": "float32",
        "scale": 1.0, "chunk_len": 16,
        "initial_state": False,
        "output_chunk_state": False, "output_sa": False,
        "compare_outputs": ["o"],
        "negative": None,
        "seed": 47,
    },
    {
        "id": "chunk_len_8",
        "tags": ["accuracy", "generalization", "chunk_len"],
        "B": 2, "H": 4, "T": 64, "K": 64, "V": 64,
        "dtype": "float32",
        "scale": 1.0, "chunk_len": 8,
        "initial_state": True,
        "output_chunk_state": True, "output_sa": True,
        "compare_outputs": ["o", "s", "sa"],
        "negative": None,
        "seed": 51,
        "note": "非默认 chunk_len=8：s 共 8 个快照；与官方 backward 不兼容，仅前向语义",
    },
    {
        "id": "zero_chunk_state",
        "tags": ["accuracy", "generalization", "boundary", "chunk_len"],
        "B": 1, "H": 2, "T": 8, "K": 64, "V": 64,
        "dtype": "float32",
        "scale": 1.0, "chunk_len": 16,
        "initial_state": True,
        "output_chunk_state": True, "output_sa": True,
        "compare_outputs": ["o", "s", "sa"],
        "negative": None,
        "seed": 52,
        "note": "T=8 < chunk_len=16 → NT=0：s 为零尺寸 (B,H,0,K,V)，"
                "storage 层 1 行占位，锁定 ctypes 通路的零快照路径",
    },
    {
        "id": "negative_shape_mismatch",
        "tags": ["negative"],
        "B": 1, "H": 1, "T": 8, "K": 64, "V": 64,
        "dtype": "float32",
        "scale": 1.0, "chunk_len": 16,
        "initial_state": False,
        "output_chunk_state": False, "output_sa": False,
        "compare_outputs": [],
        "negative": "shape",
        "seed": 48,
        "note": "六个输入 shape 不一致时 wrapper/golden 应拒绝",
    },
    {
        "id": "negative_bad_init_shape",
        "tags": ["negative"],
        "B": 1, "H": 1, "T": 8, "K": 64, "V": 64,
        "dtype": "float32",
        "scale": 1.0, "chunk_len": 16,
        "initial_state": True,
        "output_chunk_state": False, "output_sa": False,
        "compare_outputs": [],
        "negative": "initial_state",
        "seed": 49,
        "note": "initial_state 非 (B,H,K,V) 时应拒绝",
    },
    {
        "id": "negative_chunk_len_zero",
        "tags": ["negative", "chunk_len"],
        "B": 2, "H": 4, "T": 64, "K": 64, "V": 64,
        "dtype": "float32",
        "scale": 1.0, "chunk_len": 0,
        "initial_state": False,
        "output_chunk_state": False, "output_sa": False,
        "compare_outputs": [],
        "negative": "chunk",
        "seed": 52,
        "note": "chunk_len < 1 时 wrapper/aclnn 应拒绝",
    },
]

# fp32 对拍阈值：与旧 manifest tolerance 一致（golden vs 官方 GPU fixture
# 实测 rel-RMSE ~1e-7，项目口径 2e-3 留了充足余量）。
REL_RMSE_THRESHOLD = {"float32": 2e-3, "float16": 1e-2, "bfloat16": 1e-2}
