"""chunk_kda_fwd ATK 执行器 + 预计算金标加载钩子（KDA_GOLDEN_PRECOMPUTED_DIR）。

用途：T16384 export 例（case 297）的 fp64 金标在本容器 32GiB cgroup 下无法
在 ATK worker 内完成（峰值 ~32GiB 被 SIGKILL）。金标是确定性函数（同 seed
同实现），可用 /workspace/scripts/gen_kda_golden_stream.py 以流式按 head 方式
在内存受限下预计算——该脚本已在 case 287（T8192 export，含非整 chunk、
gk/h 视图保存）上与 ATK 自算金标逐输出 max|diff|=0 验证一致。

本文件不改变任何 DUT/比较逻辑：仅当环境变量 KDA_GOLDEN_PRECOMPUTED_DIR
存在时，用"加载预计算金标"替代 CPU 金标分支的重算（省 19GiB fp64 输出 +
中间量内存）；NPU 分支与 ATK 混合容差比较全部走原生链路。若 spec 要求的
可见输出超出预计算覆盖（如 final_state），直接抛错而不是静默出错值。
"""
import importlib.util
import os
import sys

_CANDIDATES = []
try:  # ATK 若按路径加载本文件
    _CANDIDATES.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "executor_chunk_kda_fwd.py"))
except NameError:
    pass
_CANDIDATES.append("/workspace/flash-linear-attention-npu/tests/atk/"
                   "chunk_kda_fwd/executor_chunk_kda_fwd.py")
_CANDIDATES.append(os.path.join(os.getcwd(), "executor_chunk_kda_fwd.py"))
_ORIG_PATH = next(p for p in _CANDIDATES if os.path.exists(p))
_spec = importlib.util.spec_from_file_location("executor_chunk_kda_fwd_orig",
                                               _ORIG_PATH)
orig = importlib.util.module_from_spec(_spec)
sys.modules["executor_chunk_kda_fwd_orig"] = orig
_spec.loader.exec_module(orig)

# 预计算文件按可见元组顺序落盘：o, gk, aqk, akk, w, u, qg, kg, v_new, h
_PRECOMP_NAMES = ("attn_out", "gk", "Aqk", "Akk", "w", "u", "qg", "kg",
                  "v_new", "h")


def _precomputed_golden(inputs, spec):
    import torch

    cache_dir = os.environ["KDA_GOLDEN_PRECOMPUTED_DIR"]
    tensors = []
    for idx, name in enumerate(_PRECOMP_NAMES):
        path = os.path.join(cache_dir, f"output_{idx}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"KDA_GOLDEN_PRECOMPUTED_DIR={cache_dir} 缺 {name}: {path}")
        tensors.append(torch.load(path, map_location="cpu", weights_only=False))
        print(f"[precomputed-golden] loaded {name} <- output_{idx}.pt "
              f"{tuple(tensors[-1].shape)}", flush=True)

    # 还原 full 12 元组（预计算目录即按 full 导出口径落盘，final_state/
    # initial_state 未预计算——297 例均为 None）
    full = [
        tensors[0],   # attn_out
        None,         # final_state（297 例 output_final_state=False）
        tensors[1],   # gk
        tensors[2],   # Aqk
        tensors[3],   # Akk
        tensors[4],   # w
        tensors[5],   # u
        tensors[6],   # qg
        tensors[7],   # kg
        tensors[8],   # v_new
        tensors[9],   # h
        None,         # initial_state_out
    ]
    applied = orig._apply_output_policy(tuple(full), spec)
    for want, got in zip(applied, full):
        if want is None and got is not None:
            continue  # 策略隐藏该输出，允许
        if want is not None and got is None:
            raise RuntimeError(
                "spec 要求的输出不在预计算金标内（如 final_state）；"
                "请用 gen_kda_golden_stream.py 按该 spec 重新生成")
    return applied


if os.environ.get("KDA_GOLDEN_PRECOMPUTED_DIR"):
    orig._torch_fp64_golden = _precomputed_golden

# 让 -p 加载的本模块对外呈现原执行器的一切注册与符号
__all__ = [n for n in dir(orig) if not n.startswith("_")]
globals().update({n: getattr(orig, n) for n in __all__})
