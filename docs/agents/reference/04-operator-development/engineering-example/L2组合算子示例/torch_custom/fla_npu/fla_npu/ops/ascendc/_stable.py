"""示例文件（形态 B/C）：torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py

交付布局依据 torch_custom/fla_npu/README.md §1.1/§1.2：真签名 wrapper（不要 `*args` / `**kwargs`）、
字符串用 `_char_code`、host 数组用 `_host_ints`、stream 用 `_current_stream_ptr()`（每次现取）。

注意事项：
  1. **场景选择不在 Python 层做**：V1/V2 是同一个 schema 的两条 aclnn 入口，由 C++ 适配层
     `run_<op>` 内部选择（见 `stable_example_scan.cpp`）；Python wrapper 只做参数校验与透传。
  2. wrapper 只负责"schema 表达不了、且要在两条后端上保持一致"的判据：
     新开关被显式打开但当前场景不支持时，直接抛错并给出场景要求，不要静默回落。
  3. 新增参数追加在末尾、带默认值；默认值等于历史语义。
  4. 组合算子（形态 C）单独一个真签名 wrapper，名字与 schema 一致，不做场景回落
     （范围不满足时由 L2 返回 `ACLNN_ERR_PARAM_INVALID`）。
  5. 本示例只展示本次新增/修改的函数；实际交付时改仓库中的 `_stable.py`。
"""


# ---------------------------------------------------------------------------
# 形态 B：在已有 npu_example_scan 的 wrapper 末尾追加参数与校验（显式参数，不用 **kwargs）
# ---------------------------------------------------------------------------
def npu_example_scan(x, g, *, a_log=None, initial_state=None, cu_seqlens=None,
                     chunk_indices=None, layout="BSND", scale=1.0, chunk_size=64,
                     epsilon=1e-6, return_saved=False, tail_mode=False):
    """V1/V2 共用入口；V2 相关的开关默认值即 V1 的历史语义。"""

    import torch

    tail_mode_value = bool(tail_mode)
    if tail_mode_value:
        # 显式打开的新开关不在 V1 支持范围内：不满足 V2 场景时直接拒绝，不静默回落。
        if (x.dtype != torch.bfloat16 or int(chunk_size) != 64
                or str(layout) not in ("BSND", "TND")):
            raise RuntimeError(
                "npu_example_scan: tail_mode requires the V2 scenario "
                "(bfloat16 x, chunk_size=64, layout BSND/TND).")

    return _op("npu_example_scan")(
        x, g, a_log, initial_state,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        _char_code("npu_example_scan", "layout", layout),
        float(scale), int(chunk_size), float(epsilon), bool(return_saved),
        tail_mode_value, _current_stream_ptr(),
    )


# ---------------------------------------------------------------------------
# 形态 C：组合算子（只有 L2、用 L0 拼接）的 wrapper
# ---------------------------------------------------------------------------
def npu_example_scan_fused(x, g, *, a_log=None, initial_state=None, cu_seqlens=None,
                           chunk_indices=None, layout="BSND", scale=1.0,
                           chunk_size=64, epsilon=1e-6, return_saved=False,
                           with_tail=False):
    """在同一个 executor 内组合 ExampleScan 与 ExampleScanTail。"""

    return _op("npu_example_scan_fused")(
        x, g, a_log, initial_state,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        _char_code("npu_example_scan_fused", "layout", layout),
        float(scale), int(chunk_size), float(epsilon), bool(return_saved),
        bool(with_tail), _current_stream_ptr(),
    )
