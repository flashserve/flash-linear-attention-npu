"""示例追加片段（形态 B/C）：fla_npu/ops/ascendc/_stable.py

注意事项：
  1. 组合入口也要有真签名 wrapper；入参名、默认值、返回 tuple 必须与 schema 一致。
  2. 形态 B（同一算子的 V2 入口）的场景选择写在本算子已有的 wrapper 里：命中 V2 场景走 V2，
     否则回落 V1；调用方只看到一套 Python 签名，不需要知道底层调了哪个 aclnn。
  3. 回落的判断依据只能是文档化的支持范围（dtype/layout/chunk_size/连续性），并且"显式打开的新开关
     不在 V2 范围内"时必须直接报错，不能静默忽略开关后回落。
  4. 形态 C（只有 L2 的组合算子）单独一个 wrapper，名字与 schema 一致；它不做场景回落，
     范围不满足时由 L2 返回 ACLNN_ERR_PARAM_INVALID。
  5. stream 每次现取 `_current_stream_ptr()`，不要缓存；新增关键字参数只能追加在末尾并带默认值。
  6. 依赖算子缺配置时（过滤构建漏依赖）这里只会看到 561103/EZ1013，属于构建问题，
     不要在本层加 try/except 把它吞掉或降级成单算子调用。
"""


# ---------------------------------------------------------------------------
# 形态 C：组合算子的 wrapper（npu_example_scan_fused）
# ---------------------------------------------------------------------------
@_op("npu_example_scan_fused")
def npu_example_scan_fused(
    x,
    g,
    a_log=None,
    initial_state=None,
    cu_seqlens=None,
    chunk_indices=None,
    layout="BSND",
    scale=1.0,
    chunk_size=64,
    epsilon=1e-6,
    return_saved=False,
    with_tail=False,
    *,
    stream=None,
):
    return _op("npu_example_scan_fused").__call__(
        x,
        g,
        a_log,
        initial_state,
        _host_ints(cu_seqlens),
        _host_ints(chunk_indices),
        _char_code("npu_example_scan_fused", "layout", layout),
        float(scale),
        int(chunk_size),
        float(epsilon),
        bool(return_saved),
        bool(with_tail),
        _current_stream_ptr() if stream is None else stream,
    )


# ---------------------------------------------------------------------------
# 形态 B：在同一算子的 wrapper 里做 V2/V1 场景选择（示例：example_scan）
# ---------------------------------------------------------------------------
def _example_scan_uses_v2(x, layout, chunk_size, return_saved, tail_mode):
    """场景选择：只依据文档化的支持范围，不做"猜"。"""
    if x.dtype != torch.bfloat16:
        return False
    if layout not in ("BSND", "TND"):
        return False
    if int(chunk_size) != 64:
        return False
    if not x.is_contiguous():
        return False
    return True


def example_scan_with_v2_fallback(x, g, layout="BSND", chunk_size=64, tail_mode=False, **kwargs):
    if _example_scan_uses_v2(x, layout, chunk_size, kwargs.get("return_saved", False), tail_mode):
        return _op("npu_example_scan_v2").__call__(
            x, g, _char_code("npu_example_scan", "layout", layout), int(chunk_size),
            int(chunk_size), bool(tail_mode), _current_stream_ptr(), **_v2_extra_kwargs(**kwargs)
        )
    if tail_mode:
        # 显式打开的新开关不在回落路径的支持范围内：直接报错，不静默忽略。
        raise ValueError("tail_mode 仅支持 BF16 + BSND/TND + chunk_size=64 + 连续输入的场景。")
    return npu_example_scan(x, g, layout=layout, chunk_size=chunk_size, **kwargs)
