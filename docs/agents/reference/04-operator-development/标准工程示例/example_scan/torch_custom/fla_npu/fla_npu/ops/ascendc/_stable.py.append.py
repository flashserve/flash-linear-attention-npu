"""示例追加片段：torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py

注意事项：
  1. 本片段是**追加**到已有 `_stable.py` 的内容，不是新文件；不要为了加算子另建模块。
  2. 函数名 = schema 里的算子名（`npu_<op>`）；参数名、顺序、默认值与 schema 完全一致。
  3. 字符串参数（layout 等）在这里用 `_char_code("<op>", "<arg>", value)` 转成 code，
     名表顺序必须与 C++ 侧的 k<Op><Arg>Names 一致。
  4. int[] 参数用 `_host_ints(...)` 转 host int64 CPU tensor；device 元数据用 optional_tensor 语义。
  5. stream 每次现取：`_current_stream_ptr()`，不要缓存成进程级变量（多线程/多 stream 会串）。
  6. 新增关键字参数只能追加在末尾并带默认值，默认值等于本算子历史行为。
  7. 返回 tuple 的长度/顺序固定；未请求的可选输出返回 None。
"""


@_op("npu_example_scan")
def npu_example_scan(
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
    *,
    out=None,
    stream=None,
):
    # 入参名与默认值必须与 kSchema_example_scan 一致；return_saved 只在 Python 侧解释成
    # "要不要申请 state/x_norm"，aclnn 只认输出指针。
    return _op("npu_example_scan").__call__(
        x,
        g,
        a_log,
        initial_state,
        _host_ints(cu_seqlens),
        _host_ints(chunk_indices),
        _char_code("npu_example_scan", "layout", layout),
        float(scale),
        int(chunk_size),
        float(epsilon),
        bool(return_saved),
        _current_stream_ptr() if stream is None else stream,
    )
