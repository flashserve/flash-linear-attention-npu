"""示例文件：torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py

交付布局依据 torch_custom/fla_npu/README.md §1.1/§1.2：
  - 真签名 wrapper，**不要 `*args` / `**kwargs`**；位置参数顺序与 schema 形参一致；
  - 字符串参数用 `_char_code`、host 数组用 `_host_ints`、stream 用 `_current_stream_ptr()`；
  - stream 每次现取，不允许缓存成进程级变量（vLLM 是多线程多 stream）；
  - schema 无法表达 `None` 的默认值在 wrapper 里补齐（参考 npu_kda_gate_cumsum 的 lower_bound）。

注意事项：
  1. 函数名 = schema 里的算子名（`npu_<op>`）；短名由 `__init__.py` 的 `_strip_npu_prefix()` 统一导出，
     不需要在这里额外写一份短名函数。
  2. 参数校验只放"schema 表达不了、且两条后端要保持一致"的判据（例如正数校验、档位互斥）；
     张量契约（shape/dtype/layout）仍由 aclnn L2 校验，避免两处规则漂移。
  3. 新增关键字参数只能追加在**末尾**并带默认值；默认值等于本算子的历史行为。
  4. 本示例只展示本次新增的函数；实际交付时把函数追加到仓库的 `_stable.py` 里。
"""


def npu_example_scan(x, g, *, a_log=None, initial_state=None, cu_seqlens=None,
                     chunk_indices=None, layout="BSND", scale=1.0, chunk_size=64,
                     epsilon=1e-6, return_saved=False):
    """Chunk 内扫描 + 可选保存中间量。

    ``return_saved=True`` 时额外导出 ``state`` 与 ``x_norm``；两者要么同时要，要么同时不要
    （传与不传只影响是否落公开 GM，计算结果逐位一致）。
    """

    epsilon_value = 1e-6 if epsilon is None else float(epsilon)
    if not epsilon_value > 0.0:
        raise RuntimeError("npu_example_scan: epsilon must be a positive finite number.")
    scale_value = 1.0 if scale is None else float(scale)
    if not scale_value > 0.0:
        raise RuntimeError("npu_example_scan: scale must be a positive finite number.")

    return _op("npu_example_scan")(
        x,
        g,
        a_log,
        initial_state,
        _host_ints(cu_seqlens),
        _host_ints(chunk_indices),
        _char_code("npu_example_scan", "layout", layout),
        scale_value,
        int(chunk_size),
        epsilon_value,
        bool(return_saved),
        _current_stream_ptr(),
    )
