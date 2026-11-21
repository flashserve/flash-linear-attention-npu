"""示例文件（形态 B/C）：torch_custom/fla_npu/fla_npu/ops/ascendc/_stable.py

交付布局依据 torch_custom/fla_npu/README.md §1.1/§1.2：真签名 wrapper（不要 `*args` / `**kwargs`）、
字符串用 `_char_code`、host 数组用 `_host_ints`、stream 用 `_current_stream_ptr()`（每次现取）。

注意事项：
  1. **场景选择不在 Python 层做**：V1/V2 是同一个 schema 的两条 aclnn 入口，由 C++ 适配层
     `run_<op>` 内部选择（见 `stable_op_name.cpp`）；Python wrapper 只做拆栈、转换、默认值补齐与透传。
  2. wrapper 不做参数、shape、dtype、layout、属性、输出组合或支持范围拦截；即使新开关或场景不支持，
     也不得直接抛错，必须让所选 aclnn L2/tiling 返回结构化错误。
  3. 新增参数追加在末尾、带默认值；默认值等于历史语义。V2 的 `tail_scale` 也必须沿同一
     schema 透传，不能因回落 V1 而丢失。
  4. 组合算子（形态 C）单独一个真签名 wrapper，名字与 schema 一致，不做场景回落
     （范围不满足时由 L2 返回 `ACLNN_ERR_PARAM_INVALID`）。
  5. 本示例只展示本次新增/修改的函数；实际交付时改仓库中的 `_stable.py`。
"""


# ---------------------------------------------------------------------------
# 形态 B：在已有 npu_op_name 的 wrapper 末尾追加参数（显式参数，不用 **kwargs）
# ---------------------------------------------------------------------------
def npu_op_name(x, g, *, a_log=None, initial_state=None, cu_seqlens=None,
                     chunk_indices=None, layout="BSND", scale=1.0, chunk_size=64,
                     epsilon=1e-6, return_saved=False, tail_mode=False, tail_scale=1.0):
    """V1/V2 共用入口；V2 相关的开关默认值即 V1 的历史语义。"""

    tail_mode_value = bool(tail_mode)

    return _op("npu_op_name")(
        x, g, a_log, initial_state,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        _char_code("npu_op_name", "layout", layout),
        float(scale), int(chunk_size), float(epsilon), bool(return_saved),
        tail_mode_value, float(tail_scale), _current_stream_ptr(),
    )


# ---------------------------------------------------------------------------
# 形态 C：组合算子（只有 L2、用 L0 拼接）的 wrapper
# ---------------------------------------------------------------------------
def npu_op_name_fused(x, g, *, a_log=None, initial_state=None, cu_seqlens=None,
                           chunk_indices=None, layout="BSND", scale=1.0,
                           chunk_size=64, epsilon=1e-6, return_saved=False,
                           with_tail=False):
    """在同一个 executor 内组合 OpName 与 OpNameTail。"""

    return _op("npu_op_name_fused")(
        x, g, a_log, initial_state,
        _host_ints(cu_seqlens), _host_ints(chunk_indices),
        _char_code("npu_op_name_fused", "layout", layout),
        float(scale), int(chunk_size), float(epsilon), bool(return_saved),
        bool(with_tail), _current_stream_ptr(),
    )
