"""CANN 9.2.1 host tiling for the manually embedded projection kernels."""
import json
import os
from pathlib import Path
import struct
import sys


def build(frames, heads, width, length, soc):
    from tbe.common.platform import set_current_compile_soc_info
    from tbe.common.utils.op_tiling import do_op_tiling

    set_current_compile_soc_info(soc)
    root = Path(os.environ["ASCEND_HOME_PATH"]) / "opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_legacy"
    def get(op, path, shapes, adj_a, adj_b, dtype, keys):
        names = ("x1", "x2", "y")
        spec = [dict(name=name, shape=shape, ori_shape=shape, format="ND", ori_format="ND", dtype=dtype)
                for name, shape in zip(names, shapes)]
        trans = ("adj_x1", "adj_x2") if op == "BatchMatMulV2" else ("transpose_x1", "transpose_x2")
        flags = [(trans[0], adj_a, "bool"), (trans[1], adj_b, "bool"), ("offset_x", 0, "int"),
                 ("_op_impl_mode", "enable_float_32_execution", "str"), ("_op_impl_mode_enum", 0, "int")]
        info = json.loads((root / path).read_text())["compileInfo"]
        tile = do_op_tiling(op, info, spec[:2], spec[2:],
                           attrs=[dict(name=k, value=v, dtype=t) for k, v, t in flags])
        key = int(tile["tiling_key"])
        if key not in keys or tile["workspaces"]:
            return None
        raw = bytearray(tile["tiling_data"])
        # The precision mode affects scheduling as well as the hardware flag.
        if dtype == "float32" and struct.unpack_from("<i",raw,176 if op=="BatchMatMulV2" else 124)[0]:
            return None
        if len(raw) > 252:
            return None
        raw.extend(bytes(256-len(raw)))
        struct.pack_into("<i", raw, 252, key)
        return raw.hex()
    mm = "mat_mul/MatMulV2_ND_ND_FP32_FP32_false_true_all.json"
    bmm = "batch_matmul_v2/BatchMatMulV2_ND_ND_FP16_FP16_true_false_all.json"
    a = get("MatMulV2", mm, [[frames,width],[128,width],[frames,128]],False,True,"float32",{65665,65731,65745,66115,66177,66243,66257})
    b = get("MatMulV2", mm, [[frames,128],[heads*128,128],[frames,heads*128]],False,True,"float32",{65665,65731,65745,66115,66177,66243,66257})
    ta = get("BatchMatMulV2", bmm.replace("FP16","FP32"), [[heads,length,128]]*2+[[heads,128,128]],True,False,"float32",{65537,65603,65665,65745,66115,66177,66257}) if length else ""
    t = get("BatchMatMulV2",bmm,[[heads,length,128]]*2+[[heads,128,128]],True,False,"bfloat16",{98305,98385,98433,98817,98883,98897,98945}) if length else ""
    return [a,b,ta+t] if None not in (a,b,ta,t) else None


if __name__ == "__main__":
    print(json.dumps(build(*map(int, sys.argv[1:5]),sys.argv[5])))
