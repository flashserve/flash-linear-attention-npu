"""Inline selected CANN 9.2.1 MatMulV2 and BatchMatMulV2 implementations."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROJECTION = (65665, 65731, 65745, 66115, 66177, 66243, 66257)
GRAM_A = (65537, 65603, 65665, 65745, 66115, 66177, 66257)
GRAM_B = (98305, 98385, 98433, 98817, 98883, 98897, 98945)


def sources(tag, build):
    if tag == "Projection":
        folder, name, op, impl = "mat_mul", "MatMulV2_ND_ND_FP32_FP32_false_true_all", "MatMulV2", "mat_mul"
    else:
        dtype = "FP32" if tag == "TextA" else "FP16"
        folder, name, op, impl = "batch_matmul_v2", f"BatchMatMulV2_ND_ND_{dtype}_{dtype}_true_false_all", "BatchMatMulV2", "batch_matmul_v2"
    root = Path(os.environ["ASCEND_HOME_PATH"]) / "opp/built-in/op_impl/ai_core/tbe"
    spec = json.loads((root / "kernel/ascend910b/ops_legacy" / folder / (name+".json")).read_text())["supportInfo"]
    target = build/tag
    target.mkdir(parents=True,exist_ok=True)
    if (target/(tag+".o")).is_file() and all(any(target.rglob(f"*_{key}.cce")) for key in
            {"Projection":PROJECTION,"TextA":GRAM_A,"TextB":GRAM_B}[tag]):
        return target
    inputs = [dict(x,paramType="required") if x else dict(value=None) for x in spec["inputs"]]
    outputs = [dict(x,paramType="required") for x in spec["outputs"]]
    item = dict(bin_filename=tag,inputs=inputs,outputs=outputs,attrs=spec["attrs"])
    (target/"input.json").write_text(json.dumps(dict(op_type=op,op_list=[item])))
    subprocess.run(["opc",str(root/"impl/ops_legacy/dynamic"/(impl+".py")),
                    "--input_param="+str(target/"input.json"),"--main_func="+impl,"--output="+str(target),
                    "--debug_dir="+str(target),"--soc_version=Ascend910B3","--op_mode=dynamic",
                    "--op_debug_config=dump_cce","--log=error"],cwd=target,check=True)
    return target


def inline_family(tag, keys, src):
    dtype = "half" if tag=="TextB" else "float"
    lines = ["namespace "+tag+" {"]
    for key in keys:
        s=next(src.rglob(f"*_{key}.cce")).read_text()
        start=re.search(r'extern "C" __global__ __aicore__ void \w+\(',s)
        end=s.index("\n}",start.end())+2
        helper=s[s.index("#endif")+6:start.start()].strip().replace("static __aicore__", "inline __aicore__")
        lines.extend([f"namespace K{key} {{",helper,"inline __aicore__ void Run("+s[start.end():end],"}"])
    lines.extend(["inline __aicore__ void Run(GM_ADDR a,GM_ADDR b,GM_ADDR c,GM_ADDR tile) {",
                  "switch(*((__gm__ int32_t*)(tile+252))) {"])
    for key in keys:
        lines.append(f"case {key}: K{key}::Run((__gm__ {dtype}*)a,(__gm__ {dtype}*)b,(__gm__ {dtype}*)c,tile); break;")
    lines.extend(["}", "}", "}"])
    return "\n".join(lines)


if __name__=="__main__":
    build=Path(sys.argv[1]).resolve()
    out=Path(sys.argv[2]).resolve()
    parts=["#pragma once","#if defined(__DAV_CUBE__)"]
    for i,(tag,keys) in enumerate((("Projection",PROJECTION),("TextA",GRAM_A),("TextB",GRAM_B))):
        src=Path(sys.argv[i+3]) if len(sys.argv)==6 else sources(tag,build)
        parts.append(inline_family(tag,keys,src))
    out.write_text("\n".join(parts+["#endif",""]))
