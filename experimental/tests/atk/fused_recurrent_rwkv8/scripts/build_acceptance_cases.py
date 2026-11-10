#!/usr/bin/env python3
"""按 tests/atk/README.md「正式验收用例包」构造 _perf.json 与 _mss.json。

三份正式验收 JSON 来源必须不同，不能互相替代：

- ``atk_<op>.json``：由 ``gen_cases``（``gen_fused_recurrent_rwkv8.py``）产出后原样使用。
- ``atk_<op>_perf.json``：来源为「用户在算子开发开始时提供的模型 case」，这里取
  转测文档第 7.1~7.6 节列出的代表 shape（同一批形态用于跨轮性能对比）。
- ``atk_<op>_mss.json``：人工依据设计/实现中的全部可达 TilingKey 构造。本算子
  ``tilingKey`` 恒为 0（``fused_recurrent_rwkv8_tiling.cpp`` 的 ``DoLibApiTiling``），
  因此覆盖点落在该 key 之下与内存/同步/复用有关的 TilingData 字段与关键路径：

  * 8 种 flags 组合（initial_state / output_chunk_state / output_sa）驱动不同的
    GM 读写分支与 s 快照槽位；
  * ``chunkLen`` 字段（8 / 16）与多 chunk 尾块，决定 s 快照写入次数与偏移；
  * V 路径：``ColBroadcast`` 的 V==8 / V%64==0 / 其余三条分支，以及 V=128 时
    ``OuterMulAdd`` / ``StridedMul`` / ``StridedAdd`` 的 64 列组拆分；
  * K 路径：``Kp_``（≥K 的最小 2 幂）行 pad，K=96/120 触发真实 pad 分支；
  * dtype：Cast 路径（fp32 / fp16 / bf16）；
  * T < chunk_len：``sSlots_`` 归 0，s 输出形状退化为 [B,H,0,K,V]；
  * K>V / K<V 极端：K 侧与 V 侧 GM 跨度不对称。

用法（在算子目录下）：

    python3 scripts/build_acceptance_cases.py

用例骨架取自同目录的 ``atk_<op>.json``（保证 ``standard`` / ``api_type`` 等与 yaml
一致），仅改写 ``id`` / ``name`` / ``default_seed`` / ``inputs`` 中的逐 case 字段。
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy

HERE = os.path.dirname(os.path.abspath(__file__))
OP_DIR = os.path.dirname(HERE)
sys.path.insert(0, OP_DIR)

import gen_fused_recurrent_rwkv8 as gen  # noqa: E402

OP = gen.OP_NAME
# 逐 case 属性：顺序与 yaml inputs 中 case_spec 之后的 attr 顺序一致
ATTRS = ("dtype", "B", "H", "T", "K", "V", "scale", "chunk_len",
         "initial_state", "output_chunk_state", "output_sa", "case_id", "seed", "soc", "route")

PERF_SEED_BASE = gen.SEED_BASE + 10000
MSS_SEED_BASE = gen.SEED_BASE + 20000


def _spec(case, index, seed_base):
    spec = {
        "name": f"{case['tag']}_{index:04d}",
        "dtype": case["dtype"],
        "B": case["B"], "H": case["H"], "T": case["T"],
        "K": case["K"], "V": case["V"],
        "scale": case["scale"], "chunk_len": case["chunk_len"],
        "initial_state": case["initial_state"],
        "output_chunk_state": case["output_chunk_state"],
        "output_sa": case["output_sa"],
        "seed": seed_base + index,
    }
    spec.update({"op": OP, "case_id": index, "route": "ascendc", "soc": "ascend910b"})
    return spec


def perf_cases():
    """用户模型 case：转测文档 7.1~7.6 的代表 shape。"""
    return gen._template_cases()


def mss_cases():
    """全部可达 TilingKey（本算子仅 key=0）下的内存/同步/复用关键路径代表用例。"""
    c = gen._c
    out = []
    for f in range(8):
        out.append(c("fp16", 1, 2, 64, 64, 64, init=bool(f & 1), s=bool(f & 2), sa=bool(f & 4),
                     tag=f"mss_flag{f}"))
    out.append(c("fp16", 1, 2, 128, 64, 64, chunk_len=8, init=True, s=True, sa=True,
                 tag="mss_chunk8_multi"))
    out.append(c("fp16", 1, 2, 128, 64, 64, chunk_len=16, init=True, s=True, sa=True,
                 tag="mss_chunk16_multi"))
    for V in (8, 32, 64, 96, 128):
        out.append(c("fp16", 1, 2, 64, 64, V, init=True, s=True, sa=True, tag=f"mss_v{V}"))
    for K in (8, 64, 96, 120, 128):
        out.append(c("fp16", 1, 2, 64, K, 64, init=True, s=True, sa=True, tag=f"mss_k{K}"))
    out.append(c("fp32", 1, 2, 64, 128, 128, init=True, s=True, sa=True, tag="mss_fp32_maxkv"))
    out.append(c("bf16", 1, 2, 64, 128, 128, init=True, s=True, sa=True, tag="mss_bf16_maxkv"))
    out.append(c("fp16", 1, 2, 8, 64, 64, init=True, s=True, sa=True, tag="mss_t_lt_chunk"))
    out.append(c("fp16", 1, 2, 4, 64, 64, init=True, s=True, sa=True, tag="mss_t_min"))
    out.append(c("fp16", 1, 2, 64, 128, 8, init=True, s=True, sa=True, tag="mss_k128_v8"))
    out.append(c("fp32", 1, 2, 64, 8, 128, init=True, s=True, sa=True, tag="mss_k8_v128"))
    return out


def _build(template, cases, seed_base, out_path):
    built = []
    for index, case in enumerate(cases):
        spec = _spec(case, index, seed_base)
        item = deepcopy(template)
        item["id"] = index
        item["default_seed"] = spec["seed"]
        item["name"] = f"{OP}_{index:04d}_{spec['name']}"
        for entry in item["inputs"]:
            cfg = entry[0] if isinstance(entry, list) else entry
            if cfg["name"] == "low_precision_marker":
                cfg["dtype"] = gen._dtype(spec["dtype"])
            elif cfg["name"] == "case_spec":
                cfg["range_values"] = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
            elif cfg["name"] in ATTRS:
                cfg["range_values"] = spec[cfg["name"]]
        built.append(item)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(built, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
    return built


def main():
    accuracy_path = os.path.join(OP_DIR, f"atk_{OP}.json")
    with open(accuracy_path, encoding="utf-8") as handle:
        template = json.load(handle)[0]

    perf = _build(template, perf_cases(), PERF_SEED_BASE,
                  os.path.join(OP_DIR, f"atk_{OP}_perf.json"))
    mss = _build(template, mss_cases(), MSS_SEED_BASE,
                 os.path.join(OP_DIR, f"atk_{OP}_mss.json"))

    for label, items in (("perf", perf), ("mss", mss)):
        keys = [(i["id"], i["name"]) for i in items]
        assert len({k[0] for k in keys}) == len(items), f"{label}: id 重复"
        assert len({k[1] for k in keys}) == len(items), f"{label}: name 重复"
        print(f"{label}: {len(items)} cases -> atk_{OP}_{label}.json")


def _attr(case, name):
    for entry in case["inputs"]:
        cfg = entry[0] if isinstance(entry, list) else entry
        if cfg["name"] == name:
            return cfg["range_values"]
    raise KeyError(name)


if __name__ == "__main__":
    main()
