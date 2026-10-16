#!/usr/bin/env python3
"""Stable-ABI 适配器生成器（**脚手架 / 评估用**，非生产）。

目标：把 `csrc/src/stable_<op>.cpp` 里**机械可推导**的部分自动生成，并对着现有的手写实现做
**分段 diff 验收**——用数据回答"到底有多少可以自动化、剩下的必须人工在哪"。

真相来源（全部取自仓库现有文件，不硬编码算子知识）：
  1. `fla_npu/ops/ascendc/_aclnn_ctypes.py::npu_<op>`  → aclnn 调用名 + **参数编组序列**（含输出）
  2. `npu_custom.yaml` 的 `func:` 行                    → torch 侧 schema
  3. 任意一个现有适配器（默认取行数最多的那个）          → 样板目录（includes/using 清单）
  4. 目标算子现有手写适配器（若存在）                    → **验收基线**（分段比较）

用法::

    python gen_stable_adapter.py --op chunk_fwd_o --root <repo>/torch_custom/fla_npu \
        [--out gen.cpp] [--report report.md]

分段口径（与设计文档 §7.6.1 一致）：
  S1 头注释  S2 includes/using 样板  S3 layout 名表  S4 schema 字符串
  S5 run_ 签名  S6 FLA_STABLE_EXEC 编组  S7 输出分配（**不生成**，标记为人工）
"""

from __future__ import annotations

import argparse
import difflib
import pathlib
import re
import sys


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def find_ctypes_wrapper(text: str, op: str) -> tuple[str, list[str]]:
    """从 _aclnn_ctypes.py 取出某算子的 aclnn 名与参数编组序列。"""

    match = re.search(rf"\ndef {op}\((.*?)\n(?=def |\Z)", text, re.S)
    if not match:
        raise SystemExit(f"未找到 ctypes wrapper: {op}")
    body = match.group(0)
    call = re.search(r'_call_aclnn\(\s*"([a-zA-Z0-9_]+)"\s*,\s*lambda ctx:\s*\[(.*?)\]\s*,', body, re.S)
    if not call:
        raise SystemExit(f"{op}: 未找到 _call_aclnn 调用")
    aclnn_name = call.group(1)
    raw_items = [item.strip() for item in re.split(r",(?![^()]*\))", call.group(2)) if item.strip()]
    return aclnn_name, raw_items


def map_item(item: str) -> str:
    """把 ctypes 编组项映射成适配器里的编组表达式。"""

    if m := re.match(r'ctx\.tensor\(\s*(\w+)\s*,\s*"(\w+)"\s*\)', item):
        var, alias = m.group(1), m.group(2)
        return f"out_tensor(meta_of({alias}))" if var == "out" else f"tensor(meta_of({var}))"
    if m := re.match(r"ctx\.int_array\(\s*(\w+)\s*\)", item):
        return f"int_array({m.group(1)})"
    if re.match(r"ctypes\.c_(double|int64|bool|int32)\(", item):
        return None  # 标量：名字要从 schema 里取，见 build_scalar_sequence
    if "c_char_p" in item or "layout_buffer" in item:
        return "CStrArg(layout)"
    return f"/* UNMAPPED: {item} */"


def yaml_func_line(yaml_text: str, op: str) -> str:
    """从 npu_custom.yaml 撷取 `- func: <op>(...) -> ...` 行（可能跨行）。"""

    match = re.search(rf"-\s*func:\s*{op}\((.*?)\n\s*op_api:", yaml_text, re.S)
    if not match:
        raise SystemExit(f"npu_custom.yaml 未找到 {op}")
    return f"{op}({match.group(1).strip()}"


def split_params(func_line: str) -> tuple[list[tuple[str, str]], str]:
    """把 `op(Tensor q, ..., *, int k=1) -> (Tensor, Tensor?)` 拆成 params 与返回类型。"""

    head, _, ret = func_line.partition("->")
    inner = head[head.index("(") + 1 : head.rindex(")")]
    params: list[tuple[str, str]] = []
    for chunk in re.split(r",(?![^()]*\))", inner):
        chunk = chunk.strip()
        if not chunk or chunk == "*":
            continue
        chunk = chunk.split("=")[0].strip()  # 丢掉默认值
        ptype, _, pname = chunk.rpartition(" ")
        params.append((ptype.strip(), pname.strip()))
    return params, ret.strip()


CPP_TYPES = {
    "Tensor": "Tensor",
    "Tensor?": "std::optional<Tensor>",
    "float": "double",
    "int": "int64_t",
    "bool": "bool",
    "str": "int64_t",  # 适配器把 str 归一化成 layout 枚举码
    "int[]?": "std::optional<Tensor>",
}


def collect_template(root: pathlib.Path) -> tuple[list[str], list[str], str]:
    """从现有适配器里收集样板：using 清单、includes、layout 名表写法。"""

    sources = sorted((root / "csrc" / "src").glob("stable_*.cpp"), key=lambda p: -len(p.read_text().splitlines()))
    using: list[str] = []
    includes: list[str] = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("#include \"stable/"):
                if line not in includes:
                    includes.append(line)
            if m := re.match(r"using fla_npu_stable::stable::(\w+);", line):
                name = m.group(1)
                if f"::{name};" not in "".join(f"::{u};\n" for u in using):
                    using.append(name)
        if using and len(includes) >= 4:
            break
    return includes, sorted(set(using)), sources[-1].name if sources else ""


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def generate(op: str, root: pathlib.Path) -> tuple[str, dict]:
    ctypes_text = (root / "fla_npu" / "ops" / "ascendc" / "_aclnn_ctypes.py").read_text(encoding="utf-8")
    yaml_text = (root / "npu_custom.yaml").read_text(encoding="utf-8")
    aclnn_name, items = find_ctypes_wrapper(ctypes_text, op)
    func_line = yaml_func_line(yaml_text, op)
    params, ret = split_params(func_line)
    includes, using_catalogue, template_from = collect_template(root)

    # schema：位置参数形式（丢掉 kwargs-only 的 `*`），尾部补 `int stream`
    positional = [f"{t} {n}" for t, n in params]
    schema = f"{op}({', '.join(positional)}, int stream) -> {ret}"

    # run_ 签名
    run_params = []
    for ptype, pname in params:
        cpp = CPP_TYPES.get(ptype, "/* TODO type */ int64_t")
        run_params.append((cpp, pname))
    sig_lines = []
    for index, (cpp, pname) in enumerate(run_params):
        sep = "," if index < len(run_params) - 1 else ", int64_t stream)"
        sig_lines.append(f"    {cpp} {pname}{sep}")
    if run_params:
        sig_lines[0] = f"Tensor run_{op}(" + sig_lines[0].strip()

    # 编组：标量按 schema 顺序回填
    scalar_names = [n for t, n in params if t in ("float", "int", "bool") and n not in ("stream",)]
    scalar_iter = iter(scalar_names)
    exec_items = []
    for item in items:
        mapped = map_item(item)
        if mapped is None:
            mapped = f"scalar({next(scalar_iter, '/*TODO*/')})"
        exec_items.append(mapped)

    used_using = sorted({name for name in using_catalogue
                         if re.search(rf"\b{name}\(", " ".join(exec_items) + " allocate_sizes ") or name in
                         ("Tensor",)})
    using_block = "\n".join(f"using fla_npu_stable::stable::{name};" for name in used_using)

    code = f"""// Stable-ABI adapter for {op}.   (GENERATED SKELETON -- review before use)
// aclnn: {aclnn_name}
//
// One operator per file: csrc/src/stable_ops.cpp #includes this file into
// the single translation unit and registers it there.  The per-operator
// contract (schema == run_ == FLA_STABLE_EXEC == aclnn order) is in
// docs/architecture/适配层接入指南.md.

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <cstring>
#include <optional>

namespace {{

{using_block}

// ---------------------------------------------------------------------------
// {op}
// ---------------------------------------------------------------------------

constexpr const char* kSchema_{op} =
    "{schema}";

{sig_lines[0] if sig_lines else f"Tensor run_{op}(int64_t stream)"}
  // ---- S7: 输出分配（**人工**：layout → shape 规则）------------------------
  // 参考实现：_aclnn_ctypes.py::{op} 里的 out_shape 计算
  Tensor out;  // TODO(human): allocate_sizes(...) with dtype = meta_of(<last input>).scalar_type
  // -------------------------------------------------------------------------

  FLA_STABLE_EXEC("{aclnn_name}", /*TODO stream-context*/, {", ".join(exec_items)});
  return out;
}}

}}  // namespace
"""
    meta = {
        "aclnn": aclnn_name,
        "schema": schema,
        "params": params,
        "return": ret,
        "exec_items": exec_items,
        "template_from": template_from,
        "includes": includes,
        "run_signature": sig_lines,
        "generated_lines": len(code.splitlines()),
    }
    return code, meta


# ---------------------------------------------------------------------------
# 分段 diff 验收
# ---------------------------------------------------------------------------
def sections(text: str) -> dict[str, list[str]]:
    """把手写适配器按语义分段（用于与服务端生成物对照）。"""

    lines = text.splitlines()
    out: dict[str, list[str]] = {"S1_header": [], "S2_boilerplate": [], "S3_layout_table": [],
                                 "S4_schema": [], "S5_run_signature": [], "S6_exec": [],
                                 "S7_output_alloc": [], "S8_registration": []}
    state = "S1_header"
    for line in lines:
        if line.startswith("#include") or line.startswith("using ") or line.startswith("namespace"):
            state = "S2_boilerplate"
        if "enum_name(" in line or "Names[]" in line:
            state = "S3_layout_table"
        if line.startswith("constexpr const char* kSchema"):
            state = "S4_schema"
        if re.match(r"Tensor run_", line):
            state = "S5_run_signature"
        if "FLA_STABLE_EXEC" in line:
            state = "S6_exec"
        if "allocate_sizes(" in line or "allocate_like(" in line:
            state = "S7_output_alloc"
        out[state].append(line.rstrip())
    return out


def normalise(lines: list[str]) -> list[str]:
    cleaned = []
    for line in lines:
        stripped = re.sub(r"//.*$", "", line).strip()
        if stripped:
            cleaned.append(re.sub(r"\s+", " ", stripped))
    return cleaned


def diff_report(op: str, generated: str, baseline_path: pathlib.Path | None) -> str:
    report = [f"# 适配器生成器 diff 验收 — `{op}`", ""]
    gen_secs = sections(generated)
    if baseline_path is None or not baseline_path.exists():
        report.append("手写基线不存在——只列出生成物分段行数（无法做 diff 验收）")
        for name, lines in gen_secs.items():
            report.append(f"- {name}: {len(lines)} 行")
        return "\n".join(report)

    base_secs = sections(baseline_path.read_text(encoding="utf-8"))
    report.append(f"手写基线：`{baseline_path.name}`（{len(baseline_path.read_text(encoding='utf-8').splitlines())} 行）")
    report.append("")
    report.append("| 分段 | 生成 | 手写 | 归一化后一致 | 说明 |")
    report.append("|---|---|---|---|---|")
    for name in ("S1_header", "S2_boilerplate", "S3_layout_table", "S4_schema",
                 "S5_run_signature", "S6_exec", "S7_output_alloc"):
        gen_n = normalise(gen_secs.get(name, []))
        base_n = normalise(base_secs.get(name, []))
        if not gen_n:
            verdict = "—（未生成，需人工）"
        elif not base_n:
            verdict = "—（基线无此段）"
        else:
            ratio = difflib.SequenceMatcher(None, gen_n, base_n).ratio()
            verdict = f"{ratio * 100:.0f}%"
        report.append(f"| {name} | {len(gen_n)} | {len(base_n)} | {verdict} | |")
    report.append("")
    report.append("## S6 编组逐项对照（最关键的一段）")
    report.append("")
    gen_exec = " ".join(normalise(gen_secs.get("S6_exec", [])))
    base_exec = " ".join(normalise(base_secs.get("S6_exec", [])))
    gen_args = re.findall(r"[a-zA-Z_]+\([^()]*\)", gen_exec)
    base_args = re.findall(r"[a-zA-Z_]+\([^()]*\)", base_exec)
    report.append(f"- 生成：{', '.join(gen_args)}")
    report.append(f"- 手写：{', '.join(base_args)}")
    report.append(f"- 逐项一致：{'**是**' if gen_args == base_args else '**否**'}")
    return "\n".join(report)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True)
    parser.add_argument("--root", type=pathlib.Path, required=True,
                        help="torch_custom/fla_npu 目录")
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args(argv[1:])

    code, meta = generate(args.op, args.root)
    baseline = args.root / "csrc" / "src" / f"stable_{args.op.replace('npu_', '')}.cpp"
    if not baseline.exists():
        baseline = args.root / "csrc" / "src" / f"stable_{args.op}.cpp"
    report = diff_report(args.op, code, baseline)

    if args.out:
        args.out.write_text(code, encoding="utf-8")
        print(f"generated {args.out}  ({meta['generated_lines']} lines)")
    else:
        print(code)
    print(report)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
