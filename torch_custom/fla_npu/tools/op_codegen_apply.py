#!/usr/bin/env python3
"""Apply a spec end-to-end: cpp + pybind + _thin wrapper + whitelist."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # torch_custom/fla_npu
SPEC_DIR = ROOT / "op_specs"
CSRC = ROOT / "csrc_thin" / "src"
THIN = ROOT / "fla_npu" / "ops" / "ascendc"


_FN_DEF_RE = re.compile(
    r"(?m)^(?:at::Tensor|std::vector<at::Tensor>)\s+(\w+)\s*\(")


def has_fn_def(text: str, name: str) -> bool:
    """True when *text* already defines a thin adapter function named *name*.

    The aggregate file contains both single-output (``at::Tensor``) and
    multi-output (``std::vector<at::Tensor>``) adapters, so the presence check
    must accept either return type to stay idempotent.
    """

    return any(m.group(1) == name for m in _FN_DEF_RE.finditer(text))


def cpp_type(kind: str, name: str) -> str:
    if kind == "tensor":
        return f"const at::Tensor& {name}"
    if kind == "optional_tensor":
        return f"const c10::optional<at::Tensor>& {name}"
    if kind == "int_array":
        return f"const std::vector<int64_t>& {name}"
    if kind == "char_ptr":
        return f"const std::string& {name}"
    return {"int64": "int64_t", "bool": "bool", "double": "double",
            "float": "float"}[kind] + f" {name}"


def _arg_cpp_name(arg: dict) -> str:
    return arg.get("cpp", arg["name"])


def patch_pybind(spec: dict) -> None:
    name = spec["python_name"]
    args = [a for a in spec["args"] if a["kind"] != "out_tensor"]
    n_out = sum(1 for a in spec["args"] if a["kind"] == "out_tensor")
    path = CSRC / "pybind.cpp"
    text = path.read_text(encoding="utf-8")
    if n_out > 1 and "#include <vector>" not in text:
        text = "#include <vector>\n" + text
    if f"{name}(" in text:
        if n_out > 1 and "#include <vector>" not in text:
            path.write_text(text, encoding="utf-8")
        return
    params = ",\n    ".join(
        cpp_type(a["kind"], _arg_cpp_name(a)) for a in args)
    ret = "std::vector<at::Tensor>" if n_out > 1 else "at::Tensor"
    decl = (f"\n{ret} {name}(\n    {params},\n    uint64_t stream);\n"
            f"\n}}  // namespace fla_npu_thin")
    text = text.replace("}  // namespace fla_npu_thin", decl, 1)
    argnames = ",\n      ".join(
        f'py::arg("{a["name"]}")' for a in args)
    block = (f'  m.def(\n      "{name}",\n      &{name}, {argnames},\n'
             f'      py::arg("stream"));\n}}')
    text = text.rstrip()
    assert text.endswith("}")
    text = text[:-1] + block
    path.write_text(text, encoding="utf-8")


def patch_thin(spec: dict) -> None:
    name = spec["python_name"]
    n_out = sum(1 for a in spec["args"] if a["kind"] == "out_tensor")
    path = THIN / "_thin.py"
    text = path.read_text(encoding="utf-8")
    if f"def {name}(" in text:
        return
    py = spec.get("python", {})
    positional = py.get("positional", [])
    defaults = py.get("defaults", {})
    ignored = py.get("ignored", [])
    kw = [a["name"] for a in spec["args"]
          if a["name"] not in positional and a["kind"] != "out_tensor"]
    kw = [k for k in kw if k not in ignored]
    sig_kw = list(kw) + [k for k in ignored if k not in kw]
    sig = ", ".join(positional)
    if sig_kw:
        sig += ", *, " + ", ".join(
            f"{k}={defaults.get(k, 'None')}" for k in sig_kw)
    lines = [f"\n\ndef {name}({sig}):",
             "    ext = _extension()"]
    python_pre = py.get("pre")
    if python_pre:
        for raw_line in python_pre.splitlines():
            lines.append(("    " + raw_line) if raw_line.strip() else "")
    # Scalar kwargs that mirror ctypes' ``_optional_*(v, default)`` handling:
    # keep the Python default ``None`` but resolve the semantic default here so
    # omitting the kwarg behaves exactly like the ctypes wrapper.
    scalar_resolve = {
        "int64": "int", "bool": "bool", "double": "float", "float": "float",
    }
    for a in spec["args"]:
        kind = a["kind"]
        arg_default = a.get("default")
        if (kind not in scalar_resolve or arg_default is None
                or defaults.get(a["name"]) != "None"):
            continue
        lines.append(
            f"    {a['name']} = ({arg_default} if {a['name']} is None "
            f"else {scalar_resolve[kind]}({a['name']}))")
    for a in spec["args"]:
        if a["kind"] == "int_array":
            v = a["name"]
            lines.append(
                f"    {v} = [] if {v} is None else "
                f"[int(v) for v in {v}]")
    if py.get("derive_chunk_indices") and "chunk_size" in {
            a["name"] for a in spec["args"]}:
        # Mirror the ctypes wrappers: when cu_seqlens is provided but
        # chunk_indices is omitted, synthesize the canonical sequence-major
        # [seq, chunk] list before launching.
        lines.append("    if cu_seqlens and not chunk_indices:")
        lines.append("        chunk_indices = []")
        lines.append("        for _seq in range(len(cu_seqlens) - 1):")
        lines.append("            _len = cu_seqlens[_seq + 1] - cu_seqlens[_seq]")
        lines.append("            for _c in range((_len + chunk_size - 1) // chunk_size):")
        lines.append("                chunk_indices.extend((_seq, _c))")
    call_args = []
    for a in spec["args"]:
        if a["kind"] == "out_tensor":
            continue
        kind, v = a["kind"], a["name"]
        if kind == "tensor" or kind == "optional_tensor":
            call_args.append(v)
        elif kind == "int_array":
            call_args.append(v)
        elif kind == "char_ptr":
            call_args.append(f"str({v})")
        elif kind in ("int64", "bool", "double", "float"):
            cast = {"int64": "int", "bool": "bool",
                    "double": "float", "float": "float"}[kind]
            call_args.append(f"{cast}({v})")
    body = "\n".join(lines)
    call = "ext.%s(\n        %s,\n        _current_stream_ptr(),\n    )" % (
        name, ",\n        ".join(call_args))
    if n_out > 1:
        body += f"\n    result = {call}"
        outputs_spec = spec.get("outputs", [])
        out_names = [a["name"] for a in spec["args"] if a["kind"] == "out_tensor"]
        return_suffix = py.get("return_suffix", [])
        return_code = py.get("return_code")
        if return_code:
            # Raw return lines: needed for operators whose Python-visible
            # return shape depends on flags that never reach aclnn (e.g.
            # disable_recompute / return_intermediate_states).
            for raw_line in return_code.splitlines():
                body += ("\n    " + raw_line) if raw_line.strip() else ""
            text = text.rstrip() + "\n" + body + "\n"
            path.write_text(text, encoding="utf-8")
            return
        return_order = py.get("return_order")
        if return_order is not None:
            terms = []
            for item in return_order:
                if isinstance(item, dict) and item.get("none"):
                    terms.append("None")
                    continue
                item_name = item if isinstance(item, str) else item["name"]
                if item_name not in out_names:
                    raise ValueError(
                        f"{name}: return_order item {item_name!r} is not an "
                        f"output arg {out_names}")
                terms.append(f"result[{out_names.index(item_name)}]")
            terms.extend(return_suffix)
            body += "\n    return (" + ", ".join(terms) + ")"
            text = text.rstrip() + "\n" + body + "\n"
            path.write_text(text, encoding="utf-8")
            return
        whens = {
            i: (outputs_spec[i].get("when")
                or outputs_spec[i].get("return_when"))
            for i in range(min(len(outputs_spec), len(out_names)))
            if ("when" in outputs_spec[i]
                or "return_when" in outputs_spec[i])
        }
        if whens:
            terms = []
            for i, out_name in enumerate(out_names):
                if i in whens:
                    py_when = outputs_spec[i].get("when_py", whens[i])
                    terms.append(f"(result[{i}] if {py_when} else None)")
                else:
                    terms.append(f"result[{i}]")
            terms.extend(return_suffix)
            body += "\n    return (" + ", ".join(terms) + ")"
        else:
            if return_suffix:
                terms = [f"result[{i}]" for i in range(len(out_names))]
                terms.extend(return_suffix)
                body += "\n    return (" + ", ".join(terms) + ")"
            else:
                body += "\n    return tuple(result)"
    else:
        out_entry = spec.get("output", {})
        when = out_entry.get("when")
        if when:
            body += f"\n    result = {call}"
            body += f"\n    return None if not {when} else result"
        else:
            body += f"\n    return {call}"
    text = text.rstrip() + "\n" + body + "\n"
    path.write_text(text, encoding="utf-8")


def patch_whitelist(spec: dict) -> None:
    # Whitelist is derived dynamically from _thin module functions; no patch.
    return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if not spec.get("enabled", True):
        print(f"skipped {spec['python_name']}: spec disabled (stays on ctypes)")
        return 0
    sys.path.insert(0, str(Path(__file__).parent))
    from op_spec_codegen import generate

    generated_path = CSRC / "ops_generated.cpp"
    text = generated_path.read_text(encoding="utf-8")
    if not has_fn_def(text, spec["python_name"]):
        with generated_path.open("a", encoding="utf-8") as fh:
            fh.write("\n// ============ generated from "
                     f"{spec['aclnn_name']} ============\n")
            fh.write(generate(spec))
            fh.write("\n")
    patch_pybind(spec)
    patch_thin(spec)
    patch_whitelist(spec)
    print(f"applied {spec['python_name']}: cpp/pybind/_thin/whitelist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
