"""只从 worker 调用边界内的实际 Host debug 行提取 key。"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

BEGIN = re.compile(r"\[finalize-arch22-key-begin\] case=(\d+)")
END = re.compile(r"\[finalize-arch22-key-end\] case=(\d+)")
KEY = re.compile(r"GDN_FINALIZE_ARCH22_TILING actualKey=(\d+) owner=(\d+) gKey=(\d+) norm=(\d+) sigmoid=(\d+) exp2=(\d+)")


def capture(root):
    observed, errors = {}, []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in (".log", ".txt"):
            continue
        active = None
        with path.open(encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                begin, end, key = BEGIN.search(line), END.search(line), KEY.search(line)
                if begin:
                    active = int(begin[1])
                if key and active is not None:
                    values = tuple(map(int, key.groups()))
                    previous = observed.get(active)
                    if previous and tuple(previous["host_fields"]) != values:
                        errors.append(f"case {active}: conflicting actual Host keys")
                    observed[active] = dict(case_id=active, actual_key=values[0], host_fields=values,
                        evidence_path=str(path.resolve()), evidence_line=number,
                        evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                if end:
                    if active != int(end[1]):
                        errors.append(f"{path}:{number}: unmatched worker boundary")
                    active = None
    if errors:
        raise ValueError("; ".join(errors))
    target = root / "actual-keys.json"
    target.write_text(json.dumps(list(observed.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
