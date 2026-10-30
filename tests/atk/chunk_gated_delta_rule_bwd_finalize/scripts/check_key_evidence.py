"""核对逐 case 的真实 Host/runtime key 记录；不生成或猜测 actual_key。"""
import argparse
import hashlib
import json
from pathlib import Path


def check(selection_path, actual_path, output):
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    observed = json.loads(actual_path.read_text(encoding="utf-8"))
    if not isinstance(observed, list):
        raise ValueError("actual JSON 必须为逐 case 记录列表")
    indexed, errors, records = {}, [], []
    for value in observed:
        key = int(value["case_id"])
        if key in indexed:
            errors.append(f"duplicate actual case {key}")
        indexed[key] = value
    for expected in selection["cases"]:
        cid = expected["case_id"]
        value = indexed.get(cid)
        if value is None:
            errors.append(f"missing actual case {cid}")
            continue
        actual = value["actual_key"]
        actual = int(actual, 0) if isinstance(actual, str) else int(actual)
        path = Path(value["evidence_path"])
        if not path.is_file():
            errors.append(f"missing original evidence {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != value["evidence_sha256"]:
            errors.append(f"evidence SHA mismatch {cid}")
        if actual != expected["expected_key"]:
            errors.append(f"case {cid}: actual={actual}, expected={expected['expected_key']}")
        records.append(dict(expected, actual_key=actual, evidence_path=str(path), evidence_sha256=digest))
    if set(indexed) != {value["case_id"] for value in selection["cases"]}:
        errors.append("actual case set differs from selected case set")
    result = dict(schema="finalize-arch22-key-evidence/v1", verdict="pass" if not errors else "fail",
                  expected_key_count=selection["expected_key_count"], cases=records, errors=errors)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if not errors else 99


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("selection", type=Path)
    p.add_argument("actual", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(check(a.selection, a.actual, a.output))
