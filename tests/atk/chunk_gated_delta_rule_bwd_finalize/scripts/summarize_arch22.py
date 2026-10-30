"""保留原 Excel，汇总 ATK 原生结果、角色身份和 no-save 后验盘点。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from native_xlsx import _xlsx_rows

PASS = {"pass", "passed", "success", "successful", "true", "达标"}
BANNED = {".pt", ".pth", ".bin", ".npy", ".npz", ".pkl", ".pickle", ".safetensors", ".tensor"}


def native_report(path, scope, count):
    rows = _xlsx_rows(path, "summary")
    if len(rows) < 2:
        raise ValueError("Excel summary 无数据行")
    header = [str(x or "") for x in rows[0]]
    def column(*needles):
        for i, name in enumerate(header):
            if any(needle in name for needle in needles):
                return i
        raise ValueError(f"Excel 缺少列 {needles}")
    total = column("总用例", "用例总", "用例个数")
    statuses = [i for i, name in enumerate(header) if "达标" in name]
    if not statuses and scope != "keys":
        raise ValueError("Excel 没有原生达标列")
    details, errors = [], []
    for values in rows[1:]:
        if not values or all(x is None for x in values):
            continue
        values += [None] * (len(header) - len(values))
        if int(values[total]) != count:
            errors.append(f"{values[0]} total {values[total]} != {count}")
        effective = [str(values[i] or "").strip() for i in statuses
                     if str(values[i] or "").strip() not in ("", "-")]
        if scope == "keys":
            success = column("执行成功用例个数")
            if int(values[success]) != count:
                errors.append(f"{values[0]} native execution count {values[success]}")
        elif not effective or any(value.lower() not in PASS for value in effective):
            errors.append(f"{values[0]} native status {effective}")
        details.append(dict(zip(header, values)))
    if scope == "accuracy" and len(details) < 2:
        errors.append("三路精度至少需要 NPU 与 cpu_lowp 两条对 FP64 的原生结果")
    return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                rows=details, errors=errors)


def summarize(root, scope, rc, count):
    errors = [] if rc == 0 else [f"native task rc={rc}"]
    reports = []
    for path in sorted(root.rglob("*.xlsx")):
        try:
            reports.append(native_report(path, scope, count))
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"{path}: {exc}")
    if not reports:
        errors.append("缺少可解析的 ATK 原生 Excel")
    errors.extend(error for report in reports for error in report["errors"])
    evidence = Path(os.getenv("FINALIZE_CPU_EVIDENCE", str(root / "role-evidence")))
    roles = ("dut", "lowp", "fp64") if scope == "accuracy" else ("dut",)
    for case_id in range(count):
        records = []
        for role in roles:
            path = evidence / f"{case_id:04d}" / f"{role}.json"
            if not path.is_file():
                errors.append(f"case {case_id}: missing role {role}")
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("case_id") != case_id or record.get("role") != role:
                errors.append(f"case {case_id}: role receipt mismatch {role}")
            if role == "fp64" and any(value["dtype"] != "torch.float64" for value in record["outputs"].values()):
                errors.append(f"case {case_id}: FP64 golden was downcast")
            if role == "dut" and not record.get("runtime", {}).get("measured_binding_sha256"):
                errors.append(f"case {case_id}: missing measured worker binding")
            records.append(record)
        if len({record["input_sha256"] for record in records}) > 1:
            errors.append(f"case {case_id}: three roles used different inputs")
    banned = []
    for path in root.rglob("*"):
        if path.is_symlink():
            banned.append(str(path))
        elif path.is_file() and path.suffix.lower() in BANNED:
            # keys 专项只允许 profiler 自身二进制；它们不是 tensor IO。
            if scope != "keys" or not any("profil" in part.lower() for part in path.parts):
                banned.append(str(path))
    if banned:
        errors.append(f"no-save 盘点发现 {len(banned)} 个禁止资产或链接")
    actual_keys = None
    if scope == "keys":
        from capture_key_evidence import capture
        from check_key_evidence import check
        try:
            captured = capture(root)
            actual_keys = str(root / "key-evidence.json")
            if check(root / "selection.json", captured, Path(actual_keys)):
                errors.append("实际 Host key 与逐 case 预期不一致或缺失；见 key-evidence.json")
        except (ValueError, KeyError, OSError) as exc:
            errors.append(f"实际 Host key 提取失败：{exc}")
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result = dict(schema="finalize-arch22-result/v1", scope=scope, expected_cases=count, case_count=count,
                  source_case_ids=[value["source_case_id"] for value in selection["cases"]],
                  mapping=str(selection_path), original_excel=[{key: value[key] for key in ("path", "bytes", "sha256")} for value in reports],
                  task_rc=rc, native_reports=reports, role_evidence=str(evidence),
                  no_save=dict(passed=not banned, forbidden_files=banned),
                  actual_key_evidence=actual_keys, verdict="pass" if not errors else "fail", errors=errors)
    (root / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(verdict=result["verdict"], expected_cases=count, errors=errors[:12]), ensure_ascii=False))
    return 0 if not errors else 99


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--scope", required=True)
    p.add_argument("--task-rc", type=int, required=True)
    p.add_argument("--expected-cases", type=int, required=True)
    a = p.parse_args()
    raise SystemExit(summarize(a.root, a.scope, a.task_rc, a.expected_cases))
