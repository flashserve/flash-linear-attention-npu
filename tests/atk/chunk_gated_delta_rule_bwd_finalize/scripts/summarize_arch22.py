"""保留原 Excel，汇总 ATK 原生结果、角色身份和 no-save 后验盘点。"""
from __future__ import annotations
import argparse
import ast
import math
import re
import hashlib
import json
import os
from pathlib import Path
from native_xlsx import _xlsx_rows

PASS = {"pass", "passed", "success", "successful", "true", "达标"}
BANNED = {".pt", ".pth", ".bin", ".npy", ".npz", ".pkl", ".pickle", ".safetensors", ".tensor"}



CV_STANDARD = {'cv_fused_double_benchmark': {'max_re_ratio': 5, 'avg_re_ratio': 1.5, 'root_mean_squared_ratio': 1.5}}


def _metrics_for_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    if isinstance(value, dict):
        return {key: _metrics_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_metrics_for_json(item) for item in value]
    return value


def _declared_equal(expected, actual):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for key, value in expected.items():
            _declared_equal(value, actual[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual)
        for left, right in zip(expected, actual):
            _declared_equal(left, right)
    else:
        assert expected == actual, '原生Excel的case字段/输入/seed/阈值不同'


def _statistic_evidence(path, cases):
    rows = _xlsx_rows(path, 'statistic')
    assert len(rows) >= 2, 'statistic没有逐case数据'
    header = rows[0]
    required = ('编号', '运行结果', '失败原因', '用例json信息')
    assert all(header.count(name) == 1 for name in required)
    cols = {name: header.index(name) for name in required}
    expected = {row['id']: row for row in cases}
    assert cases and len(expected) == len(cases)
    evidence, raw_by_id, seen = [], {}, set()
    for raw in rows[1:]:
        if not raw or all(value is None for value in raw):
            continue
        values = raw + [None] * max(0, len(header) - len(raw))
        text = values[cols['编号']]
        assert not isinstance(text, bool) and re.fullmatch(r'\d+', str(text)), 'case编号不是整数'
        cid = int(text)
        assert cid in expected and cid not in seen, '额外/重复case ID'
        seen.add(cid)
        status = str(values[cols['运行结果']] or '').strip()
        assert status == 'SUCCESS', '逐case运行结果非SUCCESS（含SKIP）'
        reason = values[cols['失败原因']]
        assert reason is None or str(reason).strip() == '', '成功行仍有失败原因'
        serialized = values[cols['用例json信息']]
        native_case = json.loads(serialized)
        # ATK用配置文件stem重写save_name；其他已声明字段必须逐项相等。
        _declared_equal({key: value for key, value in expected[cid].items() if key != 'save_name'}, native_case)
        evidence.append(dict(case_id=cid, run_result=status, failure_reason=reason,
            case_json_sha256=hashlib.sha256(serialized.encode()).hexdigest()))
        raw_by_id[cid] = values
    assert seen == set(expected) and len(evidence) == len(cases), 'statistic缺失case'
    failed = _xlsx_rows(path, 'failed cases')
    assert failed and not any(row and any(value is not None for value in row) for row in failed[1:]), '原生failed cases非空'
    return evidence, header, raw_by_id


def _run_report(path, scope, cases):
    assert scope in ('keys', 'mssanitizer') and len(cases) == 40, 'keys/MSS正式矩阵必须40条'
    rows, _, _ = _statistic_evidence(path, cases)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        evidence_kind='task_run_statistic', sheet='statistic', run_success_count=40,
        case_ids=sorted(row['case_id'] for row in rows), rows=rows, errors=[],
        MSS_detection_coverage_proven=False)


def _cv_details(path, cases):
    rows, header, by_id = _statistic_evidence(path, cases)
    for name in ('npu_finalize_精度通过', 'cpu_cpu_lowp_精度通过', '精度详情'):
        assert header.count(name) == 1
    details = []
    for cid, values in by_id.items():
        # ATK 26.7的CV是一次combined判定；NPU独立pass列留空。
        assert values[header.index('npu_finalize_精度通过')] is None
        assert values[header.index('cpu_cpu_lowp_精度通过')] is True
        text = values[header.index('精度详情')]
        class Names(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in ('inf', 'nan'):
                    return ast.copy_location(ast.Constant(float(node.id)), node)
                return node
        tree = Names().visit(ast.parse(text, mode='eval'))
        ast.fix_missing_locations(tree)
        parsed = ast.literal_eval(tree)
        assert set(parsed) == {'cpu_cpu_lowp'}, 'CV必须是单组Actual/Benchmark对同high的combined判定'
        for role, outputs in parsed.items():
            assert len(outputs) == 5 and {item['filename'] for item in outputs} == {f'output_{i}.pt' for i in range(5)}
            for item in outputs:
                assert item['result'] is True and not item['error_info'], 'CV逐输出未通过'
                metrics = item['new_benchmark_indicate']
                assert isinstance(metrics, dict)
                actual = {key[len('Actual_'):] for key in metrics if key.startswith('Actual_')}
                benchmark = {key[len('Benchmark_'):] for key in metrics if key.startswith('Benchmark_')}
                assert actual and actual == benchmark, '缺Actual/Benchmark成对原生指标'
                details.append(dict(case_id=cid, role=role, output=item['filename'], metrics=_metrics_for_json(metrics)))
    return dict(case_ids=sorted(by_id), execution_rows=rows, judgments=details)


def _cv_summary(rows, count):
    assert len(rows) == 1 and rows[0]['名称'] == 'cpu_cpu_lowp', 'CV summary必须只有一条combined记录'
    row = rows[0]
    assert row['精度是否达标'] == 'Pass'
    for name in ('总用例数', '执行成功用例个数', '通过用例个数'):
        assert int(row[name]) == count, 'CV summary成功计数不符'
    for name in ('执行失败用例个数', '错误信息匹配用例个数'):
        assert int(row[name]) == 0, 'CV summary含失败或错误匹配'


def native_report(path, scope, count, cases):
    assert len(cases) == count
    if scope == 'accuracy':
        assert all(row['standard']['acc'] == CV_STANDARD for row in cases), '新accuracy仅接受固定CV标准'
    if scope in ('keys', 'mssanitizer'):
        return _run_report(path, scope, cases)
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
    if scope == 'accuracy':
        _cv_summary(details, count)
    result = dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  evidence_kind='accuracy_summary' if scope == 'accuracy' else 'summary',
                  sheet='summary', rows=details, errors=errors)
    if scope == 'accuracy':
        result['case_evidence'] = _cv_details(path, cases)
    return result


def summarize(root, scope, rc, count, output=None):
    errors = [] if rc == 0 else [f"native task rc={rc}"]
    reports = []
    cases = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    for path in sorted(root.rglob("*.xlsx")):
        try:
            reports.append(native_report(path, scope, count, cases))
        except (AssertionError, ValueError, KeyError, TypeError) as exc:
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
    output = root / 'summary.json' if output is None else output
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(verdict=result["verdict"], expected_cases=count, errors=errors[:12]), ensure_ascii=False))
    return 0 if not errors else 99


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--scope", required=True)
    p.add_argument("--task-rc", type=int, required=True)
    p.add_argument("--expected-cases", type=int, required=True)
    p.add_argument("--output", type=Path, help="新旁路汇总文件；拒绝覆盖原失败记录")
    a = p.parse_args()
    raise SystemExit(summarize(a.root, a.scope, a.task_rc, a.expected_cases, a.output))
