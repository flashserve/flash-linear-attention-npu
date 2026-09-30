#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按级别统计精度检视的漏警率与虚警，并按阈值判定是否达标。

输入
----
issues.json（人工标注的 issue 数据集）::

    {
      "scope": "备注，可选",
      "items": [
        {"id": "issue-700", "level": "lv0", "category": "同步/事件时序",
         "file": "fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/x.h", "line": 123,
         "summary": "核心问题摘要", "related": ["索引/计数"]}
      ]
    }

findings.json（检视 agent 输出的意见）::

    {
      "findings": [
        {"id": "F1", "level": "lv0", "category": "同步/事件时序",
         "file": "fla/.../x.h", "line": 131, "summary": "意见摘要",
         "hit": "issue-700",     // 可选：显式命中某个 issue
         "valid": null}          // 人工判定：true=真实问题，false=虚警，null=未判定
      ]
    }

配对规则
--------
1. 显式 `hit` 优先：直接把该 finding 记为命中对应 issue（此时不再参与自动配对）。
2. 自动配对：级别相同（可用 --ignore-level 关闭）、类别相同、文件后缀匹配、
   行号差不超过 --tolerance（给 0 表示必须同一行）。
3. 每条 issue 最多命中一次、每条 finding 最多配对一次；多条候选时取行号最接近的。
4. 人工确认结论优先于自动配对：把人工结论写成 finding 的 `hit`（命中）即可覆盖。

输出
----
逐级别问题数、命中数、漏警率与阈值判定、虚警条目、漏警明细；--json 时输出机器可读结果。
退出码 0 = 全部已测级别达标，1 = 存在未达标级别，2 = 输入错误。
"""

from __future__ import print_function

import argparse
import io
import json
import os
import sys

LEVELS = ("lv0", "lv1", "lv2", "lv3")
DEFAULT_THRESHOLDS = {"lv0": 0.10, "lv1": 0.20, "lv2": 0.20, "lv3": 0.50}


def to_text_stream(stream):
    """在 Py3 下尽量把 stdout 固定为 UTF-8，避免中文在 Windows 控制台变成乱码。"""
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass
    return stream


def load_json(path):
    with io.open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_path(path):
    if not path:
        return ""
    text = str(path).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


def normalize_token(text):
    if text is None:
        return ""
    return "".join(str(text).split()).replace("／", "/").lower()


def path_match(issue_path, finding_path):
    left, right = normalize_path(issue_path), normalize_path(finding_path)
    if not left or not right:
        return False
    return left == right or left.endswith(right) or right.endswith(left)


def as_line(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def collect_issues(payload):
    items = payload.get("items") if isinstance(payload, dict) else None
    if items is None and isinstance(payload, list):
        items = payload
    if not isinstance(items, list):
        raise ValueError("issues.json 需要包含 items 列表")
    issues = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("issues.items[%d] 不是对象" % index)
        issue_id = item.get("id") or item.get("issue") or "issue-%d" % (index + 1)
        level = normalize_token(item.get("level"))
        if level not in LEVELS:
            raise ValueError("issue %s 的 level 非法：%r" % (issue_id, item.get("level")))
        issues.append(
            {
                "id": str(issue_id),
                "level": level,
                "category": normalize_token(item.get("category")),
                "category_raw": item.get("category"),
                "file": normalize_path(item.get("file")),
                "line": as_line(item.get("line")),
                "summary": item.get("summary") or "",
            }
        )
    return issues


def collect_findings(payload, category_alias):
    items = payload.get("findings") if isinstance(payload, dict) else None
    if items is None and isinstance(payload, list):
        items = payload
    if not isinstance(items, list):
        raise ValueError("findings.json 需要包含 findings 列表")
    findings = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError("findings[%d] 不是对象" % index)
        finding_id = item.get("id") or "F%d" % (index + 1)
        level = normalize_token(item.get("level"))
        if level and level not in LEVELS:
            raise ValueError("finding %s 的 level 非法：%r" % (finding_id, item.get("level")))
        category = normalize_token(item.get("category"))
        category = normalize_token(category_alias.get(category, category))
        findings.append(
            {
                "id": str(finding_id),
                "level": level,
                "category": category,
                "category_raw": item.get("category"),
                "file": normalize_path(item.get("file")),
                "line": as_line(item.get("line")),
                "summary": item.get("summary") or "",
                "hit": str(item["hit"]) if item.get("hit") else None,
                "valid": item.get("valid"),
            }
        )
    return findings


def explicit_matches(issues, findings):
    by_id = dict((issue["id"], issue) for issue in issues)
    matched = {}
    warnings = []
    for finding in findings:
        if not finding["hit"]:
            continue
        issue = by_id.get(finding["hit"])
        if issue is None:
            warnings.append("finding %s 的 hit=%s 不在 issue 数据集中" % (finding["id"], finding["hit"]))
            continue
        matched.setdefault(issue["id"], finding["id"])
    return matched, warnings


def auto_matches(issues, findings, matched, matched_findings, tolerance, ignore_level):
    for issue in issues:
        if issue["id"] in matched:
            continue
        candidates = []
        for finding in findings:
            if finding["id"] in matched_findings:
                continue
            if not ignore_level and finding["level"] != issue["level"]:
                continue
            if finding["category"] != issue["category"]:
                continue
            if not path_match(issue["file"], finding["file"]):
                continue
            issue_line, finding_line = issue["line"], finding["line"]
            if issue_line is not None and finding_line is not None:
                distance = abs(issue_line - finding_line)
                if distance > tolerance:
                    continue
            else:
                distance = tolerance + 1
            candidates.append((distance, finding["id"]))
        if candidates:
            candidates.sort()
            chosen = candidates[0][1]
            matched[issue["id"]] = chosen
            matched_findings.add(chosen)


def score(issues, findings, tolerance, thresholds, ignore_level):
    matched, warnings = explicit_matches(issues, findings)
    matched_findings = set(matched.values())
    auto_matches(issues, findings, matched, matched_findings, tolerance, ignore_level)

    by_level = {}
    for level in LEVELS:
        by_level[level] = {"issues": 0, "caught": 0, "missed": []}
    for issue in issues:
        entry = by_level[issue["level"]]
        entry["issues"] += 1
        if issue["id"] in matched:
            entry["caught"] += 1
        else:
            entry["missed"].append(issue)

    findings_by_id = dict((finding["id"], finding) for finding in findings)
    unmatched_findings = [f for f in findings if f["id"] not in matched_findings]
    false_alarms = [f for f in unmatched_findings if f["valid"] is False]
    undecided = [f for f in unmatched_findings if f["valid"] is None]
    bonus = [f for f in unmatched_findings if f["valid"] is True]

    report = {"levels": [], "warnings": warnings}
    overall_pass = True
    for level in LEVELS:
        entry = by_level[level]
        if entry["issues"] == 0:
            report["levels"].append(
                {"level": level, "issues": 0, "caught": 0, "miss_rate": None, "threshold": thresholds[level],
                 "status": "n/a", "missed": []}
            )
            continue
        miss_rate = (entry["issues"] - entry["caught"]) / float(entry["issues"])
        passed = miss_rate < thresholds[level] - 1e-12
        overall_pass = overall_pass and passed
        report["levels"].append(
            {
                "level": level,
                "issues": entry["issues"],
                "caught": entry["caught"],
                "miss_rate": miss_rate,
                "threshold": thresholds[level],
                "status": "pass" if passed else "fail",
                "missed": [
                    {"id": issue["id"], "category": issue["category_raw"] or issue["category"],
                     "file": issue["file"], "line": issue["line"], "summary": issue["summary"]}
                    for issue in entry["missed"]
                ],
            }
        )

    report["matched"] = matched
    report["unmatched_findings"] = [
        {"id": f["id"], "level": f["level"], "category": f["category_raw"] or f["category"],
         "file": f["file"], "line": f["line"], "summary": f["summary"], "valid": f["valid"]}
        for f in unmatched_findings
    ]
    report["counts"] = {
        "issues": len(issues),
        "findings": len(findings),
        "false_alarms": len(false_alarms),
        "undecided": len(undecided),
        "bonus": len(bonus),
    }
    report["pass"] = bool(overall_pass)
    report["matched_findings"] = findings_by_id
    return report


def render_text(report):
    lines = []
    lines.append("级别   问题数  命中数  漏警率   阈值    判定")
    for entry in report["levels"]:
        if entry["issues"] == 0:
            lines.append("%-6s %6d %7s %8s %8s %6s" % (entry["level"], 0, "-", "-", "-", "n/a"))
            continue
        lines.append(
            "%-6s %6d %7d %7.1f%% %7.0f%% %6s"
            % (entry["level"], entry["issues"], entry["caught"], entry["miss_rate"] * 100.0,
               entry["threshold"] * 100.0, "PASS" if entry["status"] == "pass" else "FAIL")
        )
    counts = report["counts"]
    lines.append("")
    lines.append(
        "意见总数 %d，虚警 %d，未判定 %d，额外发现(bonus) %d"
        % (counts["findings"], counts["false_alarms"], counts["undecided"], counts["bonus"])
    )
    for warning in report["warnings"]:
        lines.append("[警告] " + warning)
    grouped = [entry for entry in report["levels"] if entry.get("missed")]
    if grouped:
        lines.append("")
        lines.append("漏警明细：")
        for entry in grouped:
            for item in entry["missed"]:
                lines.append(
                    "  [%s] %s %s:%s %s" % (entry["level"], item["id"], item["file"], item["line"], item["summary"])
                )
    unmatched = report["unmatched_findings"]
    if unmatched:
        lines.append("")
        lines.append("未配对意见（虚警候选与 bonus 来源）：")
        for item in unmatched:
            flag = {True: "bonus", False: "虚警", None: "未判定"}.get(item["valid"], "未判定")
            lines.append(
                "  [%s] %s %s:%s %s" % (flag, item["id"], item["file"], item["line"], item["summary"])
            )
    lines.append("")
    lines.append("整体判定：" + ("达标" if report["pass"] else "未达标"))
    return "\n".join(lines)


def parse_thresholds(raw):
    thresholds = dict(DEFAULT_THRESHOLDS)
    if not raw:
        return thresholds
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("--thresholds 需要 key=value 形式：%r" % part)
        key, value = part.split("=", 1)
        key = normalize_token(key)
        if key not in LEVELS:
            raise ValueError("--thresholds 的级别非法：%r" % key)
        value = float(value)
        thresholds[key] = value / 100.0 if value > 1.0 else value
    return thresholds


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="按级别统计精度检视漏警率/虚警并判定是否达标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="数据格式见脚本头部 docstring；默认阈值 lv0<10%、lv1/lv2<20%、lv3<50%。",
    )
    parser.add_argument("--issues", required=True, help="issue 标注 JSON")
    parser.add_argument("--findings", required=True, help="检视意见 JSON")
    parser.add_argument("--tolerance", type=int, default=50, help="行号配对容差，默认 50")
    parser.add_argument("--ignore-level", action="store_true", help="配对时忽略级别差异，仅按类别/文件/行号")
    parser.add_argument("--thresholds", default=None, help='覆盖阈值，例如 "lv0=5,lv3=40"（百分数或小数）')
    parser.add_argument("--alias", default=None, help="可选的类别别名映射 JSON，例如 {\"同步\": \"同步/事件时序\"}")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args(argv)

    out = to_text_stream(sys.stdout)
    try:
        issues = collect_issues(load_json(args.issues))
        alias_raw = load_json(args.alias) if args.alias else {}
        category_alias = dict((normalize_token(k), v) for k, v in alias_raw.items())
        findings = collect_findings(load_json(args.findings), category_alias)
        thresholds = parse_thresholds(args.thresholds)
    except (ValueError, IOError, OSError) as error:
        print("输入错误：%s" % error, file=out)
        return 2

    report = score(issues, findings, args.tolerance, thresholds, args.ignore_level)
    if args.json:
        payload = dict((k, v) for k, v in report.items() if k != "matched_findings")
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=out)
    else:
        print(render_text(report), file=out)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
