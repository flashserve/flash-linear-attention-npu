#!/usr/bin/env python3
"""Fetch completed repository issues and classify them into review rules."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = "flashserve/flash-linear-attention-npu"

# Rule IDs are defined in references/issue-rule-map.md. Keep this table
# deterministic so a catalog update is auditable in a small diff.
RULES = [
    ("FUNC-SYNC", "functional", r"race|并发|同步|异步下发|\bevent\b|\bflag\b|hang|超时|死锁|hazard|\bWAR\b|\bRAW\b"),
    ("FUNC-PRECISION", "functional", r"精度|误差|cosine|rmse|\bNaN\b|\bInf\b|non-finite|溢出|舍入|累加|bitwise"),
    ("FUNC-DETERMINISM", "functional", r"不确定|determin|复现不一致|fwd/recompute"),
    ("FUNC-MEMORY", "functional", r"越界|地址|MTE|GM address|内存|workspace|OOM|out of memory|未初始化|shared-anchor|double free"),
    ("FUNC-VARLEN", "functional", r"varlen|变长|cu_seqlens|chunk_indices|TND|NTD|尾块|partial chunk|有效长度|dynamic sequence length"),
    ("FUNC-SHAPE", "functional", r"shape|形状|rank|维度|scalar|空tensor|空 tensor|empty|head|GQA|GVA|三头|展平|尺寸"),
    ("FUNC-DTYPE-FORMAT", "functional", r"dtype|format|非连续|连续|contiguous|layout|布局|混合精度|\bfp16\b|\bbf16\b|\bFP32\b"),
    ("FUNC-OPTIONAL-OUTPUT", "functional", r"可选输出|optional output|输出接口|六输出|输出布局|输出不确定"),
    ("FUNC-STATE", "functional", r"\bstate\b|状态|initialState|finalState|\bh0\b|\bdht\b|\bgate\b|衰减|v_new"),
    ("DFX-VALIDATION", "dfx", r"校验|拦截|未拦截|非法|异常场景|参数约束|不支持"),
    ("DFX-MESSAGE", "dfx", r"错误信息|报错信息|只有错误码|定位困难|报错文本|错误码|161002|561103|507035|DFX"),
    ("ENG-STRUCTURE", "engineering", r"工程结构|目录|命名|冗余|文件位置|适配层|稳定入口|ABI|接口兼容|导出"),
    ("ENG-COMPATIBILITY", "engineering", r"版本差异|版本门禁|能力探测|兼容|triton-ascend|芯片路径|路径选择|回退|移植|SoC|Ascend950|A2/A3|A3/A5"),
    ("ENG-BUILD", "engineering", r"构建|编译|安装|wheel|链接|打包|依赖|OPP|No module|import|导入|环境变量"),
    ("DOC-FORMAT", "docs", r"文档|资料|README|readme|样例链接|教程|格式|版本配套|兼容关系"),
    ("DOC-CONSTRAINT", "docs", r"资料约束|文档.*约束|约束.*实际|用法不清晰|用法|支持范围"),
    ("TEST-COVERAGE", "test", r"测试|用例|覆盖|ATK|executor|golden|标杆|复检"),
    ("CI-WORKFLOW", "ci", r"\bCI\b|workflow|runner|流水线|Actions|发布|标签|需求汇总|index\.yml"),
    ("PERF-REGRESSION", "performance", r"性能|耗时|时延|吞吐|优化|提速|速度"),
    ("FEATURE-DELIVERY", "feature", r"\[Feature\]|新增|增加|支持.*入口|迁移|移植|重构|补齐"),
]

RULES_BY_ID = {rule_id: (layer, pattern) for rule_id, layer, pattern in RULES}


def fetch_issues() -> list[dict]:
    command = [
        "gh",
        "issue",
        "list",
        "--repo",
        REPOSITORY,
        "--state",
        "closed",
        "--limit",
        "1000",
        "--json",
        "number,title,body,stateReason,closedAt,labels,url",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def classify(issue: dict) -> list[dict]:
    labels = {label["name"] for label in issue.get("labels", [])}
    haystack = " ".join(
        [issue.get("title", ""), " ".join(sorted(labels))]
    )
    matches = []
    for rule_id, layer, pattern in RULES:
        if re.search(pattern, haystack, flags=re.IGNORECASE):
            matches.append({"id": rule_id, "layer": layer})

    if "bug" in labels and not any(match["layer"] == "functional" for match in matches):
        matches.append({"id": "FUNC-REGRESSION", "layer": "functional"})

    if not matches:
        if "enhancement" in labels or "[Feature]" in issue.get("title", ""):
            matches.append({"id": "FEATURE-DELIVERY", "layer": "feature"})
        elif "question" in labels:
            matches.append({"id": "DOC-FORMAT", "layer": "docs"})
        elif "bug" in labels:
            matches.append({"id": "FUNC-REGRESSION", "layer": "functional"})
        else:
            matches.append({"id": "ENG-STRUCTURE", "layer": "engineering"})

    # Remove duplicate IDs while retaining deterministic ordering.
    unique = {}
    for match in matches:
        unique.setdefault(match["id"], match)
    return list(unique.values())


def build_catalog(raw_issues: list[dict]) -> dict:
    completed = []
    excluded = []
    for issue in raw_issues:
        record = {
            "number": issue["number"],
            "title": issue["title"],
            "url": issue["url"],
            "closed_at": issue.get("closedAt"),
            "labels": sorted(label["name"] for label in issue.get("labels", [])),
        }
        if issue.get("stateReason") != "COMPLETED":
            record["state_reason"] = issue.get("stateReason")
            excluded.append(record)
            continue
        record["rules"] = classify(issue)
        completed.append(record)

    completed.sort(key=lambda item: item["number"])
    excluded.sort(key=lambda item: item["number"])
    return {
        "schema_version": 1,
        "repository": REPOSITORY,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "completed_issue_count": len(completed),
        "excluded_not_planned_count": len(excluded),
        "issues": completed,
        "excluded_not_planned": excluded,
    }


def render_markdown(catalog: dict) -> str:
    lines = [
        "# 已完成 issue 检视分类",
        "",
        f"- 仓库：`{catalog['repository']}`",
        f"- 已完成 issue：{catalog['completed_issue_count']}",
        "- 生成方式：`python3 scripts/sync_issue_corpus.py`",
        "- 约束：每条 `COMPLETED` issue 必须至少映射一个 rule；`NOT_PLANNED` 单独列在 JSON 快照中，不纳入已解决覆盖。",
        "",
        "| Issue | 标题 | 检视规则 |",
        "| --- | --- | --- |",
    ]
    for issue in catalog["issues"]:
        title = issue["title"].replace("|", "\\|")
        rules = ", ".join(rule["id"] for rule in issue["rules"])
        lines.append(f"| [#{issue['number']}]({issue['url']}) | {title} | {rules} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "references" / "issue-corpus.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "references" / "issue-catalog.md",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    catalog = build_catalog(fetch_issues())
    rendered = json.dumps(catalog, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not args.output.exists():
            raise SystemExit(f"missing catalog: {args.output}")
        current = json.loads(args.output.read_text(encoding="utf-8"))
        if current["completed_issue_count"] != catalog["completed_issue_count"]:
            raise SystemExit(
                "completed issue count drifted: "
                f"{current['completed_issue_count']} != {catalog['completed_issue_count']}"
            )
        current_numbers = {item["number"] for item in current["issues"]}
        fetched_numbers = {item["number"] for item in catalog["issues"]}
        if current_numbers != fetched_numbers:
            missing = sorted(fetched_numbers - current_numbers)
            stale = sorted(current_numbers - fetched_numbers)
            raise SystemExit(f"issue set drifted: missing={missing}, stale={stale}")
        if not args.markdown_output.exists():
            raise SystemExit(f"missing generated markdown: {args.markdown_output}")
        if args.markdown_output.read_text(encoding="utf-8") != render_markdown(catalog):
            raise SystemExit("issue-catalog.md is stale; rerun sync_issue_corpus.py")
        print(f"issue corpus current: {len(fetched_numbers)} completed issues")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(catalog), encoding="utf-8")
    print(
        f"wrote {catalog['completed_issue_count']} completed issues to "
        f"{args.output} and {args.markdown_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
