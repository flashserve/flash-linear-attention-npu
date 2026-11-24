#!/usr/bin/env python3
"""Validate that every completed issue is mapped to a review rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALLOWED_LAYERS = {"functional", "engineering", "dfx", "docs", "test", "ci", "performance", "feature"}
REQUIRED_LAYERS = {"functional", "engineering", "dfx", "docs"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "catalog",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "references" / "issue-corpus.json",
    )
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    errors = []
    rules_seen = set()
    layers_seen = set()
    issue_numbers = []
    for issue in catalog.get("issues", []):
        issue_numbers.append(issue.get("number"))
        rules = issue.get("rules", [])
        if not rules:
            errors.append(f"#{issue.get('number')} has no review rule")
            continue
        for rule in rules:
            if not rule.get("id"):
                errors.append(f"#{issue.get('number')} has an empty rule id")
            if rule.get("layer") not in ALLOWED_LAYERS:
                errors.append(
                    f"#{issue.get('number')} has invalid layer {rule.get('layer')!r}"
                )
            rules_seen.add(rule.get("id"))
            layers_seen.add(rule.get("layer"))

    expected = catalog.get("completed_issue_count")
    if expected != len(issue_numbers):
        errors.append(
            f"completed_issue_count={expected} does not match {len(issue_numbers)} records"
        )
    if len(issue_numbers) != len(set(issue_numbers)):
        errors.append("issue corpus contains duplicate issue numbers")
    missing_layers = REQUIRED_LAYERS - layers_seen
    if missing_layers:
        errors.append(f"required review layers are missing: {sorted(missing_layers)}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(
        "issue coverage OK: "
        f"{len(issue_numbers)} completed issues, "
        f"{len(rules_seen)} rules, layers={','.join(sorted(layers_seen))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
