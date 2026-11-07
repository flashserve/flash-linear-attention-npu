"""Map actual profiler kernel keys to the public layout-case IDs."""

import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("tasks", type=Path)
parser.add_argument("manifest", type=Path)
parser.add_argument("--arch", choices=("arch22", "arch35"), required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
calls = json.loads(args.manifest.read_text(encoding="utf-8"))
assert len(calls) == 140 and all(c["iteration"] == 0 and c["measured"] for c in calls)
expected = {"ChunkFwdH": 32 if args.arch == "arch22" else 16,
            "ChunkGatedDeltaRuleBwdDhu": 4,
            "ChunkGatedDeltaRuleBwdFinalize": 64 if args.arch == "arch22" else 32}
expected.update({"RecomputeWUFwd": 2, "ChunkBwdDvLocal": 4} if args.arch == "arch22"
                else {"ChunkGdnBwdIntra": 32})
pattern = re.compile(r"(" + "|".join(expected) + r")_[a-f0-9]{32}_(\d+)")
selected = defaultdict(list)
with args.tasks.open(encoding="utf-8-sig", newline="") as handle:
    rows = sorted(csv.DictReader(handle), key=lambda row: Decimal(row["task_start(us)"].strip()))
for row in rows:
    match = pattern.fullmatch(row["kernel_name"])
    if match:
        selected[match[1]].append(int(match[2]))
assert set(selected) == set(expected)
operators = {}
for name, keys in selected.items():
    assert len(keys) == len(calls), (name, len(keys))
    mapping = defaultdict(list)
    for call, key in zip(calls, keys):
        mapping[key].append(call["case_id"])
    assert len(mapping) == expected[name], (name, len(mapping), expected[name])
    operators[name] = dict(key_count=len(mapping), calls=len(keys),
                           keys={str(key): mapping[key] for key in sorted(mapping)})
result = dict(arch=args.arch, cases=len(calls),
              task_csv_sha256=hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
              case_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              operators=operators)
args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(args.arch, {name: op["key_count"] for name, op in operators.items()})
