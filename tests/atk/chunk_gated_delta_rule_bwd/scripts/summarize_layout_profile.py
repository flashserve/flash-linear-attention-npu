import argparse
import csv
import json
from collections import Counter
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('summary', type=Path)
p.add_argument('manifest', type=Path)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
with a.summary.open(newline='', encoding='utf-8-sig') as handle:
    rows = sorted(csv.DictReader(handle), key=lambda row: float(row['Task Start Time(us)']))
groups, current = [], []
for row in rows:
    name = row['Op Name']
    if name.startswith('aclnnChunkGatedDeltaRuleBwd_'):
        current.append(row)
    elif name.startswith('aclnnAdd') and current:
        groups.append(current)
        current = []
assert not current, 'last call has no Add marker'
manifest = json.loads(a.manifest.read_text())
assert len(groups) == len(manifest), (len(groups), len(manifest))
result = []
for call, tasks in zip(manifest, groups):
    if not call['measured']:
        continue
    types = Counter(task['OP Type'] for task in tasks)
    assert types['ChunkGatedDeltaRuleBwdFinalize'] == 1, types
    item = dict(case_id=call['case_id'], name=call['name'], layout=call['layout'],
                total_us=sum(float(t['Task Duration(us)']) for t in tasks),
                tasks=len(tasks), transposes=types['Transpose'])
    for op_type in types:
        item[op_type + '_us'] = sum(float(t['Task Duration(us)']) for t in tasks
                                   if t['OP Type'] == op_type)
    result.append(item)
columns = list(dict.fromkeys(key for row in result for key in row))
with a.out.open('w', newline='', encoding='utf-8') as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(result)
print(json.dumps(result, indent=2))
