import argparse
import csv
from pathlib import Path
import json

p = argparse.ArgumentParser()
p.add_argument('old', type=Path)
p.add_argument('new', type=Path)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
old, new = [list(csv.DictReader(path.open())) for path in (a.old, a.new)]
assert len(old) == len(new) == 16
result = []
for before, after in zip(old, new):
    assert (before['case_id'], before['layout']) == (after['case_id'], after['layout'])
    seq = before['layout'] in ('BSND', 'TND')
    assert int(before['transposes']) - int(after['transposes']) == (6 if seq else 0)
    baseline, current = float(before['total_us']), float(after['total_us'])
    row = dict(case_id=before['case_id'], name=before['name'], layout=before['layout'],
               before_us=baseline, after_us=current, improvement_pct=(baseline-current)/baseline*100,
               before_tasks=before['tasks'], after_tasks=after['tasks'],
               before_transposes=before['transposes'], after_transposes=after['transposes'])
    result.append(row)
with a.out.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(result[0]))
    writer.writeheader()
    writer.writerows(result)
print(json.dumps(result, indent=2))
