"""Resolve one shared KDA smoke plan for A2, A5 and the CPU reference."""

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from random import Random


FIXED_BOUNDS = {"glm-flash": -5.0, "case1": -2.5, "bailing-v3-h2": -0.01}
# Integer hundredths ensure that rounding cannot turn a valid input into zero.
BOUND_RANGES = ((-500, -301), (-300, -101), (-100, -1))


def daily_plan(catalog, day):
    cases = catalog["cases"]
    ids = [case["id"] for case in cases]
    if len(set(ids)) != 39 or len(ids) != 39 or not FIXED_BOUNDS.keys() <= set(ids):
        raise ValueError("Expected the 39 model cases including the three fixed cases")
    # Use the same seed on every runner, independent of process, PR and retry.
    seed = int(day.strftime("%Y%m%d"))
    rng = Random(seed)
    rotating_ids = sorted(set(ids) - FIXED_BOUNDS.keys())
    rng.shuffle(rotating_ids)
    bounds = dict(FIXED_BOUNDS)
    for group, (start, end) in enumerate(BOUND_RANGES):
        samples = rng.sample(range(start, end + 1), 12)
        for case_id, hundredths in zip(rotating_ids[group * 12:(group + 1) * 12], samples):
            bounds[case_id] = hundredths / 100
    resolved = []
    for case in cases:
        case_id = case["id"]
        # Keep tensor inputs fixed across days when exercising different bounds.
        tensor_seed = int.from_bytes(hashlib.sha256(case_id.encode()).digest()[:4], "big")
        resolved.append(dict(case, lower_bound=bounds[case_id], tensor_seed=tensor_seed))
    return {
        "date": day.isoformat(), "sampling_seed": seed, "safe_gate": True,
        "head_dim": catalog["head_dim"], "cases": resolved, "cu_seqlens": catalog["cu_seqlens"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, help="Beijing run date (YYYY-MM-DD)")
    args = parser.parse_args()
    day = args.date or datetime.now(timezone(timedelta(hours=8))).date()
    catalog = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
    print(json.dumps(daily_plan(catalog, day), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
