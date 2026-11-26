"""Numerical evidence required for one passing model case (fresh or cached)."""


def passed(case, row, platform):
    if not isinstance(row, dict) or row.get("passed") is not True:
        return False
    if any(row.get(k) != value for k, value in case.items()):
        return False
    if case["suite"] == "kda":
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            return False
        checks = [metrics.get(k) for k in ("o", "final_state", "dq", "dk", "dv", "dbeta", "dg")]
    else:
        checks = [row.get("output")]
        if case["entry"] == "decode":
            checks.append(row.get("state"))
        elif platform == "a5":
            checks.append(row.get("route"))
    return all(isinstance(metric, dict) and metric.get("passed") is True for metric in checks)
