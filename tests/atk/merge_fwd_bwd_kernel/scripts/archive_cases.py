"""Archive ATK-generated cases with stable seed and identical-input dedup."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    op = root.name
    source = root / "result" / op / "json" / ("all_" + op + ".json")
    cases = json.loads(source.read_text())
    seen, result = set(), []
    for case in cases:
        key = json.dumps(case["inputs"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        case["id"], case["default_seed"] = len(result), 20261030
        result.append(case)
    target = root / ("atk_" + op + ".json")
    target.write_text("[\n" + ",\n".join(json.dumps(case, ensure_ascii=False, separators=(",", ":")) for case in result) + "\n]\n")
    target.chmod(0o644)
    print(f"{len(cases)} generated -> {len(result)} unique cases: {target.name}")


if __name__ == "__main__":
    main()
