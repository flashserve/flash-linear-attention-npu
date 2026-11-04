"""Archive ATK-generated Phase A candidates with static deduplication and a fixed seed."""
import json
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parents[1]
    op = directory.name
    source = directory / "result" / op / "json" / ("all_" + op + ".json")
    generated = json.loads(source.read_text())
    seen, selected = set(), []
    for case in generated:
        key = json.dumps(case["inputs"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        case["default_seed"] = 20261030
        selected.append(case)
    selected = selected[:1]
    target = directory / ("atk_" + op + ".json")
    target.write_text("[\n" + ",\n".join(json.dumps(case, ensure_ascii=False, separators=(",", ":")) for case in selected) + "\n]\n")
    target.chmod(0o644)
    print(f"{op}: {len(generated)} generated -> {len(selected)} retained candidates, seed=20261030")


if __name__ == "__main__":
    main()
