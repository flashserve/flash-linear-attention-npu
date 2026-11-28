import json
from functools import lru_cache


@lru_cache(maxsize=4)
def _manifest_profiles(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["id"]: json.loads(next(entry["range_values"] for entry in row["inputs"]
                                  if isinstance(entry, dict) and entry.get("name") == "case_spec"))
        for row in rows
    }


def case_profiles(task_result, path, legacy):
    config = getattr(task_result, "case_config", None)
    name = config.get("name") if isinstance(config, dict) else getattr(config, "name", None)
    prefix = path.stem.removesuffix("_10000") + "_"
    if name and name.startswith(prefix):
        return _manifest_profiles(path)
    return legacy
