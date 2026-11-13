"""Discover the closed experimental operator set; reuse the common CLI validator."""
import importlib.util
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
RESERVED = {"scripts", "tests", "cmake", "torch_custom", "docs"}
spec = importlib.util.spec_from_file_location("_common_build_ops", ROOT / "scripts/check_build_ops.py")
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def discover_supported_ops(repo_root=ROOT):
    root = Path(repo_root).resolve()
    base = root / "experimental"
    if not base.is_dir():
        return []
    if base.resolve() != base:
        raise ValueError(f"Experimental root must not redirect outside its source location: {base}")
    base = base / "ops"
    if not base.is_dir():
        return []
    if base.resolve() != base:
        raise ValueError(f"Experimental ops root must not redirect outside its source location: {base}")
    operators = []
    for entry in base.iterdir():
        if not entry.is_dir() or entry.name.startswith("_") or entry.name in RESERVED:
            continue
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", entry.name) or entry.name in {"all", "ALL"}:
            raise ValueError(f"Invalid or reserved experimental operator name: {entry.name}")
        marker = entry / "CMakeLists.txt"
        if not entry.resolve().is_relative_to(base) or not marker.resolve().is_relative_to(base):
            raise ValueError(f"Experimental source escapes its tree: {entry}")
        if not marker.is_file():
            raise ValueError(f"Experimental operator entry point is missing: {marker}")
        operators.append(entry.name)
    return sorted(operators)


def main(argv=None):
    parser = common.build_arg_parser()
    parser.set_defaults(repo_root=str(ROOT))
    args = parser.parse_args(argv)
    try:
        supported = discover_supported_ops(args.repo_root)
    except ValueError as exc:
        parser.error(str(exc))
    if args.list:
        import json
        print(json.dumps({"count": len(supported), "operators": supported}, indent=2)
              if args.json else "\n".join(supported))
        return 0
    if args.ops is None:
        parser.error("--ops is required unless --list is given")
    common.LIST_COMMAND = "bash build.sh --experimental --list-ops"
    common.SCRIPT_LIST_COMMAND = "python3 experimental/scripts/check_build_ops.py --list"
    return 0 if common.validate_ops_filter(args.ops, source=args.source, origin=args.origin,
                                         supported=supported) else 1


if __name__ == "__main__":
    sys.exit(main())
