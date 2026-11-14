"""Manage an experimental base with the common layer protocol, without importing NPU."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--source", type=Path)
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--remove-overlay", action="store_true")
    actions.add_argument("--shell-env", action="store_true")
    parser.add_argument("--opp-source", type=Path)
    parser.add_argument("--package-dir", type=Path,
                        default=Path(__file__).resolve().parent / "fla_npu_experimental")
    args = parser.parse_args(argv)
    package = args.package_dir.resolve()
    # A private namespace imports layout configuration without running package __init__.
    alias = "_experimental_management"
    module = types.ModuleType(alias)
    module.__path__ = [str(package)]
    sys.modules[alias] = module
    spec = importlib.util.spec_from_file_location(alias + "._layers", package / "_layers.py")
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    manager = layout.manager
    if args.shell_env:
        print(manager.shell_environment(package))
        return
    if args.source:
        result = manager.install(package, args.source, opp_source=args.opp_source)
    elif args.remove_overlay:
        manager.remove(package)
        result = {"state": "base"}
    else:
        result = manager.status(package)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
