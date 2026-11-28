"""Manage an experimental base with the common layer protocol, without importing NPU."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types


def layer_manager(package):
    # A private namespace imports layout configuration without running package __init__.
    alias = "_experimental_management"
    module = types.ModuleType(alias)
    module.__path__ = [str(package)]
    sys.modules[alias] = module
    spec = importlib.util.spec_from_file_location(alias + "._layers", package / "_layers.py")
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    return layout.manager


def selected_vendors(package):
    manager = layer_manager(package)
    vendors = manager.environment_vendors(package)
    return vendors if vendors is not None else [package / "opp/vendors" / manager.BASE]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--source", type=Path)
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--remove-overlay", action="store_true")
    actions.add_argument("--shell-env", action="store_true")
    parser.add_argument("--package-dir", type=Path,
                        default=Path(__file__).resolve().parent / "fla_npu_experimental")
    args = parser.parse_args(argv)
    package = args.package_dir.resolve()
    manager = layer_manager(package)
    if args.shell_env:
        print(manager.shell_environment(package))
        return
    if args.source:
        try:
            result = manager.install(package, args.source)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.exit(1, f"{exc}\n")
    elif args.remove_overlay:
        manager.remove(package)
        result = {"state": "base"}
    else:
        result = manager.status(package)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
