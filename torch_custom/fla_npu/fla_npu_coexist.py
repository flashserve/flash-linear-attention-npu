"""Coordinate two independently installed wheel vendors before CANN initialization."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
from threading import RLock

from fla_npu_wheel_overlay import abi_module


def _inventory(vendors, abi):
    """Inspect the selected OPP files, including run-only layers without adapters."""
    types, symbols = {}, {}
    for vendor in vendors:
        configs = sorted((vendor / "op_impl/ai_core/tbe/config").glob("*/aic-*-ops-info.json"))
        if not configs:
            raise ValueError(f"Missing CANN opType inventory in {vendor}")
        configs.extend((vendor / "op_impl/ai_core/tbe/kernel/config").glob("*/binary_info_config.json"))
        for config in configs:
            data = json.loads(config.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"Invalid CANN opType inventory in {config}")
            for name in data:
                types.setdefault(name, vendor)
        for name in abi.elf_exported_symbols(vendor / abi.LIBRARY):
            symbols.setdefault(name, vendor)
    return types, symbols


class InstalledPair:
    def __init__(self, formal, experimental, formal_vendors, select_experimental):
        self.formal = formal
        self.experimental = experimental
        self.vendors = {formal: tuple(formal_vendors), experimental: ()}
        self.lock = RLock()
        self.active = set()
        self.errors = {}
        self.conflicts = []
        abi = abi_module(formal)
        inventories = {}
        for root in self.vendors:
            try:
                if root == experimental:
                    self.vendors[root] = tuple(select_experimental(root))
                inventories[root] = _inventory(self.vendors[root], abi)
            except (OSError, ValueError, RuntimeError) as exc:
                self.errors[root] = str(exc)
        if not self.errors:
            for kind, first, second in zip(("CANN opType", "ACLNN"), inventories[formal], inventories[experimental]):
                for name in sorted(first.keys() & second.keys()):
                    self.conflicts.append(f"{kind} {name}: {first[name]} and {second[name]}")

    def prepare(self, vendors):
        external = []
        for value in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep):
            if value and not any(Path(value).resolve().is_relative_to(root / "opp") for root in self.vendors):
                external.append(value)
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join([*map(str, vendors), *external])

    def startup(self):
        if self.conflicts:
            self.prepare(())
        else:
            self.prepare([v for root, vendors in self.vendors.items() if root not in self.errors for v in vendors])

    @contextmanager
    def activating(self, side):
        # Serialize the two package import bodies, including native side effects.
        with self.lock:
            if side in self.errors:
                raise ImportError(f"Cannot enable {side}: {self.errors[side]}")
            if self.conflicts and self.active.difference({side}):
                raise ImportError("Conflicting operator vendors:\n" + "\n".join(self.conflicts))
            old_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
            if self.conflicts:
                self.prepare(self.vendors[side])
            try:
                yield
            except BaseException:
                if old_path is None:
                    os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
                else:
                    os.environ["ASCEND_CUSTOM_OPP_PATH"] = old_path
                raise
            self.active.add(side)
