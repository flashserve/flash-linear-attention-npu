"""Coordinate two independently installed wheel vendors before CANN initialization."""
from contextlib import contextmanager
from importlib import import_module
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


def _adapter_exports(package, manager):
    """Read public names from base and enabled wheel layers without importing operators."""
    exports = set(manager.manifest(package)["exports"])
    snapshot = manager.active(package)
    if snapshot is not None:
        adapters, _ = manager.component_vendors(package, snapshot)
        for adapter in adapters:
            exports.update(manager.manifest(adapter / manager.PACKAGE)["exports"])
    return exports


class InstalledPair:
    def __init__(self, formal, experimental, formal_vendors, select_experimental, layer_manager=None):
        self.formal = formal
        self.experimental = experimental
        self.vendors = {formal: tuple(formal_vendors), experimental: ()}
        self.lock = RLock()
        self.active = set()
        self.errors = {}
        self.conflicts = []
        self.experimental_exports = set()
        abi = abi_module(formal)
        inventories = {}
        for root in self.vendors:
            try:
                if root == experimental:
                    self.vendors[root] = tuple(select_experimental(root))
                    if layer_manager is not None:
                        self.experimental_exports = _adapter_exports(root, layer_manager(root))
                inventories[root] = _inventory(self.vendors[root], abi)
            except (OSError, ValueError, RuntimeError, SyntaxError) as exc:
                self.errors[root] = str(exc)
        if not self.errors:
            for kind, first, second in zip(("CANN opType", "ACLNN"), inventories[formal], inventories[experimental]):
                for name in sorted(first.keys() & second.keys()):
                    self.conflicts.append(f"{kind} {name}: {first[name]} and {second[name]}")

    def expose_experimental_ops(self, namespace):
        """Resolve the selected peer wrapper only when its public name is requested."""
        aliases = {}
        for name in sorted(self.experimental_exports):
            short = name.removeprefix("npu_")
            for alias in (name, short):
                if alias not in namespace:
                    aliases[alias] = short

        def resolve(name):
            if name not in aliases:
                raise AttributeError(f"module {namespace['__name__']!r} has no attribute {name!r}")
            peer = import_module("fla_npu_experimental")
            wrapper = getattr(peer, aliases[name])
            namespace[name] = wrapper
            return wrapper

        namespace["__getattr__"] = resolve
        namespace["__dir__"] = lambda: sorted(namespace.keys() | aliases.keys())
        namespace["__all__"].extend(aliases)

    def prepare(self, vendors):
        external = []
        for value in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep):
            if value and not any(Path(value).resolve().is_relative_to(root / "opp") for root in self.vendors):
                external.append(value)
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join([*map(str, vendors), *external])

    def startup(self):
        if self.conflicts:
            # A process may initialize CANN without importing either Python package.
            self.prepare(self.vendors[self.formal])
        else:
            self.prepare([v for root, vendors in self.vendors.items() if root not in self.errors for v in vendors])

    def _loaded_formal_vendor(self):
        """CANN loads the selected vendor's native libraries when it registers OPP."""
        maps = Path("/proc/self/maps")
        if not maps.exists():  # The supported CANN runtime is Linux.
            return None
        roots = tuple(str(v) + "/" for v in self.vendors[self.formal])
        with maps.open(encoding="utf-8", errors="replace") as mappings:
            for entry in mappings:
                fields = entry.rstrip("\n").split(maxsplit=5)
                if len(fields) == 6 and "x" in fields[1] and fields[5].startswith(roots):
                    return fields[5]
        return None

    @contextmanager
    def activating(self, side):
        # Serialize the two package import bodies, including native side effects.
        with self.lock:
            if side in self.errors:
                raise ImportError(f"Cannot enable {side}: {self.errors[side]}")
            if self.conflicts and self.active.difference({side}):
                raise ImportError("Conflicting operator vendors:\n" + "\n".join(self.conflicts))
            if self.conflicts and side == self.experimental and not self.active:
                loaded = self._loaded_formal_vendor()
                if loaded:
                    raise ImportError(
                        f"The formal vendor is already loaded ({loaded}). To use the experimental "
                        "vendor, import fla_npu_experimental before initializing the NPU in a new process.\n"
                        "Conflicting operator vendors:\n" + "\n".join(self.conflicts))
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
