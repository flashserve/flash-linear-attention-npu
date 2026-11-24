"""Opt-in experimental operators; configure before importing torch_npu/acl."""
import importlib
import importlib.util
from pathlib import Path
import sys

from . import _loader
from ._exports import EXPORTS

_loader.configure()
__all__ = list(EXPORTS)
for _name, _module in EXPORTS.items():
    globals()[_name] = getattr(importlib.import_module("." + _module, __name__), _name)

# An adapter component owns its wrappers and dispatcher schema. Reuse the base
# loader's immutable OPP selection, without importing the base wrappers again.
if _loader._PLAN and _loader._PLAN["layer"] and not globals().get("_NAMESPACE"):
    from . import _layers
    _path = Path(_loader._PLAN["layer"]) / _layers.manager.PACKAGE
    _alias = "_fla_npu_experimental_layer"
    _spec = importlib.util.spec_from_file_location(_alias, _path / "__init__.py",
                                                  submodule_search_locations=[str(_path)])
    _layer = importlib.util.module_from_spec(_spec)
    _layer._NAMESPACE = "fla_npu_experimental_overlay_stable"
    sys.modules[_alias] = _layer
    sys.modules[_alias + "._loader"] = _loader
    sys.modules[_alias + "._layers"] = _layers
    _spec.loader.exec_module(_layer)
    for _dispatch in _loader._PLAN["exports"]:
        _name = _dispatch.removeprefix("npu_")
        globals()[_name] = getattr(_layer, _name)
        if _name not in __all__:
            __all__.append(_name)
