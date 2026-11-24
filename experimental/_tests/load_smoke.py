"""Check wheel registration in a fresh process, without allocating an NPU tensor."""
import json
from pathlib import Path
import sys

import fla_npu_experimental
from fla_npu_experimental import _stable

_stable.load()
import torch

expected = {"fla_npu_experimental_stable::" + op for op in _stable._OPS}
names = set(torch._C._dispatch_get_all_op_names())
actual = {name for name in names if name.startswith("fla_npu_experimental_stable::")}
if expected != actual:
    raise RuntimeError(f"Experimental dispatcher mismatch: {expected} != {actual}")
if any(name.startswith("fla_npu_stable::") for name in names) or "fla_npu" in sys.modules:
    raise RuntimeError("Stable Python package or dispatcher leaked into experimental process")
if torch.npu.is_initialized():
    raise RuntimeError("Host-only load check unexpectedly initialized NPU")
print(json.dumps({"package": str(Path(fla_npu_experimental.__file__).resolve()),
                  "registered_operators": sorted(actual), "npu_initialized": False,
                  "runtime_executed": False}, indent=2))
