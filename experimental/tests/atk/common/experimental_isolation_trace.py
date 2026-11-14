"""Opt-in evidence after a real experimental NPU call; used only by manual validation."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

_seen = set()


def record_experimental_execution(device):
    directory = os.environ.get("FLA_A2_TRACE_DIR")
    if not directory or os.getpid() in _seen:
        return
    import torch
    import fla_npu_experimental
    from fla_npu_experimental import _loader
    owner = fla_npu_experimental.fused_recurrent_rwkv8.__module__.rsplit('.', 1)[0]
    _stable = sys.modules[owner + '._stable']
    torch.npu.synchronize(device)
    package = Path(fla_npu_experimental.__file__).resolve().parent
    expected_libraries = {Path(p).resolve() for p in _loader._LIBRARIES}
    expected = Path(_loader._LIBRARIES[0]).resolve()
    libraries = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = Path(fields[5].removesuffix(' (deleted)')).resolve()
            if path.name.startswith(('libcust_', 'libfla_npu_', 'libopapi')):
                libraries.add(path)
    custom = {p for p in libraries if p.name == 'libcust_opapi.so'}
    registered = sorted(n for n in torch._C._dispatch_get_all_op_names()
                        if n.startswith(('fla_npu_stable::', 'fla_npu_experimental_')))
    if custom != expected_libraries or 'fla_npu' in sys.modules or any(n.startswith('fla_npu_stable::') for n in registered):
        raise AssertionError(f'Experimental NPU execution used mixed libraries or registrations: {libraries}, {registered}')
    probe = _stable._LIB.fla_npu_experimental_queue_enqueue_available
    probe.restype = ctypes.c_int32
    report = {'pid': os.getpid(), 'device': str(device), 'package': str(package),
              'custom_opapi': str(expected), 'wrapper_module': owner,
              'opapi_sha256': hashlib.sha256(expected.read_bytes()).hexdigest(),
              'libraries': sorted(map(str, libraries)), 'dispatcher': registered,
              'launch_mode': 'queue' if probe() else 'inline', 'synchronized_after_real_call': True}
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / f'experimental-{os.getpid()}.json').write_text(json.dumps(report, indent=2))
    _seen.add(os.getpid())
