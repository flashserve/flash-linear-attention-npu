"""Opt-in evidence after a real experimental NPU call; never used by stable CI."""
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
    from fla_npu_experimental import _stable
    torch.npu.synchronize(device)
    package = Path(fla_npu_experimental.__file__).resolve().parent
    expected = (package / "opp/vendors/fla_npu_experimental_transformer/op_api/lib/libcust_opapi.so").resolve()
    libraries = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = Path(fields[5].removesuffix(' (deleted)')).resolve()
            if path.name.startswith(('libcust_', 'libfla_npu_', 'libopapi')):
                libraries.add(path)
    custom = {p for p in libraries if p.name == 'libcust_opapi.so'}
    registered = sorted(n for n in torch._C._dispatch_get_all_op_names()
                        if n.startswith(('fla_npu_stable::', 'fla_npu_experimental_stable::')))
    if custom != {expected} or 'fla_npu' in sys.modules or any(n.startswith('fla_npu_stable::') for n in registered):
        raise AssertionError(f'Experimental NPU execution used mixed libraries or registrations: {libraries}, {registered}')
    probe = _stable._LIB.fla_npu_experimental_queue_enqueue_available
    probe.restype = ctypes.c_int32
    report = {'pid': os.getpid(), 'device': str(device), 'package': str(package),
              'custom_opapi': str(expected), 'opapi_sha256': hashlib.sha256(expected.read_bytes()).hexdigest(),
              'libraries': sorted(map(str, libraries)), 'dispatcher': registered,
              'launch_mode': 'queue' if probe() else 'inline', 'synchronized_after_real_call': True}
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / f'experimental-{os.getpid()}.json').write_text(json.dumps(report, indent=2))
    _seen.add(os.getpid())


def record_stable_execution(device):
    directory = os.environ.get("FLA_A2_TRACE_DIR")
    if not directory or os.getpid() in _seen:
        return
    import torch
    import fla_npu
    torch.npu.synchronize(device)
    package = Path(fla_npu.__file__).resolve().parent
    libraries = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = Path(fields[5].removesuffix(' (deleted)')).resolve()
            if path.name.startswith(('libcust_', 'libfla_npu_', 'libopapi')):
                libraries.add(path)
    custom = {p for p in libraries if p.name == 'libcust_opapi.so'}
    expected = (package / 'opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so').resolve()
    registered = sorted(n for n in torch._C._dispatch_get_all_op_names()
                        if n.startswith(('fla_npu_stable::', 'fla_npu_experimental_stable::')))
    if custom != {expected} or 'fla_npu_experimental' in sys.modules or any(n.startswith('fla_npu_experimental_stable::') for n in registered):
        raise AssertionError(f'Stable NPU execution used mixed libraries or registrations: {libraries}, {registered}')
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / f'stable-{os.getpid()}.json').write_text(json.dumps({
        'pid': os.getpid(), 'device': str(device), 'package': str(package),
        'libraries': sorted(map(str, libraries)), 'dispatcher': registered,
        'opapi_sha256': hashlib.sha256(expected.read_bytes()).hexdigest(),
        'synchronized_after_real_call': True}, indent=2))
    _seen.add(os.getpid())
