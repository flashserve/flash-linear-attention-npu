"""Persist CPU reference tensors independently of the operator under test."""
import hashlib
import inspect
import json
import os
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

import torch
from filelock import FileLock


@lru_cache(maxsize=None)
def _reference_version(reference):
    source = Path(inspect.getfile(reference)).read_bytes()
    return hashlib.sha256(source).hexdigest(), reference.__qualname__


def _input_key(reference, inputs):
    digest = hashlib.sha256()

    def record(value):
        data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)

    record([1, _reference_version(reference), torch.__version__, torch.__config__.show(),
            torch.backends.cpu.get_cpu_capability(), torch.get_num_threads(),
            torch.get_float32_matmul_precision(), torch.backends.mkldnn.enabled,
            str(torch.get_default_dtype()), os.environ.get("MKL_CBWR")])
    for name, value in sorted(inputs.items()):
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise ValueError("CPU reference inputs must reside on CPU")
            record([name, str(value.dtype), list(value.shape), list(value.stride())])
            raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw))
        else:
            record([name, value])
    return digest.hexdigest()


def _load(path):
    if not path.is_file():
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                return None
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (EOFError, RuntimeError, ValueError, zipfile.BadZipFile):
        return None


def _restore_outputs(outputs, inputs, state_names):
    if not state_names:
        return outputs
    restored = [outputs[0]]
    if len(outputs) != len(state_names) + 1:
        raise ValueError("Unexpected CPU reference output count")
    for name, value in zip(state_names, outputs[1:]):
        inputs[name].copy_(value)
        restored.append(inputs[name])
    return tuple(restored)


def cached_cpu_outputs(reference, inputs, directory, state_names=()):
    key = _input_key(reference, inputs)
    path = Path(directory) / key[:2] / (key + ".pt")
    outputs = _load(path)
    if outputs is not None:
        return _restore_outputs(outputs, inputs, state_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        outputs = _load(path)
        if outputs is not None:
            return _restore_outputs(outputs, inputs, state_names)
        outputs = reference(**inputs)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            name = tmp.name
        try:
            torch.save(outputs, name)
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
        return outputs
