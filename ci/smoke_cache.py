"""Reusable smoke evidence. The runtime fingerprint is calculated once after the OPP is configured."""
import gzip
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import zlib


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with path.open("rb") as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runtime_fingerprint():
    image = os.environ["CI_SMOKE_IMAGE_ID"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("missing Docker image ID")
    root = Path(__file__).resolve().parent.parent
    package = Path(util.find_spec("fla_npu").origin).resolve().parent
    opp = Path(os.environ["FLA_NPU_OPP_PATH"]).resolve(strict=True)
    files = {}
    visited = {}

    def add(path):
        resolved = path.resolve(strict=True)
        if resolved not in visited:
            visited[resolved] = file_hash(resolved)
        files[str(path)] = visited[resolved]

    def scan(directory):
        # Follow installed vendor symlinks; reject directory cycles instead of omitting inputs.
        def visit(path, parents):
            real = path.resolve(strict=True)
            if real in parents:
                raise ValueError(f"recursive OPP link at {path}")
            for item in path.iterdir():
                if item.name == "__pycache__" or item.suffix == ".pyc":
                    continue
                if item.is_dir():
                    visit(item, parents | {real})
                elif item.is_file():
                    add(item)
                else:
                    raise ValueError(f"unresolved runtime input: {item}")
        visit(directory, set())

    scan(package)
    if not opp.is_relative_to(package):
        scan(opp)
    for variable in ("FLA_NPU_OP_API_LIBS", "FLA_NPU_OP_API_LIB", "ASCEND_CUSTOM_OPP_PATH"):
        for entry in filter(None, os.getenv(variable, "").split(os.pathsep)):
            p = Path(entry).resolve(strict=True)
            if p.is_file():
                add(p)
            elif not (p.is_relative_to(opp) or p.is_relative_to(package)):
                scan(p)
    for directory in ("ci/model_cases", "tests/atk/smoke", "examples", "fla"):
        for p in (root / directory).rglob("*.py"):
            add(p)
    add(Path(__file__).resolve())
    add(Path("/usr/local/Ascend/driver/version.info"))
    versions = {name: metadata.version(name) for name in ("atk", "torch", "torch_npu")}
    versions["fla"] = metadata.version("flash-linear-attention-npu-" + os.environ["CI_ACCURACY_PLATFORM"])
    env = {k: os.environ.get(k) for k in ("CI_SOC", "CI_ACCURACY_PLATFORM", "NPU_SELECTED_NAME",
           "FLA_NPU_OP_API_LIBS", "FLA_NPU_OP_API_LIB", "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH",
           "PYTHONPATH", "ATEN_CPU_CAPABILITY", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    if not env["NPU_SELECTED_NAME"]:
        raise ValueError("missing NPU model")
    board = subprocess.check_output(
        ["npu-smi", "info", "-t", "board", "-i", os.environ["NPU_SELECTED_DEVICE"]],
        text=True, timeout=15, stderr=subprocess.DEVNULL)
    hardware = dict((k.strip(), v.strip()) for line in board.splitlines() if ":" in line
                    for k, v in [line.split(":", 1)] if k.strip() in {
                        "Chip Name", "Chip Version", "Firmware Version", "Software Version", "Board ID"})
    if not hardware:
        raise ValueError("missing NPU board information")
    cpu = sorted({line.strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                  if line.partition(":")[0].strip() in ("model name", "flags", "Features", "CPU part", "microcode")})
    return digest([image, platform.machine(), platform.release(), sys.version, hardware, cpu, env, versions, files])


class ResultCache:
    def __init__(self):
        directory = os.environ.get("CI_SMOKE_CACHE_DIR")
        self.runtime = os.environ.get("CI_SMOKE_RUNTIME", "")
        self.root = Path(directory) if directory and re.fullmatch(r"[0-9a-f]{64}", self.runtime) else None

    def key(self, kind, inputs):
        return digest(["smoke-v1", self.runtime, kind, inputs])

    def get(self, key):
        if self.root is None:
            return None
        try:
            with gzip.open(self.root / (key + ".json.gz"), "rt", encoding="utf8") as f:
                entry = json.load(f)
            if entry["key"] != key or entry["checksum"] != digest([key, entry["origin"], entry["value"]]):
                raise ValueError("checksum mismatch")
            return entry
        except FileNotFoundError:
            return None
        except (OSError, EOFError, ValueError, KeyError, TypeError, zlib.error) as exc:
            print(f"[smoke-cache] Ignoring invalid {key[:12]}: {exc}", file=sys.stderr)
            return None

    def put(self, key, value):
        if self.root is None:
            return
        origin = {k: os.environ.get(v) for k, v in (
            ("head_sha", "CI_ACCURACY_HEAD_SHA"), ("run_id", "CI_ACCURACY_RUN_ID"),
            ("attempt", "CI_ACCURACY_RUN_ATTEMPT"))}
        tmp = None
        try:
            entry = dict(key=key, value=value, origin=origin, checksum=digest([key, origin, value]))
            self.root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as raw:
                tmp = Path(raw.name)
                with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=3) as f:
                    f.write(json.dumps(entry, ensure_ascii=False, allow_nan=False).encode("utf8"))
            # The host runner uploads evidence written by the container user.
            tmp.chmod(0o644)
            tmp.replace(self.root / (key + ".json.gz"))
            marker = os.environ.get("CI_SMOKE_CACHE_DIRTY")
            if marker:
                Path(marker).touch()
        except (ValueError, OSError) as exc:
            print(f"[smoke-cache] Unable to save {key[:12]}: {exc}", file=sys.stderr)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        print(runtime_fingerprint())
    except (OSError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        print(f"[smoke-cache] Runtime not identified; running fresh: {error}", file=sys.stderr)
        sys.exit(1)
