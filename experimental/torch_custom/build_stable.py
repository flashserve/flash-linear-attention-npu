"""Compile only the explicitly selected experimental registration unit."""
import importlib.util
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("_shared_stable_builder", ROOT / "torch_custom/fla_npu/csrc/build_stable.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
CSRC_INCLUDE = shared.CSRC_INCLUDE
torch_lib_dir = shared.torch_lib_dir
torch_include_dirs = shared.torch_include_dirs

def compile_launcher(units, output, stamp, *, compiler="g++", no_debug_probe=True, extra_flags=()):
    """Compile an explicit registration unit and runtime; callers own identity/stamps."""
    torch_lib = torch_lib_dir()
    cmd = [
        compiler,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        "-fvisibility=hidden",
        "-fno-gnu-unique",
        *(["-DFLA_STABLE_NO_DEBUG_PROBE"] if no_debug_probe else []),
        f'-DFLA_STABLE_SOURCE_HASH="{stamp}"',
        "-I", str(CSRC_INCLUDE),
        *[f"-I{path}" for path in torch_include_dirs()],
        *extra_flags,
        *[str(path) for path in units],
        # The runtime (dlopen + symbol cache) ships with the launcher.
        "-L", torch_lib,
        "-Wl,--no-as-needed", "-ltorch_cpu", "-lc10", "-ltorch",
        "-ldl",
        "-o", str(output),
    ]
    print(" ".join(cmd))
    if shutil.which(compiler) is None:
        raise SystemExit(f"compiler {compiler!r} not found")
    subprocess.run(cmd, check=True)
    print(f"built {output}")
