"""在实际 worker 进程校验冻结安装树和已加载动态库。"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

_ARCH22_IDENTITY = None

def _arch22_sha(path):
    h=hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def _arch22_runtime_identity():
    """由 worker 自报实际导入和加载路径；请求中的 SHA 单独记录。"""
    global _ARCH22_IDENTITY
    if _ARCH22_IDENTITY is not None:
        return _ARCH22_IDENTITY
    import fla_npu
    fla_npu.load_ascendc_opapi_libraries()
    module = Path(fla_npu.__file__).resolve()
    maps = Path("/proc/self/maps").read_text() if Path("/proc/self/maps").is_file() else ""
    paths = sorted({line.split()[-1] for line in maps.splitlines()
                    if ".so" in line and ("fla_npu" in line or "cust_opapi" in line)})
    libraries = [{"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size, "sha256": _arch22_sha(path)}
                 for path in paths if Path(path).is_file()]
    binding_path = os.environ.get("FINALIZE_BINDING_PATH")
    if not binding_path:
        raise RuntimeError("FINALIZE_BINDING_PATH is required for A2/A3 DUT identity")
    binding_file = Path(binding_path).resolve()
    measured_binding_sha = _arch22_sha(binding_file)
    if measured_binding_sha != os.environ.get("FINALIZE_BINDING_SHA256"):
        raise RuntimeError("DUT measured binding SHA differs from FINALIZE_BINDING_SHA256")
    binding = json.loads(binding_file.read_text(encoding="utf-8"))
    if not binding.get("frozen") or binding["source_commit"] != os.environ.get("FINALIZE_SOURCE_COMMIT"):
        raise RuntimeError("DUT binding is not frozen or source commit differs")
    site = Path(binding["site"]).resolve()
    expected_files = binding["installed_files"]
    actual_files = {str(path.relative_to(site)).replace(os.sep, "/"): path
                    for path in site.rglob("*") if path.is_file()
                    and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    if set(actual_files) != set(expected_files):
        raise RuntimeError("DUT installed binding tree membership differs from frozen tree")
    for relative, path in actual_files.items():
        expected = expected_files[relative]
        if path.stat().st_size != expected["bytes"] or _arch22_sha(path) != expected["sha256"]:
            raise RuntimeError(f"DUT installed file differs: {relative}")
    for path in [module, *(Path(entry["path"]) for entry in libraries)]:
        relative = path.relative_to(site).as_posix()
        if relative not in expected_files:
            raise RuntimeError(f"DUT loaded path absent from binding: {path}")
    if not any("cust_opapi" in entry["path"] for entry in libraries):
        raise RuntimeError("DUT /proc/self/maps contains no actual cust_opapi library")
    source = Path(binding["source_dir"])
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if head != binding["source_commit"] or status.strip():
        raise RuntimeError("DUT measured source HEAD/clean differs from frozen source")
    maps_file = Path(os.environ["FINALIZE_CPU_EVIDENCE"]) / "worker-identity" / str(os.getpid()) / "maps.txt"
    maps_file.parent.mkdir(parents=True, exist_ok=True)
    maps_file.write_text(maps, encoding="utf-8")
    measured = dict(binding_sha256=measured_binding_sha, source_commit=head, site=str(site),
                    package=dict(path=str(module), bytes=module.stat().st_size, sha256=_arch22_sha(module)),
                    custom_opapi=next(entry for entry in libraries if "cust_opapi" in entry["path"]),
                    libraries=libraries,
                    maps=dict(path=str(maps_file), bytes=maps_file.stat().st_size, sha256=_arch22_sha(maps_file)))
    _ARCH22_IDENTITY = dict(measured_identity=measured,
                fla_npu_file=str(module), fla_npu_file_sha256=_arch22_sha(module),
                source_commit_requested=os.getenv("FINALIZE_SOURCE_COMMIT"),
                binding_sha256_requested=os.getenv("FINALIZE_BINDING_SHA256"),
                source_commit_env=os.getenv("SOURCE_COMMIT"), binding_sha_env=os.getenv("BINDING_SHA"),
                measured_source_commit=head, measured_source_clean=not status.strip(),
                measured_binding_sha256=measured_binding_sha, measured_binding_path=str(binding_file),
                measured_site=str(site), verified_installed_file_count=len(actual_files),
                loaded_libraries=libraries, proc_maps=maps)
    return _ARCH22_IDENTITY
