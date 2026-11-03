"""在实际 worker 中校验冻结安装树、目标符号和调用后的 tiling 映射。"""
import ctypes
import hashlib
import json
import os
import subprocess
from pathlib import Path

_ARCH22_IDENTITY = None
_ARCH22_SNAPSHOT_SEQUENCE = 0
_ARCH22_DLADDR = None
_ARCH22_VERIFIED_INSTALLS = {}
_ARCH22_MAP_SNAPSHOTS = {}
_TARGET_SYMBOLS = (
    "aclnnChunkGatedDeltaRuleBwdFinalizeGetWorkspaceSize",
    "aclnnChunkGatedDeltaRuleBwdFinalize",
)
_VENDOR = "vendors/fla_npu_transformer"
_EXTERNAL_REGISTRATION_FILES = {
    "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so",
    "op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so",
}


def _arch22_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _arch22_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _arch22_file_record(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise RuntimeError(f"DUT loaded file is absent: {path}")
    return dict(path=str(path), bytes=path.stat().st_size, sha256=_arch22_sha(path))


def _arch22_bound_file(path, site, expected_files):
    path = Path(path).resolve()
    if not _arch22_within(path, site):
        raise RuntimeError(f"DUT target implementation is outside the frozen site: {path}")
    relative = path.relative_to(site).as_posix()
    if relative not in expected_files:
        raise RuntimeError(f"DUT loaded path absent from binding: {path}")
    record = _arch22_file_record(path)
    expected = expected_files[relative]
    if record["bytes"] != expected["bytes"] or record["sha256"] != expected["sha256"]:
        raise RuntimeError(f"DUT loaded file differs from frozen binding: {relative}")
    return record


def _arch22_read_maps():
    path = Path("/proc/self/maps")
    if not path.is_file():
        raise RuntimeError("DUT identity requires actual /proc/self/maps")
    return path.read_text()


def _arch22_loaded_libraries(maps, site, expected_files):
    package, external = {}, {}
    opp_value = os.environ.get("ASCEND_OPP_PATH")
    opp_root = Path(opp_value).resolve() if opp_value else None
    external_vendor = opp_root / _VENDOR if opp_root is not None else None
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or ".so" not in fields[5]:
            continue
        mapped = fields[5]
        deleted = mapped.endswith(" (deleted)")
        path = Path(mapped[:-10] if deleted else mapped).resolve()
        if _arch22_within(path, site):
            relevant = True
        else:
            relevant = "fla_npu" in str(path) or "cust_opapi" in path.name
        if not relevant:
            continue
        if deleted:
            raise RuntimeError(f"DUT identity cannot verify a deleted mapped library: {path}")
        if _arch22_within(path, site):
            package[str(path)] = _arch22_bound_file(path, site, expected_files)
            continue
        # CANN 默认 OPP 可以在新包之前装入这两个注册/proto 库；它们不是目标实现证明。
        relative = path.relative_to(external_vendor).as_posix() if (
            external_vendor is not None and _arch22_within(path, external_vendor)) else None
        if relative not in _EXTERNAL_REGISTRATION_FILES:
            raise RuntimeError(f"DUT unexpected external FLA/custom-opapi library: {path}")
        record = _arch22_file_record(path)
        record.update(cann_opp_root=str(opp_root), classification="CANN default vendor registration/prototype",
                      target_implementation=False)
        external[str(path)] = record
    return list(package.values()), list(external.values())


def _arch22_dladdr(symbol):
    global _ARCH22_DLADDR
    if _ARCH22_DLADDR is None:
        try:
            library = ctypes.CDLL("libdl.so.2")
        except OSError:
            library = ctypes.CDLL(None)

        class DlInfo(ctypes.Structure):
            _fields_ = [("dli_fname", ctypes.c_char_p), ("dli_fbase", ctypes.c_void_p),
                        ("dli_sname", ctypes.c_char_p), ("dli_saddr", ctypes.c_void_p)]

        function = library.dladdr
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        function.restype = ctypes.c_int
        _ARCH22_DLADDR = (library, function, DlInfo)
    _, function, info_type = _ARCH22_DLADDR
    address = ctypes.cast(symbol, ctypes.c_void_p).value
    info = info_type()
    if not address or not function(ctypes.c_void_p(address), ctypes.byref(info)) or not info.dli_fname:
        raise RuntimeError("DUT dladdr could not identify the resolved target symbol")
    return dict(path=str(Path(os.fsdecode(info.dli_fname)).resolve()), address=hex(address),
                resolved_name=os.fsdecode(info.dli_sname) if info.dli_sname else None)


def _arch22_symbol_sources():
    from fla_npu.ops.ascendc._runtime import runtime
    actual_runtime = runtime()
    return {name: _arch22_dladdr(actual_runtime.symbol(name)) for name in _TARGET_SYMBOLS}


def _arch22_runtime_identity(*, require_tiling=False, refresh=False):
    """首次只证 opapi；调用后 refresh 核映射，tiling 回调选择另由真实 Host marker 证明。"""
    global _ARCH22_IDENTITY, _ARCH22_SNAPSHOT_SEQUENCE
    if _ARCH22_IDENTITY is not None and not refresh:
        if (_ARCH22_IDENTITY["binding_sha256_requested"] != os.getenv("FINALIZE_BINDING_SHA256") or
                _ARCH22_IDENTITY["source_commit_requested"] != os.getenv("FINALIZE_SOURCE_COMMIT")):
            raise RuntimeError("DUT binding request changed after identity initialization")
        if not require_tiling or _ARCH22_IDENTITY["tiling_identity"]["required"]:
            return _ARCH22_IDENTITY
    import fla_npu
    fla_npu.load_ascendc_opapi_libraries()
    module = Path(fla_npu.__file__).resolve()
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
    install_key = (str(binding_file), measured_binding_sha, str(site))
    reused_install_verification = install_key in _ARCH22_VERIFIED_INSTALLS
    if not reused_install_verification:
        actual_files = {path.relative_to(site).as_posix(): path for path in site.rglob("*")
                        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
        if set(actual_files) != set(expected_files):
            raise RuntimeError("DUT installed binding tree membership differs from frozen tree")
        for relative, path in actual_files.items():
            expected = expected_files[relative]
            if path.stat().st_size != expected["bytes"] or _arch22_sha(path) != expected["sha256"]:
                raise RuntimeError(f"DUT installed file differs: {relative}")
        _ARCH22_VERIFIED_INSTALLS[install_key] = len(actual_files)
    installed_file_count = _ARCH22_VERIFIED_INSTALLS[install_key]
    package_record = _arch22_bound_file(module, site, expected_files)
    vendor = module.parent / "opp" / _VENDOR
    target_opapi = (vendor / "op_api/lib/libcust_opapi.so").resolve()
    opapi_record = _arch22_bound_file(target_opapi, site, expected_files)
    symbols = _arch22_symbol_sources()
    for name in _TARGET_SYMBOLS:
        resolved = Path(symbols[name]["path"]).resolve()
        if resolved != target_opapi:
            raise RuntimeError(f"DUT target symbol {name} resolves outside the bound opapi: {resolved}")
        symbols[name].update(_arch22_bound_file(resolved, site, expected_files))
    maps = _arch22_read_maps()
    libraries, external = _arch22_loaded_libraries(maps, site, expected_files)
    if not any(Path(entry["path"]) == target_opapi for entry in libraries):
        raise RuntimeError("DUT maps does not contain the bound target cust_opapi library")
    target_tiling = (vendor / "op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/"
                     "libcust_opmaster_rt2.0.so").resolve()
    tiling_record = _arch22_bound_file(target_tiling, site, expected_files)
    tiling_mapped = any(Path(entry["path"]) == target_tiling for entry in libraries)
    if require_tiling and not tiling_mapped:
        raise RuntimeError("DUT maps after call does not contain the frozen package tiling library")
    tiling = dict(required=bool(require_tiling), mapped=tiling_mapped,
                  mapping_verified_after_call=bool(require_tiling and tiling_mapped), library=tiling_record,
                  selected_callback_verified=False,
                  selection_evidence="requires actual new Host key marker plus this package mapping/SHA")
    source = Path(binding["source_dir"])
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if head != binding["source_commit"] or status.strip():
        raise RuntimeError("DUT measured source HEAD/clean differs from frozen source")
    phase = "post-tiling" if require_tiling else "pre-call"
    maps_digest = hashlib.sha256(maps.encode("utf-8")).hexdigest()
    reused_maps = maps_digest in _ARCH22_MAP_SNAPSHOTS
    if not reused_maps:
        _ARCH22_SNAPSHOT_SEQUENCE += 1
        maps_file = (Path(os.environ["FINALIZE_CPU_EVIDENCE"]) / "worker-identity" / str(os.getpid()) /
                     f"maps-{_ARCH22_SNAPSHOT_SEQUENCE:04d}.txt")
        maps_file.parent.mkdir(parents=True, exist_ok=True)
        maps_file.write_text(maps, encoding="utf-8")
        _ARCH22_MAP_SNAPSHOTS[maps_digest] = dict(
            path=str(maps_file), bytes=maps_file.stat().st_size, sha256=_arch22_sha(maps_file))
    maps_record = dict(_ARCH22_MAP_SNAPSHOTS[maps_digest], observed_phase=phase, reused_snapshot=reused_maps)
    measured = dict(binding_sha256=measured_binding_sha, source_commit=head, site=str(site),
                    package=package_record, custom_opapi=opapi_record, target_symbols=symbols,
                    libraries=libraries, external_dependencies=external, tiling_identity=tiling,
                    maps=maps_record, immutable_install_verification_reused=reused_install_verification)
    _ARCH22_IDENTITY = dict(measured_identity=measured, fla_npu_file=str(module),
                fla_npu_file_sha256=package_record["sha256"],
                source_commit_requested=os.getenv("FINALIZE_SOURCE_COMMIT"),
                binding_sha256_requested=os.getenv("FINALIZE_BINDING_SHA256"),
                source_commit_env=os.getenv("SOURCE_COMMIT"), binding_sha_env=os.getenv("BINDING_SHA"),
                measured_source_commit=head, measured_source_clean=not status.strip(),
                measured_binding_sha256=measured_binding_sha, measured_binding_path=str(binding_file),
                measured_site=str(site), verified_installed_file_count=installed_file_count,
                loaded_libraries=libraries, external_dependencies=external, target_symbols=symbols,
                tiling_identity=tiling, proc_maps=maps)
    return _ARCH22_IDENTITY
