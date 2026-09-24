"""Gate a built wheel before it is delivered, uploaded to PyPI included.

``python -m pip install`` trusts the wheel tag for its platform decision, so a
wheel that lies about its architecture installs cleanly and then fails at import
time on the user's machine.  Everything this script checks is a property of the
artifact that cannot be re-checked after upload, and every failure is one a user
would otherwise hit first:

* the file name, the ``dist-info`` metadata and the tier marker agree;
* the payload carries the Stable-ABI launcher, its build stamp, the OPP host
  libraries and the kernels of the SoC the wheel claims;
* every shared object matches the platform tag's architecture;
* no shared object asks for a newer glibc/libstdc++ than ``manylinux_2_34``
  promises (the tag asserts the *lower* bound; the upper bound is what breaks).

The per-object watermark rule follows section 6.5 of
`docs/architecture/torch-npu-decoupled-architecture.md`: the floor has to be judged on
every ``.so`` in the package (the OPP is the binding constraint, not the
launcher), not on one file.

Runs on Linux with binutils' ``readelf`` (the CI image has it).

Every build -- local, GitHub Release or PyPI -- produces the same distribution
name, platform tag and tier metadata this gate expects, so a developer can run
this on a wheel they built themselves before it becomes a release.

The version is the one thing that differs by build type, and the default is the
strict one: ``Version`` must equal ``__version__`` exactly, which is what a
published wheel carries.  A daily build is labelled
``<release>+<branch>_dev<commit>`` -- the public part is ``__version__`` with a
trailing ``.devN`` dropped -- and has to be declared as such with
``--allow-local-version``; the failure message says so, so a daily wheel can
never be uploaded to the real index by accident.

Usage:
    python scripts/check_pypi_wheel.py dist/*.whl \
        --expect-tier a2 --expect-arch aarch64 --require-offline-bundle
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# Measured on the release build image: glibc 2.34 is what the launcher
# (dlopen/dlsym/dlerror) and the OPP host libraries need under the pinned Ubuntu
# 22.04 toolchain, and the C++ ABI watermark is the one GCC 11 stamps.
# scripts/fla_npu_artifacts.WHEEL_PLATFORM_TAG must claim the same glibc.
DEFAULT_MAX_GLIBC = "2.34"
DEFAULT_MAX_GLIBCXX = "3.4.29"

# Kept beside ``DEFAULT_MAX_GLIBC`` so a bare checkout can check a wheel without
# importing the build helper; tests assert the two stay in step.
EXPECTED_PLATFORM_PREFIX = "manylinux_2_34"

# A development line carries ``<next>.dev0`` in fla/__init__.py; a daily build of
# it drops the ``.devN``, so the local version is the only thing that says "not
# released yet".  Kept in step with scripts/fla_npu_artifacts.daily_base_version
# by tests/test_wheel_environment.py.
DAILY_BASE_SUFFIX = re.compile(r"\.dev\d+$")

TIER_SOC = {"a2": "ascend910b", "a3": "ascend910_93", "a5": "ascend950"}

# Architecture names as the two tools spell them.
ELF_MACHINE = {"aarch64": "AArch64", "x86_64": "Advanced Micro Devices X86-64"}
ELF_CLASS = {"aarch64": "ELF64", "x86_64": "ELF64"}
OPP_HOST_LIB_ARCH = {"aarch64": "aarch64", "x86_64": "x86_64"}

VENDOR = "fla_npu_transformer"


class CheckFailure(RuntimeError):
    pass


def _version_tuple(text: str, parts: int) -> tuple[int, ...]:
    numbers = [int(part) for part in text.split(".")[:parts]]
    while len(numbers) < parts:
        numbers.append(0)
    return tuple(numbers)


def _readelf(*args: str) -> str:
    executable = shutil.which("readelf")
    if executable is None:
        raise CheckFailure(
            "readelf is required to verify the wheel's ELF payload "
            "(install binutils: apt-get install -y binutils)"
        )
    return subprocess.run(
        [executable, *args],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout


def _extract(archive: zipfile.ZipFile, root: Path) -> None:
    """Extract the archive, refusing members that escape *root*."""

    for info in archive.infolist():
        if info.is_dir():
            continue
        target = (root / info.filename).resolve()
        if root.resolve() not in target.parents and target.parent != root.resolve():
            raise CheckFailure(f"refusing to extract {info.filename!r} outside the wheel root")
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, open(target, "wb") as destination:
            shutil.copyfileobj(source, destination)


def _elf_machine(path: Path) -> str:
    match = re.search(r"^\s*Machine:\s*(.+?)\s*$", _readelf("-h", str(path)), re.MULTILINE)
    if not match:
        raise CheckFailure(f"unable to read the ELF header of {path.name}")
    return match.group(1).strip()


def _elf_class(path: Path) -> str:
    match = re.search(r"^\s*Class:\s*(.+?)\s*$", _readelf("-h", str(path)), re.MULTILINE)
    if not match:
        raise CheckFailure(f"unable to read the ELF class of {path.name}")
    return match.group(1).split(",")[0].strip()


def _symbol_versions(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Highest ``GLIBC_x.y`` and ``GLIBCXX_x.y.z`` this object asks for."""

    text = _readelf("--version-info", str(path))
    glibc = max(
        (_version_tuple(match.group(1), 2) for match in re.finditer(r"GLIBC_(\d+\.\d+)", text)),
        default=(0, 0),
    )
    glibcxx = max(
        (
            _version_tuple(match.group(1), 3)
            for match in re.finditer(r"GLIBCXX_(\d+\.\d+\.\d+)", text)
        ),
        default=(0, 0, 0),
    )
    return glibc, glibcxx


def _wheel_parts(wheel: Path) -> dict[str, str]:
    stem = wheel.name[: -len(".whl")]
    parts = stem.split("-")
    if len(parts) == 5:
        distribution, version, python_tag, abi_tag, platform_tag = parts
        build_tag = ""
    elif len(parts) == 6:
        distribution, version, build_tag, python_tag, abi_tag, platform_tag = parts
    else:
        raise CheckFailure(f"{wheel.name}: not a wheel file name")
    return {
        "distribution": distribution,
        "version": version,
        "build_tag": build_tag,
        "python_tag": python_tag,
        "abi_tag": abi_tag,
        "platform_tag": platform_tag,
    }


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _metadata(archive: zipfile.ZipFile) -> dict[str, list[str]]:
    name = next((n for n in archive.namelist() if n.endswith(".dist-info/METADATA")), None)
    if name is None:
        raise CheckFailure("wheel carries no dist-info/METADATA")
    fields: dict[str, list[str]] = {}
    for line in archive.read(name).decode("utf-8").splitlines():
        key, separator, value = line.partition(": ")
        if separator:
            fields.setdefault(key.strip(), []).append(value.strip())
    return fields


def _install_path(name: str) -> str:
    """Map a wheel member onto the path it occupies once installed.

    ``setup.py`` marks the wheel ``Root-Is-Purelib: false`` (the payload is a
    host launcher plus the OPP it loads) while the ``build_py`` output stays
    pure, so setuptools stages the package under ``<dist>.data/purelib/``.  pip
    maps that directory back onto the purelib scheme -- the same
    ``site-packages/fla_npu`` a top-level member lands in -- so the structural
    checks look through the prefix instead of demanding members at the root.
    """
    match = re.fullmatch(r"[\w.+-]+\.data/(?:purelib|platlib)/(.+)", name)
    return match.group(1) if match else name


def check_wheel(
    wheel: Path,
    *,
    expect_tier: str | None,
    expect_arch: str | None,
    expect_version: str | None,
    # Fails closed: without it the version must be the released one exactly.
    allow_local_version: bool = False,
    require_offline_bundle: bool,
    require_launcher: bool,
    max_glibc: str,
    max_glibcxx: str,
    allow_missing_readelf: bool,
) -> list[str]:
    """Check one wheel, returning the list of informational notes."""

    notes: list[str] = []
    parts = _wheel_parts(wheel)
    tier = expect_tier
    if tier is None:
        match = re.fullmatch(
            r"flash[_-]linear[_-]attention[_-]npu[_-](a[235])",
            parts["distribution"],
            re.IGNORECASE,
        )
        if not match:
            raise CheckFailure(
                f"{wheel.name}: not a tiered distribution name; pass --expect-tier"
            )
        tier = match.group(1).lower()

    arch = expect_arch
    if arch is None:
        match = re.fullmatch(r"manylinux_\d+_\d+_(\w+)", parts["platform_tag"])
        if not match:
            raise CheckFailure(
                f"{wheel.name}: platform tag {parts['platform_tag']!r} is not a "
                "PEP 600 manylinux tag; pip refuses to install such a wheel on a "
                "modern platform (PyPI also rejects the upload)"
            )
        arch = match.group(1)

    expected_distribution = f"flash_linear_attention_npu_{tier}"
    if _normalize(parts["distribution"]) != _normalize(expected_distribution):
        raise CheckFailure(
            f"{wheel.name}: distribution {parts['distribution']!r} does not match "
            f"tier {tier!r} (expected {expected_distribution})"
        )
    if (parts["python_tag"], parts["abi_tag"]) != ("py3", "none"):
        raise CheckFailure(
            f"{wheel.name}: expected the ABI-free py3-none tags, got "
            f"{parts['python_tag']}-{parts['abi_tag']}"
        )
    if parts["build_tag"]:
        raise CheckFailure(
            f"{wheel.name}: a published wheel must not carry the build tag "
            f"{parts['build_tag']!r} (tier and arch are already in the name)"
        )
    expected_platform = f"{EXPECTED_PLATFORM_PREFIX}_{arch}"
    if parts["platform_tag"] != expected_platform:
        raise CheckFailure(
            f"{wheel.name}: platform tag {parts['platform_tag']!r} != {expected_platform!r}"
        )

    soc = TIER_SOC[tier]
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        # Members read/verified below stay keyed by their real archive name so
        # archive.read() and the extraction keep working; only the matching is
        # done on the installed path.
        installed = {_install_path(name): name for name in names}
        if any(installed_path != name for installed_path, name in installed.items()):
            notes.append("payload staged under .data/purelib/ (pip installs it to site-packages)")
        fields = _metadata(archive)
        if _normalize(fields["Name"][0]) != _normalize(expected_distribution):
            raise CheckFailure(
                f"{wheel.name}: METADATA Name {fields['Name'][0]!r} does not match "
                f"the file name {parts['distribution']!r}"
            )
        if expect_version:
            version = fields["Version"][0]
            public, _, local = version.partition("+")
            if allow_local_version:
                # A daily build: the version has to be a build of this tree, and
                # the local part is what says which branch and commit it came from.
                if public != DAILY_BASE_SUFFIX.sub("", expect_version):
                    raise CheckFailure(
                        f"{wheel.name}: METADATA Version {version!r} is not a daily "
                        f"build of {expect_version!r}"
                    )
            elif version != expect_version:
                hint = (
                    " (a daily build carries a +<branch>_dev<commit> local version; "
                    "pass --allow-local-version to accept it)"
                    if local
                    else ""
                )
                raise CheckFailure(
                    f"{wheel.name}: METADATA Version {version!r} != {expect_version!r}{hint}"
                )
        requires_python = " ".join(fields.get("Requires-Python", []))
        if "3.9" not in requires_python:
            raise CheckFailure(
                f"{wheel.name}: Requires-Python {requires_python!r} does not cover 3.9"
            )
        requires = fields.get("Requires-Dist", [])
        if require_launcher:
            # build_wheel.py injects these when the wheel carries the launcher;
            # a launcher that is not backed by the declaration would fail at
            # import time on an environment pip was told was fine.
            for pin in ("torch>=2.7.1", "torch_npu>=2.7.1"):
                if pin not in requires:
                    raise CheckFailure(
                        f"{wheel.name}: the launcher needs {pin!r} in Requires-Dist "
                        f"(got: {', '.join(requires) or '<none>'})"
                    )

        def find(pattern: str) -> list[str]:
            regex = re.compile(pattern)
            return [
                original
                for installed_path, original in installed.items()
                if regex.search(installed_path)
            ]

        def require(pattern: str, what: str) -> list[str]:
            found = find(pattern)
            if not found:
                raise CheckFailure(f"{wheel.name}: missing {what} ({pattern})")
            return found

        if require_launcher:
            require(r"^fla_npu/libfla_npu_stable\.so$", "the Stable-ABI launcher")
            require(
                r"^fla_npu/ops/ascendc/_stable_hash\.py$",
                "the launcher build stamp (_stable_hash.py)",
            )
        build_meta = require(r"^fla_npu/_build_meta\.py$", "the tier marker (_build_meta.py)")
        compat = require(r"^fla_npu/_compat\.py$", "the generated version table (_compat.py)")
        tier_source = archive.read(build_meta[0]).decode("utf-8")
        tier_match = re.search(r"^TIER\s*=\s*['\"]([\w]+)['\"]", tier_source, re.MULTILINE)
        if not tier_match or tier_match.group(1) != tier:
            raise CheckFailure(
                f"{wheel.name}: _build_meta.py does not declare TIER = {tier!r} "
                f"(got {tier_source.strip()!r})"
            )
        compat_source = archive.read(compat[0]).decode("utf-8")
        if not re.search(r"^MIN_CANN\s*=", compat_source, re.MULTILINE):
            raise CheckFailure(f"{wheel.name}: _compat.py carries no MIN_CANN")
        min_torch = re.search(r"^MIN_TORCH\s*=\s*['\"]([^'\"]+)['\"]", compat_source, re.MULTILINE)
        if not min_torch:
            raise CheckFailure(f"{wheel.name}: _compat.py carries no MIN_TORCH")
        if _version_tuple(min_torch.group(1), 3) < _version_tuple("2.7.1", 3):
            raise CheckFailure(
                f"{wheel.name}: the embedded version promise (MIN_TORCH = "
                f"{min_torch.group(1)!r}) is below the 2.7.1 the release documents"
            )
        require(
            rf"^fla_npu/opp/vendors/{VENDOR}/op_api/lib/libcust_opapi\.so$",
            "the packaged OPP op_api library",
        )
        kernel = find(rf"^fla_npu/opp/vendors/{VENDOR}/op_impl/ai_core/tbe/kernel/{soc}/")
        if not kernel:
            raise CheckFailure(
                f"{wheel.name}: no {soc} kernels (tier {tier} must carry its own "
                "prebuilt kernels; check FLA_NPU_SOC during the build)"
            )
        notes.append(f"{len(kernel)} {soc} kernel files")
        config = find(rf"^fla_npu/opp/vendors/{VENDOR}/op_impl/ai_core/tbe/config/{soc}/")
        if not config:
            raise CheckFailure(
                f"{wheel.name}: no {soc} op config under op_impl/ai_core/tbe/config"
            )
        bundle = find(r"^fla_npu/offline/third_party/")
        if require_offline_bundle and not bundle:
            raise CheckFailure(
                f"{wheel.name}: the offline recompile bundle is missing "
                "(build with FLA_NPU_BUILD_OFFLINE_BUNDLE=1 and a seeded third_party)"
            )
        notes.append(f"{len(bundle)} offline bundle files" if bundle else "no offline bundle")
        shared_objects = find(r"\.so(\.[\d.]+)?$")
        if not shared_objects:
            raise CheckFailure(f"{wheel.name}: the payload carries no shared object")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _extract(archive, root)
            if shutil.which("readelf") is None and allow_missing_readelf:
                notes.append("readelf unavailable: ELF checks skipped")
                return notes
            expected_machine = ELF_MACHINE[arch]
            for name in shared_objects:
                path = root / name
                machine = _elf_machine(path)
                if machine != expected_machine:
                    raise CheckFailure(
                        f"{wheel.name}: {name} is {machine}, the {expected_platform} tag "
                        f"promises {expected_machine}"
                    )
                elf_class = _elf_class(path)
                if elf_class != ELF_CLASS[arch]:
                    raise CheckFailure(
                        f"{wheel.name}: {name} is {elf_class}, expected {ELF_CLASS[arch]}"
                    )
                glibc, glibcxx = _symbol_versions(path)
                if glibc > _version_tuple(max_glibc, 2):
                    raise CheckFailure(
                        f"{wheel.name}: {name} requires GLIBC_{'.'.join(str(p) for p in glibc)}, "
                        f"above the {EXPECTED_PLATFORM_PREFIX} promise (max {max_glibc}); "
                        "it would fail "
                        "on a host with the promised glibc"
                    )
                if glibcxx > _version_tuple(max_glibcxx, 3):
                    raise CheckFailure(
                        f"{wheel.name}: {name} requires "
                        f"GLIBCXX_{'.'.join(str(p) for p in glibcxx)}, above the promise "
                        f"(max {max_glibcxx}); rebuild it with the pinned toolchain"
                    )
            notes.append(f"{len(shared_objects)} shared objects verified for {arch}")
            host_lib_rel = (
                f"fla_npu/opp/vendors/{VENDOR}/op_impl/ai_core/tbe/op_tiling/"
                f"lib/linux/{OPP_HOST_LIB_ARCH[arch]}"
            )
            host_lib = next(
                (
                    root / original
                    for installed_path, original in installed.items()
                    if installed_path.startswith(f"{host_lib_rel}/")
                ),
                None,
            )
            if host_lib is None:
                notes.append(f"note: {host_lib_rel} is absent")
            else:
                notes.append(f"{OPP_HOST_LIB_ARCH[arch]} host libraries present")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--expect-tier", choices=sorted(TIER_SOC))
    parser.add_argument("--expect-arch", choices=sorted(ELF_MACHINE))
    parser.add_argument(
        "--expect-version",
        default=None,
        help="expected version; defaults to __version__ in fla/__init__.py",
    )
    parser.add_argument("--require-offline-bundle", action="store_true")
    parser.add_argument(
        "--allow-local-version",
        action="store_true",
        help="accept a daily build: <version>+<branch>_dev<commit>; without it the "
             "version must equal __version__ exactly (what a published wheel carries)",
    )
    parser.add_argument(
        "--without-launcher",
        action="store_true",
        help="the wheel was built with FLA_NPU_BUILD_STABLE_ABI=0",
    )
    parser.add_argument("--max-glibc", default=DEFAULT_MAX_GLIBC)
    parser.add_argument("--max-glibcxx", default=DEFAULT_MAX_GLIBCXX)
    parser.add_argument("--allow-missing-readelf", action="store_true")
    args = parser.parse_args()

    expect_version = args.expect_version
    if expect_version is None:
        init_py = REPO_ROOT / "fla" / "__init__.py"
        if init_py.is_file():
            match = re.search(
                r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]',
                init_py.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
            expect_version = match.group(1) if match else None

    failures = []
    for wheel in args.wheels:
        try:
            notes = check_wheel(
                wheel,
                expect_tier=args.expect_tier,
                expect_arch=args.expect_arch,
                expect_version=expect_version,
                allow_local_version=args.allow_local_version,
                require_offline_bundle=args.require_offline_bundle,
                require_launcher=not args.without_launcher,
                max_glibc=args.max_glibc,
                max_glibcxx=args.max_glibcxx,
                allow_missing_readelf=args.allow_missing_readelf,
            )
        except CheckFailure as exc:
            print(f"[FAIL] {exc}", file=sys.stderr)
            failures.append(str(exc))
            continue
        except zipfile.BadZipFile as exc:
            print(f"[FAIL] {wheel}: not a wheel archive ({exc})", file=sys.stderr)
            failures.append(str(exc))
            continue
        print(f"[OK] {wheel.name}: {', '.join(notes)}")

    if failures:
        print(f"\n{len(failures)} wheel(s) failed the release gate.", file=sys.stderr)
        return 1
    print(f"\n{len(args.wheels)} wheel(s) passed the release gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
