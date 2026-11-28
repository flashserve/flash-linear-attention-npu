"""Install or verify the pinned AscendTest ATK release used in FLA CI images."""

import argparse
import hashlib
from importlib import metadata
import platform
from pathlib import Path
import subprocess
import sys
import tempfile


VERSION = "26.11.11"
RELEASE_URL = f"https://gitcode.com/AscendTest/ATK/releases/download/v{VERSION}"
# SHA-256 digests published with the release assets, keyed by Python ABI and CPU.
SHA256 = {
    ("cp310", "aarch64"): "97b361a06ffb3210671dc3cfa494ecb974cbe1acf41d3bda43bc6c18984e0c67",
    ("cp310", "x86_64"): "d82ee2f71a2979662153bcb599652a8b90a6fe15ff0d504e773902e5a572743d",
    ("cp311", "aarch64"): "7668a2bb9af10c9c1af5526823afd07118a82c50df2c4a53edf081e98e154aaa",
    ("cp311", "x86_64"): "d0fd4032c6be6bc21722c5cd9b5b29ee09ef81687e7b605f153d97cd3a86d0b0",
    ("cp312", "aarch64"): "536d01c54a3cfa4a1c19eed34a8ea57e0a89021068611e0775b1d457989eaf9d",
    ("cp312", "x86_64"): "cf98e9aec143fc7f8186dc0e80f1f0c44d71874374082dd0e5ce2500e5107b99",
    ("cp313", "aarch64"): "552b51e2bd0598df1d66c35cf15d446c560f426371ee66a03924df0434823366",
    ("cp313", "x86_64"): "e9f4ea8f1553286ea31d2c1d220a2d07279d13838301aeb4ab1e129cd5184f39",
    ("cp39", "aarch64"): "d88ad993522ff6db62dac09681cd43b944560894cee6245d7b9461c4183631bb",
    ("cp39", "x86_64"): "ffac2544f483ba45a7a34d403f67e837d3527813bca24a9414a985efdc660021",
}


def wheel_info():
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    arch = platform.machine()
    if sys.platform != "linux" or sys.implementation.name != "cpython" or (abi, arch) not in SHA256:
        raise SystemExit(f"ATK {VERSION} has no CI wheel for {sys.platform}/{sys.implementation.name}/{abi}/{arch}")
    return f"atk-{VERSION}-{abi}-{abi}-linux_{arch}.whl", SHA256[abi, arch]


def check_version():
    import atk

    installed = metadata.version("atk")
    loaded = getattr(atk, "PACKAGE_VERSION", None)
    if (installed, loaded) != (VERSION, VERSION):
        raise SystemExit(f"CI requires ATK {VERSION}; installed={installed}, imported={loaded} ({atk.__file__})")
    print(f"[CI] ATK {installed} ({atk.__file__})")


def install(wheel_dir):
    name, expected = wheel_info()
    with tempfile.TemporaryDirectory() as tmp:
        wheel = (wheel_dir or Path(tmp)) / name
        if wheel_dir is None:
            subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--retry-delay", "2",
                            "--output", str(wheel), f"{RELEASE_URL}/{name}"], check=True)
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"SHA-256 mismatch for {name}")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", str(wheel),
                        "kombu==5.6.2"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--wheel-dir", type=Path, help="Install a pre-downloaded release wheel")
    args = parser.parse_args()
    if args.install:
        install(args.wheel_dir)
    check_version()
