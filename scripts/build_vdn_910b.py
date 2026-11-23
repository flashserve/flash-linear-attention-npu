"""Build the optional Ascend 910B VDN library into a wheel staging directory."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from fla_npu_artifacts import get_soc


def build(output):
    import torch

    cann = os.environ.get("ASCEND_HOME_PATH")
    if not cann:
        raise RuntimeError("Source the CANN set_env.sh before building the VDN kernels")
    source = Path(__file__).resolve().parents[1] / 'torch_custom/fla_npu/csrc/vdn'
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    include = source.parent / "include"
    with tempfile.TemporaryDirectory(prefix="vdn-build-", dir=output.parent) as temp:
        work = Path(temp)
        objects = []
        for name in ("scan", "bridge", "diagonal"):
            obj = work / (name + ".o")
            subprocess.run(["bisheng", "--npu-arch=dav-2201", "-O2", "-fPIC", "-c",
                            source / ("vdn_" + name + ".asc"), "-I" + cann + "/include", "-o", obj], check=True)
            objects.append(obj)
        library = work / output.name
        registration = work / "registration.o"
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-fPIC", "-c",
                        "-fvisibility=hidden", "-fno-gnu-unique", "-I" + str(include),
                        "-I" + str(include / "torch_stable_abi"), source / "registration.cpp", "-o", registration], check=True)
        subprocess.run(["bisheng", "-shared", registration, *objects, "-Wl,-z,defs",
                        "-L" + str(torch_lib), "-ltorch_cpu", "-lc10", "-ltorch",
                        "-L" + cann + "/lib64", "-lascendcl", "-o", library], check=True)
        library.replace(output)


def stage_optional_vdn(built_package):
    if os.environ.get("FLA_NPU_VDN_NATIVE", "0") == "1":
        if get_soc() != "ascend910b":
            raise ValueError("FLA_NPU_VDN_NATIVE=1 requires FLA_NPU_SOC=ascend910b")
        build(Path(built_package) / "ops/vdn/libfla_vdn_910b.so")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)

