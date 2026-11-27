"""Build the optional 910B3 VDN device payload into the selected wheel."""
import importlib.util
import os
from pathlib import Path
import re
import subprocess


def stage_vdn_pipeline(repo, package, build_dir):
    from fla_npu_artifacts import get_soc

    selection = set(filter(None, re.split(r"[,;\s]+", os.getenv("FLA_NPU_OPS", ""))))
    if get_soc() != "ascend910b" or (selection and "vdn_forward" not in selection):
        return
    toolkit = Path(os.getenv("ASCEND_HOME_PATH", "")) / "compiler/version.info"
    if not toolkit.is_file() or "Version=9.2.1" not in toolkit.read_text().splitlines():
        return
    spec = importlib.util.find_spec("triton")
    if not spec or not spec.origin:
        return
    bitcode = Path(spec.origin).parent / "backends/ascend/bishengir/lib/meta_op.aiv.c220.O0.bc"
    if not bitcode.is_file():
        return
    source = repo / "fla/ops/ascendc/vdn/vdn_forward_manual"
    subprocess.run(["bash", str(source / "build.sh"), str(build_dir), str(package / "ops/vdn")],
                   check=True)
