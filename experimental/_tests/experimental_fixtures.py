"""Minimal real two-phase ABI fixture for build/install contract tests."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

from tests.opp_abi_fixtures import write_elf_exports

ABI_HELPER = Path(__file__).resolve().parents[2] / "torch_custom/fla_npu/fla_npu/_op_abi.py"

def write_abi(vendor, name="probe", op_type="Probe"):
    spec = importlib.util.spec_from_file_location("fixture_op_abi", ABI_HELPER)
    abi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abi)
    header = vendor / "op_api/include/aclnnop" / ("aclnn_" + name + ".h")
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_text(
        f"aclnnStatus aclnn{op_type}GetWorkspaceSize(const aclTensor* input, uint64_t* workspaceSize, aclOpExecutor** executor);\n"
        f"aclnnStatus aclnn{op_type}(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);\n",
        encoding="utf-8")
    exports = {"aclnn" + op_type, "aclnn" + op_type + "GetWorkspaceSize"}
    write_elf_exports(vendor / abi.LIBRARY, exports)
    with patch.object(abi, "exported_symbols", return_value=exports):
        abi.write_manifest(vendor)
