"""ChunkKdaFwdPrepare 的 direct launch 静态合同。"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
MANIFEST = ROOT / "tests/op_cases/chunk_kda_fwd_prepare.json"
DIRECT = (
    ROOT
    / "tests/operators/chunk_kda_fwd_prepare/routes"
    / "test_direct_chunk_kda_fwd_prepare.cpp"
)


def test_direct_launch_covers_declared_route_cases():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    case_ids = {
        case["id"]
        for case in manifest["cases"]
        if "direct_launch" in case["run_on"]
    }
    source = DIRECT.read_text(encoding="utf-8")
    assert case_ids == {
        "prepare_dense_bnsd_raw",
        "prepare_dense_bsnd_fused",
    }
    assert all(case_id in source for case_id in case_ids)
    assert "<<<blockDim, nullptr, stream>>>" in source
    assert "ChunkKdaFwdPrepareTilingData" in source
    assert "SetSysWorkspaceForce" in source
    assert "RunPrepare" in source
