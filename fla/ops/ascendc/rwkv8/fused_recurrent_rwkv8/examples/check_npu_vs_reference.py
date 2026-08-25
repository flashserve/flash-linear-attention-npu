"""NPU dump vs Python CPU golden 本体对拍。

读 RWKV8_DUMP_DIR 下 example 落盘的 case{ i}_*.bin（raw fp32）+ meta.txt，
逐 case 调 tests/reference/fused_recurrent_rwkv8_reference.py（精度真值锚点，
已与 fla 竞品 GPU fixture 对齐 ~1e-7）重算 golden，与 NPU 输出比 rel-RMSE。

用法: python check_npu_vs_reference.py <dump_dir>
退出码: 全部 case rel-RMSE <= 0.002 → 0，否则 1。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))

from fused_recurrent_rwkv8_reference import fused_recurrent_rwkv8_reference  # noqa: E402

TOL = 0.002
IO_NAMES = ["q", "w", "k", "v", "z", "b"]


def load_bin(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    arr = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    assert arr.size == expected, f"{path}: expect {expected} floats, got {arr.size}"
    return torch.from_numpy(arr.reshape(shape).copy())


def rel_rmse(out: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (out.double() - ref.double()).pow(2).sum().item()
    den = ref.double().pow(2).sum().item()
    if den == 0.0:
        return 0.0 if diff == 0.0 else 1e30
    return float(np.sqrt(diff / den))


def parse_meta(path: Path) -> dict:
    meta = {}
    for tok in path.read_text().split():
        key, _, val = tok.partition("=")
        # 同名键保留首次出现：desc 自由文本里可能再出现 "chunkLen=..." 之类字样
        if key not in meta:
            meta[key] = val
    return meta


def main() -> int:
    dump_dir = Path(sys.argv[1])
    metas = sorted(dump_dir.glob("case*_meta.txt"))
    assert metas, f"no case*_meta.txt under {dump_dir}"

    failed = 0
    for meta_path in metas:
        tag = meta_path.name[: -len("_meta.txt")]
        meta = parse_meta(meta_path)
        B, T, H = (int(meta[k]) for k in "BTH")
        K = int(meta.get("K", meta.get("N", 0)))   # 兼容旧 dump（N = K = V）
        V = int(meta.get("V", meta.get("N", 0)))
        scale = float(meta["scale"])
        has_init = meta["hasInit"] == "1"
        k_shape = (B, H, T, K)   # BHTC
        v_shape = (B, H, T, V)
        state_shape = (B, H, K, V)   # initial_state 接口朝向（= 内核 Sᵀ）

        inputs = {}
        for name in IO_NAMES:
            shape = v_shape if name == "v" else k_shape
            inputs[name] = load_bin(dump_dir / f"{tag}_{name}.bin", shape)
        initial_state = (
            load_bin(dump_dir / f"{tag}_initial_state.bin", state_shape) if has_init else None
        )
        o_npu = load_bin(dump_dir / f"{tag}_o_npu.bin", v_shape)

        # 训练预埋输出（meta 里的开关决定是否存在；缺省按关处理以兼容旧 dump）
        reverse = meta.get("reverse", "0") == "1"
        out_s = meta.get("outputS", "0") == "1"
        out_sa = meta.get("outputSa", "0") == "1"
        chunk_len = int(meta.get("chunkLen", "16"))   # 缺省 16 兼容旧 dump

        res = fused_recurrent_rwkv8_reference(
            inputs["q"], inputs["w"], inputs["k"], inputs["v"], inputs["z"], inputs["b"],
            scale=scale, initial_state=initial_state,
            reverse=reverse, output_chunk_state=out_s, output_sa=out_sa,
            chunk_len=chunk_len,
        )
        o_ref = res.o

        o_err = rel_rmse(o_npu, o_ref)
        ok = o_err <= TOL
        extras = ""
        if out_s:
            s_shape = (B, H, T // chunk_len, K, V)
            s_npu = load_bin(dump_dir / f"{tag}_s_npu.bin", s_shape)
            chunk_err = rel_rmse(s_npu, res.s) if T // chunk_len > 0 else 0.0
            ok = ok and chunk_err <= TOL
            extras += f", s rel-RMSE={chunk_err:.3e}"
        if out_sa:
            sa_npu = load_bin(dump_dir / f"{tag}_sa_npu.bin", v_shape)
            sa_err = rel_rmse(sa_npu, res.sa)
            ok = ok and sa_err <= TOL
            extras += f", sa rel-RMSE={sa_err:.3e}"
        failed += 0 if ok else 1
        print(f"{tag} B{B} T{T} H{H} K{K} V{V} scale={scale:g} hasInit={int(has_init)} ({meta.get('desc', '')}): "
              f"o rel-RMSE={o_err:.3e}{extras}, {'PASS' if ok else 'FAIL'}")

    if failed:
        print(f"{failed} case(s) FAILED (vs python reference)")
        return 1
    print("all cases PASS (vs python reference)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
