#!/usr/bin/env python3
"""Verify that Stable-ABI exceptions keep their ACLNN error details."""

import argparse

import torch
import torch_npu  # noqa: F401

from fla_npu.ops import ascendc
from fla_npu.ops.ascendc import _runtime


def _empty(shape, dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="npu")


def _kda_fixture(dtype=torch.bfloat16):
    return (
        _empty((1, 4, 1, 128), dtype),
        _empty((1, 4, 1, 128)),
        _empty((1, 4, 1, 128)),
        _empty((1, 4, 1, 128), torch.float32),
        _empty((1, 4, 1), torch.float32),
    )


def _capture_error(call):
    try:
        call()
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("invalid operator call was accepted")


def _recent_aclnn_error():
    return _runtime.runtime()._recent_error_message().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    q, k, v, g, beta = _kda_fixture(torch.float32)
    message = _capture_error(
        lambda: ascendc.chunk_kda_fwd(
            q, k, v, g, beta, scale=128 ** -0.5, layout="BSND"))
    assert "aclnnChunkKdaFwd" in message, message
    assert "Error message:" in message and "EZ" in message, message
    assert message.count("Error message:") == 1, message
    assert _recent_aclnn_error() in message, message
    assert ascendc.BACKENDS["npu_chunk_kda_fwd"] == "stable", ascendc.BACKENDS
    print("PASS ChunkKdaFwd Stable-ABI exception contains ACLNN details", flush=True)

    # A launcher-side precheck has no ACLNN status/error message. It must not
    # inherit details from the preceding failure.
    q, k, v, g, beta = _kda_fixture()
    saved = _empty((1, 4, 1, 64))
    q = q[..., :64]
    k = k[..., :64]
    v = v[..., :64]
    g = g[..., :64]
    message = _capture_error(
        lambda: ascendc.chunk_kda_fwd(
            q, k, v, g, beta, scale=0.125, layout="BSND",
            q_hat_out=saved))
    assert "Error message:" not in message, message
    print("PASS launcher-side errors do not inherit stale ACLNN details",
          flush=True)

    cp_q = _empty((1, 1, 4, 64))
    cp_k = _empty((1, 1, 4, 64))
    cp_w = _empty((1, 1, 4, 64))
    cp_do = _empty((1, 1, 4, 64))
    cp_dv = _empty((1, 1, 4, 64))
    cp_g = _empty((1, 1, 4), torch.float32)
    message = _capture_error(
        lambda: ascendc.pre_process_bwd_kernel_merged(
            cp_q, cp_k, cp_w, cp_do, cp_dv, scale=0.125,
            chunk_size=64, g=cp_g, cu_seqlens=[0, 4]))
    assert "aclnnPreProcessBwdKernelMergedGetWorkspaceSize failed: 161002" in message, message
    assert "EZ0027" in message and "q.shape[3]" in message, message
    assert message.count("Error message:") == 1, message
    assert _recent_aclnn_error() in message, message
    assert ascendc.BACKENDS[
        "npu_pre_process_bwd_kernel_merged"] == "stable", ascendc.BACKENDS
    print("PASS PreProcessBwdKernelMerged Stable-ABI exception contains "
          "EZ0027 details", flush=True)


if __name__ == "__main__":
    main()
