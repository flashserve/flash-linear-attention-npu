"""Check framework metadata producers without a caller-side synchronization."""
import argparse
import sys
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executor_recurrent_kda import build_inputs, call_options, run_cpu
from gen_recurrent_kda import base_spec
from fla_npu.ops.ascendc import _aclnn_ctypes, _stable
from atk.tasks.post_process.mixed_tolerance_benchmark_compare import MixedToleranceBenchmarkAccuracyCompare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    compare = MixedToleranceBenchmarkAccuracyCompare(None)
    count = 0
    for backend in (_stable, _aclnn_ctypes):
        for custom_stream in (False, True):
            stream = torch.npu.Stream(device=device) if custom_stream else torch.npu.current_stream(device)
            for packed in (False, True):
                spec = base_spec(B=1, T=2, H=1, HV=1, cu=packed,
                                 indices="packed", accepted=1, pattern="signal")
                with torch.npu.stream(stream):
                    inputs = build_inputs(spec, device)
                    inputs["num_accepted_tokens"].fill_(1)
                    # No .cpu(), item(), or stream/device synchronize before the call.
                    actual = backend.npu_recurrent_kda(**inputs, **call_options(spec))
                    stream.synchronize()
                    golden = run_cpu(spec, high_precision=True)
                    assert all(compare.compute_accuracy_result(a.cpu(), b.float(), f"output_{i}.pt").result
                               for i, (a, b) in enumerate(zip(actual, golden)))
                    count += 1
                    for invalid in ("accepted", "cu") if packed else ("accepted",):
                        inputs["num_accepted_tokens"].fill_(1)
                        name = "num_accepted_tokens" if invalid == "accepted" else "cu_seqlens"
                        if invalid == "accepted":
                            inputs[name].fill_(0)
                        else:
                            inputs[name][0] = 1
                        try:
                            backend.npu_recurrent_kda(**inputs, **call_options(spec))
                        except RuntimeError as exc:
                            message = str(exc)
                            assert name in message and "EZ0037" in message, message
                        else:
                            raise AssertionError((backend.__name__, custom_stream, packed, invalid))
                        count += 1
    print(f"METADATA_READINESS_OK cases={count} backends=stable,ctypes streams=default,custom")


if __name__ == "__main__":
    main()
