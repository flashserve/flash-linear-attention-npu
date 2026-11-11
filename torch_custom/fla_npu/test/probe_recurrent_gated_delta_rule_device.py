"""Print the native error for CPU inputs; use --device npu for a valid control.

Run with an installed fla_npu package and a configured CANN environment.
An operator exception is printed with its traceback and exits with status 1.
"""

import argparse
import inspect
import traceback

import torch
import torch_npu  # noqa: F401

from fla_npu.ops import ascendc
from fla_npu.ops.ascendc import _stable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    torch.npu.set_device(args.device_id)
    _stable.load()
    print("wrapper:", inspect.getsourcefile(_stable.npu_recurrent_gated_delta_rule), flush=True)
    print("launcher:", _stable._loaded_path, flush=True)

    query = torch.zeros((1, 1, 128), dtype=torch.bfloat16)
    query[..., 0] = 1
    inputs = dict(
        query=query,
        key=torch.zeros_like(query),
        value=torch.zeros_like(query),
        state=torch.ones((1, 1, 128, 128), dtype=torch.float32),
        beta=torch.zeros((1, 1), dtype=torch.bfloat16),
        g=torch.zeros((1, 1), dtype=torch.float32),
        actual_seq_lengths=torch.tensor([0, 1], dtype=torch.int32),
        ssm_state_indices=torch.tensor([0], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
    )
    inputs = {name: tensor.to(args.device) for name, tensor in inputs.items()}
    for name, tensor in inputs.items():
        print(f"{name}: device={tensor.device}, shape={tuple(tensor.shape)}, dtype={tensor.dtype}", flush=True)

    try:
        output = ascendc.recurrent_gated_delta_rule(**inputs, scale=1.0)
        torch.npu.synchronize()
    except Exception as error:
        print(f"operator error: {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        return 1

    print(f"operator returned: device={output.device}, shape={tuple(output.shape)}", flush=True)
    if args.device == "cpu":
        print("CPU inputs were not rejected.", flush=True)
        return 2
    torch.testing.assert_close(output.cpu(), torch.ones((1, 1, 128), dtype=torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(inputs["state"].cpu(), torch.ones((1, 1, 128, 128)), rtol=0, atol=0)
    print("NPU control: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
