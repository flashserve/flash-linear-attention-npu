"""Exercise native RecurrentGDN device checks through the public Python API."""

import argparse

import torch
import torch_npu  # noqa: F401

from fla_npu.ops import ascendc


def fixture(device):
    query = torch.zeros((1, 1, 128), dtype=torch.bfloat16, device=device)
    query[..., 0] = 1
    return dict(
        query=query,
        key=torch.zeros_like(query),
        value=torch.zeros_like(query),
        state=torch.ones((1, 1, 128, 128), dtype=torch.float32, device=device),
        beta=torch.zeros((1, 1), dtype=torch.bfloat16, device=device),
        actual_seq_lengths=torch.tensor([0, 1], dtype=torch.int32, device=device),
        ssm_state_indices=torch.tensor([0], dtype=torch.int32, device=device),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        g=torch.zeros((1, 1), dtype=torch.float32, device=device),
        gk=None,
        A_log=torch.zeros((1,), dtype=torch.float32, device=device),
        dt_bias=torch.zeros((1,), dtype=torch.float32, device=device),
    )


def rejected(label, inputs, parameter, reason):
    state = inputs["state"]
    version = state._version
    before = state.cpu().clone()
    try:
        ascendc.recurrent_gated_delta_rule(**inputs, scale=1.0)
    except RuntimeError as error:
        message = str(error)
        assert "fla_npu(stable): npu_recurrent_gated_delta_rule: " + parameter in message, message
        assert reason in message, message
    else:
        raise AssertionError(f"{label}: invalid input was accepted")
    # A rejected call must leave the stream usable and State untouched.
    torch.npu.synchronize()
    assert state._version == version, (label, version, state._version)
    torch.testing.assert_close(state.cpu(), before, rtol=0, atol=0)
    print(f"PASS {label}: {message}", flush=True)


def valid_control(label, inputs):
    state = inputs["state"]
    version = state._version
    output = ascendc.recurrent_gated_delta_rule(**inputs, scale=1.0)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), torch.ones_like(output.cpu()), rtol=0, atol=0)
    torch.testing.assert_close(state.cpu(), torch.ones_like(state.cpu()), rtol=0, atol=0)
    assert state._version == version + 1
    print(f"PASS {label}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--other-device", type=int)
    args = parser.parse_args()
    if args.other_device == args.device:
        parser.error("--other-device must differ from --device")
    torch.npu.set_device(args.device)
    cpu = fixture("cpu")
    rejected("all_cpu", cpu, "query", "must be on NPU")
    npu = fixture(f"npu:{args.device}")
    names = tuple(npu)
    for name in names:
        replacement = cpu[name] if name != "gk" else cpu["g"]
        rejected(name + "_cpu", {**npu, name: replacement}, name, "must be on NPU")
    if args.other_device is not None:
        for name in names:
            source = cpu[name] if name != "gk" else cpu["g"]
            replacement = source.to(f"npu:{args.other_device}")
            parameter = "key" if name == "query" else name
            rejected(name + "_other_device", {**npu, name: replacement}, parameter,
                     "must be on the same device as query")
    valid_control("npu_after_rejections", npu)
    minimal = {**npu, "num_accepted_tokens": None, "g": None, "A_log": None, "dt_bias": None}
    valid_control("optional_inputs_absent", minimal)


if __name__ == "__main__":
    main()
