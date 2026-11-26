"""Adapt the KDA case profiles to the bundled CPU references and public NPU APIs."""

import torch


def finalize_device(inputs, spec, cu, chunks, device):
    from fla_npu.ops.ascendc import npu_chunk_kda_bwd_finalize

    dev = {}
    for name, value in inputs.items():
        if name not in ("a_log", "dt_bias"):
            if spec["lengths"] is not None and spec["rank"] == 4:
                value = value.unsqueeze(0)
            elif spec["lengths"] is None and spec["rank"] == 3:
                value = value.squeeze(0)
        dev[name] = value.contiguous().to(device)
    dev["d_aqk"] = dev.pop("dAqk")
    indices = (
        [n for item in chunks for n in item[:2]]
        if cu and spec["metadata"] != "auto"
        else None
    )
    return npu_chunk_kda_bwd_finalize(
        **dev,
        scale=spec["scale"],
        lower_bound=spec["lower_bound"],
        chunk_size=64,
        safe_gate=True,
        use_gate_in_kernel=True,
        use_exp2=True,
        state_v_first=False,
        output_layout=spec["output_layout"],
        cu_seqlens=cu or None,
        chunk_indices=indices,
        use_beta_sigmoid_in_kernel=spec["beta_mode"] != 0,
        allow_neg_eigval=spec["beta_mode"] == 2,
    )


def run_kda_profile(op, reference, spec, input_data, cpu, high):
    if op == "chunk_kda_bwd_prepare":
        return (
            reference.run_cpu(spec, high_precision=high)
            if cpu
            else reference.run_npu(spec, input_data)
        )

    device = next(x.device for x in input_data.kwargs.values() if torch.is_tensor(x))
    if op == "chunk_kda_bwd_finalize":
        inputs, cu, chunks = reference.build_inputs(spec, spec["seed"])
        if cpu:
            return reference.reference(inputs, spec, chunks, high)
        return finalize_device(inputs, spec, cu, chunks, device)

    inputs = reference.build_inputs(spec)
    if cpu:
        outputs = reference.cpu_reference(inputs, spec, high_precision=high)
        if not high:
            outputs = reference._cast_cpu_outputs_to_abi(outputs, spec)
    else:
        from fla_npu.ops.ascendc import chunk_kda_bwd

        outputs = chunk_kda_bwd(**reference.npu_inputs(inputs, spec, device))
    return reference._finite_outputs(outputs, probe_device=not cpu)
