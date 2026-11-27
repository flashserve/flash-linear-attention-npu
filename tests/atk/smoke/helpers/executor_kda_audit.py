"""Adapt the KDA case profiles to the bundled CPU references and public NPU APIs."""

import torch


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
        return reference.finalize_device(inputs, spec, cu, chunks, device)

    inputs = reference.build_inputs(spec)
    if cpu:
        outputs = reference.cpu_reference(inputs, spec, high_precision=high)
        if not high:
            outputs = reference._cast_cpu_outputs_to_abi(outputs, spec)
    else:
        from fla_npu.ops.ascendc import chunk_kda_bwd

        outputs = chunk_kda_bwd(**reference.npu_inputs(inputs, spec, device))
    return reference._finite_outputs(outputs, probe_device=not cpu)
