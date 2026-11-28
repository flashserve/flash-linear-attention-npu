"""Shared ATK role and packed-sequence handling for GDN backward."""

import torch
from atk.tasks.api_execute.base_api import BaseApi

from _ascendc_common_executor import _case_spec, _finite_tuple, _marker_device, _rand


def case_spec(input_data, op):
    spec = _case_spec(input_data, op)
    for key in ("B", "T", "HK", "HV", "K", "V", "chunk_size", "seed"):
        if key not in spec:
            raise ValueError(f"{op}: missing case_spec field {key}")
    return spec


def packed_metadata(spec):
    cu = spec.get("cu_seqlens")
    if cu is None:
        return None, None, None
    lengths = [end - start for start, end in zip(cu, cu[1:])]
    if spec["B"] != 1 or len(cu) < 2 or cu[0] != 0 or cu[-1] != spec["T"] or min(lengths) <= 0:
        raise ValueError("Packed inputs require B=1 and strictly increasing offsets spanning T")
    chunk_size = spec["chunk_size"]
    indices = tuple(value for seq, length in enumerate(lengths)
                    for chunk in range((length + chunk_size - 1) // chunk_size)
                    for value in (seq, chunk))
    return lengths, tuple(cu), indices


def uniform(shape, dtype, device, seed, low, high):
    name = {torch.bfloat16: "bf16", torch.float32: "fp32"}[dtype]
    return _rand(shape, name, dtype, device, seed, low, high)


def gate_values(batch, heads, tokens, seqlens, seed, dtype, device):
    """Generate the FLAN chain's negative gate cumsum within each chunk."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    steps = -(torch.rand((batch, heads, tokens), generator=generator) * 0.01 + 0.001)
    gate = torch.empty_like(steps)
    begin = 0
    for length in seqlens or (tokens,):
        end = begin + length
        for start in range(begin, end, 64):
            stop = min(start + 64, end)
            gate[..., start:stop] = steps[..., start:stop].cumsum(dim=-1)
        begin = end
    return gate.to(dtype=dtype, device=device)


class GdnApi(BaseApi):
    def __init__(self, task_result):
        super().__init__(task_result)
        self.inputs = None

    def init_by_input_data(self, input_data):
        self.spec = case_spec(input_data, self.op_name)
        inputs = self.build_inputs(self.spec, torch.device("cpu"))
        self.expected_outputs = self.output_contract(inputs)
        if self.device in ("npu", "pyaclnn"):
            device = _marker_device(input_data)
            if device.type != "npu":
                raise RuntimeError(f"NPU DUT requires an NPU marker tensor, got {device}")
            inputs = self.prepare_npu_inputs(self.spec, inputs)
            inputs = {name: value.to(device) if isinstance(value, torch.Tensor) else value
                      for name, value in inputs.items()}
        elif self.device != "cpu":
            raise RuntimeError(f"Unsupported ATK device {self.device!r}")
        self.inputs = inputs

    @staticmethod
    def prepare_npu_inputs(spec, inputs):
        return inputs

    def __call__(self, input_data, with_output=False):
        if self.inputs is None:
            self.init_by_input_data(input_data)
        high_precision = self.device == "cpu" and bool(self.task_result.is_benchmark_task)
        outputs = (self.run_cpu(self.spec, self.inputs, high_precision)
                   if self.device == "cpu" else self.run_npu(self.spec, self.inputs))
        if not with_output:
            return None
        expected = self.expected_outputs
        if len(outputs) != len(expected):
            raise RuntimeError(f"{self.op_name}: expected {len(expected)} outputs, got {len(outputs)}")
        for index, (value, contract) in enumerate(zip(outputs, expected)):
            if contract is None and value is None:
                continue
            if contract is None or not isinstance(value, torch.Tensor):
                raise RuntimeError(f"{self.op_name}: unexpected output presence at slot {index}")
            shape, dtype = contract
            dtype = torch.float64 if high_precision else dtype
            if value.shape != shape or value.dtype != dtype:
                raise RuntimeError(f"{self.op_name}: slot {index} expected {shape}/{dtype}, "
                                   f"got {value.shape}/{value.dtype}")
        values = _finite_tuple(outputs)
        return values if high_precision else tuple(value.float() for value in values)
