"""ATK precision coverage for the fused Ascend 950 KDA operator."""

import json
import os
import sys
from itertools import accumulate
from pathlib import Path

import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

TESTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TESTS / "reference"))
sys.path.insert(0, str(TESTS / "atk" / "common"))
from cpu_golden_cache import cached_cpu_outputs
from mega_recurrent_kda_reference import _make_case, mega_recurrent_kda_golden
from _ascendc_common_executor import _finite_tuple, _marker_device


def expected_outputs(inputs):
    directory = os.environ.get("FLA_KDA_GOLDEN_CACHE", TESTS.parent / "build" / "cpu-golden-cache")
    return cached_cpu_outputs(
        mega_recurrent_kda_golden, inputs, directory, ("conv_state", "replay_u", "replay_k", "replay_decay")
    )


def build_inputs(spec):
    lengths = spec["lengths"]
    batch, sequence, heads = len(lengths), max(lengths), spec["heads"]
    slots = spec.get("slots", batch if spec["name"].startswith("main") else batch + 1)
    inputs, hidden, _ = _make_case(batch, sequence, heads, slots, spec["conv"], seed=spec["seed"])
    hidden = torch.cat([hidden[i * sequence : i * sequence + n] for i, n in enumerate(lengths)])
    inputs["hidden_states"] = hidden.reshape(batch, sequence, -1) if spec["layout"] == "BSH" else hidden
    prefix_dtype = torch.int64 if spec["name"] == "cross" else torch.int32
    inputs["cu_seqlens"] = (
        torch.tensor([0, *accumulate(lengths)], dtype=prefix_dtype) if spec["layout"] == "TH" else None
    )
    inputs["state_indices"] = torch.arange(slots - batch, slots, dtype=torch.int32).flip([0])
    inputs["conv_num_accepted_tokens"] = torch.arange(batch, dtype=torch.int32) % min(sequence + 1, 4)
    inputs["A_log"] = torch.linspace(-0.75, 0.65, heads)
    inputs["dt_bias"] = torch.linspace(-0.5, 0.5, heads * 128).reshape(heads, 128)
    inputs["output_norm_weight"] = torch.linspace(0.8, 1.2, 128)
    inputs["rms_norm_eps"] = 1e-5
    if "qkv_scale_alg" in spec:
        cols = heads * 384
        original = inputs["qkv_projection_weight"].T.contiguous()
        # Vary the E8M0 scale independently over K group, output column and pair lane.
        pair = torch.arange(112)[:, None, None]
        scale = (121 + (pair + torch.arange(cols)[None, :, None] + 2 * torch.arange(2)) % 5).byte()
        factors = torch.exp2(scale.transpose(1, 2).reshape(224, 1, cols).float() - 127)
        weights = (original.reshape(224, 32, cols).float() / factors).reshape(1, 7168, cols)
        inputs.update(qkv_projection_weight=weights.to(torch.float8_e4m3fn),
                      qkv_projection_weight_scale=scale.unsqueeze(0), qkv_scale_alg=spec["qkv_scale_alg"])
    if spec["name"] == "slot_gaps":
        for name, gap in (("conv_state", 16), ("replay_u", 24), ("replay_k", 32), ("replay_decay", 40)):
            pool = inputs[name]
            inner = pool[0].numel()
            stride = inner + gap
            backing = torch.full(((slots - 1) * stride + inner,), 17.0, dtype=pool.dtype)
            view = backing.as_strided(pool.shape, (stride, *pool.stride()[1:]))
            inputs[name] = view.copy_(pool)
    return inputs


def to_device(value, device):
    if not isinstance(value, torch.Tensor):
        return value
    if value.is_contiguous():
        return value.to(device)
    size = (value.shape[0] - 1) * value.stride(0) + value[0].numel()
    storage = value.as_strided((size,), (1,)).to(device)
    return storage.as_strided(value.shape, value.stride())


def device_inputs(host, device):
    inputs = {key: to_device(value, device) for key, value in host.items()}
    inputs["a_log"] = inputs.pop("A_log")
    if "qkv_projection_weight_scale" in inputs:
        import torch_npu
        inputs["qkv_projection_weight"] = torch_npu.npu_format_cast(inputs["qkv_projection_weight"], 29)
    return inputs


@register("executor_mega_recurrent_kda")
class FunctionApi(BaseApi):
    def __call__(self, input_data, with_output=False):
        # ATK omits non_param attributes from the runtime tensor dataset.
        item = next(cfg for cfg in self.task_result.case_config.inputs if cfg.name == "case_spec")
        spec = json.loads(item.range_values)
        inputs = build_inputs(spec)
        if self.device == "cpu":
            return _finite_tuple(expected_outputs(inputs), golden=True)
        if self.device != "npu":
            raise ValueError(f"Unsupported ATK backend {self.device}")
        from fla_npu.ops.ascendc import npu_mega_recurrent_kda

        device = _marker_device(input_data)
        inputs = device_inputs(inputs, device)
        checkpoint = inputs["checkpoint_state"].clone()
        outputs = npu_mega_recurrent_kda(**inputs)
        mutable = ("conv_state", "replay_u", "replay_k", "replay_decay")
        if any(out.data_ptr() != inputs[key].data_ptr() for out, key in zip(outputs[1:], mutable)):
            raise AssertionError("State outputs must alias the supplied state pools")
        if not torch.equal(inputs["checkpoint_state"], checkpoint):
            raise AssertionError("The checkpoint changed during the call")
        # Check gapped slot views on CPU after copying their complete physical span.
        logical = tuple(t if t.is_contiguous() else to_device(t, "cpu").contiguous() for t in outputs)
        return _finite_tuple(logical)
