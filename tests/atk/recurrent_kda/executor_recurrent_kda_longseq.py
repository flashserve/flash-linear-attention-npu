"""Long-sequence regression using the existing independent CPU recurrence."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from executor_recurrent_kda import (
    BaseApi, InputDataset, TaskResult, _case_spec, _finite_tuple, _marker_device,
    recurrent_kda_reference, register,
)
from test_l2norm_epsilon import compare
from fla_npu.ops.ascendc import recurrent_kda

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
LENGTHS = [9, 15, 16, 17, 31, 32, 33, 64, 128, 256, 512, 1024]


def specifications():
    configs = []
    for i, tokens in enumerate(LENGTHS):
        for j, layout in enumerate(["BSND", "TND"]):
            configs.append(dict(
                T=tokens, layout=layout, state_v_first=(i+j) % 2 == 0,
                state_dtype="bf16" if i % 2 else "fp32",
                V=256 if i % 3 == 0 else 128,
                norm=i % 3 != 0, raw_gate=i % 3 == 1,
                beta_sigmoid=i % 2 == 0, negative=i % 4 == 0,
                safe_gate=i % 6 == 1,
                gate_dtype=["fp32", "bf16", "fp16"][i % 3],
                beta_dtype=["fp16", "fp32", "bf16"][i % 3],
            ))
    configs += [
        dict(T=17, B=2, cu=False),
        dict(T=9, B=2, state_v_first=False, state_dtype="bf16"),
        dict(T=17, layout="TND", cu=False, H=1, HV=1, norm=True),
        dict(T=33, H=1, HV=1, beta_dtype="bf16", beta_sigmoid=True),
        dict(T=17, H=48, HV=96, norm=True),
        dict(T=17, B=2, strided=True, norm=True),
        dict(T=33, strided=True, state_v_first=False, state_dtype="bf16"),
        dict(T=33, lengths=[0, 1, 8, 9, 17, 0, 33], layout="TND", padding=7),
        dict(T=33, lengths=[0, 1, 8, 9, 17, 0, 33], padding=7, norm=True),
        dict(T=17, indices="unique", accepted=9, metadata_dtype="int32"),
        dict(T=33, indices="repeat", accepted=33, metadata_dtype="int64",
             norm=True, state_dtype="bf16", inplace=True),
        dict(T=33, indices="repeat", accepted=9, metadata_dtype="int32",
             state_v_first=False, inplace=True),
        dict(T=17, lengths=[0, 17, 9], layout="TND", indices="repeat",
             accepted=9, metadata_dtype="int64", indices_packed=True),
        dict(T=33, lengths=[9, 0, 33], indices="unique", accepted=9,
             state_v_first=False, state_dtype="bf16", metadata_dtype="int32"),
        dict(T=17, output=False, inplace=True, norm=True),
        dict(T=17, output=False, inplace=False),
        dict(T=17, initial_none=True, norm=True),
    ]
    # Append new structures so the original 123 case IDs remain stable.
    for i, tokens in enumerate([63, 65, 127, 129, 255, 257, 511, 513, 1023]):
        for j, layout in enumerate(["BSND", "TND"]):
            configs.append(dict(
                T=tokens, layout=layout, state_v_first=(i+j) % 2 == 0,
                state_dtype="bf16" if (i+j) % 2 else "fp32",
                H=1 if i % 4 == 0 else 3, HV=1 if i % 4 == 0 else 6,
                V=256 if i % 3 == 0 else 128,
                norm=(i+j) % 2 == 0, raw_gate=i % 3 == 0,
                safe_gate=i % 3 == 0 and j == 1,
                beta_sigmoid=i % 2 == 0, negative=i % 4 == 0,
                gate_dtype=["fp32", "bf16", "fp16"][i % 3],
                beta_dtype=["fp16", "fp32", "bf16"][(i+j) % 3],
            ))
    configs += [
        dict(T=65, B=3, cu=False, V=256, state_v_first=False,
             state_dtype="bf16", norm=True, gate_dtype="bf16", beta_dtype="fp16"),
        dict(T=129, layout="TND", cu=False, H=3, HV=12, V=256,
             state_dtype="bf16", strided=True, norm=True, raw_gate=True,
             safe_gate=True, beta_sigmoid=True, negative=True),
        dict(T=65, lengths=[0, 9, 16, 17, 0, 33, 65], padding=9,
             norm=True, raw_gate=True, gate_dtype="bf16", beta_dtype="fp16"),
        dict(T=65, lengths=[0, 9, 16, 17, 0, 33, 65], layout="TND",
             indices="repeat", indices_packed=True, accepted=17,
             metadata_dtype="int64", state_dtype="bf16", V=256, norm=True,
             raw_gate=True, safe_gate=True, beta_sigmoid=True,
             gate_dtype="bf16", beta_dtype="fp16", inplace=True, output=False),
        dict(T=129, lengths=[9, 0, 65, 129], H=1, HV=1, norm=True,
             indices="unique", accepted=65, metadata_dtype="int32"),
        dict(T=257, H=2, HV=8, V=256, state_v_first=False, state_dtype="bf16",
             strided=True, norm=True, raw_gate=True, beta_sigmoid=True,
             negative=True, gate_dtype="fp16", beta_dtype="bf16"),
        dict(T=513, layout="TND", V=256, initial_none=True, norm=True,
             raw_gate=True, safe_gate=True, beta_sigmoid=True,
             gate_dtype="bf16", beta_dtype="fp16"),
    ]
    cases = []
    for config in configs:
        for seed in [20261017, 20261018, 20261019]:
            spec = dict(B=1, H=2, HV=4, V=128, layout="BSND",
                        state_v_first=True, state_dtype="fp32", norm=False,
                        raw_gate=False, beta_sigmoid=False, negative=False,
                        safe_gate=False, gate_dtype="fp32", beta_dtype="fp32",
                        cu=True, inplace=False, output=True)
            spec.update(config)
            spec.update(seed=seed, case_id=len(cases), op="recurrent_kda")
            cases.append(spec)
    # Two extra seeds stress the long tail while every structure keeps >=3 seeds.
    long_tail = next(spec for spec in cases
                     if spec["T"] == 1023 and spec["layout"] == "BSND")
    for seed in [20261020, 20261021]:
        spec = dict(long_tail, seed=seed, case_id=len(cases))
        cases.append(spec)
    return cases


def make_inputs(spec, device):
    rng = torch.Generator().manual_seed(spec["seed"])
    normal = lambda shape, scale: (torch.randn(shape, generator=rng) * scale)
    lengths = spec.get("lengths", [spec["T"]] * spec["B"])
    valid = sum(lengths)
    capacity = valid + spec.get("padding", 0)
    shape = ((capacity,) if spec["layout"] == "TND" else
             (1, capacity) if "lengths" in spec else (spec["B"], spec["T"]))
    h, hv, vdim = spec["H"], spec["HV"], spec["V"]
    q = normal((*shape, h, 128), 0.08).to(torch.bfloat16)
    k = normal((*shape, h, 128), 0.035).to(torch.bfloat16)
    v = normal((*shape, hv, vdim), 0.2).to(torch.bfloat16)
    g = (normal((*shape, hv, 128), 0.3) if spec["raw_gate"] else
         -torch.rand((*shape, hv, 128), generator=rng) * 0.04)
    beta = (normal((*shape, hv), 0.6) if spec["beta_sigmoid"] else
            0.1 + torch.rand((*shape, hv), generator=rng) * 0.6)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    meta_dtype = torch.int32 if spec.get("metadata_dtype") == "int32" else torch.int64
    indices = None
    accepted = None
    state_capacity = len(lengths)
    if spec.get("indices"):
        max_step = max(lengths) + 2
        indices = torch.full((len(lengths), max_step), -1, dtype=meta_dtype)
        cursor = 0
        for seq, length in enumerate(lengths):
            for token in range(length):
                slot = token if spec["indices"] == "unique" else token % 3
                # Explicitly repeat the destination across a window boundary and at the tail.
                if spec["indices"] == "repeat" and (token in [7, 8] or token >= length - 2):
                    slot = 0
                indices[seq, token] = cursor + slot
            cursor += max(length, 1) if spec["indices"] == "unique" else 3
        state_capacity = cursor + 2
        if spec.get("indices_packed"):
            indices = torch.cat([indices[i, :n] for i, n in enumerate(lengths)])
        accepted = torch.tensor(
            [min(spec.get("accepted", 1), n) if n else 0 for n in lengths], dtype=meta_dtype)
    tail = (vdim, 128) if spec["state_v_first"] else (128, vdim)
    state = normal((state_capacity, hv, *tail), 0.02).to(DTYPES[spec["state_dtype"]])
    if spec.get("initial_none"):
        state = None
    x = dict(q=q, k=k, v=v, g=g.to(DTYPES[spec["gate_dtype"]]),
             beta=beta.to(DTYPES[spec["beta_dtype"]]), initial_state=state,
             cu_seqlens=torch.tensor(offsets, dtype=meta_dtype) if spec["cu"] else None,
             ssm_state_indices=indices, num_accepted_tokens=accepted)
    if spec["raw_gate"]:
        x["A_log"] = normal((hv,), 0.1) - 2.0
        x["dt_bias"] = normal((hv, 128), 0.05)
    if device.type != "cpu":
        x = {name: value.to(device) if isinstance(value, torch.Tensor) else value
             for name, value in x.items()}
    if spec.get("strided"):
        for name in ["q", "k", "v"]:
            tensor = x[name]
            storage = torch.zeros((*tensor.shape[:-1], tensor.shape[-1] * 2),
                                  dtype=tensor.dtype, device=device)
            view = storage[..., :tensor.shape[-1]]
            view.copy_(tensor)
            x[name] = view
        tensor = x["initial_state"]
        storage = torch.zeros((tensor.shape[0], hv + 1, *tensor.shape[2:]),
                              dtype=tensor.dtype, device=device)
        x["initial_state"] = storage[:, :hv]
        x["initial_state"].copy_(tensor)
    kw = dict(layout=spec["layout"], state_v_first=spec["state_v_first"],
              output_final_state=spec["output"], inplace_final_state=spec["inplace"],
              use_qk_l2norm_in_kernel=spec["norm"], use_gate_in_kernel=spec["raw_gate"],
              use_beta_sigmoid_in_kernel=spec["beta_sigmoid"],
              allow_neg_eigval=spec["negative"] and spec["beta_sigmoid"],
              safe_gate=spec["safe_gate"])
    return x, kw, valid


def execute(spec, device, high_precision=False):
    x, kw, valid = make_inputs(spec, device)
    if high_precision:
        x = {name: value.to(torch.float64)
             if isinstance(value, torch.Tensor) and value.is_floating_point() else value
             for name, value in x.items()}
    before = x["initial_state"].clone() if x["initial_state"] is not None else None
    if device.type == "cpu":
        # CPU reference requires the final offset to equal its token count.
        reference = dict(x)
        if spec.get("padding"):
            for name in ["q", "k", "v", "g", "beta"]:
                reference[name] = x[name][:valid] if spec["layout"] == "TND" else x[name][:, :valid]
        outputs = recurrent_kda_reference(**reference, **kw)
        if not spec["output"] and not spec["inplace"]:
            outputs = outputs[0], before
    else:
        out, state = recurrent_kda(**x, **kw)
        torch_npu.npu.synchronize()
        if not spec["inplace"] and before is not None:
            assert torch.equal(x["initial_state"], before), "non-inplace call modified initial state"
        state = x["initial_state"] if state is None else state
        out = out[:valid] if spec["layout"] == "TND" else out
        if spec.get("padding") and spec["layout"] == "BSND":
            out = out[:, :valid]
        outputs = out, state
    if spec["output"] and not spec["inplace"]:
        # Non-inplace output slots with no token write are unspecified, unlike
        # the input pool, which is checked unchanged above.
        lengths = spec.get("lengths", [spec["T"]] * spec["B"])
        if x["ssm_state_indices"] is None:
            slots = [i for i, length in enumerate(lengths) if length]
        else:
            indices = x["ssm_state_indices"].cpu()
            if indices.ndim == 1:
                slots = sorted(set(indices.tolist()))
            else:
                slots = sorted({int(indices[i, j]) for i, length in enumerate(lengths)
                                for j in range(length)})
        selection = torch.tensor(slots, dtype=torch.int64, device=outputs[1].device)
        outputs = outputs[0], outputs[1].index_select(0, selection)
    return outputs


@register("executor_recurrent_kda_longseq")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self.high_precision = self.device == "cpu" and bool(task_result.is_benchmark_task)
        if self.device == "cpu":
            torch.set_num_threads(1)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        spec = _case_spec(input_data, "recurrent_kda")
        if "raw_gate" not in spec:
            restored = specifications()[int(spec["case_id"])]
            for name in ["B", "T", "H", "HV", "V", "layout", "state_v_first", "seed"]:
                if name in spec and spec[name] != restored[name]:
                    raise ValueError(f"restored long-sequence spec mismatch: {name}")
            spec = restored
        device = torch.device("cpu") if self.device == "cpu" else _marker_device(input_data)
        return _finite_tuple(execute(spec, device, self.high_precision))


def generate(path):
    template = json.loads((Path(__file__).with_name("atk_recurrent_kda.json")).read_text())[0]
    output = []
    for spec in specifications():
        case = copy.deepcopy(template)
        case["standard"]["acc"] = {"cv_fused_double_benchmark": {
            "max_re_ratio": 5, "avg_re_ratio": 1.5,
            "root_mean_squared_ratio": 1.5, "number_count_radio": 4}}
        case.update(id=spec["case_id"], default_seed=spec["seed"],
                    name=f"recurrent_kda_longseq_{spec['case_id']:03d}",
                    api_type="executor_recurrent_kda_longseq")
        for value in case["inputs"]:
            if value["name"] == "case_spec":
                value["range_values"] = json.dumps(spec)
            elif value["name"] in spec:
                value["range_values"] = spec[value["name"]]
        output.append(case)
    Path(path).write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print("LONGSEQ_CASES_GENERATED", len(output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate")
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.generate:
        generate(args.generate)
        return
    spec = specifications()[args.case_id]
    torch_npu.npu.set_device(args.device)
    expected = execute(spec, torch.device("cpu"))
    actual = tuple(value.cpu() for value in execute(spec, torch.device(f"npu:{args.device}")))
    compare(actual, expected)
    result = dict(case_id=args.case_id, spec=spec, passed=True)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    print("LONGSEQ_SMOKE_PASS", args.case_id)


if __name__ == "__main__":
    main()
