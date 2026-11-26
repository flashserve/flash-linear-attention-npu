"""Use the FLA reference implementation on the exact tensors passed to the DUT."""

import json
import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
import conv_reference as gold


@register("executor_conv_backward")
class ConvBackward(BaseApi):
    def __call__(self, input_data, with_output=False):
        s = json.loads(
            self.task_result.case_config.get_input_data_config(
                name="case_spec"
            ).range_values
        )
        layout = s["input_layout"]
        packed = layout in ("NTD", "TND")
        t, d, w = s["T"], s["D"], s["W"]
        dtype = getattr(
            torch,
            {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[s["dtype"]],
        )
        lengths = [t // 2, t - t // 2] if packed else [t] * s["B"]
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)

        def make(shape, off):
            return gold.make_tensor(shape, dtype=dtype, offset=s["seed"] % 37 + off)

        weight = make((w, d), 11)
        shape = (len(lengths), w, d)
        h0 = make(shape, 23) if s["with_initial"] else None
        dht = make(shape, 7) if s["with_dht"] else None
        x = make((t, d) if packed else (len(lengths), t, d), 9)
        dy = make(tuple(x.shape), 17)
        activation = s["activation"]
        args = []
        ys = []
        for i, n in enumerate(lengths):
            xi = x[offsets[i] : offsets[i + 1]].unsqueeze(0) if packed else x[i : i + 1]
            gi = (
                dy[offsets[i] : offsets[i + 1]].unsqueeze(0)
                if packed
                else dy[i : i + 1]
            )
            hi = h0[i : i + 1] if h0 is not None else None
            fi = dht[i : i + 1] if dht is not None else None
            yi = (
                gold.causal_conv1d_preactivation_ref(xi, weight, hi).to(dtype)
                if activation or s["with_y"]
                else None
            )
            args.append((xi, yi, weight, gi, hi, fi, activation))
            ys.append(yi)
        if self.device == "cpu":
            grad = [
                gold.causal_conv1d_bwd_ref(*a, cast_reduction_outputs=False)
                for a in args
            ]
            dx = torch.cat([r[0].squeeze(0) if packed else r[0] for r in grad], dim=0)
            required = (
                dx,
                sum(r[1] for r in grad).to(dtype),
                sum(r[2] for r in grad).to(dtype),
            )
            return (
                required + (torch.cat([r[3] for r in grad]),)
                if h0 is not None
                else required
            )
        from fla_npu.ops.ascendc import npu_causal_conv1d_bwd

        y = (
            torch.cat([z.squeeze(0) if packed else z for z in ys])
            if ys[0] is not None
            else None
        )
        fmt = (
            gold.logical_to_ntd
            if layout == "NTD"
            else gold.logical_to_bnsd
            if layout == "BNSD"
            else None
        )
        if fmt:
            dy = fmt(dy, 2)
            if y is not None:
                y = fmt(y, 2)
        device = next(
            v.device for v in input_data.kwargs.values() if torch.is_tensor(v)
        )
        inputs = [
            v.to(device) if v is not None else None for v in (x, y, weight, dy, h0, dht)
        ]
        values = npu_causal_conv1d_bwd(
            *inputs,
            query_start_loc=offsets if packed else None,
            activation=activation,
            input_layout=layout,
        )
        # The ACLNN contract only defines dh0 when an initial state is provided.
        return values if h0 is not None else values[:3]
