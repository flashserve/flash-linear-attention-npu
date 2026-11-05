"""Benchmark the issue #679 A5 model shapes on the V1 and V2 public kernels."""

import argparse
import gc
import itertools
import json
import statistics

import torch
import torch_npu

from fla_npu.ops.ascendc import _stable


MODEL_SHAPES = tuple(
    itertools.product((32, 64, 72), (32, 48, 128), (1, 2, 4, 8))
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("event", "profile"), default="event")
    parser.add_argument("--impl", choices=("v1", "v2", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    return parser.parse_args()


def make_inputs(bs, hv, tokens_per_batch, device):
    torch.manual_seed(679 + bs * 100 + hv + tokens_per_batch)
    hk = 16
    dim = 128
    total_tokens = bs * tokens_per_batch
    query = (torch.randn(total_tokens, hk, dim, device=device) * 0.05).to(torch.bfloat16)
    key = (torch.randn(total_tokens, hk, dim, device=device) * 0.05).to(torch.bfloat16)
    value = (torch.randn(total_tokens, hv, dim, device=device) * 0.05).to(torch.bfloat16)
    beta = torch.rand(total_tokens, hv, dtype=torch.bfloat16, device=device)
    gate = -torch.rand(total_tokens, hv, dtype=torch.float32, device=device) * 0.02
    state = torch.zeros(total_tokens, hv, dim, dim, dtype=torch.float32, device=device)
    actual_seq_lengths = torch.tensor(
        [0] + [tokens_per_batch] * bs, dtype=torch.int32, device=device
    )
    state_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)
    return query, key, value, beta, gate, state, actual_seq_lengths, state_indices


def make_callers(inputs):
    query, key, value, beta, gate, state, actual_seq_lengths, state_indices = inputs
    stream = _stable._current_stream_ptr()
    v1_op = _stable._op("npu_recurrent_gated_delta_rule")
    v2_op = _stable._op("npu_recurrent_gated_delta_rule_v2")
    scale = 128 ** -0.5

    def call_v1():
        return v1_op(
            query, key, value, state, beta, actual_seq_lengths, state_indices,
            None, gate, None, scale, stream,
        )

    def call_v2():
        return v2_op(
            query, key, value, state, beta, actual_seq_lengths, state_indices,
            None, gate, None, None, None, scale, False, False, False, False, stream,
        )

    return state, {"v1": call_v1, "v2": call_v2}


def event_measure(call, warmup, loops, samples):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    elapsed = []
    for _ in range(samples):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(loops):
            call()
        end.record()
        torch.npu.synchronize()
        elapsed.append(start.elapsed_time(end) * 1000.0 / loops)
    return elapsed


def profile_calls(call, warmup, loops):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    for _ in range(loops):
        call()
    torch.npu.synchronize()


def main():
    args = parse_args()
    implementations = ("v1", "v2") if args.impl == "both" else (args.impl,)
    device = torch.device("npu:0")
    for case_index, (bs, hv, tokens_per_batch) in enumerate(MODEL_SHAPES):
        case = {
            "case_index": case_index,
            "bs": bs,
            "hk": 16,
            "hv": hv,
            "tokens_per_batch": tokens_per_batch,
            "t": bs * tokens_per_batch,
            "k": 128,
            "v": 128,
        }
        inputs = make_inputs(bs, hv, tokens_per_batch, device)
        state, callers = make_callers(inputs)
        for implementation in implementations:
            state.zero_()
            torch.npu.synchronize()
            print("CASE_BEGIN", json.dumps({**case, "impl": implementation}), flush=True)
            if args.mode == "event":
                samples = event_measure(
                    callers[implementation], args.warmup, args.loops, args.samples
                )
                print(
                    "EVENT_RESULT",
                    json.dumps(
                        {
                            **case,
                            "impl": implementation,
                            "median_us": statistics.median(samples),
                            "samples_us": samples,
                        }
                    ),
                    flush=True,
                )
            else:
                profile_calls(callers[implementation], args.warmup, args.loops)
            print("CASE_END", case_index, implementation, flush=True)
        del callers, state, inputs
        gc.collect()
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
