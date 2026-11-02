"""Measure metadata enqueue/E2E and unrelated-stream isolation (not kernel timing)."""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import torch
import torch_npu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fla_npu.ops.ascendc import recurrent_kda
    from executor_recurrent_kda import build_inputs, call_options
    from gen_recurrent_kda import model_specs

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    result = {"warmup": 3, "samples": 5, "package": str(args.package), "cases": []}
    for case_id in (0, 7, 35):
        spec = dict(model_specs()[case_id], layout="TND", state_dtype="bf16", state_v_first=True)
        inputs = build_inputs(spec, device)
        opts = call_options(spec)
        current = torch.npu.current_stream(device)
        for _ in range(3):
            recurrent_kda(**inputs, **opts)
        current.synchronize()
        entry = {"id": case_id, "spec": spec, "single": [], "decode8": [], "two_heads": []}
        for mode, calls in (("single", 1), ("decode8", 8)):
            for _ in range(5):
                start = time.perf_counter_ns()
                for _ in range(calls):
                    recurrent_kda(**inputs, **opts)
                submitted = time.perf_counter_ns()
                current.synchronize()
                done = time.perf_counter_ns()
                entry[mode].append({"enqueue_us": (submitted-start)/1000,
                                    "e2e_us": (done-start)/1000, "calls": calls})

        # Independent State and input ownership per head; one framework stream
        # per host thread. Barrier separates preparation from measured work.
        head_inputs = [build_inputs(spec, device) for _ in range(2)]
        current.synchronize()
        streams = [torch.npu.Stream(device=device) for _ in range(2)]
        for _ in range(5):
            ready = threading.Barrier(3)
            errors = []
            times = [None, None]

            def head(index):
                try:
                    torch.npu.set_device(args.device)
                    with torch.npu.stream(streams[index]):
                        # Warm the host thread's runtime/context as well as
                        # the operator before measuring concurrent submissions.
                        for _ in range(3):
                            recurrent_kda(**head_inputs[index], **opts)
                        streams[index].synchronize()
                        ready.wait()
                        start = time.perf_counter_ns()
                        for _ in range(8):
                            recurrent_kda(**head_inputs[index], **opts)
                        submit = time.perf_counter_ns()
                        streams[index].synchronize()
                        end = time.perf_counter_ns()
                        times[index] = (start, submit, end)
                except BaseException as exc:
                    errors.append(repr(exc))
                    ready.abort()

            threads = [threading.Thread(target=head, args=(i,)) for i in range(2)]
            for thread in threads:
                thread.start()
            ready.wait(timeout=60)
            for thread in threads:
                thread.join(timeout=60)
            assert not errors and all(t is not None for t in times), errors
            origin = min(t[0] for t in times)
            entry["two_heads"].append({"enqueue_us": (max(t[1] for t in times)-origin)/1000,
                                       "e2e_us": (max(t[2] for t in times)-origin)/1000, "calls": 16})
        result["cases"].append(entry)
        args.output.write_text(json.dumps(result, indent=2))

    # A long unrelated stream makes a device-wide wait observable. Do not wait
    # for it on the current stream. The final explicit drain is outside timing.
    spec = dict(model_specs()[0], layout="TND", state_dtype="bf16", state_v_first=True)
    inputs = build_inputs(spec, device)
    opts = call_options(spec)
    background = torch.npu.Stream(device=device)
    a = torch.randn((4096, 4096), device=device, dtype=torch.float16)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    current.synchronize()
    result["isolation"] = []
    for _ in range(5):
        event = torch.npu.Event()
        with torch.npu.stream(background):
            for _ in range(128):
                torch.mm(a, b, out=c)
            event.record()
        start = time.perf_counter_ns()
        recurrent_kda(**inputs, **opts)
        elapsed = (time.perf_counter_ns()-start)/1000
        completed = event.query()
        result["isolation"].append({"enqueue_us": elapsed, "unrelated_complete_at_return": completed})
        background.synchronize()
        current.synchronize()
    args.output.write_text(json.dumps(result, indent=2))
    print("METADATA_STREAM_PERF_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
