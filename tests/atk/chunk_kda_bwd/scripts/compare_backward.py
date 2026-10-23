import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch
import torch_npu
from atk.configs.case_config import CaseConfig
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--reference', type=Path)
parser.add_argument('--mode', choices=('correctness', 'performance'), default='correctness')
parser.add_argument('--lengths', default='1024,2048,4095,4096,4097,8191,8192,8193,16384,16385,32768,65536,131072')
parser.add_argument('--seeds', default='0,20260630,7')
parser.add_argument('--role', required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source / 'tests/atk/chunk_kda_bwd'))
from executor_chunk_kda_bwd import FunctionApi
from gen_chunk_kda_bwd import case

torch.set_num_threads(4)
torch.npu.set_device(0)
args.output.parent.mkdir(parents=True, exist_ok=True)
references = {}
if args.reference:
    references = {r['key']: r for r in json.loads(args.reference.read_text())}


def digest(tensor):
    value = tensor.detach().contiguous().cpu()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy()).hexdigest(),
                finite=bool(torch.isfinite(value).all()), max_abs=float(value.float().abs().max()))


records = []
lengths = [int(v) for v in args.lengths.split(',')]
seeds = [int(v) for v in args.seeds.split(',')]
total = len(lengths) * len(seeds)
for tokens in lengths:
    for seed in seeds:
        spec = dict(op='chunk_kda_bwd', B=1, H=32, T=tokens, dtype='bf16', beta_dtype='bf16',
                    dt_bias=True, seed=seed, soc='ascend910b', safe_gate=True)
        config = CaseConfig(**case(0, spec))
        api = FunctionApi(TaskResult(case_config=config, backend='npu', device_id=0))
        inputs = InputDataset(kwargs={'case_spec': json.dumps(spec),
                                     'low_precision_marker': torch.zeros(1, device='npu', dtype=torch.bfloat16)})
        api.init_by_input_data(inputs)
        input_records = [digest(t) for t in api.args if isinstance(t, torch.Tensor)]
        input_records += [dict(name=n, **digest(t)) for n, t in sorted(api.kwargs.items()) if isinstance(t, torch.Tensor)]
        output = api(inputs)
        torch.npu.synchronize()
        result = [digest(t) for t in output]
        if not all(r['finite'] for r in result):
            raise AssertionError('Non-finite gradient')
        reference_output = output
        for repeat in range(4):
            del output
            output = api(inputs)
            torch.npu.synchronize()
            if len(output) != len(reference_output) or any(
                a.shape != b.shape or a.dtype != b.dtype or
                not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                for a, b in zip(output, reference_output)
            ):
                raise AssertionError(f'Non-deterministic gradient: T={tokens}, seed={seed}, repeat={repeat + 1}')
        del reference_output
        key = f'{tokens}:{seed}'
        if key in references:
            reference = references[key]
            if input_records != reference['inputs']:
                raise AssertionError(f'Backward inputs differ: {key}')
            if result != reference['outputs']:
                raise AssertionError(f'Gradient bytes differ: {key}')
        row = dict(key=key, role=args.role, spec=spec, inputs=input_records, outputs=result,
                   deterministic_repeats=5, bitwise_reference=key in references)
        if args.mode == 'performance':
            for _ in range(10):
                output = api(inputs)
            torch.npu.synchronize()
            events = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(50)]
            for start, end in events:
                start.record()
                output = api(inputs)
                end.record()
            torch.npu.synchronize()
            times = [start.elapsed_time(end) for start, end in events]
            row.update(event_ms=times, p50_ms=statistics.median(times), p95_ms=sorted(times)[47],
                       mean_ms=statistics.mean(times), warmup=10, samples=50)
        records.append(row)
        args.output.write_text(json.dumps(records, indent=2) + '\n')
        print(f'[{args.role}] [{len(records)}/{total}] T={tokens} seed={seed} deterministic=5/5 '
              f'bitwise={row["bitwise_reference"]} p50_ms={row.get("p50_ms", "- ")}', flush=True)
        del output, inputs, api
        torch.npu.empty_cache()
print('COMPLETE', args.role, len(records), flush=True)
