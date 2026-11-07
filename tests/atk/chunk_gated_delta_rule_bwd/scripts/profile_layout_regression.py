"""Profile layout regressions; an Add marker separates each complete executor call."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
p.add_argument('--case-json', type=Path)
p.add_argument('--iterations', type=int, default=4)
p.add_argument('--out-dir', type=Path, required=True)
a = p.parse_args()
op = a.repo / 'tests/atk/chunk_gated_delta_rule_bwd'
ex = load(op / 'executor_chunk_gated_delta_rule_bwd.py', 'profile_executor')
eq = load(op / 'scripts/check_layout_equivalence.py', 'profile_layout')
assert a.iterations > 0
cases = json.loads((a.case_json or (op / 'atk_chunk_gated_delta_rule_bwd.json')).read_text(encoding='utf-8'))
specs = [json.loads(next(i['range_values'] for i in case['inputs'] if i['name'] == 'case_spec'))
         for case in cases]
for layout in ([] if a.case_json else ['BNSD', 'BSND', 'NTD', 'TND']):
    spec = dict(name='layout_control_' + layout, B=1, HK=2, HV=4, T=65,
                with_state=True, with_dht=True, use_exp2=True, scalar_dtype='fp32',
                seed=20261107, layout=layout)
    if layout in ('NTD', 'TND'):
        spec['seqlens'] = [1, 64]
    specs.append(spec)

torch.npu.set_device(0)
a.out_dir.mkdir(exist_ok=False, parents=True)
marker = torch.ones(1, device='npu')
marker_out = torch.empty_like(marker)
manifest = []
for case_id, spec in enumerate(specs):
    inputs = ex.build_inputs(spec, torch.device('npu:0'))
    for iteration in range(a.iterations):
        out = eq.launch(ex, inputs, spec, spec['layout'] in ('BSND', 'TND'))
        torch.npu.synchronize()
        # A separate Add task delimits each call in op_summary. It is excluded
        # from operator durations, along with all input-construction tasks.
        torch.add(marker, marker, out=marker_out)
        torch.npu.synchronize()
        manifest.append(dict(case_id=case_id, name=spec['name'], layout=spec['layout'],
                             measured=iteration == a.iterations - 1, iteration=iteration, spec=spec))
    torch.save(tuple(value.cpu() if value is not None else None for value in out),
               a.out_dir / f'case{case_id}_outputs.pt')
    print(case_id, spec['name'], 'profiled', flush=True)
(a.out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
