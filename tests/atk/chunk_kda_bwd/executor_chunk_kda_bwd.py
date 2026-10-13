import math
import sys
from pathlib import Path

import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from _ascendc_common_executor import _case_spec, _marker_device, _rand, _randn


def build_inputs(spec, device, high_precision=False):
    shape = (int(spec['B']), int(spec['H']), int(spec['T']), 128)
    seed = int(spec['seed'])
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}[spec['dtype']]
    calc = torch.float64 if high_precision else dtype
    beta_type = spec.get('beta_dtype', 'bf16')
    beta_calc = torch.float64 if high_precision else {'bf16': torch.bfloat16, 'fp32': torch.float32}[beta_type]
    fp = torch.float64 if high_precision else torch.float32
    values = {
        'q': _randn(shape, spec['dtype'], calc, device, seed + 1, 0.05),
        'k': _randn(shape, spec['dtype'], calc, device, seed + 2, 0.05),
        'v': _randn(shape, spec['dtype'], calc, device, seed + 3, 0.05),
        'raw_g': _randn(shape, 'fp32', fp, device, seed + 4, 1.25),
        'beta': _rand(shape[:-1], beta_type, beta_calc, device, seed + 5, 0.1, 0.9),
        'A_log': _randn((shape[1],), 'fp32', fp, device, seed + 6, 0.12),
        'd_o': _randn(shape, spec['dtype'], calc, device, seed + 7, spec.get('d_o_std', 51.2)),
    }
    if spec.get('dt_bias', True):
        values['dt_bias'] = (_randn((shape[1], 128), 'fp32', torch.float32, device, seed + 8, 1.65) - 3.0).to(fp).reshape(-1)
    return values


@torch.enable_grad()
def run_cpu(values, spec=None):
    spec = spec or {}
    names = ['q', 'k', 'v', 'beta', 'raw_g', 'A_log']
    if 'dt_bias' in values:
        names.append('dt_bias')
    leaves = {name: values[name].detach().double().requires_grad_(True) for name in names}
    raw = leaves['raw_g']
    if 'dt_bias' in leaves:
        raw = raw + leaves['dt_bias'].view(1, raw.shape[1], 1, 128)
    decay = leaves['A_log'].exp()[None, :, None, None]
    gate = (-5.0 * torch.sigmoid(decay * raw) if spec.get('safe_gate', True)
            else -decay * torch.nn.functional.softplus(raw))
    batch, heads, tokens, dim = leaves['q'].shape
    state = torch.zeros(batch, heads, dim, dim, dtype=torch.float64)
    outputs = []
    starts = set(spec.get('cu_seqlens', [0, tokens])[:-1])
    for token in range(tokens):
        if token in starts:
            state = torch.zeros_like(state)
        state = state * gate[:, :, token].exp().unsqueeze(-1)
        key = leaves['k'][:, :, token]
        residual = leaves['v'][:, :, token] - torch.einsum('bhkv,bhk->bhv', state, key)
        delta = residual * leaves['beta'][:, :, token].unsqueeze(-1)
        state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(torch.einsum('bhkv,bhk->bhv', state, leaves['q'][:, :, token]) / math.sqrt(dim))
    loss = (torch.stack(outputs, dim=2) * values['d_o'].double()).sum()
    gradients = torch.autograd.grad(loss, [leaves[name] for name in names])
    if spec.get('cu_seqlens'):
        gradients = tuple(t.squeeze(0) if i < 5 else t for i, t in enumerate(gradients))
    return tuple(t.float() for t in gradients)


@register('executor_chunk_kda_bwd')
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        self.spec = _case_spec(input_data, 'chunk_kda_bwd')
        device = torch.device('cpu') if self.device == 'cpu' else _marker_device(input_data)
        self.values = build_inputs(self.spec, device, high_precision=self.device == 'cpu')
        if self.device == 'cpu':
            return
        from fla_npu.ops.ascendc import chunk_kda_bwd, chunk_kda_fwd
        self.operator = chunk_kda_bwd
        v = self.values
        if not self.spec.get('safe_gate', True):
            self.args = (v['q'], v['k'], v['v'], v['beta'], None, None, None, None,
                         None, None, None, None, v['d_o'], 1.0 / math.sqrt(128))
            self.kwargs = dict(safe_gate=False, implementation='legacy')
            return
        packed = bool(self.spec.get('cu_seqlens'))
        tokens = {name: v[name].squeeze(0) if packed else v[name]
                  for name in ('q', 'k', 'v', 'raw_g', 'beta', 'd_o')}
        cu = self.spec.get('cu_seqlens')
        saved = chunk_kda_fwd(
            tokens['q'], tokens['k'], tokens['v'], tokens['raw_g'], tokens['beta'], 1.0 / math.sqrt(128), 64,
            layout='NTD' if packed else 'BNSD', safe_gate=self.spec.get('safe_gate', True), lower_bound=-5.0,
            cu_seqlens=cu,
            use_gate_in_kernel=True, A_log=v['A_log'], dt_bias=v.get('dt_bias'),
            disable_recompute=True, return_intermediate_states=True,
            state_v_first=False,
        )
        _, _, gk, aqk, akk, w, _, qg, kg, v_new, h, _ = saved
        self.args = (tokens['q'], tokens['k'], tokens['v'], tokens['beta'], gk, aqk, akk, w, qg, kg,
                     v_new, h, tokens['d_o'], 1.0 / math.sqrt(128))
        self.kwargs = dict(raw_g=tokens['raw_g'], A_log=v['A_log'],
                           dt_bias=None if 'dt_bias' not in v else v['dt_bias'].view(self.spec['H'], 128),
                           chunk_size=64, safe_gate=self.spec.get('safe_gate', True), lower_bound=-5.0,
                           cu_seqlens=cu,
                           use_gate_in_kernel=True, disable_recompute=True, use_exp2=True,
                           state_v_first=False, implementation='legacy')
        torch.npu.synchronize()

    def __call__(self, input_data: InputDataset, with_output=False):
        if self.device == 'cpu':
            return run_cpu(self.values, self.spec)
        outputs = self.operator(*self.args, **self.kwargs)
        return tuple(outputs[i] for i in (0, 1, 2, 3, 4, 6, 7) if outputs[i] is not None)

    def export_custom_data(self, input_data: InputDataset):
        return self.spec
