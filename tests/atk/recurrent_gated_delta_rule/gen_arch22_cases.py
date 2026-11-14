"""Generate 200 arch22 cases, eight dtype keys and all legal fusion flags."""
import itertools
import json
from pathlib import Path
from gen_recurrent_gated_delta_rule import (
    CaseConfig, CaseGenerator, GENERATOR_REGISTRY, build_specs, _case_payload,
)

FLAGS = ('use_qk_l2norm_in_kernel', 'use_gate_in_kernel',
         'use_beta_sigmoid_in_kernel', 'allow_neg_eigval')

def build_arch22_specs():
    flags = [x for x in itertools.product((False, True), repeat=4) if not x[3] or x[2]]
    specs = []
    for i, spec in enumerate(build_specs()[:200]):
        # gk is explicitly outside the current API. These are new cases under
        # that contract, keeping the existing value ranges and tolerance.
        spec.update(arch22=True, design_id=f'RGDR-A{i+1:03}',
                    name=f'arch22_{i:03}', platform_focus='ascend910b,ascend910_93',
                    gate_mode='g', beta_dtype='fp32' if (i // 2) % 2 else 'bf16',
                    g_dtype='bf16' if (i // 4) % 2 else 'fp32',
                    state_dtype='fp32' if i % 2 else 'bf16')
        if spec['HV'] // spec['HK'] > 16:
            spec['HK'] = spec['HV'] // 16
        spec.update(zip(FLAGS, flags[(i // 8) % len(flags)]))
        if i % 17 == 0 and not spec['use_gate_in_kernel']:
            spec['gate_mode'] = 'none'
        if i >= 192:
            spec['seq_lengths'] = [[9], [16], [17], [9, 0, 17]][(i - 192) % 4]
            spec['B'] = len(spec['seq_lengths'])
            spec['T'] = spec.get('prefix_tokens', 0) + sum(spec['seq_lengths'])
            spec['block_num'] = spec['T']
            spec['state_indices'] = list(range(spec['T']))
            spec.pop('accepted_tokens', None)
        specs.append(spec)
    return specs


@GENERATOR_REGISTRY.register('generator_recurrent_gated_delta_rule_arch22')
class RecurrentGatedDeltaRuleArch22Generator(CaseGenerator):
    def after_case_config(self, case_config):
        del case_config
        specs = build_arch22_specs()
        index = max(int(self.index) - 1, 0) % len(specs)
        return CaseConfig(**_case_payload(index, specs[index]))


def main():
    folder = Path(__file__).resolve().parent
    cases = [_case_payload(i, spec) for i, spec in enumerate(build_arch22_specs())]
    def write(name, records):
        (folder / name).write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write('atk_recurrent_gated_delta_rule_arch22.json', cases)
    # All twelve legal flag combinations, all eight keys and window rollover.
    selected = [g * 8 + (g % 2) * 4 + j for g in range(12) for j in range(4)] + [198, 199]
    write('atk_recurrent_gated_delta_rule_arch22_mss.json', [cases[i] for i in selected])
    write('atk_recurrent_gated_delta_rule_arch22_perf.json', cases[:8])
    print('accuracy=200 determinism/memory=50 performance=8')

if __name__ == '__main__':
    main()
