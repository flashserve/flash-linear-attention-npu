import json
from pathlib import Path

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
except ModuleNotFoundError as exc:
    if exc.name != 'atk':
        raise
    GENERATOR_REGISTRY = None

LENGTHS = (1024, 2048, 4095, 4096, 4097, 8191, 8192, 8193, 16384, 16385, 32768, 65536, 131072)
SEEDS = (0, 20260630, 7)
PROFILES = [(1, 32, t, 'bf16', 'bf16', True) for t in (1, 15, 16, 63, 64, 65, 127, 128, 129)]
PROFILES += [(1, h, 65, 'bf16', 'bf16', True) for h in (2, 16, 48, 96)]
PROFILES += [(2, 32, 65, 'bf16', 'bf16', True), (1, 32, 65, 'fp16', 'bf16', True),
             (1, 32, 65, 'bf16', 'fp32', True), (1, 32, 65, 'bf16', 'bf16', False)]
PROFILES += [(1, 32, 65, 'bf16', 'bf16', True),
             (1, 32, 128, 'bf16', 'bf16', True), (1, 32, 128, 'bf16', 'bf16', True)]
PROFILES += [(1, 32, 65, 'bf16', 'bf16', True), (1, 2, 65, 'bf16', 'bf16', True)]
PROFILES += [(1, 3, 65, 'bf16', 'bf16', True), (1, 32, 66, 'bf16', 'bf16', True)]


def spec(index):
    b, h, t, dtype, beta_dtype, bias = PROFILES[(index // len(SEEDS)) % len(PROFILES)]
    profile = (index // len(SEEDS)) % len(PROFILES)
    values = dict(op='chunk_kda_bwd', B=b, H=h, T=t, dtype=dtype, beta_dtype=beta_dtype,
                dt_bias=bias, seed=SEEDS[index % len(SEEDS)], soc='ascend910b',
                safe_gate=profile not in (17, 19), d_o_std=204.8 if profile in (20, 21) else 51.2)
    if profile in (18, 19):
        values['cu_seqlens'] = [0, 64, 128]
    if profile == 23:
        values['cu_seqlens'] = [0, 1, 66]
    return values


def case(index, config):
    dtypes = {str(i): config['dtype'] for i in range(7 if config.get('dt_bias', True) else 6)}
    if config.get('beta_dtype', 'bf16') == 'bf16':
        dtypes['3'] = 'bf16'
    return dict(id=index, default_seed=config['seed'], name='chunk_kda_bwd',
                aclnn_name='ChunkKdaBwd', version='v2.1', api='pytorch',
                api_type='executor_chunk_kda_bwd',
                expected_error_msg=None if config.get('safe_gate', True) else
                    'safe_gate=False is reserved but not supported', backward=False,
                standard={'acc': {'mixed_tolerance_bm': {'output_dtype_overrides':
                    dtypes}},
                    'perf': 'not_key'}, outputs=None,
                inputs=[dict(name='low_precision_marker', type='tensor', required=True,
                             dtype=config['dtype'], shape=[1], range_values=[0, 0], backward=False),
                        dict(name='fp32_marker', type='tensor', required=True,
                             dtype='fp32', shape=[1], range_values=[0, 0], backward=False),
                        dict(name='case_spec', type='attr', required=True, dtype='non_param',
                             shape=None, range_values=json.dumps(config, separators=(',', ':')), backward=False)])


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register('generator_chunk_kda_bwd')
    class Generator(CaseGenerator):
        def after_case_config(self, config):
            index = max(int(self.index) - 1, 0)
            values = spec(index)
            config.id = index
            config.default_seed = values['seed']
            config.expected_error_msg = case(index, values)['expected_error_msg']
            config.standard = type(config.standard)(**case(index, values)['standard'])
            for item in config.inputs:
                item = item[0] if isinstance(item, list) else item
                if item.name == 'low_precision_marker':
                    item.dtype = values['dtype']
                elif item.name == 'case_spec':
                    item.range_values = json.dumps(values, separators=(',', ':'))
            return config


if __name__ == '__main__':
    root = Path(__file__).parent
    accuracy = [case(i, spec(i)) for i in range(len(PROFILES) * len(SEEDS))]
    perf = [case(i, dict(op='chunk_kda_bwd', B=1, H=32, T=t, dtype='bf16', beta_dtype='bf16',
                         dt_bias=True, seed=20260630, soc='ascend910b')) for i, t in enumerate(LENGTHS)]
    mss = [case(i, spec(i)) for i in (0, 12, 24, 39, 42, 45, 48, 54, 66, 69)]
    mss += [case(len(accuracy) + i, json.loads(perf[j]['inputs'][-1]['range_values']))
            for i, j in enumerate((0, 4))]
    for suffix, values in (('', accuracy), ('_perf', perf), ('_mss', mss)):
        (root / ('atk_chunk_kda_bwd' + suffix + '.json')).write_text(json.dumps(values, indent=2) + '\n')
