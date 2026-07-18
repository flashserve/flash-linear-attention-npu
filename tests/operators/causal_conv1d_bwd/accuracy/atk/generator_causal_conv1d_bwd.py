# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


X_INDEX = 0
Y_INDEX = 1
WEIGHT_INDEX = 2
DY_INDEX = 3
INITIAL_STATE_INDEX = 4
DHT_INDEX = 5
QUERY_START_LOC_INDEX = 6
ACTIVATION_INDEX = 7
INPUT_LAYOUT_INDEX = 8


def _round_up(value: int, align: int) -> int:
    return ((int(value) + align - 1) // align) * align


MAX_B = 8
MAX_T = 32768
MAX_D = 1536
MAX_X_ELEMS = 4 * 1024 * 1024


@GENERATOR_REGISTRY.register("generator_causal_conv1d_bwd")
class CausalConv1dBwdGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        input_dtype = case_config.inputs[X_INDEX].dtype
        for idx in (Y_INDEX, WEIGHT_INDEX, DY_INDEX, INITIAL_STATE_INDEX, DHT_INDEX):
            case_config.inputs[idx].dtype = input_dtype

        b, t, d = case_config.inputs[X_INDEX].shape
        w = case_config.inputs[WEIGHT_INDEX].shape[0]
        input_layout = str(case_config.inputs[INPUT_LAYOUT_INDEX].range_values).upper()
        if input_layout not in ("BSND", "TND", "BNSD", "NTD"):
            input_layout = "BSND"

        b = min(max(1, int(b)), MAX_B)
        t = min(max(64, int(t)), MAX_T)
        d = min(max(16, _round_up(d, 16)), MAX_D)
        max_t_by_elems = max(64, MAX_X_ELEMS // max(1, b * d))
        t = min(t, max_t_by_elems)
        w = 4 if int(w) >= 4 else 2
        if t < w:
            t = w

        state_b = b
        if input_layout == "BNSD":
            n = 2 if d >= 32 else 1
            head_dim = _round_up((d + n - 1) // n, 16)
            if n * head_dim > MAX_D:
                n = 1
                head_dim = d
            d = n * head_dim
            x_shape = [b, t, d]
            grad_shape = [b, n, t, head_dim]
            state_shape = [b, w, d]
            query_start_loc = [0, t]
        elif input_layout == "NTD":
            n = 2 if d >= 32 else 1
            head_dim = _round_up((d + n - 1) // n, 16)
            if n * head_dim > MAX_D:
                n = 1
                head_dim = d
            d = n * head_dim
            state_b = 1
            x_shape = [t, d]
            grad_shape = [n, t, head_dim]
            state_shape = [state_b, w, d]
            query_start_loc = [0, t]
        elif input_layout == "TND":
            state_b = 1
            x_shape = [t, d]
            grad_shape = x_shape
            state_shape = [state_b, w, d]
            query_start_loc = [0, t]
        else:
            x_shape = [b, t, d]
            grad_shape = x_shape
            state_shape = [b, w, d]
            query_start_loc = [0, t]
        case_config.inputs[X_INDEX].shape = x_shape
        case_config.inputs[Y_INDEX].shape = grad_shape
        case_config.inputs[DY_INDEX].shape = grad_shape
        case_config.inputs[WEIGHT_INDEX].shape = [w, d]
        case_config.inputs[INITIAL_STATE_INDEX].shape = state_shape
        case_config.inputs[DHT_INDEX].shape = state_shape
        case_config.inputs[QUERY_START_LOC_INDEX].range_values = query_start_loc

        activation = int(case_config.inputs[ACTIVATION_INDEX].range_values)
        if activation not in (0, 1, 2):
            case_config.inputs[ACTIVATION_INDEX].range_values = 0

        return case_config
