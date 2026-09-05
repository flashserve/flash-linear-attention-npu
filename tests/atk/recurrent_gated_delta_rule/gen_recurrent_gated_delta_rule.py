"""Generate the reviewed recurrent_gated_delta_rule ATK case matrix.

The committed accuracy matrix is frozen by detailed-design ID:
  * 0-99: RGDR-P001 through RGDR-P100
  * 100-199: RGDR-G001 through RGDR-G100

All positive cases use dimensions aligned to the 32B BF16 transfer block;
unaligned dimensions are invalid and are covered by op_host interception tests.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise

    class CaseGenerator:
        """Fallback base used only by the standalone JSON materializer."""

    class CaseConfig:
        """Fallback marker used only by the standalone JSON materializer."""

    class _FallbackRegistry:
        def register(self, _name):
            return lambda generator: generator

    GENERATOR_REGISTRY = _FallbackRegistry()
    ATK_AVAILABLE = False
else:
    ATK_AVAILABLE = True


OP_NAME = "recurrent_gated_delta_rule"
SEED_BASE = 20260826
DIM_REQUIRED = 128
STANDARD = {"acc": "mixed_tolerance_bm", "perf": "not_key"}

STATE_LAYOUTS = {
    "contiguous",
    "noncontiguous",
    "head_padded",
    "block_padded",
    "head_block_padded",
}
INPUT_LAYOUTS = {
    "contiguous",
    "noncontiguous_qkv_beta_g",
    "noncontiguous_gk_metadata",
    "noncontiguous_qk",
    "noncontiguous_v_gates",
    "noncontiguous_all",
}


def _profile(name: str, **overrides) -> dict:
    profile = {
        "name": name,
        "seq_lengths": [2],
        "HK": 1,
        "HV": 1,
        "K": 128,
        "V": 128,
        "gate_mode": "both",
        "state_dtype": "fp32",
        "state_layout": "contiguous",
        "input_layout": "contiguous",
        "data_profile": "random",
        "platform_focus": "all",
        "design_routes": "P+A",
    }
    profile.update(overrides)
    return profile


def _build_positive_profiles() -> list[dict]:
    profiles: list[dict] = []

    def add(name: str, **overrides) -> None:
        profile = _profile(name, **overrides)
        profile["design_id"] = f"RGDR-P{len(profiles) + 1:03d}"
        profile["tags"] = "accuracy,generalization,positive"
        profiles.append(profile)

    # P001-P006: gate and state dtype.
    add("minimal_g_bf16_state", seq_lengths=[1], gate_mode="g", state_dtype="bf16")
    add("minimal_g_fp32_state", seq_lengths=[1], gate_mode="g", design_routes="P+A+D")
    add("two_token_g", gate_mode="g", state_dtype="bf16")
    add("two_token_gk", gate_mode="gk", state_dtype="bf16")
    add("two_token_both", state_dtype="bf16", design_routes="P+A+D")
    add("a2_a3_add_fold_k128", seq_lengths=[3], state_dtype="bf16", platform_focus="ascend910b,ascend910_93")

    # P007-P022: sequence metadata, prefix and accepted tokens.
    add("single_sequence_no_prefix", seq_lengths=[1], gate_mode="g")
    add("single_prefix_token", prefix_tokens=1, seq_lengths=[1], gate_mode="g")
    add("multi_token_prefix", prefix_tokens=3, seq_lengths=[2], gate_mode="g")
    add("two_varlen_sequences", seq_lengths=[2, 1], HK=1, HV=2)
    add("three_increasing_sequences", seq_lengths=[1, 2, 3], HK=1, HV=2)
    add("three_decreasing_sequences", seq_lengths=[3, 2, 1], HK=1, HV=2)
    add("middle_zero_length_sequence", seq_lengths=[2, 0, 1], HK=1, HV=2)
    add("prefix_and_multiple_zero_lengths", prefix_tokens=2, seq_lengths=[0, 2, 0, 1], HK=1, HV=2)
    add("accepted_single_token", seq_lengths=[1], accepted_tokens=[1], gate_mode="g", design_routes="P+A+D")
    add("accepted_sequence_head", seq_lengths=[3], accepted_tokens=[1], gate_mode="g", design_routes="P+A+D")
    add("accepted_sequence_middle", seq_lengths=[3], accepted_tokens=[2], gate_mode="g", design_routes="P+A+D")
    add("accepted_sequence_tail", seq_lengths=[3], accepted_tokens=[3], gate_mode="g", design_routes="P+A+D")
    add("accepted_tokens_gqa", seq_lengths=[3, 2, 1], accepted_tokens=[2, 1, 1], HK=2, HV=4, gate_mode="g")
    add("prefix_and_accepted", prefix_tokens=2, seq_lengths=[2, 3], accepted_tokens=[2, 3], HK=1, HV=2)
    add("max_length_accepted_ends", seq_lengths=[8, 8], accepted_tokens=[1, 8], HK=1, HV=2)
    add("complex_metadata_four_routes", prefix_tokens=1, seq_lengths=[1, 2, 4], accepted_tokens=[1, 2, 3], HK=2, HV=4, design_routes="P+A+D+F")

    # P023: the only supported key/value dimension and UB profile.
    dimensions = [
        (128, 128, "dims_128_128"),
    ]
    for key_dim, value_dim, name in dimensions:
        add(name, K=key_dim, V=value_dim, state_dtype="bf16")

    # P024-P039: head mapping and GVA basics.
    head_shapes = [
        (1, 1, "heads_1_1", "P+A+D"),
        (1, 2, "heads_1_2", "P+A"),
        (2, 4, "heads_2_4", "P+A"),
        (3, 6, "heads_3_6", "P+A"),
        (1, 3, "heads_1_3", "P+A"),
        (2, 6, "heads_2_6", "P+A"),
        (1, 4, "heads_1_4", "P+A"),
        (2, 8, "heads_2_8", "P+A"),
        (4, 8, "heads_4_8", "P+A"),
        (8, 16, "heads_8_16", "P+A"),
        (16, 64, "heads_16_64", "P+A"),
        (32, 96, "heads_32_96", "P+A"),
        (48, 96, "heads_48_96", "P+A"),
        (64, 128, "heads_64_128", "P+A"),
        (1, 256, "heads_1_256", "P+A"),
        (256, 256, "heads_256_256", "P+A"),
    ]
    for key_heads, value_heads, name, routes in head_shapes:
        add(name, HK=key_heads, HV=value_heads, design_routes=routes, data_profile="traceable_gva")

    # P040-P055: state layout and state-index mapping.
    add("state_bf16_contiguous", gate_mode="g", state_dtype="bf16")
    add("state_fp32_contiguous", gate_mode="g")
    add("state_unused_blocks", gate_mode="g", block_num=6, state_indices=[0, 1])
    add("state_sparse_indices", seq_lengths=[3], gate_mode="g", block_num=6, state_indices=[1, 3, 5])
    add("state_head_padding", HK=2, HV=4, state_layout="head_padded", design_routes="P+A+D+F")
    add("state_block_padding", seq_lengths=[2, 1], HK=1, HV=2, state_layout="block_padded", design_routes="P+A+D+F")
    add("state_head_block_padding", seq_lengths=[2, 1], HK=2, HV=4, state_layout="head_block_padded", design_routes="P+A+D+F")
    add("state_noncontiguous_fp32", HK=2, HV=4, state_layout="noncontiguous", design_routes="P+A+D+F")
    add("state_indices_reversed", seq_lengths=[3], state_indices=[2, 1, 0])
    add("state_indices_permuted_per_sequence", seq_lengths=[2, 2], state_indices=[1, 0, 3, 2], HK=1, HV=2)
    add("accepted_sparse_state_index", seq_lengths=[3], accepted_tokens=[2], block_num=6, state_indices=[1, 3, 5])
    add("disjoint_sequence_state_blocks", seq_lengths=[2, 2], block_num=6, state_indices=[0, 1, 4, 5], HK=1, HV=2)
    add("repeated_state_index", seq_lengths=[3], block_num=1, state_indices=[0, 0, 0])
    add("zero_length_without_state_read", seq_lengths=[2, 0, 1], HK=1, HV=2)
    add("functional_contiguous_state", HK=2, HV=4, design_routes="D+F")
    add("functional_padded_state", HK=2, HV=4, state_layout="head_block_padded", design_routes="D+F")

    # P056-P070: platform, route and regression representatives. ATK executes
    # the stable Python route; design_routes records the originating matrix row.
    add("a2_baseline_matrix", seq_lengths=[8], state_dtype="bf16", platform_focus="ascend910b", design_routes="P")
    add("a3_baseline_matrix", seq_lengths=[3, 2, 1], accepted_tokens=[2, 1, 1], HK=2, HV=4, gate_mode="g", platform_focus="ascend910_93", design_routes="P")
    add("aclnn_example_shape", seq_lengths=[2], HK=2, HV=4, design_routes="A")
    add("fast_mutable_baseline_shape", seq_lengths=[2], HK=2, HV=4, design_routes="D")
    add("fast_functional_baseline_shape", seq_lengths=[2], HK=2, HV=4, state_layout="head_padded", design_routes="F")
    add("four_route_g_bf16_shape", gate_mode="g", state_dtype="bf16", design_routes="P+A+D+F")
    add("four_route_gk_fp32_shape", gate_mode="gk", design_routes="P+A+D+F")
    add("four_route_both_gva3_varlen", seq_lengths=[2, 1], HK=2, HV=6, design_routes="P+A+D+F")
    add("four_route_accepted_prefix_gva2", prefix_tokens=1, seq_lengths=[2, 3], accepted_tokens=[1, 3], HK=2, HV=4, design_routes="P+A+D+F")
    add("noncontiguous_qkv_beta_g", HK=2, HV=4, gate_mode="g", input_layout="noncontiguous_qkv_beta_g", design_routes="P")
    add("noncontiguous_gk_metadata", seq_lengths=[2, 1], HK=2, HV=4, input_layout="noncontiguous_gk_metadata", design_routes="P")
    add("nondefault_stream_shape", HK=32, HV=96, platform_focus="all", design_routes="P", stream_mode="nondefault")
    add("reuse_state_two_calls", HK=2, HV=4, repeat_calls=2, design_routes="P")
    add("independent_stream_state_shape", seq_lengths=[2, 1], HK=2, HV=4, design_routes="P", stream_mode="independent")
    add("accepted_tokens_gqa_regression", seq_lengths=[3, 2, 1], accepted_tokens=[2, 1, 1], HK=2, HV=4, gate_mode="g", design_routes="P+A+D+F")

    # P071-P082: gate, state layout and input layout cross coverage.
    add("bf16_state_head_padding_g", HK=2, HV=4, gate_mode="g", state_dtype="bf16", state_layout="head_padded")
    add("bf16_state_block_padding_gk", seq_lengths=[2, 1], HK=2, HV=4, gate_mode="gk", state_dtype="bf16", state_layout="block_padded")
    add("bf16_state_head_block_padding_both", HK=2, HV=6, state_dtype="bf16", state_layout="head_block_padded")
    add("bf16_state_noncontiguous_both", HK=2, HV=8, state_dtype="bf16", state_layout="noncontiguous")
    add("fp32_state_head_padding_gk", HK=2, HV=6, gate_mode="gk", state_layout="head_padded")
    add("fp32_state_block_padding_both", seq_lengths=[2, 1], HK=2, HV=8, state_layout="block_padded")
    add("fp32_state_noncontiguous_g", HK=4, HV=8, gate_mode="g", state_layout="noncontiguous")
    add("noncontiguous_qk_g", HK=2, HV=4, gate_mode="g", input_layout="noncontiguous_qk")
    add("noncontiguous_v_gates_gk", HK=2, HV=6, gate_mode="gk", input_layout="noncontiguous_v_gates")
    add("noncontiguous_all_both", seq_lengths=[2, 1], HK=2, HV=8, input_layout="noncontiguous_all")
    add("head_padding_noncontiguous_qkv", HK=2, HV=4, state_layout="head_padded", input_layout="noncontiguous_qkv_beta_g")
    add("accepted_noncontiguous_gk_metadata", seq_lengths=[3], accepted_tokens=[2], HK=2, HV=6, gate_mode="gk", input_layout="noncontiguous_gk_metadata")

    # P083-P091: longer metadata chains, index reuse and scale values.
    add("sixteen_single_token_sequences", seq_lengths=[1] * 16, HK=2, HV=4, gate_mode="g")
    add("alternating_zero_length_sequences", seq_lengths=[1, 0] * 8, HK=2, HV=4, gate_mode="gk")
    add("long_prefix_max_mtp", prefix_tokens=8, seq_lengths=[8], HK=2, HV=4)
    add("eight_max_mtp_sequences", seq_lengths=[8] * 8, HK=1, HV=2, state_dtype="bf16")
    add("increasing_lengths_and_accepted", seq_lengths=list(range(1, 9)), accepted_tokens=list(range(1, 9)), HK=2, HV=4)
    add("disjoint_state_across_batches", seq_lengths=[1] * 8, block_num=8, state_indices=[7, 0, 6, 1, 5, 2, 4, 3], HK=2, HV=4, gate_mode="g")
    add("prefix_accepted_sparse_indices", prefix_tokens=4, seq_lengths=[2, 3], accepted_tokens=[1, 3], block_num=10, state_indices=[8, 7, 6, 5, 0, 2, 4, 6, 8], HK=2, HV=4)
    add("unit_scale", seq_lengths=[4], HK=2, HV=4, scale=1.0, gate_mode="g")
    add("small_scale", seq_lengths=[4], HK=2, HV=4, scale=0.03125, gate_mode="gk")

    # P092-P096: additional divisible head mappings.
    add("heads_5_10", HK=5, HV=10, data_profile="traceable_gva")
    add("heads_7_21", HK=7, HV=21, data_profile="traceable_gva")
    add("heads_12_48", HK=12, HV=48, data_profile="traceable_gva")
    add("heads_25_100", HK=25, HV=100, data_profile="traceable_gva")
    add("heads_128_256", seq_lengths=[1], HK=128, HV=256, state_dtype="bf16", data_profile="traceable_gva")

    # P097-P100: repeated calls and numerical profiles.
    add("repeat_two_calls_bf16_state", HK=2, HV=4, state_dtype="bf16", repeat_calls=2)
    add("repeat_three_calls_fp32_state", seq_lengths=[1], HK=2, HV=4, gate_mode="g", repeat_calls=3)
    add("per_head_gate_and_beta", HK=4, HV=8, gate_mode="g", gate_profile="per_head", beta_profile="per_head")
    add("strong_decay_accepted_padded_state", seq_lengths=[4, 2], accepted_tokens=[3, 1], HK=2, HV=6, gate_profile="strong_decay", state_layout="head_padded")

    if len(profiles) != 100:
        raise AssertionError(f"expected 100 P profiles, got {len(profiles)}")
    return profiles


def _build_gva_profiles() -> list[dict]:
    profiles: list[dict] = []

    def add(name: str, **overrides) -> None:
        overrides.setdefault("HK", 2)
        overrides.setdefault("HV", 4)
        overrides.setdefault("data_profile", "traceable_gva")
        profile = _profile(name, **overrides)
        profile["design_id"] = f"RGDR-G{len(profiles) + 1:03d}"
        profile["tags"] = "accuracy,generalization,gva,positive"
        profiles.append(profile)

    # G001-G016: mapping and group sizes.
    mappings = [
        (1, 2, "gva_1_2", "all", "P+A+D"),
        (2, 4, "gva_2_4", "all", "P+A"),
        (3, 6, "gva_3_6", "all", "P+A"),
        (48, 96, "gva_48_96", "ascend950", "P+A"),
        (64, 128, "gva_64_128", "all", "P+A"),
        (1, 3, "gva_1_3", "all", "P+A+D"),
        (2, 6, "gva_2_6", "ascend950", "P+A"),
        (32, 96, "gva_32_96", "ascend950", "P+A"),
        (1, 4, "gva_1_4", "all", "P+A"),
        (2, 8, "gva_2_8", "all", "P+A+D"),
        (24, 96, "gva_24_96", "ascend950", "P+A"),
        (1, 8, "gva_1_8", "all", "P+A"),
        (6, 96, "gva_6_96", "ascend950", "P+A"),
        (3, 96, "gva_3_96", "ascend950", "P+A"),
        (1, 256, "gva_1_256", "all", "P+A"),
        (256, 256, "gva_256_256", "all", "P+A+D"),
    ]
    for key_heads, value_heads, name, platform, routes in mappings:
        add(name, HK=key_heads, HV=value_heads, platform_focus=platform, design_routes=routes)

    # G017-G032: gate and state dtype.
    add("gva_g_bf16_state", HK=2, HV=8, gate_mode="g", state_dtype="bf16")
    add("gva_g_fp32_state", HK=2, HV=8, gate_mode="g")
    add("gva_gk_bf16_state", HK=2, HV=8, gate_mode="gk", state_dtype="bf16")
    add("gva_gk_fp32_state", HK=2, HV=8, gate_mode="gk")
    add("gva_both_bf16_state", HK=2, HV=8, state_dtype="bf16")
    add("gva_both_fp32_state", HK=2, HV=8)
    add("gva_per_head_g", HK=2, HV=6, gate_mode="g", gate_profile="per_head")
    add("gva_gk_column_pulse", HK=2, HV=6, gate_mode="gk", gate_profile="column_pulse")
    add("gva_near_zero_g", HK=32, HV=96, gate_mode="g", gate_profile="near_zero", platform_focus="ascend950")
    add("gva_strong_decay_g", HK=32, HV=96, gate_mode="g", gate_profile="strong_decay", platform_focus="ascend950")
    add("gva_near_zero_gk", HK=24, HV=96, gate_mode="gk", gate_profile="near_zero", platform_focus="ascend950")
    add("gva_strong_decay_gk", HK=24, HV=96, gate_mode="gk", gate_profile="strong_decay", platform_focus="ascend950")
    add("gva_per_head_beta", HK=1, HV=8, beta_profile="per_head")
    add("gva_state_pulse_hv2", HK=2, HV=6, state_profile="pulse_hv2")
    add("gva_state_pulse_hv3", HK=2, HV=6, state_profile="pulse_hv3")
    add("gva_bf16_fp32_state_pair", HK=48, HV=96, state_profile="traceable")

    # G033-G048: sequence metadata, accepted tokens and prefix.
    add("gva_single_token", seq_lengths=[1], HK=2, HV=6)
    add("gva_two_tokens", HK=2, HV=6)
    add("gva_max_mtp", seq_lengths=[8], HK=2, HV=6)
    add("gva_two_varlen_sequences", seq_lengths=[2, 1], HK=2, HV=8)
    add("gva_three_varlen_sequences", seq_lengths=[1, 2, 3], HK=2, HV=6)
    add("gva_zero_length_sequence", seq_lengths=[2, 0, 1], HK=2, HV=8)
    add("gva_prefix_varlen", prefix_tokens=2, seq_lengths=[1, 2], HK=2, HV=8)
    add("gva_prefix_multiple_zero_lengths", prefix_tokens=4, seq_lengths=[0, 2, 0, 1], HK=2, HV=6)
    add("gva_accepted_head", seq_lengths=[3], accepted_tokens=[1], HK=2, HV=6, design_routes="P+A+D")
    add("gva_accepted_middle", seq_lengths=[3], accepted_tokens=[2], HK=2, HV=6, design_routes="P+A+D")
    add("gva_accepted_tail", seq_lengths=[3], accepted_tokens=[3], HK=2, HV=6, design_routes="P+A+D")
    add("gva_case4_metadata", seq_lengths=[3, 2, 1], accepted_tokens=[2, 1, 1], HK=2, HV=4)
    add("gva_a5_max_chain_accepted", prefix_tokens=1, seq_lengths=[8, 8], accepted_tokens=[1, 8], HK=32, HV=96, platform_focus="ascend950")
    add("gva_accepted_traceable_state", seq_lengths=[3], accepted_tokens=[2], HK=2, HV=6, state_profile="traceable")
    add("gva_disjoint_sparse_blocks", seq_lengths=[2, 2], block_num=10, state_indices=[1, 3, 6, 8], HK=2, HV=8)
    add("gva_complex_metadata", prefix_tokens=1, seq_lengths=[2, 3], accepted_tokens=[1, 3], HK=24, HV=96, design_routes="P+A+D+F")

    # G049: aligned key/value dimensions and UB slicing at the supported 128 dimension.
    gva_dimensions = [
        (128, 128, 32, 96, "gva_dims_128_128", "all"),
    ]
    for key_dim, value_dim, key_heads, value_heads, name, platform in gva_dimensions:
        add(name, K=key_dim, V=value_dim, HK=key_heads, HV=value_heads, platform_focus=platform)

    # G050-G065: state layout, indices and route representatives.
    add("gva_state_bf16_contiguous", HK=2, HV=8, state_dtype="bf16")
    add("gva_state_fp32_contiguous", HK=2, HV=8)
    add("gva_state_head_padding_bf16", HK=2, HV=6, state_dtype="bf16", state_layout="head_padded", design_routes="P+A+D+F")
    add("gva_state_head_padding_fp32", HK=2, HV=6, state_layout="head_padded", design_routes="P+A+D+F")
    add("gva_state_block_padding", seq_lengths=[2, 1], HK=2, HV=8, state_layout="block_padded", design_routes="P+A+D+F")
    add("gva_state_head_block_padding", HK=32, HV=96, state_layout="head_block_padded", design_routes="P+A+D+F")
    add("gva_sparse_state_indices", block_num=4, state_indices=[1, 3], HK=2, HV=8)
    add("gva_reversed_state_indices", state_indices=[1, 0], HK=2, HV=6)
    add("gva_repeated_state_index", block_num=1, state_indices=[0, 0], HK=2, HV=8)
    add("gva_accepted_permuted_index", seq_lengths=[3], accepted_tokens=[2], state_indices=[2, 0, 1], HK=2, HV=6)
    add("gva_three_route_shape", HK=2, HV=8, design_routes="P+A+D")
    add("gva_mutable_functional_shape", HK=2, HV=6, state_layout="head_padded", design_routes="D+F")
    add("gva_noncontiguous_qk", HK=2, HV=8, input_layout="noncontiguous_qk", design_routes="P")
    add("gva_noncontiguous_v_gates", HK=2, HV=8, input_layout="noncontiguous_v_gates", design_routes="P")
    add("gva_nondefault_stream_shape", HK=32, HV=96, stream_mode="nondefault", design_routes="P")
    add("gva_all_complex_views", HK=24, HV=96, state_layout="head_block_padded", input_layout="noncontiguous_all", design_routes="P+A+D+F")

    # G066-G075: additional GVA ratios and upper head-count boundaries.
    add("gva_5_10", HK=5, HV=10, gate_mode="g")
    add("gva_7_14", HK=7, HV=14, gate_mode="gk")
    add("gva_10_20", HK=10, HV=20)
    add("gva_12_36", HK=12, HV=36)
    add("gva_16_48", HK=16, HV=48, gate_mode="g")
    add("gva_25_100", HK=25, HV=100, gate_mode="gk")
    add("gva_32_128", HK=32, HV=128)
    add("gva_64_256", seq_lengths=[1], HK=64, HV=256, state_dtype="bf16")
    add("gva_128_256", seq_lengths=[1], HK=128, HV=256, gate_mode="g", state_dtype="bf16")
    add("gva_4_256", seq_lengths=[1], HK=4, HV=256, gate_mode="gk", state_dtype="bf16")

    # G076-G083: gate, state dtype and numerical-profile cross coverage.
    add("gva_g_bf16_head_padding", HK=4, HV=12, gate_mode="g", state_dtype="bf16", state_layout="head_padded")
    add("gva_g_fp32_block_padding", seq_lengths=[2, 1], HK=4, HV=12, gate_mode="g", state_layout="block_padded")
    add("gva_gk_bf16_head_block_padding", HK=6, HV=18, gate_mode="gk", state_dtype="bf16", state_layout="head_block_padded")
    add("gva_gk_fp32_noncontiguous_state", HK=6, HV=18, gate_mode="gk", state_layout="noncontiguous")
    add("gva_both_bf16_near_zero", HK=8, HV=24, state_dtype="bf16", gate_profile="near_zero")
    add("gva_both_fp32_strong_decay", HK=8, HV=24, gate_profile="strong_decay")
    add("gva_per_head_gate_and_beta", HK=4, HV=12, gate_mode="g", gate_profile="per_head", beta_profile="per_head")
    add("gva_gk_pulse_traceable_state", HK=6, HV=18, gate_mode="gk", gate_profile="column_pulse", state_profile="traceable")

    # G084-G091: long varlen, accepted-token and state-index chains.
    add("gva_sixteen_single_token_sequences", seq_lengths=[1] * 16, HK=4, HV=12)
    add("gva_alternating_zero_length_sequences", seq_lengths=[1, 0] * 8, HK=4, HV=12)
    add("gva_long_prefix_max_mtp", prefix_tokens=8, seq_lengths=[8], HK=4, HV=12)
    add("gva_increasing_lengths_accepted", seq_lengths=list(range(1, 9)), accepted_tokens=list(range(1, 9)), HK=4, HV=12)
    add("gva_eight_max_mtp_accepted", seq_lengths=[8] * 8, accepted_tokens=list(range(1, 9)), HK=2, HV=8, state_dtype="bf16")
    add("gva_repeated_state_per_batch", seq_lengths=[2] * 4, block_num=4, state_indices=[0, 0, 1, 1, 2, 2, 3, 3], HK=4, HV=12)
    add("gva_sparse_state_chain", seq_lengths=[4, 3, 2, 1], block_num=12, state_indices=[9, 0, 8, 1, 7, 2, 6, 3, 5, 4], HK=4, HV=12)
    add("gva_prefix_accepted_complex_chain", prefix_tokens=4, seq_lengths=[4, 3, 2, 1], accepted_tokens=[2, 3, 1, 1], HK=24, HV=96)

    # G092-G099: combined state and input views plus repeat calls.
    add("gva_bf16_head_padding_noncontiguous_qk", HK=4, HV=12, state_dtype="bf16", state_layout="head_padded", input_layout="noncontiguous_qk")
    add("gva_fp32_block_padding_noncontiguous_v", HK=4, HV=12, state_layout="block_padded", input_layout="noncontiguous_v_gates")
    add("gva_bf16_head_block_noncontiguous_all", HK=4, HV=12, state_dtype="bf16", state_layout="head_block_padded", input_layout="noncontiguous_all")
    add("gva_fp32_noncontiguous_state_and_qkv", HK=4, HV=12, state_layout="noncontiguous", input_layout="noncontiguous_qkv_beta_g")
    add("gva_accepted_noncontiguous_metadata", seq_lengths=[3], accepted_tokens=[2], HK=4, HV=12, gate_mode="gk", input_layout="noncontiguous_gk_metadata")
    add("gva_head_padding_repeat_two", HK=4, HV=12, state_layout="head_padded", repeat_calls=2)
    add("gva_block_padding_repeat_three", seq_lengths=[1], HK=4, HV=12, state_layout="block_padded", repeat_calls=3)
    add("gva_large_complex_views", HK=16, HV=48, state_layout="head_block_padded", input_layout="noncontiguous_all")

    # G100: combined regression representative.
    add("gva_full_regression_chain", prefix_tokens=1, seq_lengths=[8, 4, 2, 1], accepted_tokens=[5, 3, 1, 1], HK=8, HV=24, state_dtype="bf16", state_layout="head_block_padded", input_layout="noncontiguous_all", design_routes="P+A+D+F")

    if len(profiles) != 100:
        raise AssertionError(f"expected 100 G profiles, got {len(profiles)}")
    return profiles


def _finalize(profile: dict, case_id: int) -> dict:
    spec = deepcopy(profile)
    prefix_tokens = int(spec.get("prefix_tokens", 0))
    seq_lengths = [int(value) for value in spec["seq_lengths"]]
    total_tokens = prefix_tokens + sum(seq_lengths)
    state_indices = [
        int(value) for value in spec.get("state_indices", range(total_tokens))
    ]
    minimum_blocks = max(state_indices, default=0) + 1
    spec.update(
        {
            "op": OP_NAME,
            "case_id": case_id,
            "seed": SEED_BASE + case_id,
            "route": "ascendc",
            "soc": "all",
            "dtype": "bf16",
            "B": len(seq_lengths),
            "T": total_tokens,
            "block_num": int(spec.get("block_num", max(total_tokens, minimum_blocks))),
            "scale": float(spec.get("scale", 1.0 / math.sqrt(spec["K"]))),
            "seq_lengths": seq_lengths,
            "state_indices": state_indices,
        }
    )
    return spec


def _validate_specs(specs: list[dict]) -> None:
    positive_count = sum(str(spec["design_id"]).startswith("RGDR-P") for spec in specs)
    gva_count = sum(str(spec["design_id"]).startswith("RGDR-G") for spec in specs)
    expected_ids = [f"RGDR-P{index:03d}" for index in range(1, positive_count + 1)]
    expected_ids += [f"RGDR-G{index:03d}" for index in range(1, gva_count + 1)]
    actual_ids = [str(spec["design_id"]) for spec in specs]
    if actual_ids != expected_ids:
        raise ValueError("detailed-design IDs are missing, duplicated or out of order")
    names = [str(spec["name"]) for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("case names must be unique")

    for case_id, spec in enumerate(specs):
        if int(spec["case_id"]) != case_id:
            raise ValueError(f"{spec['design_id']}: non-contiguous case_id")
        seq_lengths = [int(value) for value in spec["seq_lengths"]]
        if not seq_lengths or any(value < 0 or value > 8 for value in seq_lengths):
            raise ValueError(f"{spec['design_id']}: seq_lengths must be in [0, 8]")
        if int(spec.get("prefix_tokens", 0)) < 0:
            raise ValueError(f"{spec['design_id']}: prefix_tokens must be non-negative")
        if int(spec["T"]) <= 0:
            raise ValueError(f"{spec['design_id']}: T must be positive")
        if not (1 <= int(spec["HK"]) <= int(spec["HV"]) <= 256):
            raise ValueError(f"{spec['design_id']}: invalid HK/HV range")
        if int(spec["HV"]) % int(spec["HK"]) != 0:
            raise ValueError(f"{spec['design_id']}: HV must be divisible by HK")
        if int(spec["K"]) != DIM_REQUIRED or int(spec["V"]) != DIM_REQUIRED:
            raise ValueError(f"{spec['design_id']}: K/V must equal {DIM_REQUIRED}")
        if spec["gate_mode"] not in {"g", "gk", "both"}:
            raise ValueError(f"{spec['design_id']}: invalid gate_mode")
        if spec["state_dtype"] not in {"bf16", "fp32"}:
            raise ValueError(f"{spec['design_id']}: invalid state_dtype")
        if spec["state_layout"] not in STATE_LAYOUTS:
            raise ValueError(f"{spec['design_id']}: invalid state_layout")
        if spec["input_layout"] not in INPUT_LAYOUTS:
            raise ValueError(f"{spec['design_id']}: invalid input_layout")
        state_indices = [int(value) for value in spec["state_indices"]]
        if len(state_indices) != int(spec["T"]):
            raise ValueError(f"{spec['design_id']}: state_indices length must equal T")
        if any(value < 0 or value >= int(spec["block_num"]) for value in state_indices):
            raise ValueError(f"{spec['design_id']}: state index out of range")
        accepted = spec.get("accepted_tokens")
        if accepted is not None:
            accepted = [int(value) for value in accepted]
            if len(accepted) != len(seq_lengths):
                raise ValueError(f"{spec['design_id']}: accepted_tokens length must equal B")
            if any(value < 1 or value > seq_lengths[index] for index, value in enumerate(accepted)):
                raise ValueError(f"{spec['design_id']}: accepted token out of range")


def build_specs() -> list[dict]:
    profiles = _build_positive_profiles() + _build_gva_profiles()
    specs = [_finalize(profile, case_id) for case_id, profile in enumerate(profiles)]
    _validate_specs(specs)
    return specs


def _spec(index: int) -> dict:
    specs = build_specs()
    return deepcopy(specs[index % len(specs)])


def _input(name: str, dtype: str, value, *, input_type: str = "attr", shape=None):
    return {
        "name": name,
        "type": input_type,
        "required": True,
        "dtype": dtype,
        "shape": shape,
        "range_values": value,
        "backward": False,
    }


def _case_payload(case_id: int, spec: dict) -> dict:
    inputs = [
        _input("low_precision_marker", "bf16", [0, 0], input_type="tensor", shape=[1]),
        _input("fp32_marker", "fp32", [0, 0], input_type="tensor", shape=[1]),
        _input(
            "case_spec",
            "string",
            json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ),
        _input("design_id", "string", spec["design_id"]),
        _input("dtype", "string", spec["dtype"]),
        _input("state_dtype", "string", spec["state_dtype"]),
        _input("state_layout", "string", spec["state_layout"]),
        _input("input_layout", "string", spec["input_layout"]),
        _input("B", "int", spec["B"]),
        _input("T", "int", spec["T"]),
        _input("HK", "int", spec["HK"]),
        _input("HV", "int", spec["HV"]),
        _input("K", "int", spec["K"]),
        _input("V", "int", spec["V"]),
        _input("block_num", "int", spec["block_num"]),
        _input("gate_mode", "string", spec["gate_mode"]),
        _input("use_accepted_tokens", "bool", "accepted_tokens" in spec),
        _input("prefix_tokens", "int", int(spec.get("prefix_tokens", 0))),
        _input("case_id", "int", case_id),
        _input("seed", "int", spec["seed"]),
        _input("soc", "string", spec["soc"]),
        _input("route", "string", spec["route"]),
    ]
    return {
        "id": case_id,
        "default_seed": spec["seed"],
        "name": OP_NAME,
        "aclnn_name": "RecurrentGatedDeltaRule",
        "version": "v2.1",
        "api": "pytorch",
        "api_type": f"executor_{OP_NAME}",
        "expected_error_msg": None,
        "backward": False,
        "standard": STANDARD,
        "outputs": None,
        "inputs": inputs,
        "save_name": OP_NAME,
    }


def build_cases() -> list:
    if not ATK_AVAILABLE:
        raise RuntimeError("ATK is required to instantiate CaseConfig objects.")
    return [
        CaseConfig(**_case_payload(case_id, spec))
        for case_id, spec in enumerate(build_specs())
    ]


@GENERATOR_REGISTRY.register("generator_recurrent_gated_delta_rule")
class RecurrentGatedDeltaRuleGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        del case_config
        index = max(int(self.index) - 1, 0)
        spec = _spec(index)
        payload = _case_payload(index % len(build_specs()), spec)
        return CaseConfig(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the recurrent_gated_delta_rule ATK matrix."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name(f"atk_{OP_NAME}.json"),
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    specs = build_specs()
    payloads = [
        _case_payload(case_id, spec) for case_id, spec in enumerate(specs)
    ]
    args.output.write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.summary:
        print(f"total={len(payloads)} ids=0-{len(payloads) - 1}")
        for payload, spec in zip(payloads, specs):
            print(
                f"case_id={payload['id']} design_id={spec['design_id']} "
                f"name={spec['name']} "
                f"shape=T{spec['T']}-HK{spec['HK']}-HV{spec['HV']}-"
                f"K{spec['K']}-V{spec['V']} "
                f"state={spec['state_dtype']}/{spec['state_layout']} "
                f"input={spec['input_layout']} gate={spec['gate_mode']}"
            )


if __name__ == "__main__":
    main()
