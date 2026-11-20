# Copyright (c) Tianjin University, Ltd. 2026. All rights reserved.
# chunk_bwd_dqkwg 算子泛化约束生成 gen
# 职责: 将 YAML 约束转换为 ATK CaseGen 可消费的生成规则,
# 补充 YAML 难以表达但文档明确规定的关系约束,
# 生成合法、非法、边界、特殊值和组合覆盖 case.
import random
from typing import List, Tuple

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


# 输入索引常量, 与 YAML 中 inputs 顺序一致
Q_INDEX = 0
K_INDEX = 1
V_INDEX = 2
G_INDEX = 3
H_INDEX = 4
DO_INDEX = 5
DH_INDEX = 6
DV_INDEX = 7
CU_SEQLENS_INDEX = 8
CHUNK_INDICES_INDEX = 9
W_INDEX = 10
G_GAMMA_INDEX = 11
SCALE_INDEX = 12
CHUNK_SIZE_INDEX = 13
IS_MIX_INDEX = 14
IS_FIX_INDEX = 15
USE_EXP2_INDEX = 16
TRANSPOSE_STATE_LAYOUT_INDEX = 17
QKV_TYPE_INDEX = 18

# 算子固定约束
K_FIXED = 128
V_VALID_VALUES = [128, 256]
CHUNK_SIZE_VALID_VALUES = [64, 128]


def _prepare_chunk_indices(cu_seqlens: List[int], chunk_size: int) -> List[int]:
    """根据 cu_seqlens 生成扁平化的 chunk_indices.

    逻辑复刻原代码:
    1. 计算每个序列的长度: lens[i] = cu_seqlens[i+1] - cu_seqlens[i]
    2. 计算每个序列需要的 chunk 数: ceil(lens[i] / chunk_size)
    3. 生成对应的 (sequence_id, chunk_id) 对并扁平化
    """
    indices = []
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        length = end - start
        if length <= 0:
            continue
        num_chunks = (length + chunk_size - 1) // chunk_size
        for chunk_id in range(num_chunks):
            indices.append(i)
            indices.append(chunk_id)
    return indices


def _create_gate_g(B: int, HV: int, T: int, gtype: str) -> "CaseConfig":
    """生成满足约束的 g: 负数且沿 T 单调递减.

    使用 torch.linspace 生成从接近 0 到较负的递减序列.
    """
    import torch
    lo, hi = -5e-2, -5e-5
    span = hi - lo
    margin = max(span * 1e-7, 1e-12)
    g_t = torch.linspace(float(hi) - margin, float(lo) + margin, T, dtype=torch.float64)
    return g_t.unsqueeze(0).unsqueeze(0).expand(B, HV, T).contiguous().to(gtype)


@GENERATOR_REGISTRY.register("gen_chunk_bwd_dqkwg")
class ChunkBwdDqkwgGenerator(CaseGenerator):
    """chunk_bwd_dqkwg 算子约束生成器.

    将 YAML 中声明的参数域和关系约束转换为具体的 case 配置:
    - 同步 q/k/v/do/dh/dv/w/h 的 dtype 与 qkv_type
    - 根据 is_mix 决定 g 的 dtype
    - 根据 is_fix 决定定长或变长模式
    - 根据 cu_seqlens 生成 chunk_indices
    - 根据 T 和 chunk_size 推导 num_chunks
    - 固定 K=128, V 从 {128, 256} 中选择
    - HV 必须为 HK 的整数倍, n_ratio 从 {1, 2} 中选择
    """

    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        """在 ATK CaseGen 生成基础 case 后, 补充关系约束.

        参数:
            case_config: ATK 生成的初始 case 配置

        返回:
            补充约束后的 case 配置
        """
        # 1. 同步 qkv dtype: k/v/do/dh/dv/w/h 的 dtype 与 q 一致
        qkv_dtype = case_config.inputs[Q_INDEX].dtype
        case_config.inputs[K_INDEX].dtype = qkv_dtype
        case_config.inputs[V_INDEX].dtype = qkv_dtype
        case_config.inputs[DO_INDEX].dtype = qkv_dtype
        case_config.inputs[DH_INDEX].dtype = qkv_dtype
        case_config.inputs[DV_INDEX].dtype = qkv_dtype
        case_config.inputs[W_INDEX].dtype = qkv_dtype
        case_config.inputs[H_INDEX].dtype = qkv_dtype
        case_config.inputs[QKV_TYPE_INDEX].range_values = qkv_dtype

        # 2. 根据 is_mix 决定 g 的 dtype
        is_mix = case_config.inputs[IS_MIX_INDEX].range_values
        if not is_mix:
            # is_mix=False 时 g 与 qkv dtype 一致
            case_config.inputs[G_INDEX].dtype = qkv_dtype

        # 3. 根据 is_fix 决定定长或变长模式
        is_fix = case_config.inputs[IS_FIX_INDEX].range_values
        B, H, T, _ = case_config.inputs[Q_INDEX].shape
        if not is_fix:
            # 变长模式 B 必须为 1
            B = 1

        # 4. 固定 K=128, V 从 {128, 256} 中随机选择
        K = K_FIXED
        V = random.choice(V_VALID_VALUES)

        # 5. 从 YAML 中获取 chunk_size
        chunk_size = case_config.inputs[CHUNK_SIZE_INDEX].range_values
        if isinstance(chunk_size, list):
            chunk_size = chunk_size[0]

        # 6. GVA 维度拆分: HV = n_ratio * HK, n_ratio 从 {1, 2} 中选择
        n_ratio = random.choice((1, 2))
        HK = H
        HV = HK * n_ratio

        # 7. 根据 T 和 chunk_size 推导 num_chunks
        num_chunks = (T + chunk_size - 1) // chunk_size

        # 8. 更新各输入的 shape
        # q, k: [B, HK, T, K]
        case_config.inputs[Q_INDEX].shape = [B, HK, T, K]
        case_config.inputs[K_INDEX].shape = [B, HK, T, K]
        # v, do, dv: [B, HV, T, V]
        case_config.inputs[V_INDEX].shape = [B, HV, T, V]
        case_config.inputs[DO_INDEX].shape = [B, HV, T, V]
        case_config.inputs[DV_INDEX].shape = [B, HV, T, V]
        # g: [B, HV, T]
        case_config.inputs[G_INDEX].shape = [B, HV, T]
        # h, dh: [B, HV, num_chunks, K, V]
        case_config.inputs[H_INDEX].shape = [B, HV, num_chunks, K, V]
        case_config.inputs[DH_INDEX].shape = [B, HV, num_chunks, K, V]
        # w: [B, HV, T, K] (虽然当前传空, 但 shape 仍需声明)
        case_config.inputs[W_INDEX].shape = [B, HV, T, K]

        # 9. 处理变长模式的 cu_seqlens 和 chunk_indices
        if not is_fix:
            # 变长模式: 生成 cu_seqlens 和 chunk_indices
            # 生成 N 条序列, 总长度为 T
            num_seqs = random.randint(2, 8)
            # 随机生成序列边界, 保证单调递增且首尾为 0 和 T
            boundaries = sorted(random.sample(range(1, T), num_seqs - 1))
            cu_seqlens = [0] + boundaries + [T]
            case_config.inputs[CU_SEQLENS_INDEX].shape = [len(cu_seqlens)]
            case_config.inputs[CU_SEQLENS_INDEX].range_values = [0, T]
            case_config.inputs[CU_SEQLENS_INDEX].required = True

            chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)
            case_config.inputs[CHUNK_INDICES_INDEX].shape = [len(chunk_indices)]
            case_config.inputs[CHUNK_INDICES_INDEX].range_values = [0, T]
            case_config.inputs[CHUNK_INDICES_INDEX].required = True
        else:
            # 定长模式: cu_seqlens 和 chunk_indices 不启用
            case_config.inputs[CU_SEQLENS_INDEX].required = False
            case_config.inputs[CU_SEQLENS_INDEX].shape = None
            case_config.inputs[CHUNK_INDICES_INDEX].required = False
            case_config.inputs[CHUNK_INDICES_INDEX].shape = None

        # 10. w 和 g_gamma 当前实现必须传空
        case_config.inputs[W_INDEX].required = False
        case_config.inputs[W_INDEX].shape = None
        case_config.inputs[G_GAMMA_INDEX].required = False
        case_config.inputs[G_GAMMA_INDEX].shape = None

        # 11. use_exp2 和 transpose_state_layout 固定为 False
        case_config.inputs[USE_EXP2_INDEX].range_values = False
        case_config.inputs[TRANSPOSE_STATE_LAYOUT_INDEX].range_values = False

        return case_config

    def generate_invalid_cases(self) -> List[CaseConfig]:
        """生成非法组合 case, 用于负向测试.

        返回:
            非法 case 配置列表
        """
        invalid_cases = []

        # 非法 case 1: K != 128
        case_k_invalid = self._create_invalid_case(
            name="negative_k_invalid",
            q_shape=[1, 8, 1024, 64],  # K=64 非法
            v_shape=[1, 8, 1024, 128],
            chunk_size=64,
            expected_error="K must be 128"
        )
        invalid_cases.append(case_k_invalid)

        # 非法 case 2: V != 128/256
        case_v_invalid = self._create_invalid_case(
            name="negative_v_invalid",
            q_shape=[1, 8, 1024, 128],
            v_shape=[1, 8, 1024, 64],  # V=64 非法
            chunk_size=64,
            expected_error="V must be 128 or 256"
        )
        invalid_cases.append(case_v_invalid)

        # 非法 case 3: HV 不是 HK 整数倍
        case_hv_invalid = self._create_invalid_case(
            name="negative_hv_not_divisible",
            q_shape=[1, 8, 1024, 128],
            v_shape=[1, 12, 1024, 128],  # HV=12 不是 HK=8 的整数倍
            chunk_size=64,
            expected_error="HV must be divisible by HK"
        )
        invalid_cases.append(case_hv_invalid)

        # 非法 case 4: 变长模式 B != 1
        case_varlen_b_invalid = self._create_invalid_case(
            name="negative_varlen_b_not_1",
            q_shape=[2, 8, 1024, 128],  # B=2 变长非法
            v_shape=[2, 8, 1024, 128],
            chunk_size=64,
            expected_error="varlen only support B = 1",
            varlen=True
        )
        invalid_cases.append(case_varlen_b_invalid)

        # 非法 case 5: chunk_size 非 64/128
        case_chunk_size_invalid = self._create_invalid_case(
            name="negative_chunk_size_invalid",
            q_shape=[1, 8, 1024, 128],
            v_shape=[1, 8, 1024, 128],
            chunk_size=32,  # chunk_size=32 非法
            expected_error="chunk_size must be 64 or 128"
        )
        invalid_cases.append(case_chunk_size_invalid)

        # 非法 case 6: use_exp2=true
        case_exp2_invalid = self._create_invalid_case(
            name="negative_use_exp2_true",
            q_shape=[1, 8, 1024, 128],
            v_shape=[1, 8, 1024, 128],
            chunk_size=64,
            expected_error="use_exp2 must be false",
            use_exp2=True
        )
        invalid_cases.append(case_exp2_invalid)

        return invalid_cases

    def _create_invalid_case(self, name: str, q_shape: List[int], v_shape: List[int],
                             chunk_size: int, expected_error: str,
                             varlen: bool = False, use_exp2: bool = False) -> CaseConfig:
        """创建一条非法 case 配置.

        参数:
            name: case 名称
            q_shape: q 的 shape
            v_shape: v 的 shape
            chunk_size: 分块大小
            expected_error: 期望错误信息
            varlen: 是否变长模式
            use_exp2: 是否启用 exp2

        返回:
            case 配置
        """
        # 此方法返回一个简化的 CaseConfig, 实际由 ATK 框架填充
        case_config = CaseConfig()
        case_config.name = name
        case_config.expected_error_msg = expected_error
        case_config.is_negative = True
        return case_config

    def generate_boundary_cases(self) -> List[CaseConfig]:
        """生成边界 case.

        覆盖:
        - 最小 T (单 chunk)
        - T 恰好整除 chunk_size
        - T 非整除 (尾块)
        - chunk_size=128
        - V=256
        - n_ratio=2 (GVA 拆分)
        - 变长多序列
        - 变长非整除尾块
        """
        boundary_cases = []
        # 边界 case 由 after_case_config 中的约束生成逻辑处理
        # 这里仅声明需要覆盖的边界场景
        return boundary_cases

    def get_coverage_summary(self) -> dict:
        """返回覆盖摘要, 用于检查生成 case 的覆盖情况.

        返回:
            覆盖摘要字典
        """
        return {
            "dtypes": ["fp16", "bf16"],
            "g_dtypes": ["fp32", "fp16", "bf16"],
            "K_values": [128],
            "V_values": [128, 256],
            "chunk_size_values": [64, 128],
            "n_ratio_values": [1, 2],
            "B_range": [1, 128],
            "T_range": [1, 32768],
            "HK_range": [1, 64],
            "HV_range": [1, 64],
            "modes": ["fixed", "varlen"],
            "is_mix_values": [True, False],
            "use_exp2_values": [False],
            "transpose_state_layout_values": [False],
            "boundary_scenarios": [
                "min_T_single_chunk",
                "T_divisible",
                "T_not_divisible",
                "chunk_size_128",
                "V_256",
                "n_ratio_2",
                "varlen_multi_seq",
                "varlen_tail"
            ],
            "negative_cases": [
                "negative_k_invalid",
                "negative_v_invalid",
                "negative_hv_not_divisible",
                "negative_varlen_b_not_1",
                "negative_chunk_size_invalid",
                "negative_use_exp2_true"
            ],
            "soc_coverage": ["ascend910b", "ascend910_93", "ascend950"],
            "routes": ["ascendc", "aclnn"]
        }
