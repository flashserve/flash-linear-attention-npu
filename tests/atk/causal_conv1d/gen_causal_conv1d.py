"""causal_conv1d 的 ATK 极简用例生成器"""

from __future__ import annotations

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None

OP_NAME = "causal_conv1d"

if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_causal_conv1d")
    class Generator(CaseGenerator):
        # 此时连 __init__ 和 __len__ 都可以省略，ATK 基类会自动根据 YAML 的笛卡尔积计算 length
        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            # 1. 获取 YAML 引擎当前枚举出的真实属性值 (从 inputs 中遍历 type == 'attr')
            attr_dict = {}
            for item in case_config.inputs:
                obj = item[0] if isinstance(item, list) else item
                if hasattr(obj, 'name') and hasattr(obj, 'range_values') and getattr(obj, 'type', None) == 'attr':
                    attr_dict[obj.name] = obj.range_values
            run_mode = attr_dict.get("run_mode", 0)
            d = attr_dict.get("D", 16)
            w = attr_dict.get("W", 4)
            b = attr_dict.get("B", 1)
            t = attr_dict.get("T", 8)
            head_num = attr_dict.get("head_num", 0)
            is_varlen = attr_dict.get("is_varlen", False)
            # ==========================================
            # 2. 物理与数学约束拦截 (返回 None 即告诉 ATK 丢弃该用例)
            # ==========================================
            # headNum 必须整除 D，且 headDim 必须 16 字节对齐
            if head_num > 0 and (d % head_num != 0 or (d // head_num) % 16 != 0):
                head_num = 0
            # Decode 模式 (且非 varlen) 时，T 必须严格为 1
            if run_mode == 1 and not is_varlen and t != 1:
                t = 1
            # Varlen 模式在目前的逻辑下，如果与某些特性不兼容，也可直接拦截
            if is_varlen and head_num > 0:
                is_varlen = False
            # ==========================================
            # 3. 动态修正 Tensor 的 Required 状态与物理 Shape
            # ==========================================
            for item in case_config.inputs:
                tensor_in = item[0] if isinstance(item, list) else item
                if tensor_in.name == "x":
                    if is_varlen:
                        # varlen 模式下 x 通常会被展平或变成 2D
                        tensor_in.shape = [b * t, d]
                    else:
                        tensor_in.shape = [b, t, d]
                elif tensor_in.name == "weight":
                    tensor_in.shape = [w, d]
                elif tensor_in.name == "bias":
                    # 可以通过 case_config.id 奇偶性随机决定是否测 bias，或者全部必选
                    tensor_in.required = (int(case_config.id) % 2 == 0)
                    if tensor_in.required:
                        tensor_in.shape = [d]
                elif tensor_in.name == "conv_states":
                    if run_mode == 1:
                        # Decode 模式：必须有状态输入
                        tensor_in.required = True
                        tensor_in.shape = [b, w - 1, d]
                    else:
                        # Prefill 模式：置为不必须
                        tensor_in.required = False
            # 可选：规范化用例名称，方便打点和 debug
            case_config.name = f"{OP_NAME}_{case_config.id:04d}_mode{run_mode}_D{d}_W{w}_T{t}_H{head_num}"
            return case_config