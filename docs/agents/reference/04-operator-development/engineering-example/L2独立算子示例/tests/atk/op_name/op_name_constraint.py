"""示例文件：tests/atk/op_name/op_name_constraint.py

规范来源：ATK 仓 skill/atk-quality-guard（SKILL.md 步骤 3b、references/op-engineering.md「提取拦截条件与约束」）

注意事项：
  1. 本文件是**约束生成器**，只做三类修正：修 dtype、修 shape、修 attr 的 range_values。
     YAML 负责参数空间（dtypes/dim_values/ranges/attr 候选值），这里**不得**再定义范围，
     否则 YAML 形同空壳、泛化性差。
  2. 每条 C++ 硬校验都要在 `after_case_config` 里有对应修正，并且**维度索引精确**：
     只约束末维的 assert 只能改 `shape[-1]`，不能把同一裁剪泛化到所有维度。
  3. 约束来源优先级：C++ 内核 assert > Python 层校验 > ATK 框架限制；不要用 PyTorch 语义去"猜" assert。
  4. INT32 溢出看护：扫到 size/offset 乘法用 int32 时把 `OVERFLOW_GUARD_ENABLED` 置 True，
     并对超大 shape 档位**跳过** 2^31 修正（溢出是需要主动构造的看护场景，不是要抹掉的缺陷）。
  5. 注册名必须与 YAML 顶层 `generate` 一致（本示例：`op_name_constraint`）。
"""

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig

# 是否启用 INT32 溢出看护：扫描算子 tiling/kernel 里 size/offset 乘法是否使用 int32。
# 本示例算子 tiling 用 int64 累加 workspace 与任务数（见 op_name_tiling_processor.h 的 MulChecked），
# 未命中 int32 乘法 pattern，因此保持关闭；若目标算子命中则改为 True。
OVERFLOW_GUARD_ENABLED = False

# 平台相关的 dtype 支持列表：与 aclnn L2 的 CheckDtype 保持一致（不同 SoC 可能不同）。
# 真实实现里请从算子源码提取，不要凭记忆填写。
GATE_DTYPE_SUPPORT_LIST_ASCEND910B = ("bf16",)
GATE_DTYPE_SUPPORT_LIST_ASCEND950 = ("bf16", "fp32")

SOC_ASCEND950 = "ascend950"


def _total_elems(shape):
    total = 1
    for dim in shape or ():
        total *= int(dim)
    return total


def _shrink_to(shape, limit):
    """把 shape 等比缩到元素数不超过 limit（保持维度数不变）。"""
    ratio = (float(limit) / float(_total_elems(shape))) ** (1.0 / len(shape))
    return [max(1, int(dim * ratio)) for dim in shape]


@GENERATOR_REGISTRY.register("op_name_constraint")
class OpNameConstraintGenerator(CaseGenerator):
    """把 ATK 随机生成的参数修正到不触发算子硬校验的合法范围。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        tensors = [item for item in case_config.inputs if item.type == "tensor"]
        attrs = [item for item in case_config.inputs if item.type == "attr"]
        by_name = dict((item.name, item) for item in case_config.inputs)

        # ---- 约束一：末维只支持 128（对应 tiling 的 D 校验；只改末维，不外推到其它维度）----
        if "x" in by_name and by_name["x"].shape:
            by_name["x"].shape[-1] = 128
        if "initial_state" in by_name and by_name["initial_state"].shape:
            by_name["initial_state"].shape[-1] = 128

        # ---- 约束二：g 与 x 的 token/head 轴对齐（g 比 x 少最后一维）----
        if "x" in by_name and "g" in by_name and by_name["x"].shape:
            by_name["g"].shape = list(by_name["x"].shape[:-1])

        # ---- 约束三：dtype 必须落在当前平台的支持列表内（平台差异来自 aclnn L2 的 CheckDtype）----
        soc = str(getattr(case_config, "soc", "") or SOC_ASCEND950)
        gate_supported = (GATE_DTYPE_SUPPORT_LIST_ASCEND950 if soc == SOC_ASCEND950
                          else GATE_DTYPE_SUPPORT_LIST_ASCEND910B)
        if "g" in by_name and by_name["g"].dtype not in gate_supported:
            by_name["g"].dtype = gate_supported[0]
        if "initial_state" in by_name and "x" in by_name:
            # state 必须与主输入同 dtype。
            by_name["initial_state"].dtype = by_name["x"].dtype

        # ---- 约束四：变长元数据成对出现，且 cu_seqlens 必须以 0 开头、单调不减、末元素等于总 token 数 ----
        has_cu = "cu_seqlens" in by_name and by_name["cu_seqlens"].shape is not None
        has_chunk = "chunk_indices" in by_name and by_name["chunk_indices"].shape is not None
        if has_cu != has_chunk:
            # 只留一边没有意义：把缺席的一边按总 chunk 数补成最简形态（具体形状由用例语义决定）。
            if has_cu and not has_chunk:
                by_name["chunk_indices"].shape = [max(1, int(by_name["cu_seqlens"].shape[0]) - 1), 2]
            else:
                by_name["cu_seqlens"].shape = [2]
        if has_cu and by_name["cu_seqlens"].shape:
            by_name["cu_seqlens"].shape = [max(2, int(by_name["cu_seqlens"].shape[0]))]
            by_name["cu_seqlens"].range_values = None  # 数值由执行插件的 golden 按用例语义构造

        # ---- 约束五：attr 取值域与相互约束 ----
        for item in attrs:
            if item.name == "chunk_size":
                item.range_values = 64 if int(item.range_values or 64) not in (64, 128) else int(item.range_values)
            if item.name == "layout" and str(item.range_values) not in ("BSND", "BNSD", "TND", "NTD"):
                item.range_values = "BSND"
            if item.name == "scale" and float(item.range_values or 1.0) <= 0.0:
                item.range_values = 1.0
            if item.name == "epsilon" and float(item.range_values or 1.0e-6) <= 0.0:
                item.range_values = 1.0e-6

        # ---- 约束六：总元素数上限（ATK 框架限制 + 算子 int32 乘法风险）----
        # 精度标准要求单用例不超过 2^34；常规用例进一步压到 2^31，规避算子侧 int32 乘法溢出。
        # 若开启溢出看护，则保留 > 2^31 的档位，只保留 2^33 的 OOM 硬上限防护。
        for item in tensors:
            if not item.shape:
                continue
            total = _total_elems(item.shape)
            if total > 2 ** 33:
                item.shape = _shrink_to(item.shape, 2 ** 33)
            elif total > 2 ** 31 and not OVERFLOW_GUARD_ENABLED:
                item.shape = _shrink_to(item.shape, 2 ** 31)

        return case_config
