"""示例文件：tests/atk/op_name/gen_op_name.py

说明：文件名按本仓 tests/atk/README.md 的 `gen_<算子>.py`；内容是 ATK 的 `generate` 插件，
按 ATK 仓 skill/atk-quality-guard 的要求**只做参数修正**（skill 只提供内容要求，不改命名）。

注意事项：
  1. 注册名必须与 op_name.yaml 顶层 `generate` 一致（本示例：`generator_op_name`）。
  2. 只做三类修正：修 dtype、修 shape、修 attr 的 range_values。
     YAML 负责参数空间（dtypes/dim_values/ranges/attr 候选值），这里**不得**再定义范围，
     也不要再用 `case_spec` 之类的属性承载 shape/dtype/档位——用例必须来自 YAML 声明的真实输入。
  3. 每条 C++ 硬校验都要有对应修正，且维度索引精确：只约束末维的 assert 只能改 `shape[-1]`，
     不能把同一裁剪泛化到所有维度。
  4. 约束来源优先级：C++ 内核 assert > Python 层校验 > ATK 框架限制。
  5. INT32 溢出看护：扫到 size/offset 乘法用 int32 时把 `OVERFLOW_GUARD_ENABLED` 置 True，
     并对超大 shape 档位**跳过** 2^31 修正（溢出是主动构造的看护场景，不是要抹掉的缺陷）。
  6. 这里只保证"变长元数据成对出现"；把它归一成合法前缀和、并按 chunk_size 重算
     chunk_indices 是 executor 的职责（生成期拿不到最终 token 数，修正后也会被 executor 再统一一次）。
"""

import os

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig

# 是否启用 INT32 溢出看护：扫描 tiling/kernel 里 size/offset 乘法是否用 int32。
# 本示例算子 tiling 用 int64 累加（见 op_name_tiling_processor.h 的 MulChecked），未命中 int32 pattern。
OVERFLOW_GUARD_ENABLED = False

# 平台相关 dtype 支持列表：与 aclnn L2 的 CheckDtype 保持一致（真实实现从算子源码提取）。
GATE_DTYPE_SUPPORT_LIST_ASCEND910B = ("bf16",)
GATE_DTYPE_SUPPORT_LIST_ASCEND950 = ("bf16", "fp32")
SOC_ASCEND950 = "ascend950"
# 目标平台：生成期没有 NPU，用环境变量显式指定，并与执行期的 `-soc` 保持一致。
SOC = os.environ.get("ATK_SOC", SOC_ASCEND950)

# layout → (token 轴, head 轴)；D 轴在所有 layout 下都是最后一维。
LAYOUT_AXES = {
    "BSND": (1, 2),
    "BNSD": (2, 1),
    "TND": (0, 1),
    "NTD": (1, 0),
}
# layout → x 的 rank：BSND/BNSD 为 4 维，TND/NTD 为打包 token 的 3 维。
LAYOUT_RANK = {"BSND": 4, "BNSD": 4, "TND": 3, "NTD": 3}


def _total_elems(shape):
    total = 1
    for dim in shape or ():
        total *= int(dim)
    return total


def _shrink_to(shape, limit):
    ratio = (float(limit) / float(_total_elems(shape))) ** (1.0 / len(shape))
    return [max(1, int(dim * ratio)) for dim in shape]


@GENERATOR_REGISTRY.register("generator_op_name")
class OpNameGenerator(CaseGenerator):
    """把 ATK 随机生成的参数修正到不触发算子硬校验的合法范围。"""

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        tensors = [item for item in case_config.inputs if item.type == "tensor"]
        attrs = [item for item in case_config.inputs if item.type == "attr"]
        by_name = dict((item.name, item) for item in case_config.inputs)

        # ---- 约束一：末维只支持 128（tiling 的 D 校验；只改末维，不外推到其它维度）----
        for name in ("x", "initial_state"):
            item = by_name.get(name)
            if item is not None and item.shape:
                item.shape[-1] = 128

        # ---- 约束二：layout 决定 x 的 rank，以及各辅助输入的轴对齐（g 比 x 少 D 维；
        #      a_log 是 [H]；initial_state 去掉 token 轴后与 x 的 B/H 一致，varlen 下 B=1）----
        layout = str(by_name["layout"].range_values) if "layout" in by_name else "BSND"
        token_axis, head_axis = LAYOUT_AXES.get(layout, LAYOUT_AXES["BSND"])
        if "x" in by_name and by_name["x"].shape:
            x_shape = list(by_name["x"].shape)
            rank = LAYOUT_RANK.get(layout, 4)
            if len(x_shape) > rank:                 # 只保留前 rank-1 维 + 末维 D
                x_shape = x_shape[: rank - 1] + [x_shape[-1]]
            while len(x_shape) < rank:              # 缺维度时补 1（补齐 batch 维）
                x_shape.insert(0, 1)
            by_name["x"].shape = x_shape
            if "g" in by_name:
                by_name["g"].shape = x_shape[:-1]
            if "a_log" in by_name:
                by_name["a_log"].shape = [x_shape[head_axis]]
            if "initial_state" in by_name:
                by_name["initial_state"].shape = [
                    dim for index, dim in enumerate(x_shape)
                    if index not in (token_axis, len(x_shape) - 1)] + [128]

        # ---- 约束三：dtype 落在当前平台支持列表内；state 必须与 x 同 dtype ----
        gate_supported = (GATE_DTYPE_SUPPORT_LIST_ASCEND950 if SOC == SOC_ASCEND950
                          else GATE_DTYPE_SUPPORT_LIST_ASCEND910B)
        if "g" in by_name and by_name["g"].dtype not in gate_supported:
            by_name["g"].dtype = gate_supported[0]
        if "initial_state" in by_name and "x" in by_name:
            by_name["initial_state"].dtype = by_name["x"].dtype

        # ---- 约束四：变长元数据成对出现（cu_seqlens 与 chunk_indices 必须同时给出）----
        has_cu = "cu_seqlens" in by_name and by_name["cu_seqlens"].shape
        has_chunk = "chunk_indices" in by_name and by_name["chunk_indices"].shape
        if bool(has_cu) != bool(has_chunk):
            if has_cu:
                by_name["chunk_indices"].shape = [
                    max(1, int(by_name["cu_seqlens"].shape[0]) - 1), 2]
            else:
                by_name["cu_seqlens"].shape = [2]

        # ---- 约束五：attr 取值域与相互约束 ----
        for item in attrs:
            if item.name == "chunk_size" and int(item.range_values or 64) not in (64, 128):
                item.range_values = 64
            if item.name == "layout" and str(item.range_values) not in ("BSND", "BNSD", "TND", "NTD"):
                item.range_values = "BSND"
            if item.name == "scale" and float(item.range_values or 1.0) <= 0.0:
                item.range_values = 1.0
            if item.name == "epsilon" and float(item.range_values or 1.0e-6) <= 0.0:
                item.range_values = 1.0e-6

        # ---- 约束六：总元素数上限（ATK 框架 2^34；常规用例压到 2^31 规避 int32 乘法溢出）----
        for item in tensors:
            if not item.shape:
                continue
            total = _total_elems(item.shape)
            if total > 2 ** 33:
                item.shape = _shrink_to(item.shape, 2 ** 33)
            elif total > 2 ** 31 and not OVERFLOW_GUARD_ENABLED:
                item.shape = _shrink_to(item.shape, 2 ** 31)

        return case_config
