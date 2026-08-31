#!/usr/bin/env python3
"""静态检查 QK-mask RegBase VF 的两段行循环边界。"""

from __future__ import annotations

import re
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "op_kernel"
    / "arch35"
    / "epilogue"
    / "block"
    / "block_epilogue_gdn_fwdo_qkmask.hpp"
)


def iter_aiv_stages(m_actual: int):
    """复现 kernel 中两个 AIV sub-block 及其单/双 stage 行划分。"""
    rows_per_subblock = (m_actual + 1) // 2
    for subblock in range(2):
        rows_this_subblock = (
            rows_per_subblock
            if subblock == 0
            else m_actual - rows_per_subblock
        )
        subblock_begin = subblock * rows_per_subblock
        if rows_this_subblock <= 32:
            yield subblock, 0, subblock_begin, rows_this_subblock, "single"
            continue

        rows_per_stage = (rows_this_subblock + 1) // 2
        for stage in range(2):
            rows_this_stage = (
                rows_per_stage
                if stage == 0
                else rows_this_subblock - rows_per_stage
            )
            yield (
                subblock,
                stage,
                subblock_begin + stage * rows_per_stage,
                rows_this_stage,
                "staged",
            )


def check_source_shape() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    safe_begin = (
        "const uint32_t secondLoopBegin = "
        "gbrcStart > VL ? gbrcStart : VL;"
    )
    assert source.count(safe_begin) == 2, "full/tail VF 必须都使用安全起点"
    assert source.count(
        "for (uint32_t absRow = secondLoopBegin;"
    ) == 2, "full/tail VF 必须都从 secondLoopBegin 迭代"
    assert not re.search(
        r"for\s*\(uint32_t\s+absRow\s*=\s*64U?\s*;",
        source,
    ), "禁止恢复固定从 64 开始的第二段循环"


def check_partition_bounds() -> None:
    covered_vf_paths: set[tuple[str, int]] = set()
    regression_case_seen = False

    for full_chunk in (64, 128):
        for m_actual in range(1, full_chunk + 1):
            aligned_n = ((m_actual + 15) // 16) * 16
            for v_dim in (128, 256):
                for subblock, stage, start, rows, mode in iter_aiv_stages(
                    m_actual
                ):
                    end = start + rows
                    first_rows = list(range(start, min(end, 64)))
                    second_begin = start if start > 64 else 64
                    second_rows = list(range(second_begin, end))

                    # 两段必须无重无漏地覆盖当前 AIV stage 的绝对行。
                    assert first_rows + second_rows == list(range(start, end)), (
                        full_chunk,
                        m_actual,
                        v_dim,
                        subblock,
                        stage,
                        start,
                        rows,
                    )

                    for abs_row in second_rows:
                        row = abs_row - start
                        mask_row = abs_row - 64
                        assert 0 <= row < rows
                        assert 0 <= mask_row < 64
                        assert (row + 1) * aligned_n <= rows * aligned_n

                    vf_path = (
                        "tail"
                        if mode == "single" or aligned_n != 128
                        else "full"
                    )
                    covered_vf_paths.add((vf_path, v_dim))
                    if (
                        full_chunk,
                        m_actual,
                        v_dim,
                        subblock,
                        stage,
                        start,
                        rows,
                    ) == (128, 128, 256, 1, 1, 96, 32):
                        regression_case_seen = True
                        assert second_begin == 96
                        assert second_rows[0] - start == 0

    assert covered_vf_paths == {
        ("full", 128),
        ("full", 256),
        ("tail", 128),
        ("tail", 256),
    }
    assert regression_case_seen


def main() -> None:
    check_source_shape()
    check_partition_bounds()
    print("qkmask VF boundary check: PASS")


if __name__ == "__main__":
    main()
