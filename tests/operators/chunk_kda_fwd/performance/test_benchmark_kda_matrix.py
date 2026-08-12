from __future__ import annotations

import csv
import importlib.util
import sys
import zipfile
from argparse import Namespace
from pathlib import Path
from xml.etree import ElementTree


REPO_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = REPO_ROOT / "scripts" / "benchmark_kda_matrix.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("benchmark_kda_matrix", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_metric(path: Path, header: list[str], row: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerow(row)


def write_default_metrics(directory: Path) -> None:
    directory.mkdir(parents=True)
    tables = {
        "OpBasicInfo": (
            ["Op Name", "Op Type", "Task Duration(us)"],
            ["ChunkKdaFwd", "mix", "12.5"],
        ),
        "PipeUtilization": (
            [
                "block_id",
                "sub_block_id",
                "aic_cube_time(us)",
                "aic_mte2_time(us)",
                "aic_mte3_time(us)",
                "aic_fixpipe_time(us)",
                "aiv_vec_time(us)",
            ],
            ["0", "cube0", "8.0", "3.0", "2.0", "1.0", "4.0"],
        ),
        "ArithmeticUtilization": (
            ["block_id", "sub_block_id", "aic_cube_ratio", "aiv_vec_ratio"],
            ["0", "cube0", "0.8", "0.6"],
        ),
        "Memory": (
            ["block_id", "sub_block_id", "aic_main_mem_read_bw(GB/s)"],
            ["0", "cube0", "1"],
        ),
        "MemoryL0": (
            ["block_id", "sub_block_id", "aic_l0a_read_bw(GB/s)"],
            ["0", "cube0", "2"],
        ),
        "MemoryUB": (
            ["block_id", "sub_block_id", "aiv_ub_read_bw_vector(GB/s)"],
            ["0", "cube0", "3"],
        ),
        "L2Cache": (
            ["block_id", "sub_block_id", "aic_read_hit_rate(%)"],
            ["0", "cube0", "4"],
        ),
        "ResourceConflictRatio": (
            ["block_id", "sub_block_id", "aic_cube_wait_ratio"],
            ["0", "cube0", "5"],
        ),
    }
    for name, (header, row) in tables.items():
        write_metric(directory / f"{name}_sample.csv", header, row)


def test_pr297_case_matrix_contract():
    runner = load_runner()
    assert len(runner.CASES) == 48
    assert [case.atk_case_id for case in runner.CASES] == list(range(250, 298))
    assert {case.sequence for case in runner.CASES} == {
        1024,
        1536,
        2048,
        4096,
        8192,
        16384,
    }
    assert {case.distribution for case in runner.CASES} == {
        "single",
        "balanced8",
        "mixed_tail",
        "short64",
    }
    for case in runner.CASES:
        assert case.cu_seqlens[0] == 0
        assert case.cu_seqlens[-1] == case.sequence
        assert case.case_key.startswith("ascend950_h96_")
    assert runner.selected_cases("prefill_fwd_b1_s16384") == [runner.CASE_BY_ID["290"]]


def test_flat_default_metrics_are_merged_and_aliased(tmp_path):
    runner = load_runner()
    profile_dir = tmp_path / "OPPROF_flat"
    write_default_metrics(profile_dir)

    rows = runner.read_kernel_detail_rows(profile_dir)

    assert len(rows) == 1
    assert set(rows[0]["metric_file_types"].split(",")) == set(
        runner.PROFILE_METRIC_NAMES
    )
    result = {
        **runner.asdict(runner.CASES[0]),
        "sequence_count": 1,
        "chunk_count": 16,
        "status": "PASS",
        "profile_mode": "application_mstx",
        "aic_metrics": "Default",
        "_kernel_detail": rows,
    }
    detail = runner.build_kernel_detail_rows([result])[0]
    assert detail["mac_time_us"] == "8.0"
    assert detail["aic_mte2_time_us"] == "3.0"
    assert detail["aic_mte3_time_us"] == "2.0"
    assert detail["aic_fixpipe_time_us"] == "1.0"
    assert detail["vec_time_us"] == "4.0"
    assert detail["aic_main_mem_read_bw(GB/s)"] == "1"


def test_hierarchical_default_metrics_are_merged(tmp_path):
    runner = load_runner()
    profile_dir = tmp_path / "profile"
    write_default_metrics(profile_dir / "kernel_0" / "0")

    rows = runner.read_kernel_detail_rows(profile_dir)

    assert len(rows) == 1
    assert rows[0]["kernel_instance"] == "kernel_0"
    assert rows[0]["replay"] == "0"
    assert rows[0]["block_id"] == "0"
    assert rows[0]["sub_block_id"] == "cube0"


def test_workbook_has_one_sheet_per_pr297_case(tmp_path):
    runner = load_runner()
    result = {
        **runner.asdict(runner.CASES[0]),
        "sequence_count": 1,
        "chunk_count": 16,
        "status": "PASS",
        "profile_mode": "application_mstx",
        "aic_metrics": "Default",
        "_kernel_detail": [
            {
                "metric_file_types": ",".join(runner.PROFILE_METRIC_NAMES),
                "source_csvs": "OPPROF/PipeUtilization.csv",
                "kernel_instance": "OPPROF",
                "replay": "0",
                "block_id": "0",
                "sub_block_id": "cube0",
                "Op Name": "ChunkKdaFwd",
                "Op Type": "mix",
                "aic_cube_time(us)": "8.0",
                "aic_mte2_time(us)": "3.0",
                "aiv_vec_time(us)": "4.0",
            }
        ],
    }
    runner.write_kernel_detail_workbook(Namespace(output_dir=tmp_path), [result])

    workbook_path = tmp_path / "kernel_detail.xlsx"
    with zipfile.ZipFile(workbook_path) as workbook:
        namespace = {
            "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        }
        tree = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
        names = [sheet.attrib["name"] for sheet in tree.findall(".//x:sheet", namespace)]
        assert names == [f"case_{case_id}" for case_id in range(250, 298)]
        assert len(names) == 48
        first_sheet = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")
        assert "mac_time_us" in first_sheet
        assert "aic_mte2_time_us" in first_sheet
        assert "vec_time_us" in first_sheet
        assert "ChunkKdaFwd" in first_sheet
