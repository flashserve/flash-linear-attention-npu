"""检查冻结 ATK 用例与统一 JSON 中的声明完全一致。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[4]
MANIFEST_PATH = ROOT / "tests/op_cases/chunk_kda_fwd_prepare.json"
GENERATOR_PATH = (
    ROOT / "tests/atk/chunk_kda_fwd_prepare/gen_chunk_kda_fwd_prepare.py"
)
ATK_DIR = GENERATOR_PATH.parent
VERIFIER_PATH = ATK_DIR / "scripts/verify_matrix.py"
BUILD_WHEEL_PATH = ROOT / "scripts/build_wheel.py"


def _load_generator():
    module_name = "chunk_kda_fwd_prepare_atk_generator_test"
    spec = importlib.util.spec_from_file_location(module_name, GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


def _load_verifier():
    module_name = "chunk_kda_fwd_prepare_atk_matrix_verifier_test"
    spec = importlib.util.spec_from_file_location(module_name, VERIFIER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {VERIFIER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load_verifier()


def _load_build_wheel():
    module_name = "chunk_kda_fwd_prepare_build_wheel_test"
    spec = importlib.util.spec_from_file_location(module_name, BUILD_WHEEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BUILD_WHEEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BUILD_WHEEL = _load_build_wheel()


class ChunkKdaFwdPrepareAtkGenerationTest(unittest.TestCase):
    def test_sanitizer_build_keeps_required_debug_artifacts(self):
        args = SimpleNamespace(
            build_args=[], debug=False, sanitizer=True, oom=False
        )
        self.assertEqual(
            BUILD_WHEEL._assemble_build_args(args),
            "--bisheng_flags=sanitizer,dump_cce",
        )
        generator = (
            ROOT / "cmake/scripts/util/ascendc_bin_param_build.py"
        ).read_text(encoding="utf-8")
        self.assertIn('build_cmd_var += " --op_debug_level=1"', generator)

    def test_generation_config_comes_from_canonical_manifest(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(GENERATOR.GENERATION, manifest["atk_generation"])
        source = GENERATOR_PATH.read_text(encoding="utf-8")
        self.assertNotIn("FUNCTIONAL_SPECS", source)
        self.assertNotIn("PERF_SPECS", source)

    def test_frozen_json_matches_generator(self):
        suites = (
            (
                ATK_DIR / "atk_chunk_kda_fwd_prepare.json",
                GENERATOR.build_accuracy_specs(),
                200,
            ),
            (
                ATK_DIR / "atk_chunk_kda_fwd_prepare_perf.json",
                GENERATOR.build_perf_specs(),
                9,
            ),
            (
                ATK_DIR / "atk_chunk_kda_fwd_prepare_mss.json",
                GENERATOR.build_mss_specs(),
                432,
            ),
        )
        for path, specs, expected_count in suites:
            with self.subTest(path=path.name):
                frozen = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(len(frozen), expected_count)
                self.assertEqual(frozen, GENERATOR._payloads(specs))

    def test_output_modes_cover_accuracy_and_inference_performance(self):
        accuracy_modes = {
            spec["backward_mode"]
            for spec in GENERATOR.build_accuracy_specs()
        }
        self.assertEqual(accuracy_modes, {"none", "recompute", "save"})
        self.assertEqual(
            {
                spec["backward_mode"]
                for spec in GENERATOR.build_perf_specs()
            },
            {"none"},
        )
        self.assertEqual(
            {
                spec["backward_mode"]
                for spec in GENERATOR.build_mss_specs()
            },
            {"none", "recompute", "save"},
        )

    def test_accuracy_has_200_fixed_seed_cases(self):
        specs = GENERATOR.build_accuracy_specs()
        logical_counts = Counter(spec["logical_case_key"] for spec in specs)
        self.assertEqual(len(specs), 200)
        self.assertEqual(len(logical_counts), 56)
        self.assertGreaterEqual(min(logical_counts.values()), 3)
        self.assertEqual(len({spec["seed"] for spec in specs}), 200)

    def test_mss_covers_every_reachable_tiling_key_and_pipeline_profile(self):
        specs = GENERATOR.build_mss_specs()
        keys = {spec["expected_tiling_key"] for spec in specs}
        self.assertEqual(len(specs), 432)
        self.assertEqual(len(keys), 432)
        self.assertEqual(min(keys), 2570)
        self.assertEqual(max(keys), 24452638)
        self.assertEqual(Counter(spec["backward_mode"] for spec in specs), {
            "none": 144,
            "recompute": 144,
            "save": 144,
        })
        self.assertEqual({spec["layout"] for spec in specs}, {
            "BNSD", "BSND", "NTD", "TND",
        })
        self.assertEqual(
            set(Counter(spec["mss_profile"] for spec in specs).values()),
            {36},
        )
        for profile in {spec["mss_profile"] for spec in specs}:
            profile_specs = [
                spec for spec in specs if spec["mss_profile"] == profile
            ]
            self.assertEqual({spec["gate_dtype"] for spec in profile_specs}, {
                "bf16", "fp32",
            })
            self.assertEqual({spec["beta_dtype"] for spec in profile_specs}, {
                "bf16", "fp32",
            })
            self.assertEqual({
                spec["use_qk_l2norm_in_kernel"] for spec in profile_specs
            }, {False, True})
            self.assertEqual({spec["use_exp2"] for spec in profile_specs}, {
                False, True,
            })
            self.assertEqual({
                (
                    spec["use_exp2"],
                    spec["dt_bias"],
                )
                for spec in profile_specs
                if spec["use_gate_in_kernel"]
            }, {
                (False, False), (False, True),
                (True, False), (True, True),
            })
        digest = hashlib.sha256(
            "".join(
                f"{spec['case_id']},{spec['expected_tiling_key']}\n"
                for spec in specs
            ).encode("ascii")
        ).hexdigest()
        self.assertEqual(
            digest,
            "f28a869f07d8e3f65e8d0c85768759bde635759a09d047cd78b4a9b7136d3b19",
        )
        tags = {tag for spec in specs for tag in spec["tags"].split(",")}
        for required in (
            "dense", "varlen", "auto_indices", "explicit_indices",
            "grid_stride", "slot_reuse", "head_split", "gva",
            "active1", "active2", "active3", "active4",
        ):
            self.assertIn(required, tags)

    def test_mss_profiles_cover_real_schedule_branches(self):
        profiles = {
            spec["mss_profile"]: spec
            for spec in GENERATOR.build_mss_specs()
        }

        def schedule(spec, core_count=128):
            if spec["cu_seqlens"]:
                cu = [int(value) for value in spec["cu_seqlens"].split(",")]
                chunks = sum(
                    (end - begin + 63) // 64
                    for begin, end in zip(cu, cu[1:])
                )
            else:
                chunks = int(spec["B"]) * ((int(spec["T"]) + 63) // 64)
            head_split = chunks < core_count and int(spec["HK"]) > 1
            heads_per_partition = (
                int(spec["HV"]) // int(spec["HK"])
                if head_split else int(spec["HV"])
            )
            total_work = chunks * (int(spec["HK"]) if head_split else 1)
            return chunks, heads_per_partition, total_work

        for name in (
            "varlen_ntd_grid_stride_auto",
            "varlen_tnd_grid_stride_explicit",
        ):
            chunks, _, total_work = schedule(profiles[name])
            self.assertEqual(chunks, 129)
            self.assertEqual(total_work, 129)

        chunks, heads_per_partition, total_work = schedule(
            profiles["dense_bnsd_grid_stride_gva_chunk_only"]
        )
        self.assertEqual(chunks, 130)
        self.assertEqual(heads_per_partition, 4)
        self.assertEqual(total_work, 130)

        chunks, heads_per_partition, total_work = schedule(
            profiles["dense_bsnd_gva"]
        )
        self.assertEqual(chunks, 2)
        self.assertEqual(heads_per_partition, 2)
        self.assertEqual(total_work, 4)

    def test_executor_checks_public_output_contract_without_extra_npu_kernels(self):
        source = (
            ATK_DIR / "executor_chunk_kda_fwd_prepare.py"
        ).read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count("return result.contiguous()"), 2)
        self.assertIn("_validate_output_contract(", source)
        self.assertIn("torch.npu.synchronize()", source)
        self.assertIn("if not with_output:", source)
        self.assertIn("output.detach().cpu()", source)

    def test_matrix_verifier_requires_matching_host_and_launch_keys(self):
        key = 8391178
        log = (
            "ChunkKdaFwdPrepare tiling: outputMode=1, "
            f"tilingKey={key}, workspace=1024\n"
            "OpName:[aclnnChunkKdaFwdPrepare_0_ChunkKdaFwdPrepare] "
            f"Tiling Key: {key}, len: 48\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "console.log"
            path.write_text(log, encoding="utf-8")
            self.assertEqual(VERIFIER._logged_keys(path), ({key}, {key}))

            path.write_text(
                log.replace("outputMode=1", "outputMode=0"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "outputMode"):
                VERIFIER._logged_keys(path)

    def test_matrix_verifier_requires_active_sanitizer_kernel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            console = root / "console.log"
            sanitizer = root / "memcheck.log"
            console.write_text(
                "Start memcheck sanitizer on kernel "
                "ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_2570\n",
                encoding="utf-8",
            )
            sanitizer.write_text("memcheck completed\n", encoding="utf-8")
            count, keys, console_hash, sanitizer_hash = VERIFIER._sanitizer_evidence(
                console, sanitizer, "memcheck", {2570}
            )
            self.assertEqual(count, 1)
            self.assertEqual(keys, {2570})
            self.assertEqual(len(console_hash), 64)
            self.assertEqual(len(sanitizer_hash), 64)

            sanitizer.write_text(
                "No active sanitizer tool on kernel "
                "ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_2570\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "未加载 sanitizer"):
                VERIFIER._sanitizer_evidence(
                    console, sanitizer, "memcheck", {2570}
                )

    def test_runtime_manifest_binds_compiled_tiling_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            kernel_dir = (
                root
                / "vendors/fla_npu_transformer/op_impl/ai_core/tbe/kernel"
                / "ascend950/chunk_kda_fwd_prepare"
            )
            kernel_dir.mkdir(parents=True)
            (kernel_dir / "ChunkKdaFwdPrepare_hash.o").write_bytes(b"kernel")
            (kernel_dir / "ChunkKdaFwdPrepare_hash.json").write_text(
                json.dumps(
                    {
                        "binFileName": "ChunkKdaFwdPrepare_hash",
                        "binFileSuffix": ".o",
                        "kernelList": [
                            {
                                "tilingKey": 2570,
                                "kernelName": "ChunkKdaFwdPrepare_hash_2570",
                            },
                            {
                                "tilingKey": 8391178,
                                "kernelName": "ChunkKdaFwdPrepare_hash_8391178",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            op_api_lib = (
                root
                / "vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so"
            )
            op_api_lib.parent.mkdir(parents=True)
            op_api_lib.write_bytes(b"op_api")
            case_file = root / "cases.json"
            case_file.write_text(
                json.dumps(
                    [
                        {
                            "id": 0,
                            "inputs": [
                                {
                                    "name": "case_spec",
                                    "range_values": json.dumps(
                                        {"expected_tiling_key": 2570}
                                    ),
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(
                "os.environ",
                {
                    "FLA_NPU_OPP_PATH": str(root),
                    "FLA_NPU_OP_API_LIB": str(op_api_lib),
                    "ASCEND_CUSTOM_OPP_PATH": "",
                },
                clear=False,
            ):
                manifest = VERIFIER._runtime_manifest(
                    case_file, "ascend950", False, False
                )
            self.assertEqual(manifest["platform"], "ascend950")
            self.assertEqual(manifest["object_count"], 1)
            self.assertEqual(manifest["compiled_tiling_key_count"], 2)
            with mock.patch.dict(
                "os.environ",
                {
                    "FLA_NPU_OPP_PATH": str(root),
                    "FLA_NPU_OP_API_LIB": str(op_api_lib),
                    "ASCEND_CUSTOM_OPP_PATH": "",
                },
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "编译 key"):
                    VERIFIER._runtime_manifest(
                        case_file, "ascend950", False, True
                    )

    @unittest.skipUnless(
        importlib.util.find_spec("openpyxl"), "需要 openpyxl 生成测试报告"
    )
    def test_matrix_verifier_parses_mssanitizer_columns_by_name(self):
        import openpyxl

        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "mss.xlsx"
            workbook = openpyxl.Workbook()
            statistic = workbook.active
            statistic.title = "statistic"
            statistic.append(
                ["编号", "npu_npu_dut_内存检测通过", "运行结果"]
            )
            statistic.append([0, True, "SUCCESS"])
            statistic.append([1, True, "SUCCESS"])
            summary = workbook.create_sheet("summary")
            summary.append(
                ["名称", "总用例数", "内存检测通过率", "内存检测是否达标"]
            )
            summary.append(["npu_npu_dut", 2, 100.0, "Pass"])
            workbook.save(report)
            workbook.close()

            self.assertEqual(
                VERIFIER._report_case_ids(report, "mssanitizer"), ([0, 1], 2)
            )

    def test_sanitizer_suite_requires_all_four_tools(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_file = root / "cases.json"
            case_file.write_text(
                json.dumps(
                    [
                        {
                            "id": 0,
                            "inputs": [
                                {
                                    "name": "case_spec",
                                    "range_values": json.dumps(
                                        {"expected_tiling_key": 2570}
                                    ),
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            aggregates = []
            for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
                path = root / f"{tool}.json"
                path.write_text(
                    json.dumps(
                        {
                            "schema": "kda-prepare-atk-matrix/v2",
                            "scope": "mssanitizer",
                            "tool": tool,
                            "timeout_seconds": 60,
                            "runtime_manifest_sha256": "runtime",
                            "expected_cases": 1,
                            "observed_cases": 1,
                            "case_json_sha256": hashlib.sha256(
                                case_file.read_bytes()
                            ).hexdigest(),
                            "expected_tiling_key_count": 1,
                            "host_tiling_key_count": 1,
                            "launch_tiling_key_count": 1,
                            "sanitizer_started_tiling_key_count": 1,
                            "key_pair_sha256": hashlib.sha256(
                                b"0,2570\n"
                            ).hexdigest(),
                            "complete": True,
                            "passed": True,
                        }
                    ),
                    encoding="utf-8",
                )
                aggregates.append(path)

            args = SimpleNamespace(
                case_file=case_file,
                aggregate=aggregates,
                output=root / "suite.json",
            )
            self.assertEqual(VERIFIER.verify_sanitizer_suite(args), 0)
            suite = json.loads(args.output.read_text(encoding="utf-8"))
            self.assertTrue(suite["passed"])
            args.aggregate = aggregates[:-1]
            with self.assertRaisesRegex(ValueError, "四种 sanitizer"):
                VERIFIER.verify_sanitizer_suite(args)


if __name__ == "__main__":
    unittest.main()
