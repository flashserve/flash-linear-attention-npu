"""检查冻结 ATK 用例与统一 JSON 中的声明完全一致。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
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


def _runtime_manifest_fixture(
    keys: set[int], *, sanitizer: bool = False, complete: bool = False
) -> dict:
    compile_hash = "171e9a19df4c0d44ca00a177d8fe300c"
    binary_name = f"ChunkKdaFwdPrepare_{compile_hash}"
    kernel_names = {f"{binary_name}_{key}" for key in keys}
    digest = "a" * 64
    return {
        "schema": "kda-prepare-runtime/v2",
        "platform": "ascend950",
        "runtime_sha256": digest,
        "op_api_sha256": digest,
        "test_artifact_sha256": {"runner.py": digest},
        "python_wrapper_sha256": {
            "__init__.py": digest,
            "ops/ascendc/__init__.py": digest,
            "ops/ascendc/_aclnn_ctypes.py": digest,
            "ops/ascendc/_runtime.py": digest,
        },
        "object_count": 1,
        "metadata_count": 1,
        "compiled_tiling_key_count": len(keys),
        "compiled_tiling_key_sha256": VERIFIER._integer_digest(keys),
        "kernel_name_count": len(kernel_names),
        "kernel_name_sha256": VERIFIER._string_digest(kernel_names),
        "kernel_binaries": [
            {
                "metadata_file": f"{binary_name}.json",
                "metadata_sha256": digest,
                "metadata_kernel_name": binary_name,
                "metadata_bin_sha256": digest,
                "bin_file_name": binary_name,
                "bin_file_suffix": ".o",
                "bin_file_sha256": digest,
                "compile_hash": compile_hash,
                "kernels": [
                    {
                        "kernel_name": f"{binary_name}_{key}",
                        "tiling_key": key,
                    }
                    for key in sorted(keys)
                ],
            }
        ],
        "complete_key_set_required": complete,
        "sanitizer_required": sanitizer,
        "sanitizer_object_count": 1 if sanitizer else 0,
    }


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

    def test_timed_atk_scopes_have_configurable_process_scheduling(self):
        source = (ROOT / "tests/atk/run_test_cpu.sh").read_text(
            encoding="utf-8"
        )
        accuracy = source.split("if should_run accuracy; then", 1)[1].split(
            "if should_run performance; then", 1
        )[0]
        determinism = source.split("if should_run determinism; then", 1)[1].split(
            "if should_run mssanitizer; then", 1
        )[0]
        self.assertIn('"${SINGLE_PROCESS_ARGS[@]}"', accuracy)
        self.assertIn('"${SINGLE_PROCESS_ARGS[@]}"', determinism)

    def test_mssanitizer_writes_outer_and_atk_logs_to_same_file(self):
        source = (ROOT / "tests/atk/run_test_cpu.sh").read_text(
            encoding="utf-8"
        )
        mssanitizer = source.split("if should_run mssanitizer; then", 1)[1]
        self.assertIn(
            'mssanitizer --tool="$MSS_TOOL" --log-file "$MSS_LOG_PATH" --',
            mssanitizer,
        )
        self.assertIn('-msl "$MSS_LOG_PATH"', mssanitizer)
        self.assertIn('"${SINGLE_PROCESS_ARGS[@]}"', mssanitizer)
        self.assertIn('"${MSS_TIMEOUT_ARGS[@]}"', mssanitizer)

    def test_readme_binds_sanitizer_build_to_matrix_soc(self):
        source = (ATK_DIR / "README.md").read_text(encoding="utf-8")
        self.assertIn("export KDA_PREPARE_ATK_SOC=ascend950", source)
        self.assertIn('FLA_NPU_SOC="$KDA_PREPARE_ATK_SOC"', source)
        self.assertIn(
            "A2 使用 ascend910b，A3 使用 ascend910_93，A5 使用 ascend950",
            source,
        )

    def test_matrix_runtime_manifest_binds_runner_and_verifier(self):
        source = (ATK_DIR / "scripts/run_matrix.sh").read_text(
            encoding="utf-8"
        )
        for artifact in (
            '"$op_dir/executor_chunk_kda_fwd_prepare.py"',
            '"$op_dir/chunk_kda_fwd_prepare.yaml"',
            '"$runner"',
            '"$script_dir/run_matrix.sh"',
            '"$verifier"',
        ):
            self.assertIn(artifact, source)
        self.assertIn(
            'runtime_args+=(--test-artifact "$artifact")', source
        )
        self.assertIn(
            'aggregate_runtime_args+=(--test-artifact "$artifact")', source
        )
        self.assertIn('aggregate_runtime_args=(--soc "$soc")', source)
        self.assertIn("正式矩阵必须显式设置 KDA_PREPARE_ATK_SOC", source)

    def test_matrix_resume_revalidates_prefix_at_shard_boundaries(self):
        source = (ATK_DIR / "scripts/run_matrix.sh").read_text(
            encoding="utf-8"
        )
        prefix = source.split(
            "# 续跑前先从原始报告和日志重验完整前缀", 1
        )[1].split("for ((start = matrix_start;", 1)[0]
        self.assertIn("start < matrix_start", prefix)
        self.assertIn('python3 "$verifier" shard', prefix)
        self.assertIn("续跑前缀缺少完整分片", prefix)
        self.assertIn("matrix_start % shard_size", source)
        self.assertIn('[[ "$shard_size" == "1" ]]', source)
        self.assertIn("ATK_SINGLE_PROCESS=off", source)
        self.assertIn("--single-process-mode off", source)

    def test_runner_rejects_requested_and_physical_soc_mismatch(self):
        source = (ROOT / "tests/atk/run_test_cpu.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'detected_soc="$(detect_soc_from_npu "$NPU_DEVICE_ID")"', source
        )
        self.assertIn("resolve_physical_device_id()", source)
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES", source)
        self.assertIn('awk -F\'|\' -v target="$physical_id"', source)
        self.assertIn('elif [[ "$SOC" != "$detected_soc" ]]', source)
        self.assertIn("目标 SoC 与实际 NPU 不一致", source)
        self.assertIn("实际 NPU SoC", source)

    def test_runner_detects_selected_a2_a3_a5_device(self):
        bash = shutil.which("bash")
        if os.name == "nt":
            git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            git_bash /= "Git/bin/bash.exe"
            if git_bash.is_file():
                bash = str(git_bash)
        if bash is None:
            self.skipTest("需要 Bash 执行 SoC 解析 fixture")

        script = r"""
set -euo pipefail
source <(sed -n '/^resolve_physical_device_id()/,/^}/p' "$1")
source <(sed -n '/^detect_soc_from_npu()/,/^}/p' "$1")
MOCK_NPU_SMI='| 4     910_9391 | OK | data |
| 5 | Ascend950PR | OK | data |
| 6     910B3 | OK | data |
| 7     9579 | OK | data |'
npu-smi() { printf '%s\n' "$MOCK_NPU_SMI"; }
export ASCEND_RT_VISIBLE_DEVICES=4,5
[[ "$(detect_soc_from_npu 0)" == ascend910_93 ]]
[[ "$(detect_soc_from_npu 1)" == ascend950 ]]
[[ -z "$(detect_soc_from_npu 2)" ]]
unset ASCEND_RT_VISIBLE_DEVICES
[[ "$(detect_soc_from_npu 6)" == ascend910b ]]
[[ "$(detect_soc_from_npu 7)" == ascend950 ]]
"""
        result = subprocess.run(
            [bash, "-c", script, "soc-fixture", "tests/atk/run_test_cpu.sh"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

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

    def test_matrix_shards_require_canonical_single_case_cover(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "shard_0_1").mkdir()
            (root / "shard_1_2").mkdir()
            self.assertEqual(
                [(start, end) for _, start, end in VERIFIER._matrix_shards(root, 2)],
                [(0, 1), (1, 2)],
            )

        invalid_layouts = (
            ("shard_00_1",),
            ("shard_0_2",),
            ("shard_1_2",),
            ("shard_0_1", "shard_2_3"),
        )
        for names in invalid_layouts:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                for name in names:
                    (root / name).mkdir()
                with self.assertRaisesRegex(
                    ValueError, "非法|单 case|不连续|未覆盖"
                ):
                    VERIFIER._matrix_shards(root, 3)

    @unittest.skipUnless(
        importlib.util.find_spec("openpyxl"), "需要 openpyxl 生成测试报告"
    )
    def test_existing_shard_is_rebuilt_from_raw_evidence(self):
        import openpyxl

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases = [
                {
                    "id": case_id,
                    "inputs": [
                        {"name": "case_spec", "range_values": "{}"}
                    ],
                }
                for case_id in range(200)
            ]
            case_file = root / "cases.json"
            case_file.write_text(json.dumps(cases), encoding="utf-8")
            runtime_manifest = root / "runtime_manifest.json"
            runtime_manifest.write_text("{}\n", encoding="utf-8")
            console_log = root / "console.log"
            console_log.write_text("ATK finished\n", encoding="utf-8")
            report = (
                root
                / "accuracy/atk_output/atk_chunk_kda_fwd_prepare_test"
                / "report/result.xlsx"
            )
            report.parent.mkdir(parents=True)
            workbook = openpyxl.Workbook()
            statistic = workbook.active
            statistic.title = "statistic"
            statistic.append(["编号", "npu_npu_dut_精度通过", "运行结果"])
            statistic.append([0, True, "SUCCESS"])
            summary_sheet = workbook.create_sheet("summary")
            summary_sheet.append(
                [
                    "名称",
                    "总用例数",
                    "执行失败用例个数",
                    "通过用例个数",
                    "精度是否达标",
                ]
            )
            summary_sheet.append(["cpu_cpu_golden", 1, 0, 1, "Pass"])
            workbook.save(report)
            workbook.close()

            args = SimpleNamespace(
                case_file=case_file,
                scope="accuracy",
                tool="",
                runtime_manifest=runtime_manifest,
                timeout=60,
                loop_nums=1,
                gm_init_mode="off",
                single_process_mode="off",
                require_tiling_log=False,
                shard_root=root,
                console_log=console_log,
                summary=root / "summary.json",
                start=0,
                end=1,
            )
            with self.assertRaisesRegex(ValueError, "schema 不是 v2"):
                VERIFIER.summarize_shard(args)
            runtime_manifest.write_text(
                json.dumps(_runtime_manifest_fixture({2570})),
                encoding="utf-8",
            )
            self.assertEqual(VERIFIER.summarize_shard(args), 0)
            console_log.write_text("ATK evidence changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "原始证据不一致"):
                VERIFIER.summarize_shard(args)

            console_log.write_text("ATK finished\n", encoding="utf-8")
            report.unlink()
            with self.assertRaisesRegex(ValueError, "没有找到"):
                VERIFIER.summarize_shard(args)

    def test_aggregate_rebuilds_every_shard_before_trusting_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_file = root / "cases.json"
            cases = [
                {
                    "id": case_id,
                    "inputs": [
                        {"name": "case_spec", "range_values": "{}"}
                    ],
                }
                for case_id in range(200)
            ]
            case_file.write_text(json.dumps(cases), encoding="utf-8")
            runtime_manifest = root / "runtime_manifest.json"
            runtime_manifest.write_text("{}\n", encoding="utf-8")
            runtime_hash = hashlib.sha256(runtime_manifest.read_bytes()).hexdigest()
            case_hash = hashlib.sha256(case_file.read_bytes()).hexdigest()

            canonical = {}
            for case_id in range(200):
                shard = root / f"shard_{case_id}_{case_id + 1}"
                shard.mkdir()
                value = {
                    "schema": "kda-prepare-atk-shard/v2",
                    "scope": "accuracy",
                    "tool": "",
                    "timeout_seconds": 60,
                    "loop_nums": 1,
                    "gm_init_mode": "off",
                    "single_process_mode": "off",
                    "require_tiling_log": False,
                    "runtime_manifest_sha256": runtime_hash,
                    "start": case_id,
                    "end": case_id + 1,
                    "expected_cases": 1,
                    "case_ids": [case_id],
                    "case_json_sha256": case_hash,
                    "expected_tiling_keys": [],
                    "host_tiling_keys": [],
                    "launch_tiling_keys": [],
                    "key_pair_sha256": hashlib.sha256(b"").hexdigest(),
                    "passed": True,
                }
                canonical[case_id] = value
                (shard / "summary.json").write_text(
                    json.dumps(value), encoding="utf-8"
                )

            args = SimpleNamespace(
                case_file=case_file,
                scope="accuracy",
                tool="",
                runtime_manifest=runtime_manifest,
                timeout=60,
                loop_nums=1,
                gm_init_mode="off",
                single_process_mode="off",
                require_tiling_log=False,
                matrix_root=root,
                soc="ascend950",
                test_artifact=[root / "runner.py"],
                output=root / "aggregate_summary.json",
            )

            def rebuild(shard_args):
                return canonical[shard_args.start]

            with mock.patch.object(
                VERIFIER, "_verify_current_runtime", return_value={}
            ), mock.patch.object(
                VERIFIER, "_build_shard_summary", side_effect=rebuild
            ) as builder:
                self.assertEqual(VERIFIER.aggregate_matrix(args), 0)
                self.assertEqual(builder.call_count, 200)

            tampered = dict(canonical[0])
            tampered["passed"] = False
            (root / "shard_0_1/summary.json").write_text(
                json.dumps(tampered), encoding="utf-8"
            )
            with mock.patch.object(
                VERIFIER, "_verify_current_runtime", return_value={}
            ), mock.patch.object(
                VERIFIER, "_build_shard_summary", side_effect=rebuild
            ):
                with self.assertRaisesRegex(ValueError, "原始证据不一致"):
                    VERIFIER.aggregate_matrix(args)

    def test_matrix_verifier_requires_active_sanitizer_kernel(self):
        kernel_name = (
            "ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_2570"
        )
        binary_name = "ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c"
        runtime_manifest = {
            "schema": "kda-prepare-runtime/v2",
            "kernel_binaries": [
                {
                    "metadata_kernel_name": binary_name,
                    "metadata_bin_sha256": "a" * 64,
                    "bin_file_name": binary_name,
                    "bin_file_sha256": "a" * 64,
                    "compile_hash": "171e9a19df4c0d44ca00a177d8fe300c",
                    "kernels": [
                        {"kernel_name": kernel_name, "tiling_key": 2570}
                    ],
                }
            ],
            "kernel_name_count": 1,
            "kernel_name_sha256": VERIFIER._string_digest({kernel_name}),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            console = root / "console.log"
            sanitizer = root / "memcheck.log"
            console.write_text(
                f"Start memcheck sanitizer on kernel {kernel_name}\n",
                encoding="utf-8",
            )
            sanitizer.write_text("memcheck completed\n", encoding="utf-8")
            count, keys, names, console_hash, sanitizer_hash = (
                VERIFIER._sanitizer_evidence(
                    console,
                    sanitizer,
                    "memcheck",
                    {2570},
                    runtime_manifest,
                )
            )
            self.assertEqual(count, 1)
            self.assertEqual(keys, {2570})
            self.assertEqual(names, {kernel_name})
            self.assertEqual(len(console_hash), 64)
            self.assertEqual(len(sanitizer_hash), 64)

            stale_kernel = (
                "ChunkKdaFwdPrepare_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_2570"
            )
            console.write_text(
                f"Start memcheck sanitizer on kernel {stale_kernel}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "manifest 之外"):
                VERIFIER._sanitizer_evidence(
                    console,
                    sanitizer,
                    "memcheck",
                    {2570},
                    runtime_manifest,
                )

            sanitizer.write_text(
                "No active sanitizer tool on kernel "
                "ChunkKdaFwdPrepare_171e9a19df4c0d44ca00a177d8fe300c_2570\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "未加载 sanitizer"):
                VERIFIER._sanitizer_evidence(
                    console,
                    sanitizer,
                    "memcheck",
                    {2570},
                    runtime_manifest,
                )

    def test_matrix_verifier_rejects_noncanonical_scope_contracts(self):
        valid = {
            "accuracy": SimpleNamespace(
                scope="accuracy",
                timeout=60,
                loop_nums=1,
                gm_init_mode="off",
                single_process_mode="off",
                tool="",
                require_tiling_log=False,
            ),
            "determinism": SimpleNamespace(
                scope="determinism",
                timeout=60,
                loop_nums=50,
                gm_init_mode="not_applicable",
                single_process_mode="off",
                tool="",
                require_tiling_log=True,
            ),
            "mssanitizer": SimpleNamespace(
                scope="mssanitizer",
                timeout=60,
                loop_nums=1,
                gm_init_mode="not_applicable",
                single_process_mode="off",
                tool="memcheck",
                require_tiling_log=True,
            ),
        }
        case_counts = {"accuracy": 200, "determinism": 432, "mssanitizer": 432}
        for scope, args in valid.items():
            with self.subTest(scope=scope):
                VERIFIER._validate_scope_contract(args, case_counts[scope])

        invalid = (
            ("timeout", valid["accuracy"], 200, {"timeout": 0}),
            ("timeout", valid["accuracy"], 200, {"timeout": 61}),
            ("正式矩阵", valid["accuracy"], 199, {}),
            ("loop_nums", valid["accuracy"], 200, {"loop_nums": 2}),
            ("gm_init_mode", valid["accuracy"], 200, {"gm_init_mode": "on"}),
            (
                "single_process_mode",
                valid["accuracy"],
                200,
                {"single_process_mode": "on"},
            ),
            ("tool", valid["accuracy"], 200, {"tool": "memcheck"}),
            (
                "require_tiling_log",
                valid["accuracy"],
                200,
                {"require_tiling_log": True},
            ),
            (
                "require_tiling_log",
                valid["determinism"],
                432,
                {"require_tiling_log": False},
            ),
            ("tool", valid["mssanitizer"], 432, {"tool": "invalid"}),
        )
        for message, base_args, case_count, override in invalid:
            values = vars(base_args).copy()
            values.update(override)
            with self.subTest(message=message, values=values):
                with self.assertRaisesRegex(ValueError, message):
                    VERIFIER._validate_scope_contract(
                        SimpleNamespace(**values), case_count
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
            compile_hash = "171e9a19df4c0d44ca00a177d8fe300c"
            binary_name = f"ChunkKdaFwdPrepare_{compile_hash}"
            object_path = kernel_dir / f"{binary_name}.o"
            object_path.write_bytes(b"kernel")
            object_sha256 = hashlib.sha256(object_path.read_bytes()).hexdigest()
            (kernel_dir / f"{binary_name}.json").write_text(
                json.dumps(
                    {
                        "binFileName": binary_name,
                        "binFileSuffix": ".o",
                        "kernelName": binary_name,
                        "sha256": object_sha256,
                        "kernelList": [
                            {
                                "tilingKey": 2570,
                                "kernelName": f"{binary_name}_2570",
                            },
                            {
                                "tilingKey": 8391178,
                                "kernelName": f"{binary_name}_8391178",
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
            package_root = root / "python/fla_npu"
            for relative in (
                "__init__.py",
                "ops/ascendc/__init__.py",
                "ops/ascendc/_aclnn_ctypes.py",
                "ops/ascendc/_runtime.py",
            ):
                path = package_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {relative}\n", encoding="utf-8")
            package_spec = SimpleNamespace(
                submodule_search_locations=[str(package_root)]
            )
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
            with mock.patch.object(
                VERIFIER.importlib.util, "find_spec", return_value=package_spec
            ), mock.patch.dict(
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
            self.assertEqual(manifest["schema"], "kda-prepare-runtime/v2")
            self.assertEqual(manifest["kernel_name_count"], 2)
            self.assertEqual(
                manifest["kernel_binaries"][0]["compile_hash"], compile_hash
            )
            self.assertEqual(
                manifest["kernel_binaries"][0]["bin_file_sha256"],
                object_sha256,
            )
            with mock.patch.object(
                VERIFIER.importlib.util, "find_spec", return_value=package_spec
            ), mock.patch.dict(
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

    def test_all_scopes_reject_forged_runtime_manifest(self):
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
            manifest_path = root / "runtime_manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            for scope in ("accuracy", "determinism", "mssanitizer"):
                with self.subTest(scope=scope), self.assertRaisesRegex(
                    ValueError, "schema 不是 v2"
                ):
                    VERIFIER._load_runtime_manifest(
                        manifest_path,
                        case_file,
                        scope == "mssanitizer",
                        scope != "accuracy",
                    )

    def test_aggregate_rejects_stale_runtime_manifest(self):
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
            manifest = _runtime_manifest_fixture({2570})
            manifest_path = root / "runtime_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            args = SimpleNamespace(
                case_file=case_file,
                runtime_manifest=manifest_path,
                scope="accuracy",
                soc="ascend950",
                test_artifact=[root / "runner.py"],
            )
            with mock.patch.object(
                VERIFIER, "_runtime_manifest", return_value=manifest
            ):
                self.assertEqual(VERIFIER._verify_current_runtime(args), manifest)

            changed = dict(manifest)
            changed["runtime_sha256"] = "b" * 64
            with mock.patch.object(
                VERIFIER, "_runtime_manifest", return_value=changed
            ):
                with self.assertRaisesRegex(ValueError, "当前实际"):
                    VERIFIER._verify_current_runtime(args)

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
            cases = [
                {
                    "id": case_id,
                    "inputs": [
                        {
                            "name": "case_spec",
                            "range_values": json.dumps(
                                {"expected_tiling_key": 1000 + case_id}
                            ),
                        }
                    ],
                }
                for case_id in range(432)
            ]
            case_file.write_text(
                json.dumps(cases),
                encoding="utf-8",
            )
            pair_digest = hashlib.sha256(
                "".join(
                    f"{case_id},{1000 + case_id}\n" for case_id in range(432)
                ).encode("ascii")
            ).hexdigest()
            aggregates = []
            canonical = {}
            for tool in ("memcheck", "racecheck", "initcheck", "synccheck"):
                matrix_root = root / tool
                matrix_root.mkdir()
                path = matrix_root / "aggregate_summary.json"
                summary = {
                    "schema": "kda-prepare-atk-matrix/v2",
                    "scope": "mssanitizer",
                    "tool": tool,
                    "timeout_seconds": 60,
                    "loop_nums": 1,
                    "gm_init_mode": "not_applicable",
                    "single_process_mode": "off",
                    "require_tiling_log": True,
                    "runtime_manifest_sha256": "runtime",
                    "expected_cases": 432,
                    "observed_cases": 432,
                    "case_json_sha256": hashlib.sha256(
                        case_file.read_bytes()
                    ).hexdigest(),
                    "expected_tiling_key_count": 432,
                    "host_tiling_key_count": 432,
                    "launch_tiling_key_count": 432,
                    "sanitizer_started_tiling_key_count": 432,
                    "sanitizer_started_kernel_name_count": 432,
                    "sanitizer_started_kernel_name_sha256": "kernel-names",
                    "key_pair_sha256": pair_digest,
                    "complete": True,
                    "passed": True,
                }
                canonical[tool] = summary
                path.write_text(json.dumps(summary), encoding="utf-8")
                aggregates.append(path)

            args = SimpleNamespace(
                case_file=case_file,
                aggregate=aggregates,
                soc="ascend950",
                test_artifact=[root / "runner.py"],
                output=root / "suite.json",
            )
            with mock.patch.object(
                VERIFIER,
                "_build_matrix_summary",
                side_effect=lambda matrix_args: canonical[matrix_args.tool],
            ) as builder:
                self.assertEqual(VERIFIER.verify_sanitizer_suite(args), 0)
                self.assertEqual(builder.call_count, 4)
            suite = json.loads(args.output.read_text(encoding="utf-8"))
            self.assertTrue(suite["passed"])

            forged = dict(canonical["memcheck"])
            forged["passed"] = False
            aggregates[0].write_text(json.dumps(forged), encoding="utf-8")
            with mock.patch.object(
                VERIFIER,
                "_build_matrix_summary",
                side_effect=lambda matrix_args: canonical[matrix_args.tool],
            ):
                with self.assertRaisesRegex(ValueError, "原始证据不一致"):
                    VERIFIER.verify_sanitizer_suite(args)

            aggregates[0].write_text(
                json.dumps(canonical["memcheck"]), encoding="utf-8"
            )
            args.aggregate = aggregates[:-1]
            with self.assertRaisesRegex(ValueError, "四种 sanitizer"):
                VERIFIER.verify_sanitizer_suite(args)


if __name__ == "__main__":
    unittest.main()
