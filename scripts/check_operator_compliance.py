#!/usr/bin/env python3
"""Validate repository operator deliverables against the development standard."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASCENDC_ROOT = ROOT / "fla" / "ops" / "ascendc"
CASE_ROOT = ROOT / "tests" / "op_cases"
TEST_ROOT = ROOT / "tests" / "operators"
REQUIRED_SOCS = {"ascend910b", "ascend910_93", "ascend950"}
REQUIRED_ROUTES = {"ascendc", "aclnn", "direct_launch"}
MODEL_SHAPE_SYMBOLS = {
    "gdn": {
        "B", "N", "T", "H_k", "H_v", "R_h", "K", "V", "C", "N_c", "N_{c,b}", "S_n",
        "D", "W", "L_s", "M", "P", "B_T", "N_b", "D_s", "Q_a",
    },
    "kda": {"B", "N", "T", "H_k", "H_v", "R_h", "K", "V", "C", "N_c", "S_n", "D_0", "D_1", "D_2", "D_3", "D_4"},
}
KNOWN_RETURN_CODES = {
    "ACLNN_SUCCESS",
    "ACLNN_ERR_PARAM_NULLPTR",
    "ACLNN_ERR_PARAM_INVALID",
    "ACLNN_ERR_INNER",
    "RuntimeError",
}
REQUIRED_DOC_SECTIONS = {
    "README.md": ("功能概述", "数学定义", "支持范围", "已知限制", "shape-symbols"),
    "docs/design.md": ("目标与非目标", "能力边界", "Tiling 设计", "流水与同步", "测试设计"),
    "docs/api.md": ("API 总览", "fla_npu.ops.ascendc", "aclnn", "<<<>>>", "异常与返回码"),
}
PLACEHOLDERS = ("<op_name>", "<OpName>", "<input>", "<output>", "<description>", "<limitation>")
CTYPES_PATH = ROOT / "torch_custom" / "fla_npu" / "fla_npu" / "ops" / "ascendc" / "_aclnn_ctypes.py"


def discover_operators() -> dict[str, Path]:
    operators: dict[str, Path] = {}
    for definition in ASCENDC_ROOT.rglob("*_def.cpp"):
        text = definition.read_text(encoding="utf-8")
        if "OP_ADD(" not in text:
            continue
        root = definition.parent.parent
        operators[root.name] = root
    return dict(sorted(operators.items()))


def read_text(path: Path, errors: list[str]) -> str:
    if not path.is_file():
        errors.append(f"missing file: {path.relative_to(ROOT)}")
        return ""
    return path.read_text(encoding="utf-8")


def symbol_rows(text: str) -> dict[str, str]:
    return {
        symbol: description.strip()
        for symbol, description in re.findall(r"(?m)^\| `([^`]+)` \| ([^|]+) \|", text)
    }


def validate_documents(op: str, root: Path, errors: list[str]) -> None:
    for relative, sections in REQUIRED_DOC_SECTIONS.items():
        path = root / relative
        text = read_text(path, errors)
        if not text:
            continue
        for placeholder in PLACEHOLDERS:
            if placeholder in text:
                errors.append(f"{path.relative_to(ROOT)}: unresolved placeholder {placeholder}")
        for section in sections:
            if section not in text:
                errors.append(f"{path.relative_to(ROOT)}: missing required content {section!r}")
    readme = root / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        model = "kda" if "/kda/" in root.as_posix() else "gdn"
        if f"{model}-shape-v1" not in text:
            errors.append(f"{readme.relative_to(ROOT)}: missing {model}-shape-v1 symbol-table version")
        if "附录：Shape 变量说明" not in text:
            errors.append(f"{readme.relative_to(ROOT)}: README must own the operator Shape appendix")
        family_path = ASCENDC_ROOT / model / "README.md"
        family_symbols = symbol_rows(read_text(family_path, errors))
        appendix = text.split("附录：Shape 变量说明", 1)[-1]
        for symbol, description in symbol_rows(appendix).items():
            if symbol not in family_symbols:
                errors.append(f"{readme.relative_to(ROOT)}: symbol {symbol!r} is absent from the {model} model table")
            elif family_symbols[symbol] != description:
                errors.append(
                    f"{readme.relative_to(ROOT)}: symbol {symbol!r} description differs from the {model} model table"
                )
    for relative in ("docs/design.md", "docs/api.md"):
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "../README.md#shape-symbols" not in text:
            errors.append(f"{path.relative_to(ROOT)}: must link to the README Shape appendix")
        if "附录：Shape 变量说明" in text:
            errors.append(f"{path.relative_to(ROOT)}: Shape appendix must only appear in README")
    for relative, heading in (("README.md", "### 3.3 属性"), ("docs/api.md", "### 2.3 属性")):
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if heading not in text:
            errors.append(f"{path.relative_to(ROOT)}: missing attribute section {heading!r}")
            continue
        attribute_section = text.split(heading, 1)[1].split("\n## ", 1)[0]
        table_header = next((line for line in attribute_section.splitlines() if line.startswith("|")), "")
        if "取值范围" not in table_header:
            errors.append(f"{path.relative_to(ROOT)}: attribute table must include a value-range column")
    for path in root.glob("docs/aclnn*.md"):
        errors.append(f"legacy standalone API document: {path.relative_to(ROOT)}")


def validate_manifest(op: str, errors: list[str]) -> None:
    path = CASE_ROOT / f"{op}.json"
    if not path.is_file():
        errors.append(f"missing file: {path.relative_to(ROOT)}")
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path.relative_to(ROOT)}: invalid JSON: {exc}")
        return
    if manifest.get("op") != op:
        errors.append(f"{path.relative_to(ROOT)}: op must equal {op!r}")
    if manifest.get("schema_version") != 1:
        errors.append(f"{path.relative_to(ROOT)}: schema_version must be 1")
    if manifest.get("implementation") != "ascendc":
        errors.append(f"{path.relative_to(ROOT)}: implementation must be ascendc")
    if not isinstance(manifest.get("tolerance"), dict) or not manifest["tolerance"]:
        errors.append(f"{path.relative_to(ROOT)}: tolerance must be a non-empty object")
    if not isinstance(manifest.get("data_generation"), dict) or not manifest["data_generation"]:
        errors.append(f"{path.relative_to(ROOT)}: data_generation must be a non-empty object")
    coverage = manifest.get("coverage_requirements", {})
    for field in (
        "public_modes",
        "known_limits",
        "required_outputs",
        "accuracy_case_ids",
        "generalization_case_ids",
        "negative_case_ids",
        "route_case_ids",
        "performance_case_ids",
    ):
        if not coverage.get(field):
            errors.append(f"{path.relative_to(ROOT)}: coverage_requirements.{field} must be non-empty")
    capability = manifest.get("capability", {})
    if not REQUIRED_SOCS.issubset(set(capability.get("soc", ()))):
        errors.append(f"{path.relative_to(ROOT)}: capability.soc must include A2/A3/A5")
    if not REQUIRED_ROUTES.issubset(set(capability.get("routes", ()))):
        errors.append(f"{path.relative_to(ROOT)}: capability.routes must include ascendc/aclnn/direct_launch")
    for field in ("dtypes", "layouts", "features"):
        value = capability.get(field)
        if not isinstance(value, list) or not value:
            errors.append(f"{path.relative_to(ROOT)}: capability.{field} must be a non-empty list")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append(f"{path.relative_to(ROOT)}: cases must be a non-empty list")
        return
    ids: set[str] = set()
    all_tags: set[str] = set()
    generalization_ids: list[str] = []
    generalization_shapes: set[str] = set()
    model = manifest.get("model")
    allowed_shape_symbols = MODEL_SHAPE_SYMBOLS.get(model, set())
    for index, case in enumerate(cases):
        prefix = f"{path.relative_to(ROOT)}: cases[{index}]"
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            errors.append(f"{prefix}: id must be a non-empty string")
        elif case_id in ids:
            errors.append(f"{prefix}: duplicate id {case_id!r}")
        ids.add(case_id)
        tags = set(case.get("tags", ()))
        all_tags.update(tags)
        for field in ("shape", "dtype", "layout", "attrs", "soc", "run_on", "reference", "expect"):
            if field not in case:
                errors.append(f"{prefix}: missing field {field!r}")
        unknown_shape_symbols = set(case.get("shape", {})) - allowed_shape_symbols
        if unknown_shape_symbols:
            errors.append(f"{prefix}: unknown {model} Shape symbols {sorted(unknown_shape_symbols)}")
        if not isinstance(case.get("seed"), int):
            errors.append(f"{prefix}: seed must be an integer")
        if not REQUIRED_SOCS.issubset(set(case.get("soc", ()))):
            errors.append(f"{prefix}: every case must run on A2/A3/A5")
        if "accuracy" in tags and "ascendc" not in case.get("run_on", ()):
            errors.append(f"{prefix}: accuracy cases must run on ascendc")
        if "generalization" in tags:
            if isinstance(case_id, str):
                generalization_ids.append(case_id)
            generalization_shapes.add(json.dumps(case.get("shape", {}), sort_keys=True))
            if "ascendc" not in case.get("run_on", ()):
                errors.append(f"{prefix}: generalization cases must run on ascendc")
            if case.get("expect", {}).get("return_code") != "ACLNN_SUCCESS":
                errors.append(f"{prefix}: generalization cases must be positive execution cases")
        if "accuracy" in tags:
            expect = case.get("expect", {})
            if expect.get("finite_outputs") is not True:
                errors.append(f"{prefix}: accuracy cases must require finite outputs")
            if set(expect.get("compare_outputs", ())) != set(coverage.get("required_outputs", ())):
                errors.append(f"{prefix}: accuracy cases must compare every public output")
        if "negative" in tags:
            if not case.get("expect", {}).get("return_code"):
                errors.append(f"{prefix}: negative case must define expect.return_code")
            if not case.get("expect", {}).get("message_contains"):
                errors.append(f"{prefix}: negative case must define expect.message_contains")
        if "performance" in tags:
            expect = case.get("expect", {})
            if expect.get("metric") != "msopprof":
                errors.append(f"{prefix}: performance metric must be msopprof")
            if expect.get("requirement") not in {"faster_than_triton", "no_regression"}:
                errors.append(f"{prefix}: invalid performance requirement")
        return_code = case.get("expect", {}).get("return_code")
        if return_code not in KNOWN_RETURN_CODES:
            errors.append(f"{prefix}: unknown expect.return_code {return_code!r}")
    legacy_cases = [case for case in cases if "legacy" in case]
    legacy_ids = [case["id"] for case in legacy_cases]
    declared_legacy_ids = coverage.get("legacy_regression_case_ids", [])
    if set(declared_legacy_ids) != set(legacy_ids):
        errors.append(f"{path.relative_to(ROOT)}: legacy_regression_case_ids must match migrated cases")
    for case in legacy_cases:
        legacy = case["legacy"]
        asset = legacy.get("asset")
        if not isinstance(asset, str) or not (ROOT / asset).is_file():
            errors.append(f"{path.relative_to(ROOT)}: migrated case {case['id']!r} has no archived execution asset")
        if legacy.get("format") == "atk-v2.1" and "/accuracy/atk/" not in f"/{asset}":
            errors.append(f"{path.relative_to(ROOT)}: ATK case {case['id']!r} is outside accuracy/atk")
        if legacy.get("enabled") is False and not legacy.get("migration_note"):
            errors.append(f"{path.relative_to(ROOT)}: disabled migrated case {case['id']!r} needs migration_note")
        if "raw" not in legacy:
            errors.append(f"{path.relative_to(ROOT)}: migrated case {case['id']!r} must preserve its raw source")
    required_tags = {"accuracy", "generalization", "boundary", "negative", "route", "performance", "example"}
    missing_tags = required_tags - all_tags
    if missing_tags:
        errors.append(f"{path.relative_to(ROOT)}: case matrix missing tags {sorted(missing_tags)}")
    covered_routes = {route for case in cases for route in case.get("run_on", ())}
    for route in capability.get("optional_routes", ()):
        if route not in covered_routes:
            errors.append(f"{path.relative_to(ROOT)}: optional route {route!r} is not covered by any case")
    if len(generalization_ids) < 2 or len(generalization_shapes) < 2:
        errors.append(f"{path.relative_to(ROOT)}: require at least two distinct generalization Shapes")
    if set(coverage.get("generalization_case_ids", ())) != set(generalization_ids):
        errors.append(f"{path.relative_to(ROOT)}: generalization_case_ids must match tagged cases")


def validate_test_tree(op: str, op_root: Path, errors: list[str]) -> None:
    root = TEST_ROOT / op
    expected = (
        root / "README.md",
        root / "common" / "case_matrix.py",
        root / "accuracy" / f"test_{op}.py",
        root / "routes" / f"test_aclnn_{op}.cpp",
        root / "routes" / f"test_direct_{op}.cpp",
        root / "ut" / "op_host" / "test_contract.py",
        root / "ut" / "op_kernel" / "test_contract.py",
        root / "performance" / "profile.py",
        root / "st" / "test_example.py",
    )
    for path in expected:
        if not path.is_file():
            errors.append(f"missing file: {path.relative_to(ROOT)}")
    accuracy_path = root / "accuracy" / f"test_{op}.py"
    if accuracy_path.is_file():
        accuracy_text = accuracy_path.read_text(encoding="utf-8")
        for required in ('tags=("generalization",)', "run_generalization_cases(OP, cases)"):
            if required not in accuracy_text:
                errors.append(f"{accuracy_path.relative_to(ROOT)}: missing executable JSON generalization hook")
    shared_runner = TEST_ROOT / "_shared" / "npu_generalization.py"
    if shared_runner.is_file():
        shared_text = shared_runner.read_text(encoding="utf-8")
        if f'"{op}": _run_' not in shared_text:
            errors.append(f"{shared_runner.relative_to(ROOT)}: missing runner mapping for {op}")
    manifest_path = CASE_ROOT / f"{op}.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        legacy_path = root / "routes" / f"test_legacy_{op}.py"
        has_legacy = "torch_ops_npu" in manifest.get("capability", {}).get("optional_routes", ())
        if has_legacy and not legacy_path.is_file():
            errors.append(f"missing file: {legacy_path.relative_to(ROOT)}")
        if not has_legacy and legacy_path.exists():
            errors.append(f"unexpected legacy route: {legacy_path.relative_to(ROOT)}")
    direct_path = root / "routes" / f"test_direct_{op}.cpp"
    if direct_path.is_file():
        direct_text = direct_path.read_text(encoding="utf-8")
        if re.search(r"<<<[^>]+>>>\([^;]*(?:\[[A-Z]|\bdata\))", direct_text, re.S):
            errors.append(f"{direct_path.relative_to(ROOT)}: malformed generated direct-launch argument list")
        if "<<<blockDim" not in direct_text or "workspace" not in direct_text or "tiling" not in direct_text:
            errors.append(f"{direct_path.relative_to(ROOT)}: incomplete blockDim/workspace/tiling launch contract")
        direct_entries = set(
            re.findall(r"__global__\s+__aicore__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", direct_text)
        )
        local_entries: set[str] = set()
        for source in (op_root / "op_kernel").glob("*.cpp"):
            local_entries.update(
                re.findall(
                    r"__global__\s+__aicore__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                    source.read_text(encoding="utf-8"),
                )
            )
        repository_entries: set[str] = set()
        for source in ASCENDC_ROOT.rglob("op_kernel/*.cpp"):
            repository_entries.update(
                re.findall(
                    r"__global__\s+__aicore__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                    source.read_text(encoding="utf-8"),
                )
            )
        if (
            not direct_entries
            or not (direct_entries & local_entries)
            or not direct_entries.issubset(repository_entries)
        ):
            errors.append(
                f"{direct_path.relative_to(ROOT)}: direct entries {sorted(direct_entries)} "
                f"do not match local/repository kernel entries"
            )


def _ctypes_functions():
    tree = ast.parse(CTYPES_PATH.read_text(encoding="utf-8"))
    functions = {node.name: node.args for node in tree.body if isinstance(node, ast.FunctionDef)}
    aliases = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Name):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                aliases[target.id] = node.value.id
    return functions, aliases


def _signature_key(arguments):
    return ast.dump(arguments, annotate_fields=True, include_attributes=False)


def validate_python_signature(op: str, root: Path, functions, aliases, errors: list[str]) -> None:
    api_path = root / "docs" / "api.md"
    if not api_path.is_file():
        return
    text = api_path.read_text(encoding="utf-8")
    match = re.search(r"## 4\. `fla_npu\.ops\.ascendc` API.*?```python\n([^\n]+)\n```", text, re.S)
    if not match:
        errors.append(f"{api_path.relative_to(ROOT)}: missing Python signature code block")
        return
    signature = match.group(1).strip()
    try:
        documented = ast.parse(f"def {signature}:\n    pass\n").body[0]
    except SyntaxError as exc:
        errors.append(f"{api_path.relative_to(ROOT)}: invalid Python signature: {exc}")
        return
    if documented.name != op:
        errors.append(f"{api_path.relative_to(ROOT)}: Python API name must be {op!r}, got {documented.name!r}")

    function_name = f"npu_{op}"
    function_name = aliases.get(function_name, function_name)
    actual = functions.get(function_name)
    if actual is None:
        errors.append(f"{api_path.relative_to(ROOT)}: ctypes function npu_{op} is not implemented")
        return
    if _signature_key(documented.args) != _signature_key(actual):
        errors.append(
            f"{api_path.relative_to(ROOT)}: documented Python parameters/defaults do not match {function_name}"
        )


def validate_source_rules(operators: dict[str, Path], errors: list[str]) -> None:
    for op, root in operators.items():
        for directory in root.rglob("*"):
            if directory.is_dir() and directory.name in {"test", "tests", "ATK"}:
                errors.append(
                    f"{directory.relative_to(ROOT)}: operator-local test directory is forbidden; "
                    f"migrate it to tests/operators/{op}"
                )
        example_test_dir = ROOT / "examples" / "fast_kernel_launch_example" / "tests" / op
        if example_test_dir.exists():
            errors.append(
                f"{example_test_dir.relative_to(ROOT)}: mainline operator harness must live in tests/operators/{op}/routes"
            )
    ci_contracts = {
        ROOT / "ci" / "run_operator_build_matrix.py": (*REQUIRED_SOCS, "build.sh", "--ops="),
        ROOT / "ci" / "run_operator_generalization.py": (
            *REQUIRED_SOCS,
            "FLA_NPU_RUN_OPERATOR_TESTS",
            "FLA_NPU_CASE_IDS",
            "json_generalization_cases",
            "start_new_session=True",
        ),
        ROOT / "ci" / "run_operator_accuracy.py": (
            *REQUIRED_SOCS,
            "FLA_NPU_RUN_OPERATOR_TESTS",
            "main_ascendc_accuracy_backend",
            "start_new_session=True",
        ),
        ROOT / "ci" / "run_checks.sh": (
            "run_operator_generalization.py",
            "run_operator_accuracy.py",
            "CI_RUN_OPERATOR_GENERALIZATION",
            "CI_RUN_OPERATOR_ACCURACY",
            "operator_matrix_scope_enabled",
        ),
        ROOT / "ci" / "run_ci_container.sh": (
            "CI_RUN_OPERATOR_GENERALIZATION",
            "CI_RUN_OPERATOR_ACCURACY",
        ),
        ROOT / ".github" / "workflows" / "ci.yml": (
            "CI_RUN_OPERATOR_GENERALIZATION:",
            "CI_RUN_OPERATOR_ACCURACY:",
            "needs.prepare.outputs.ci_mode == 'full'",
            "needs.prepare.outputs.ops != ''",
            "quick 模式未指定 `ops=`",
        ),
    }
    for path, required_tokens in ci_contracts.items():
        text = read_text(path, errors)
        for token in required_tokens:
            if token not in text:
                errors.append(f"{path.relative_to(ROOT)}: missing CI contract token {token!r}")
    for cmake in ASCENDC_ROOT.rglob("CMakeLists.txt"):
        text = cmake.read_text(encoding="utf-8")
        if "--cce-auto-sync=on" in text:
            errors.append(f"{cmake.relative_to(ROOT)}: --cce-auto-sync must be off")
    for op, root in operators.items():
        definition = next(root.glob("op_host/*_def.cpp"), None)
        if definition is None:
            errors.append(f"{root.relative_to(ROOT)}: missing op definition")
            continue
        text = definition.read_text(encoding="utf-8")
        missing = [soc for soc in REQUIRED_SOCS if f'AddConfig("{soc}"' not in text]
        if missing:
            errors.append(f"{definition.relative_to(ROOT)}: missing SOC configs {missing}")
        class_match = re.search(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*public\s+OpDef", text)
        if class_match is None:
            errors.append(f"{definition.relative_to(ROOT)}: cannot determine OpDef class")
            continue
        op_type = class_match.group(1)
        host_cmake = root / "op_host" / "CMakeLists.txt"
        cmake_text = read_text(host_cmake, errors)
        if not cmake_text:
            continue
        if "add_op_to_compiled_list()" not in cmake_text:
            errors.append(f"{host_cmake.relative_to(ROOT)}: operator is absent from compiled list")
        if re.search(rf"add_modules_sources\s*\(\s*OPTYPE\s+{re.escape(op)}\b", cmake_text, re.S) is None:
            errors.append(f"{host_cmake.relative_to(ROOT)}: missing add_modules_sources for {op}")
        compile_names = re.findall(
            r"add_ops_compile_options\s*\(\s*OP_NAME\s+([A-Za-z_][A-Za-z0-9_]*)", cmake_text, re.S
        )
        if not compile_names or op_type not in compile_names:
            errors.append(f"{host_cmake.relative_to(ROOT)}: missing compile options for {op_type}")
        wrong_names = sorted(set(compile_names) - {op_type})
        if wrong_names:
            errors.append(
                f"{host_cmake.relative_to(ROOT)}: compile options target other operators {wrong_names}"
            )
        if "--cce-auto-sync=off" not in cmake_text:
            errors.append(f"{host_cmake.relative_to(ROOT)}: missing --cce-auto-sync=off")
        kernel_sources = list((root / "op_kernel").glob("*.cpp"))
        if not kernel_sources or not any(
            re.search(r"__global__\s+__aicore__\s+void\s+", source.read_text(encoding="utf-8"))
            for source in kernel_sources
        ):
            errors.append(f"{root.relative_to(ROOT)}: missing Ascend C kernel entry")


def main() -> int:
    operators = discover_operators()
    errors: list[str] = []
    functions, aliases = _ctypes_functions()
    for op, root in operators.items():
        validate_documents(op, root, errors)
        validate_manifest(op, errors)
        validate_test_tree(op, root, errors)
        validate_python_signature(op, root, functions, aliases, errors)
    validate_source_rules(operators, errors)
    if errors:
        print("operator compliance check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"operator compliance check passed: {len(operators)} Ascend C operators")
    return 0


if __name__ == "__main__":
    sys.exit(main())
