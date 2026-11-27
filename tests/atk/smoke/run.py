"""Discover and run the passing FLA ATK smoke suites. Requires Python 3.10+."""

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / "ci"))
from smoke_cache import ResultCache

REFERENCE_SHA = "72d372032214b8866cd6073431b0d478e7744ba5"
STANDARD = "cv_fused_double_benchmark"


def read(path):
    return json.loads(path.read_bytes())


def source_roots(suites):
    roots = read(ROOT / "source_roots.json")
    if set(roots) != {suite["op"] for suite in suites}:
        raise ValueError("Every smoke operator needs one source directory in source_roots.json")
    return roots


def affected_pairs(pairs, roots, base, head, repo):
    # Disabling rename detection returns both the deleted and the added path.
    # NUL delimiters preserve file names containing tabs, spaces and newlines.
    data = subprocess.check_output(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, head, "--", "fla/ops/ascendc/", "tests/atk/"],
        cwd=repo,
    )
    paths = {os.fsdecode(raw) for raw in data.split(b"\0") if raw and not raw.lower().endswith(b".md")}
    # Bundled references can import other operators' references. Exercise their
    # consumers too when changing the Python harness, reference code or catalog.
    if paths & {"tests/atk/smoke/suites.json", "tests/atk/smoke/source_roots.json"} or any(
        path.endswith(".py") and path.startswith(("tests/atk/smoke/", "tests/atk/common/"))
        for path in paths
    ):
        return pairs
    changed_ops = {
        op for op, root in roots.items()
        if any(path == root or path.startswith(root + "/")
               or (path.startswith(f"tests/atk/{op}/") and path.endswith(".py"))
               for path in paths)
    }
    if "tests/atk/chunk_fwd_h/fwd_h_cpu.py" in paths:
        changed_ops.add("chunk_gated_delta_rule_bwd")
    return {(op, platform) for op, platform in pairs
            if op in changed_ops or f"tests/atk/smoke/cases/{op}/{platform}/cases.json" in paths}


def case_path(suite):
    return ROOT / "cases" / suite["op"] / suite["platform"] / "cases.json"


def validate_cases(suite, cases):
    identities = {(case["id"], case["name"]) for case in cases}
    if not cases or len(identities) != len(cases) or len(cases) != suite["count"]:
        raise ValueError(
            f"Invalid case count or duplicate identity in {suite['group']}"
        )
    for case in cases:
        acc = case["standard"]["acc"]
        accepted_apis = {"fla_smoke", Path(suite["reference"]).stem}
        if (list(acc) if isinstance(acc, dict) else [acc]) != [STANDARD] or case["api_type"] not in accepted_apis:
            raise ValueError(
                f"Unexpected executor or standard in {suite['group']}/{case['id']}"
            )


def groups_for(entry):
    path = case_path(entry)
    groups = read(path)
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"Expected a nonempty list of ATK groups in {path}")
    seen = set()
    for group in groups:
        cases = group.pop("cases")
        name = group["group"]
        if not name or Path(name).name != name or name == ".." or name in seen:
            raise ValueError(f"Invalid or duplicate ATK group {name!r} in {path}")
        seen.add(name)
        reference = group["reference"]
        reference_dir = ROOT.parent / entry["op"] if group.get("reference_source") == "repository" else path.parent.parent
        if (
            Path(reference).name != reference
            or not reference.startswith("executor_")
            or not reference.endswith(".py")
            or not (reference_dir / reference).is_file()
        ):
            raise ValueError(f"Invalid or missing reference {reference!r} for {name}")
        suite = {**group, **entry}
        validate_cases(suite, cases)
        yield suite, cases


def verify_upstream_checkout(path):
    path = path.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != REFERENCE_SHA:
        raise ValueError(f"work_tool HEAD must be {REFERENCE_SHA}; received {commit}")
    subprocess.run(
        ["git", "-C", str(path), "diff", "--quiet", "HEAD", "--", "FLA_ATK"], check=True
    )
    if not (path / "FLA_ATK/common").is_dir():
        raise ValueError("work_tool checkout must include FLA_ATK")


def metrics_passed(row, shapes):
    details = row["精度详情"]
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except ValueError:
            details = ast.literal_eval(details)
    outputs = [
        entry
        for group in details.values()
        if isinstance(group, list)
        for entry in group
        if isinstance(entry, dict)
    ]
    expected_files = {f"output_{i}.pt" for i in range(len(shapes))}
    return not (
        row.get("cpu_cpu_golden_精度通过") is not True
        or len(outputs) != len(expected_files)
        or {e.get("filename") for e in outputs} != expected_files
        or any(
            e.get("result") is not True
            or not (
                e.get("new_benchmark_indicate")
                or any(
                    "Benchmark_" in str(value) or "相对误差比例" in str(value)
                    for value in e.values()
                )
            )
            for e in outputs
        )
    )


def check_report(folder, cases, store=None):
    reports = list(folder.glob("atk_output/*/report/*.json")) or list(
        folder.glob("atk_output/*/report/*.xlsx")
    )
    if len(reports) != 1:
        raise ValueError(f"Expected one ATK report in {folder}, found {len(reports)}")
    report = reports[0]
    if report.suffix == ".xlsx":
        import openpyxl

        book = openpyxl.load_workbook(report, read_only=True, data_only=True)
        try:
            it = iter(book.active.values)
            header = next(it)
            rows = [dict(zip(header, row)) for row in it]
        finally:
            book.close()
    else:
        rows = read(report)
        if isinstance(rows, dict):
            rows = rows.get("statistic")
        if not isinstance(rows, list):
            raise ValueError(f"Expected ATK accuracy rows in {report}")
    expected = {(c["id"], c["name"]) for c in cases}
    roles = {}
    for role in ("dut", "low", "high"):
        seen = {}
        for line in (folder / f"events.{role}").read_text(encoding="utf8").splitlines():
            ident, name, tensors = json.loads(line)
            if (ident, name) in seen or not tensors:
                raise ValueError(
                    f"Missing outputs or duplicate {role} record for {ident}/{name}"
                )
            seen[ident, name] = [shape for shape, dtype in tensors]
        if set(seen) != expected:
            raise ValueError(f"{role} did not execute the complete suite in {folder}")
        roles[role] = seen
    if roles["dut"] != roles["low"] or roles["dut"] != roles["high"]:
        raise ValueError(f"DUT and CPU reference shapes differ in {folder}")
    evidence = {}
    seen = set()
    failures = []
    for row in rows:
        config = row["用例json信息"]
        if isinstance(config, str):
            config = json.loads(config)
        acc = config["standard"]["acc"]
        if (list(acc) if isinstance(acc, dict) else [acc]) != [STANDARD]:
            raise ValueError(f"Unexpected accuracy standard in {report}")
        key = config["id"], config["name"]
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected report entry {key} in {report}")
        seen.add(key)
        shapes = roles["dut"][key]
        if metrics_passed(row, shapes):
            # ATK's unused performance columns contain NaN in accuracy-only runs.
            accuracy = {field: row[field] for field in ("用例json信息", "cpu_cpu_golden_精度通过", "精度详情")}
            evidence[key] = {"row": accuracy, "shapes": shapes}
        else:
            failures.append(key)
    if seen == expected and store is not None:
        for ident, value in evidence.items():
            store(ident, value)
    if seen != expected or failures:
        raise ValueError(
            f"{len(expected - seen)} missing, {len(failures)} failed: {failures[:20]} ({report})"
        )


def run_suite(suite, cases, output, device):
    folder = output / suite["group"]
    folder.mkdir()
    cache = ResultCache()
    pending = []
    reused = []
    keys = {}
    for case in cases:
        # Without an explicit input seed we cannot prove the next ATK tensors are identical.
        if case.get("default_seed") is None or cache.root is None:
            pending.append(case)
            continue
        ident = case["id"], case["name"]
        key = cache.key("atk", [{k: v for k, v in suite.items() if k != "count"}, case])
        keys[ident] = key
        entry = cache.get(key)
        try:
            value = entry["value"] if entry else None
            valid = value and value["shapes"] and metrics_passed(value["row"], value["shapes"])
        except (KeyError, TypeError, SyntaxError, ValueError, AttributeError):
            valid = False
        if not valid:
            pending.append(case)
        else:
            reused.append({"id": ident[0], "name": ident[1], "key": key, "origin": entry["origin"],
                           "evidence": entry["value"]})
    if reused:
        report_dir = folder / "cache" / "report"
        report_dir.mkdir(parents=True)
        (report_dir / "reused.json").write_text(json.dumps(reused, ensure_ascii=False), encoding="utf8")
    print(f"[smoke-cache] ATK {suite['group']}: {len(reused)} reused, {len(pending)} to run", flush=True)
    if not pending:
        return
    # ATK consumes a flat list and one execution configuration per invocation.
    config = folder / "pending.json"
    instrumented = [{**case, "api_type": "fla_smoke"} for case in pending]
    config.write_text(json.dumps(instrumented, ensure_ascii=False), encoding="utf8")
    env = os.environ.copy()
    for flag in (
        "KDA_ATK_KERNEL_SEMANTICS",
        "KDA_ATK_VISIBLE_OUTPUTS",
        "CHUNK_FWD_O_A5_OPT",
    ):
        env.pop(flag, None)
    env.update(
        FLA_ATK_SUITE=json.dumps(suite),
        FLA_ATK_TRACE=str(folder / "events"),
        GDN_ATK_BENCHMARK_ARCH="arch22" if suite["platform"] == "A2" else "arch35",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        KDA_ATK_REFERENCE_WORKERS="1",
    )
    if suite.get("mode") == "state":
        env.update(
            KDA_ATK_KERNEL_SEMANTICS="state",
            KDA_ATK_VISIBLE_OUTPUTS="attn_out,final_state",
        )
    if suite.get("mode") == "opt":
        env["CHUNK_FWD_O_A5_OPT"] = "1"
    command = [
        "atk",
        "node",
        "--name",
        "npu_dut",
        "--backend",
        "npu",
        "--devices",
        str(device),
        "-o",
        str(folder),
        "node",
        "--name",
        "cpu_golden",
        "--backend",
        "cpu",
        "-o",
        str(folder),
        "task",
        "-c",
        str(config),
        "-p",
        str(ROOT / "executor.py"),
        "--task",
        "accuracy",
        "-sp",
        "-mt",
        "1",
        "-to",
        "300",
        "--bm_device",
        "cpu",
        "--save_report_json",
    ]
    print(f"Running {suite['group']} ({len(pending)} cases): {folder}", flush=True)
    with (folder / "atk.log").open("w", encoding="utf8") as log:
        run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)

    def store(ident, value):
        if ident in keys:
            cache.put(keys[ident], value)

    # Only complete, checked three-role evidence can contribute passing cases.
    check_report(folder, pending, store if run.returncode == 0 else None)
    run.check_returncode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    matrix = actions.add_parser("matrix", help="Print a GitHub Actions include matrix")
    matrix.add_argument("--git-range", nargs=2, metavar=("BASE", "HEAD"))
    affected = actions.add_parser("affected", help="List changed smoke operators for one CI platform")
    affected.add_argument("--git-range", nargs=2, metavar=("BASE", "HEAD"), required=True)
    affected.add_argument("--platform", choices=("A2", "A5"), required=True)
    actions.add_parser("validate", help="Check every case file without an NPU")
    run = actions.add_parser("run", help="Run the suites for one platform and operator")
    run.add_argument("--op", required=True)
    run.add_argument("--platform", choices=("A2", "A5"), required=True)
    run.add_argument(
        "--reference",
        type=Path,
        help="Optionally verify the pinned upstream checkout; execution uses the bundled references",
    )
    run.add_argument(
        "--device", type=int, default=0, help="Logical NPU in the CI container"
    )
    run.add_argument("--output", type=Path, default=Path("atk-results"))
    run.add_argument("--group", help="Run a single suite for local reproduction")
    args = parser.parse_args()
    suites = read(ROOT / "suites.json")
    indexed = {case_path(s) for s in suites}
    if len(indexed) != len(suites) or indexed != set(
        (ROOT / "cases").glob("*/*/*.json")
    ):
        raise ValueError("Expected one cases.json per operator and platform in suites.json")
    if args.action in ("matrix", "affected"):
        roots = source_roots(suites)
        pairs = {(s["op"], s["platform"]) for s in suites if s.get("ci_enabled", True)}
        if args.git_range:
            pairs = affected_pairs(pairs, roots, *args.git_range, ROOT.parents[2])
        pairs = sorted(pairs)
        if args.action == "affected":
            for op, platform in pairs:
                if platform == args.platform:
                    print(op)
            return
        print(
            json.dumps(
                {
                    "include": [
                        {"op": op, "platform": platform} for op, platform in pairs
                    ]
                },
                separators=(",", ":"),
            )
        )
    elif args.action == "validate":
        source_roots(suites)
        counts = [len(cases) for entry in suites for _, cases in groups_for(entry)]
        print(f"{sum(counts)} cases in {len(counts)} groups across {len(suites)} platform files")
    else:
        selected = [
            (suite, cases)
            for entry in suites
            if entry["platform"] == args.platform and entry["op"] == args.op
            for suite, cases in groups_for(entry)
            if not args.group or args.group == suite["group"]
        ]
        if not selected:
            parser.error("No passing cases are registered for this selection")
        if args.reference is not None:
            verify_upstream_checkout(args.reference)
        args.output.mkdir(parents=True, exist_ok=True)
        output = Path(
            tempfile.mkdtemp(
                prefix=f"{args.op}-{args.platform}-", dir=args.output.resolve()
            )
        )
        for suite, cases in selected:
            run_suite(suite, cases, output, args.device)


if __name__ == "__main__":
    main()
