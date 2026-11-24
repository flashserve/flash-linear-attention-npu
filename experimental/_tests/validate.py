#!/usr/bin/env python3
"""Manual experimental build, installation, load and CPU-reference validation."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OP = "fused_recurrent_rwkv8"
PHASES = ("dependencies", "build-host", "load-isolation", "accuracy-queue", "accuracy-inline")


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


ISOLATION = module(ROOT / "experimental/_tests/isolation.py")


def clean_environment(case, profile):
    env = ISOLATION.environment(case, profile)
    # Ignore an inherited FLA vendor library search path.
    # Keep toolkit/driver dependencies, but do not let ELF lookup reuse a FLA vendor.
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            path for path in env["LD_LIBRARY_PATH"].split(os.pathsep) if path and not any(
                part in {"fla_npu_transformer", "fla_npu_experimental_transformer"}
                or part.startswith("fla_npu_overlay_") for part in Path(path).parts))
    return env


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def check_accuracy(result, expected):
    if (expected <= 0 or not result.get("complete")
            or len(result.get("results", [])) != expected
            or result.get("case_indices") != list(range(expected))
            or any(item.get("index") != index or item.get("status") != "PASS"
                   for index, item in enumerate(result.get("results", [])))
            or not result.get("found") or not result.get("all_pass")
            or result.get("total") != expected or result.get("check_pass") != expected
            or result.get("exec_fail") != 0 or result.get("check_fail") != 0):
        raise RuntimeError(f"CPU-reference accuracy did not pass exactly {expected} planned cases")
    return result


class Validation:
    def __init__(self, output, *, soc="ascend910b", mode="quick", device=0, jobs=4, build_only=False):
        self.output = output
        self.build_only = build_only or soc == "ascend950"
        self.soc = soc
        self.mode = mode
        self.jobs = jobs
        self.env = clean_environment({"mode": "stable", "scenario": "stable"}, {})
        for key in ("ATK_ENV", "CANN_ENV", "FLA_NPU_ENV", "FLA_NPU_OPP_ENV", "FLA_NPU_STABLE_LAUNCH"):
            self.env.pop(key, None)
        self.report = {"schema": "experimental-validation-v1", "complete": False, "status": "running",
                       "soc": self.soc, "mode": self.mode, "output": str(output),
                       "device": str(device),
                       "build_only": self.build_only,
                       "phases": {name: {"status": "not_run"} for name in PHASES}}
        self.profile = {}
        self.log = None

    def save(self):
        write_json(self.output / "summary.json", self.report)

    def run(self, command, env=None, timeout=7200):
        subprocess.run(list(map(str, command)), cwd=ROOT, env=env or self.env,
                       stdout=self.log, stderr=subprocess.STDOUT, check=True, timeout=timeout)

    def phase(self, name, action):
        item = self.report["phases"][name]
        item.update(status="running", log=str(self.output / (name + ".log")))
        self.save()
        started = time.monotonic()
        print(f"[experimental][validation] START {name}", flush=True)
        try:
            with Path(item["log"]).open("w", encoding="utf-8") as self.log:
                detail = action()
            item.update(status="success", result=detail)
        except Exception as exc:
            item.update(status="failure", reason=str(exc))
            item["log_tail"] = Path(item["log"]).read_text(encoding="utf-8", errors="replace")[-6000:]
            print(item["log_tail"], file=sys.stderr, flush=True)
            raise
        finally:
            item["seconds"] = round(time.monotonic() - started, 3)
            self.save()
            print(f"[experimental][validation] {item['status'].upper()} {name}", flush=True)

    def dependencies(self):
        output = self.output / "dependencies.json"
        try:
            self.run([sys.executable, ROOT / "experimental/_tests/check_environment.py", "--output", output],
                     timeout=720)
        finally:
            if output.exists():
                self.report["phases"]["dependencies"]["result"] = json.loads(output.read_text())
        result = json.loads(output.read_text())
        for warning in result["warnings"]:
            print(f"[experimental][WARNING] {warning}", flush=True)
        return result

    def build(self):
        if not self.env.get("ASCEND_HOME_PATH"):
            raise RuntimeError("Source the CANN toolkit set_env.sh first")
        self.run(["bash", "build.sh", "--pkg", "--experimental", "--soc=" + self.soc,
                  "--ops=all", "-j" + str(self.jobs)])
        run_package = ROOT / ("build_out/fla_npu_experimental_linux-" + os.uname().machine + ".run")
        install = self.output / "opp"
        install.mkdir()
        self.run([run_package, "--install-path=" + str(install), "--quiet"])
        manifest = ROOT / "build/experimental-build-manifest.json"
        audit = module(ROOT / "experimental/scripts/experimental_manifest.py")
        package = audit.audit_package(install, self.soc, manifest)
        wheel_root = self.output / "wheel"
        self.run([sys.executable, "experimental/_framework/build.py", "wheel", "--manifest", manifest,
                  "--install-root", install, "--output", wheel_root])
        wheels = list(wheel_root.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Expected exactly one experimental wheel")
        self.products = {"opp": str(install), "wheel": str(wheels[0])}
        self.prepare()
        env = clean_environment({"mode": "experimental", "scenario": "experimental"}, self.profile)
        self.run([self.profile["python"], ROOT / "experimental/_tests/load_smoke.py"], env)
        shutil.copy2(manifest, self.output)
        return {"products": self.products, "package": package, "host_loaded": True}

    def prepare(self):
        env_root = self.output / "env"
        self.run([sys.executable, "-m", "venv", "--without-pip", env_root])
        python = env_root / "bin/python"
        purelib = Path(subprocess.check_output([str(python), "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))"], env=self.env, text=True).strip())
        # Share dependencies without inheriting any FLA package or startup hook.
        for directory in site.getsitepackages():
            if not Path(directory).is_dir():
                continue
            for entry in Path(directory).iterdir():
                if (entry.name.startswith(("fla_npu", "flash_linear_attention", "__editable__"))
                        or entry.name in {"fla", "sitecustomize.py", "usercustomize.py", "__pycache__"}
                        or entry.suffix == ".pth" or (purelib / entry.name).exists()):
                    continue
                (purelib / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        self.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                  "--target", purelib, self.products["wheel"]])
        self.profile = {"python": str(python), "experimental_package": str(purelib / "fla_npu_experimental"),
                        "external_experimental_opp": self.products["opp"],
                        "device": int(self.report["device"]), "timeout": 120}
        build = json.loads((purelib / "fla_npu_experimental/_build.json").read_text())
        if [op["name"] for op in build["operators"]] != [OP]:
            raise RuntimeError("Update manual accuracy coverage for the new experimental operator set")

    def isolation(self):
        # These are rejection fixtures, never a compiled/installed formal operator.
        stable = self.output / "protected/fla_npu"
        library = stable / "opp/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so"
        library.parent.mkdir(parents=True)
        (stable / "__init__.py").write_text("# protected formal-package fixture\n")
        source = self.output / "foreign.c"
        source.write_text("void isolation_fixture(void) {}\n")
        self.run(["gcc", "-shared", "-fPIC", source, "-o", library])
        shutil.copy2(library, stable / "libfla_npu_stable.so")
        self.profile["stable_package"] = str(stable)
        before = ISOLATION.snapshot([stable])
        run_package = ROOT / ("build_out/fla_npu_experimental_linux-" + os.uname().machine + ".run")
        for target in (stable, stable / "opp", stable / "new-empty-dir"):
            result = subprocess.run([str(run_package), "--install-path=" + str(target), "--quiet"],
                                    cwd=ROOT, env=self.env, text=True, capture_output=True, timeout=120)
            if result.returncode == 0 or "stable" not in (result.stdout + result.stderr).lower():
                raise RuntimeError("Experimental run did not reject a formal-package destination: " + str(target))
        if before != ISOLATION.snapshot([stable]):
            raise RuntimeError("Rejected run installation changed the protected formal-package fixture")
        result = ISOLATION.validate(self.profile, self.output / "isolation")
        self.report["phases"]["load-isolation"]["result"] = result
        if not result["complete"] or result["counts"] != {"PASS": len(ISOLATION.CASES), "FAIL": 0, "BLOCKED": 0}:
            raise RuntimeError("Incomplete experimental load isolation matrix")
        result["installation_rejections"] = 3
        return result

    def accuracy(self, launch):
        profile = self.profile
        env = clean_environment({"mode": "experimental", "scenario": "experimental"}, profile)
        env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   FLA_A2_TRACE_DIR=str(self.output / ("accuracy-" + launch + "-trace")))
        env.pop("FLA_NPU_STABLE_LAUNCH", None)
        if launch == "inline":
            env["FLA_NPU_STABLE_LAUNCH"] = "inline"
        expected = 3 if self.mode == "quick" else len(json.loads(
            (ROOT / f"experimental/tests/atk/{OP}/atk_{OP}.json").read_text()))
        result_file = self.output / ("accuracy-" + launch + ".json")
        try:
            self.run([profile["python"], ROOT / "experimental/_tests/accuracy.py",
                      "--mode", self.mode, "--device", self.report["device"], "--output", result_file],
                     env, timeout=660 if self.mode == "quick" else 7260)
        finally:
            if result_file.exists():
                # Preserve per-case numerical diagnostics in the uploaded stage artifact.
                self.report["phases"]["accuracy-" + launch]["result"] = json.loads(result_file.read_text())
        result = json.loads(result_file.read_text())
        result["execution_evidence"] = [json.loads(path.read_text()) for path in
                                        Path(env["FLA_A2_TRACE_DIR"]).glob("experimental-*.json")]
        if not result["execution_evidence"]:
            raise RuntimeError("Missing evidence of real experimental NPU execution")
        return check_accuracy(result, expected)

    def execute(self):
        self.save()
        try:
            if self.soc not in {"ascend910b", "ascend950"} or self.mode not in {"quick", "full"}:
                raise ValueError(f"Unsupported experimental validation configuration: {self.soc}/{self.mode}")
            self.phase("dependencies", self.dependencies)
            self.phase("build-host", self.build)
            if not self.build_only:
                self.phase("load-isolation", self.isolation)
                for launch in ("queue", "inline"):
                    self.phase("accuracy-" + launch, lambda launch=launch: self.accuracy(launch))
            else:
                for name in PHASES[2:]:
                    self.report["phases"][name] = {"status": "skipped", "reason": "Build-only validation; NPU accuracy was not run"}
            self.report.update(complete=True, status="success")
        except Exception as exc:
            self.report.update(status="failure", reason=str(exc))
            for item in self.report["phases"].values():
                if item["status"] == "not_run":
                    item.update(status="not_run", reason="Previous experimental check failed")
        finally:
            self.save()
        print(f"[experimental][validation] {self.report['status']}: {self.output / 'summary.json'}", flush=True)
        if self.report.get("reason"):
            print(self.report["reason"], file=sys.stderr, flush=True)
        return 0 if self.report["status"] == "success" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true", help="Build/package/host checks; no NPU accuracy claim")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--soc", choices=("ascend910b", "ascend950"), default="ascend910b")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("-j", "--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1 or args.device < 0:
        parser.error("--jobs must be positive and --device must be nonnegative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    return Validation(output, soc=args.soc, mode=args.mode, device=args.device,
                      jobs=args.jobs, build_only=args.build_only).execute()


if __name__ == "__main__":
    raise SystemExit(main())
