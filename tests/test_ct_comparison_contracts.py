"""Execute legacy comparison blocks against the CT 0.9.1 return contract.

Only the actual CT calls and their result checks are compiled from each script.
The NPU setup, input generation and CPU mathematics are never imported or run.
"""
import ast
from dataclasses import dataclass
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "torch_custom/fla_npu/test"


def comparison_block(filename, function_name):
    path = SCRIPTS / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == function_name)

    def calls_ct(node):
        return any(isinstance(item, ast.Call)
                   and isinstance(item.func, ast.Attribute)
                   and isinstance(item.func.value, ast.Name)
                   and item.func.value.id == "ct"
                   and item.func.attr in {"isclose", "dual"}
                   for item in ast.walk(node))

    calls = [node for node in ast.walk(function)
             if isinstance(node, (ast.Assign, ast.Expr, ast.Assert)) and calls_ct(node)]
    result_names = {name.id for node in calls if isinstance(node, ast.Assign)
                    for target in node.targets for name in ast.walk(target)
                    if isinstance(name, ast.Name)}
    checks = [node for node in ast.walk(function) if isinstance(node, ast.Assert)
              and not calls_ct(node)
              and any(isinstance(item, ast.Name) and item.id in result_names
                      for item in ast.walk(node.test))]
    statements = sorted(calls + checks, key=lambda node: node.lineno)
    if not calls:
        raise AssertionError(f"No CT comparisons found in {path}:{function_name}")
    return compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec")


@dataclass(frozen=True)
class TensorRole:
    output: str
    role: str

    def cpu(self):
        return self


class FakeCT:
    """No numeric oracle: only CT's documented result types and input roles."""

    def __init__(self, failed_output=None, failed_value=None):
        self.failed_output = failed_output
        self.failed_value = failed_value
        self.calls = []

    def isclose(self, real, expect, **kwargs):
        self.calls.append(("isclose", (real, expect), kwargs))
        value = self.failed_value if real.output == self.failed_output else "success"
        return {"result": value}

    def dual(self, test, gt, bench, **kwargs):
        self.calls.append(("dual", (test, gt, bench), kwargs))
        value = self.failed_value if test.output == self.failed_output else True
        return {"success": value}


CASES = (
    ("test_npu_recompute_w_u_fwd.py", "test_recompute_w_u_fwd", "isclose", ("w", "u")),
    ("test_npu_chunk_bwd_dv_local.py", "test_variable", "dual", ("dv",)),
    ("test_npu_chunk_bwd_dv_local.py", "test_fix", "dual", ("dv",)),
    ("test_npu_prepare_wy_repr_bwd_full.py", "test_prepare_wy_repr_bwd_full", "dual",
     ("dv", "dk", "dg", "dbeta")),
)


def namespace_for(outputs, ct):
    namespace = {"ct": ct}
    for output in outputs:
        namespace[output] = TensorRole(output, "dut")
        namespace[f"cpu_{output}"] = TensorRole(output, "benchmark")
        namespace[f"cpu_{output}_high_precision"] = TensorRole(output, "high_precision")
        namespace[f"{output}_golden"] = namespace[f"cpu_{output}"]
        namespace[f"{output}_golden_high_precision"] = namespace[f"cpu_{output}_high_precision"]
    return namespace


class CTComparisonContractTest(unittest.TestCase):
    def test_success_compares_every_output_with_correct_roles(self):
        for filename, function, method, outputs in CASES:
            with self.subTest(script=filename, function=function):
                ct = FakeCT()
                exec(comparison_block(filename, function), namespace_for(outputs, ct))
                roles = ("dut", "benchmark") if method == "isclose" else (
                    "dut", "high_precision", "benchmark")
                options = {"diff_thd": 0.1} if method == "isclose" else {}
                expected = [(method, tuple(TensorRole(output, role) for role in roles), options)
                            for output in outputs]
                self.assertEqual(ct.calls, expected)

    def test_failure_in_any_output_raises(self):
        for filename, function, method, outputs in CASES:
            for output in outputs:
                with self.subTest(script=filename, function=function, output=output):
                    failure = "failed" if method == "isclose" else False
                    ct = FakeCT(failed_output=output, failed_value=failure)
                    with self.assertRaises(AssertionError):
                        exec(comparison_block(filename, function), namespace_for(outputs, ct))
                    self.assertEqual(ct.calls[-1][1][0].output, output)

    def test_isclose_requires_success_string(self):
        filename, function, _, outputs = CASES[0]
        for output in outputs:
            for invalid in (True, "failed", "Failed", "Pass", None, ""):
                with self.subTest(output=output, result=invalid):
                    ct = FakeCT(failed_output=output, failed_value=invalid)
                    with self.assertRaises(AssertionError):
                        exec(comparison_block(filename, function), namespace_for(outputs, ct))

    def test_dual_requires_boolean_true(self):
        for filename, function, method, outputs in CASES:
            if method != "dual":
                continue
            for output in outputs:
                for invalid in (1, "true", "success", None):
                    with self.subTest(script=filename, function=function, output=output, result=invalid):
                        ct = FakeCT(failed_output=output, failed_value=invalid)
                        with self.assertRaises(AssertionError):
                            exec(comparison_block(filename, function), namespace_for(outputs, ct))


if __name__ == "__main__":
    unittest.main()
