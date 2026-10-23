"""CPU checks for public contracts restored during Stable-only migration.

The existing NPU cumsum/Conv1d suites remain the numerical acceptance tests.
These tests independently check forwarding, explicit validation and absence of
launches or mutation on rejected metadata; they do not replace NPU acceptance.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PACKAGE = Path(__file__).resolve().parents[1] / "torch_custom/fla_npu/fla_npu"


class Tensor:
    def __init__(self, values=None, *, shape=None, dtype="int32", device="npu:0"):
        self.values = [] if values is None else values
        self.shape = (len(self.values),) if shape is None else shape
        self.dtype = dtype
        self.device = types.SimpleNamespace(type=device.split(":")[0], name=device)
        self.requires_grad = False
        self.reads = 0
        self.copies = 0

    def dim(self):
        return len(self.shape)

    def detach(self):
        return self

    def cpu(self):
        self.reads += 1
        return self

    def tolist(self):
        return list(self.values)

    def copy_(self, value):
        self.copies += 1
        return self


class StablePublicContractTests(unittest.TestCase):
    def setUp(self):
        self.old_modules = {
            name: module for name, module in sys.modules.items()
            if name == "fla_npu" or name.startswith("fla_npu.")
        }
        for name in self.old_modules:
            sys.modules.pop(name)
        package = types.ModuleType("fla_npu")
        package.__path__ = [str(PACKAGE)]
        package.load_ascendc_opapi_libraries = lambda: []
        sys.modules["fla_npu"] = package
        self.stable = importlib.import_module("fla_npu.ops.ascendc._stable")
        torch = types.ModuleType("torch")
        torch.Tensor = Tensor
        torch.int32, torch.int64, torch.bool = "int32", "int64", "bool"
        self.torch_mock = mock.patch.dict(sys.modules, {"torch": torch})
        self.torch_mock.start()
        self.x = Tensor(shape=(5, 16), dtype="float16")
        self.state = Tensor(shape=(3, 2, 16), dtype="float16")
        self.weight = Tensor(shape=(3, 16), dtype="float16")
        self.dispatch = mock.patch.object(self.stable, "_op")
        self.resolve = self.dispatch.start()
        self.stream = mock.patch.object(self.stable, "_current_stream_ptr", return_value=99)
        self.get_stream = self.stream.start()
        self.ints = mock.patch.object(self.stable, "_host_ints", side_effect=lambda value: value)
        self.ints.start()
        self.bump = mock.patch.object(self.stable, "_INC_VERSION", mock.Mock())
        self.bump.start()
        self.stable._OP_CACHE.clear()

    def tearDown(self):
        self.bump.stop()
        self.ints.stop()
        self.stream.stop()
        self.dispatch.stop()
        self.torch_mock.stop()
        for name in list(sys.modules):
            if name == "fla_npu" or name.startswith("fla_npu."):
                sys.modules.pop(name)
        sys.modules.update(self.old_modules)

    def update(self, **kwargs):
        return self.stable.npu_causal_conv1d_update(
            self.x, self.state, self.weight, **kwargs)

    def assert_no_launch_or_mutation(self):
        self.resolve.assert_not_called()
        self.get_stream.assert_not_called()
        self.stable._INC_VERSION.assert_not_called()
        self.assertEqual((self.x.copies, self.state.copies), (0, 0))

    def test_cumsum_historical_and_documented_dtype_aliases(self):
        spellings = {
            0: [None, "", "float", "float32", "fp32", "torch.float", "torch.float32"],
            1: ["bfloat16", "bf16", "torch.bfloat16"],
            2: ["float16", "fp16", "half", "torch.float16", "torch.half"],
            3: ["same", "same_as_input", "input", "none", "None", "null"],
        }
        for code, names in spellings.items():
            for name in names:
                with self.subTest(output_dtype=name):
                    self.stable.npu_chunk_local_cumsum(
                        self.x, 64, cu_seqlens=[0, 5], reverse=True, scale=0.5,
                        head_first=False, output_dtype=name)
                    self.resolve.return_value.assert_called_with(
                        self.x, [0, 5], None, 64, True, 0.5, False, code, 99)

    def test_cumsum_dtype_objects_and_same_follow_input_without_python_allocation(self):
        class Dtype:
            def __init__(self, name):
                self.name = name

            def __str__(self):
                return "torch." + self.name

        for name, code in [("float32", 0), ("bfloat16", 1), ("float16", 2)]:
            self.stable.npu_chunk_local_cumsum(self.x, 64, output_dtype=Dtype(name))
            self.assertEqual(self.resolve.return_value.call_args.args[-2], code)
            self.x.dtype = name
            self.stable.npu_chunk_local_cumsum(self.x, 64, output_dtype="same")
            self.assertEqual(self.resolve.return_value.call_args.args[-2], 3)

    def test_cumsum_invalid_output_dtype_never_launches(self):
        for value in ("float64", "int32", "unsupported", 2):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "Unsupported.*output_dtype"):
                self.stable.npu_chunk_local_cumsum(self.x, 64, output_dtype=value)
        self.resolve.return_value.assert_not_called()

    def test_conv1d_backward_bsh_alias_matches_documented_bsnd_arguments(self):
        for layout in ("BSH", "BSND"):
            self.stable.npu_causal_conv1d_bwd(
                self.x, None, self.weight, self.x,
                initial_state=self.state, dht=self.state,
                activation=2, input_layout=layout)
            self.resolve.return_value.assert_called_with(
                self.x, None, self.weight, self.x, self.state, self.state,
                None, 2, 0, 99)
        with self.assertRaisesRegex(RuntimeError, "input_layout must be one of"):
            self.stable.npu_causal_conv1d_bwd(
                self.x, None, self.weight, self.x, input_layout="unknown")

    def test_update_undersized_bound_rejects_device_metadata_before_mutation(self):
        qsl = Tensor([0, 2, 5])
        with self.assertRaisesRegex(ValueError, "observed segment length 3"):
            self.update(query_start_loc=qsl, max_query_len=2, validate_data=True)
        self.assertEqual(qsl.reads, 1)
        self.assert_no_launch_or_mutation()

    def test_conv1d_none_disables_null_slot_and_default_still_reserves_zero(self):
        # The failing head_num=2 case uses cache slot zero with explicit None;
        # translating None to the public default 0 incorrectly skips that row.
        x = Tensor(shape=(5, 32), dtype="float16")
        weight = Tensor(shape=(4, 32), dtype="float16")
        state = Tensor(shape=(2, 3, 32), dtype="float16")
        for kwargs, expected in [({}, 0), ({"null_block_id": None}, -1),
                                 ({"null_block_id": 0}, 0), ({"null_block_id": 3}, 3)]:
            with self.subTest(kwargs=kwargs):
                self.stable.npu_causal_conv1d_fn(
                    x, weight, None, state, query_start_loc_cpu=[0, 2, 5],
                    cache_indices_cpu=[0, 1], head_num=2, **kwargs)
                self.assertEqual(self.resolve.return_value.call_args.args[12], expected)
                self.assertEqual(self.resolve.return_value.call_args.args[13], 2)
                self.update(conv_state_indices_cpu=[0, 1],
                            query_start_loc_cpu=[0, 2, 5], max_query_len=3, **kwargs)
                self.assertEqual(self.resolve.return_value.call_args.args[9], expected)

    def test_update_undersized_bound_rejects_cpu_metadata_before_mutation(self):
        for qsl in ([0, 2, 5], (0, 2, 5), Tensor([0, 2, 5], dtype="int64", device="cpu")):
            with self.subTest(type=type(qsl)), self.assertRaisesRegex(ValueError, "observed segment length 3"):
                self.update(query_start_loc_cpu=qsl, max_query_len=2, validate_data=True)
        self.assert_no_launch_or_mutation()

    def test_fn_and_update_reject_invalid_boundaries_without_launch(self):
        for values in ([], [0], [1, 2, 5], [0, 2, 4], [0, 4, 3, 5]):
            for mode in ("fn", "update"):
                with self.subTest(values=values, mode=mode), self.assertRaises(ValueError):
                    if mode == "fn":
                        self.stable.npu_causal_conv1d_fn(
                            self.x, self.weight, None, self.state,
                            query_start_loc_cpu=values, validate_data=True)
                    else:
                        self.update(query_start_loc_cpu=values, max_query_len=5, validate_data=True)
        self.assert_no_launch_or_mutation()

    def test_bound_type_and_required_value_reject_before_launch(self):
        for value, error in [(True, TypeError), (2.0, TypeError), (-2, ValueError), (-1, ValueError)]:
            with self.subTest(value=value), self.assertRaises(error):
                self.update(query_start_loc_cpu=[0, 2, 5], max_query_len=value)
        self.assert_no_launch_or_mutation()

    def test_validation_is_opt_in_and_valid_bound_reaches_same_metadata(self):
        qsl = Tensor([0, 2, 5])
        self.update(query_start_loc=qsl, max_query_len=2, validate_data=False)
        self.assertEqual(qsl.reads, 0)
        self.assertEqual(self.resolve.return_value.call_args.args[8], 2)
        self.update(query_start_loc=qsl, max_query_len=3, validate_data=True)
        self.assertEqual(qsl.reads, 1)
        self.assertIs(self.resolve.return_value.call_args.args[7], qsl)
        self.assertEqual(self.resolve.return_value.call_args.args[8], 3)
        self.assertEqual(self.x.copies, 2)

    def test_validation_rejects_ambiguous_wrong_device_rank_and_dtype_metadata(self):
        cases = [
            {"query_start_loc": Tensor([0, 2, 5]), "query_start_loc_cpu": [0, 2, 5]},
            {"query_start_loc": Tensor([0, 2, 5], device="npu:1")},
            {"query_start_loc": Tensor([0, 2, 5], shape=(1, 3))},
            {"query_start_loc": Tensor([0, 2, 5], dtype="int64")},
            {"query_start_loc_cpu": Tensor([0, 2, 5])},
            {"query_start_loc_cpu": Tensor([0, 2, 5], dtype="float32", device="cpu")},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                self.update(max_query_len=3, validate_data=True, **kwargs)
        self.assert_no_launch_or_mutation()


if __name__ == "__main__":
    unittest.main()
