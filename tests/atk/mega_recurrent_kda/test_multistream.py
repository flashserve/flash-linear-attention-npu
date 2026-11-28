"""Cross-stream regression for mixed Cube/Vector barriers and mutable replay buffers."""
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
from executor_mega_recurrent_kda import build_inputs, device_inputs

MUTABLE = ("conv_state", "replay_u", "replay_k", "replay_decay")


@pytest.mark.parametrize("quantized,lengths", [(False, [4, 4]), (False, [63, 2]),
                                             (True, [4] * 8), (True, [3, 65, 7])])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_independent_streams_match_serial(quantized, lengths, graph_mode):
    if not torch.npu.is_available():
        pytest.skip("Requires Ascend device")
    device_name = torch.npu.get_device_name()
    a5 = "950" in device_name
    if not a5 and not device_name.startswith(("Ascend910B", "Ascend910_93", "Ascend910C")):
        pytest.skip("Requires Atlas A2/A3 or Ascend 950")
    from fla_npu.ops.ascendc import npu_mega_recurrent_kda, npu_mega_recurrent_kda_metadata

    streams = [torch.npu.Stream() for _ in range(2)]
    args, initial, references, checkpoints = [], [], [], []
    with torch.inference_mode():
        for i in range(2):
            spec = dict(name="streams", lengths=lengths, heads=6, hidden=4160,
                        layout="TH", conv="flat", seed=17+i)
            if quantized and a5:
                spec["qkv_scale_alg"] = i
            host = build_inputs(spec)
            current = device_inputs(host, torch.device("npu", torch.npu.current_device()))
            if quantized and not a5:
                from test_int8 import quantized_weights
                quantized_weights(host, current, True)
            current["metadata"] = npu_mega_recurrent_kda_metadata(current["cu_seqlens"], sum(lengths))
            args.append(current)
            initial.append({key: current[key].clone() for key in MUTABLE})
            checkpoints.append(current["checkpoint_state"].cpu())
            references.append([x.cpu() for x in npu_mega_recurrent_kda(**current)])
        torch.npu.synchronize()
        graphs, outputs = [], [None, None]
        if graph_mode:
            for i in range(2):
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=streams[i]):
                    for key in MUTABLE:
                        args[i][key].copy_(initial[i][key])
                    outputs[i] = npu_mega_recurrent_kda(**args[i])
                graphs.append(graph)
            torch.npu.synchronize()
        for _ in range(12):
            for i in range(2):
                with torch.npu.stream(streams[i]):
                    if graph_mode:
                        graphs[i].replay()
                    else:
                        for key in MUTABLE:
                            args[i][key].copy_(initial[i][key])
                        outputs[i] = npu_mega_recurrent_kda(**args[i])
            torch.npu.synchronize()
            for i in range(2):
                assert all(x.data_ptr() == args[i][key].data_ptr()
                           for x, key in zip(outputs[i][1:], MUTABLE))
                assert all(torch.equal(x.cpu(), expected) for x, expected in zip(outputs[i], references[i]))
                assert torch.equal(args[i]["checkpoint_state"].cpu(), checkpoints[i])
