from pathlib import Path

import torch

from cpu_golden_cache import cached_cpu_outputs


def test_cached_state_updates_and_invalidation(tmp_path):
    calls = []

    def reference(x, state):
        calls.append(1)
        state.add_(x)
        return x + state, state

    def run(x, state):
        result = cached_cpu_outputs(reference, {"x": x, "state": state}, tmp_path, ("state",))
        assert result[1].data_ptr() == state.data_ptr()
        return result

    x = torch.tensor([1, 2], dtype=torch.bfloat16)
    expected = run(x, torch.zeros_like(x))
    state = torch.zeros_like(x)
    actual = run(x, state)
    assert len(calls) == 1 and torch.equal(state, x)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))

    path = next(Path(tmp_path).rglob("*.pt"))
    path.write_bytes(b"partial")
    repaired = run(x, torch.zeros_like(x))
    assert len(calls) == 2 and torch.equal(repaired[0], expected[0])

    run(x + 1, torch.zeros_like(x))
    run(x, torch.ones_like(x))
    assert len(calls) == 4
