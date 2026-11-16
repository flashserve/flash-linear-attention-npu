"""Small analytical checks for rank order, state layout and in-place alias."""
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location("merge_cpu", Path(__file__).with_name("merge_fwd_bwd_kernel_cpu.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    count = 0
    for dtype in (torch.float32, torch.bfloat16):
        for forward in (False, True):
            for transpose in (False, True):
                for world, n, rank in ((1, 1, 0), (4, 1, 2), (4, 2, 2 if forward else 0)):
                    ag = torch.zeros(world, 2, 128, 256, dtype=dtype)
                    base = torch.arange(128, dtype=torch.float32).view(128, 1).expand(128, 128)
                    for source in range(world):
                        ag[source, :, :, :128] = (base + source).to(dtype)
                        ag[source, :, :, 128:] = torch.eye(128, dtype=dtype) * 2
                    sources = ([0] if world == 1 else
                               list(range(rank - n, rank)) if forward else
                               list(range(rank + n, rank, -1)))
                    expected = ag[sources[0], ..., :128].to(torch.float64)
                    for source in sources[1:]:
                        expected = 2 * expected + ag[source, ..., :128].to(torch.float64)
                    if transpose:
                        expected = expected.transpose(-1, -2)
                    h = torch.full((2, 128, 128), float("nan"), dtype=dtype)
                    result = module.merge_fwd_bwd_kernel_cpu(h, ag, n, rank, forward, transpose)
                    assert result.data_ptr() == h.data_ptr()
                    assert result.shape == (2, 128, 128) and result.dtype == dtype
                    torch.testing.assert_close(result, expected.to(dtype), rtol=0, atol=0)
                    count += 1
    print(f"merge: {count} analytical checks passed")


if __name__ == "__main__":
    main()
