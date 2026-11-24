"""Check the fixed legacy public return contract and packed metadata."""
import importlib.util
from pathlib import Path
import torch

path = Path(__file__).resolve().parents[1] / "executor_chunk_kda_bwd.py"
spec = importlib.util.spec_from_file_location("bwd_executor", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    count = 0
    for packed in (False, True):
        for with_bias in (False, True):
            for tokens in (1, 63, 64, 65, 129):
                for heads in (1, 3):
                    for beta_dtype in (torch.bfloat16, torch.float32):
                        prefix = (heads, tokens) if packed else (2, heads, tokens)
                        q = torch.empty((*prefix, 128), dtype=torch.bfloat16, device="meta")
                        values = module.build_inputs(dict(q=q, k=q, v=q,
                            beta=torch.empty(prefix, dtype=beta_dtype, device="meta"),
                            dt_bias=torch.empty((heads, 128), device="meta"),
                            chunk_size=64, is_varlen=packed, with_bias=with_bias))
                        out = module.run_cpu(values)
                        expected = ((q.shape, torch.float32), (q.shape, torch.float32),
                            (q.shape, torch.bfloat16), (prefix, torch.float32),
                            (q.shape, torch.float32), None, ((heads,), torch.float32),
                            ((heads, 128), torch.float32) if with_bias else None)
                        assert len(out) == 8
                        for value, contract in zip(out, expected):
                            if contract is None:
                                assert value is None
                            else:
                                assert tuple(value.shape) == tuple(contract[0]) and value.dtype == contract[1]
                                assert not torch.count_nonzero(value)
                        if packed:
                            cu, indices = values["cu_seqlens"], values["chunk_indices"]
                            assert cu == [0, tokens // 2, tokens]
                            canonical = []
                            for seq in range(2):
                                for chunk in range((cu[seq + 1] - cu[seq] + 63) // 64):
                                    canonical.extend((seq, chunk))
                            assert indices == canonical
                        else:
                            assert values["cu_seqlens"] is None and values["chunk_indices"] is None
                        count += 1
    print(f"legacy backward: {count} shape/dtype/metadata checks passed (zero placeholder)")


if __name__ == "__main__":
    main()
