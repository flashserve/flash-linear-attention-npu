"""Manual two-wheel NPU smoke test with recompute_w_u_fwd and RWKV8."""
import argparse
import importlib
from pathlib import Path


def check_formal(torch, device):
    from fla_npu.ops.ascendc import npu_recompute_w_u_fwd

    # Identity A, unit beta and zero gate give W=K and U=V exactly.
    rng = torch.Generator().manual_seed(43)
    k, v = [torch.randn(1, 1, 64, 128, generator=rng).bfloat16() for _ in range(2)]
    beta = torch.ones(1, 1, 64, device=device)
    matrix = torch.eye(64).bfloat16().view(1, 1, 64, 64).to(device)
    w, u = npu_recompute_w_u_fwd(k.to(device), v.to(device), beta, matrix, 64,
                                  g=torch.zeros_like(beta))
    torch.testing.assert_close(w.cpu(), k)
    torch.testing.assert_close(u.cpu(), v)


def check_experimental(torch, device):
    from fla_npu_experimental import fused_recurrent_rwkv8
    from accuracy import load_cases
    import rwkv8_reference as reference

    for case in load_cases("quick"):
        cpu = reference._rwkv8_inputs(case["spec"], torch.device("cpu"), high_precision=False)
        golden = reference.fused_recurrent_rwkv8_reference(**{
            k: v.double() if isinstance(v, torch.Tensor) else v for k, v in cpu.items()
        })
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cpu.items()}
        outputs = fused_recurrent_rwkv8(**inputs)
        results = reference.check_outputs(outputs, golden, inputs)
        if not all(item["pass"] for item in results.values()):
            raise AssertionError((case["name"], results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, choices=("formal", "experimental"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--conflict", help="Expected diagnostic, using prepared conflicting packages")
    parser.add_argument("--npu-first", action="store_true", help="Run a PyTorch NPU operation before package imports")
    args = parser.parse_args()
    order = ["fla_npu", "fla_npu_experimental"]
    if args.first == "experimental":
        order.reverse()
    if args.npu_first:
        import torch
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
        value = torch.ones(4, device=f"npu:{args.device}")
        (value + value).cpu()

    winner = "fla_npu" if args.npu_first else order[0]
    enabled = []
    for name in order:
        if args.conflict and name != winner:
            try:
                importlib.import_module(name)
            except ImportError as exc:
                if args.conflict not in str(exc):
                    raise
            else:
                raise AssertionError(f"{name} unexpectedly accepted the conflict")
        else:
            importlib.import_module(name)
            enabled.append(name)

    import torch
    import torch_npu  # noqa: F401
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    checks = {"fla_npu": check_formal, "fla_npu_experimental": check_experimental}
    with torch.inference_mode():
        for name in enabled:
            checks[name](torch, device)
            print(f"{name}: accuracy PASS", flush=True)
    libraries = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                 if line.endswith("/libcust_opapi.so")}
    expected = set()
    if "fla_npu" in enabled:
        import fla_npu
        expected.update(str(v / "op_api/lib/libcust_opapi.so") for v in fla_npu._resolve_vendor_dirs())
    if "fla_npu_experimental" in enabled:
        from fla_npu_experimental import _loader
        expected.update(_loader._LIBRARIES)
    if libraries != expected:
        raise AssertionError(f"Mapped libraries {libraries} differ from selection {expected}")
    print("Mapped op_api libraries:", *sorted(libraries), sep="\n")


if __name__ == "__main__":
    main()
