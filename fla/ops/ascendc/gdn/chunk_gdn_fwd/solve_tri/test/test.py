"""
test.py - Test SolveTri operator on NPU.

Computes (I + A)^{-1} where A is strictly lower triangular.
Compares NPU output against CPU golden (numpy inverse).
"""
import torch
import torch_npu
import numpy as np
import fla_npu

torch.npu.utils.set_device(0)


def solve_tril_golden(A_tensor):
    """CPU golden: compute (I + A)^{-1} for each chunk."""
    A = A_tensor.float().numpy()
    B, H, T, BT = A.shape
    num_chunks = T // BT
    result = np.zeros_like(A)

    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                row_start = c * BT
                row_end = row_start + BT
                block = A[b, h, row_start:row_end, :BT]
                eye = np.eye(BT, dtype=np.float32)
                M = eye + block
                M_inv = np.linalg.inv(M)
                result[b, h, row_start:row_end, :BT] = M_inv

    return torch.from_numpy(result).half()


def generate_lower_tri_input(B, H, T, BT, dtype=torch.float16):
    """Generate random strictly lower triangular input."""
    A = torch.randn(B, H, T, BT, dtype=dtype) * 0.1
    num_chunks = T // BT
    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                row_start = c * BT
                for i in range(BT):
                    for j in range(i, BT):
                        A[b, h, row_start + i, j] = 0.0
    return A


def verify_inverse(A, result, atol=1e-3):
    """Verify (I+A) * result ≈ I."""
    A_f = A.float().numpy()
    R_f = result.float().numpy()
    B, H, T, BT = A_f.shape
    num_chunks = T // BT
    max_err = 0.0

    for b in range(B):
        for h in range(H):
            for c in range(num_chunks):
                s = c * BT
                block = A_f[b, h, s:s+BT, :BT]
                inv_block = R_f[b, h, s:s+BT, :BT]
                eye = np.eye(BT, dtype=np.float32)
                product = (eye + block) @ inv_block
                err = np.abs(product - eye).max()
                max_err = max(max_err, err)

    return max_err < atol, max_err


def test_solve_tri(B, H, T, BT, dtype=torch.float16):
    """Run one test case."""
    print(f"  Test: B={B}, H={H}, T={T}, BT={BT}, dtype={dtype}")

    A = generate_lower_tri_input(B, H, T, BT, dtype)
    golden = solve_tril_golden(A)

    # Call NPU operator
    A_npu = A.npu()
    out_npu = torch.ops.npu.npu_solve_tri(A_npu, layout="bhtd")
    out_cpu = out_npu.cpu()

    # Compare with golden
    diff = (out_cpu.float() - golden.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Verify inverse property
    passed, verify_err = verify_inverse(A, out_cpu)
    status = "PASS" if passed else "FAIL"
    print(f"    [{status}] max_diff={max_diff:.6f}, mean_diff={mean_diff:.8f}, verify_err={verify_err:.6f}")
    return passed


def main():
    print("=" * 60)
    print("SolveTri NPU Test")
    print("=" * 60)

    test_cases = [
        # B, H, T, BT
        (1, 1, 16, 16),
        (2, 4, 32, 16),
        (1, 2, 32, 32),
        (2, 4, 64, 32),
        (1, 2, 64, 64),
        (2, 4, 128, 64),
        (1, 2, 128, 128),
    ]

    results = []
    for B, H, T, BT in test_cases:
        passed = test_solve_tri(B, H, T, BT)
        results.append(passed)

    print("\n" + "=" * 60)
    total = len(results)
    passed = sum(results)
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("All tests PASSED!")
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
