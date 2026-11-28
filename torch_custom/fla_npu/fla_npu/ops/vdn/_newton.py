"""The FP32 Newton–Schulz inverse for VDN systems with spectral bound 405."""

import math

import torch

SPECTRAL_BOUND = 405.0
TARGET_RESIDUAL = 1e-7
_SCALE = 2.0 / (2.0 + SPECTRAL_BOUND)
ITERATIONS = math.ceil(math.log2(math.log(TARGET_RESIDUAL) / math.log1p(-_SCALE)))


def inverse(system, eye):
    # X1 = c I (2 I - M c I). Both initial products reduce to elementwise work.
    x = torch.add(eye * (2.0 * _SCALE), system, alpha=-(_SCALE * _SCALE))
    two_eye = eye * 2.0
    for _ in range(1, ITERATIONS):
        correction = torch.bmm(system, x)
        correction.neg_().add_(two_eye)
        x = torch.bmm(x, correction)
        del correction
    return x


def inverse_sub(system, eye):
    """Use one broadcast subtraction to form each FP32 Newton correction."""
    x = torch.add(eye * (2.0 * _SCALE), system, alpha=-(_SCALE * _SCALE))
    two_eye = eye * 2.0
    for _ in range(1, ITERATIONS):
        correction = torch.bmm(system, x)
        torch.sub(two_eye, correction, out=correction)
        x = torch.bmm(x, correction)
    return x
