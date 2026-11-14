#!/usr/bin/env python3
"""Shared, byte-identical FP16 fixtures and independent CPU verification."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SIDE = 16
CASES = ("zero", "identity", "pattern")


def make_inputs(case):
    grid = np.arange(SIDE * SIDE, dtype=np.int32).reshape(SIDE, SIDE)
    a = (((grid * 7 + 3) % 17 - 8) / 8).astype("<f2")
    b = (((grid * 5 + 1) % 13 - 6) / 8).astype("<f2")
    if case == "zero":
        a.fill(0)
    elif case == "identity":
        a = np.eye(SIDE, dtype="<f2")
    elif case != "pattern":
        raise ValueError(f"unknown case: {case}")
    return a, b


def generate(root):
    for case in CASES:
        directory = root / case
        directory.mkdir(parents=True, exist_ok=True)
        a, b = make_inputs(case)
        a.tofile(directory / "a.bin")
        b.tofile(directory / "b.bin")
        # All inputs are multiples of 1/8; these 16-term dot products are
        # exactly representable in FP32. Exact equality is intentional.
        reference = (a.astype(np.float64) @ b.astype(np.float64)).astype("<f4")
        np.save(directory / "reference.npy", reference)
        if case == "zero":
            np.testing.assert_array_equal(reference, np.zeros_like(reference))
        elif case == "identity":
            np.testing.assert_array_equal(reference, b.astype(np.float32))
        print(f"generated {directory}")


def read_inputs(directory):
    values = []
    for name in ("a", "b"):
        path = directory / f"{name}.bin"
        if path.stat().st_size != SIDE * SIDE * 2:
            raise ValueError(f"{path}: expected 512 bytes")
        values.append(np.fromfile(path, dtype="<f2").reshape(SIDE, SIDE))
    return values


def check(root, backend):
    records = []
    for case in CASES:
        directory = root / case
        a, b = read_inputs(directory)
        expected = (a.astype(np.float64) @ b.astype(np.float64)).astype("<f4")
        path = directory / f"{backend}_output.bin"
        if path.stat().st_size != SIDE * SIDE * 4:
            raise ValueError(f"{path}: expected 1024 bytes")
        actual = np.fromfile(path, dtype="<f4").reshape(SIDE, SIDE)
        np.testing.assert_array_equal(actual, expected)
        record = {
            "case": case, "backend": backend, "passed": True,
            "max_abs_error": float(np.max(np.abs(actual - expected))),
            "input_sha256": hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest(),
            "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        }
        records.append(record)
        print(json.dumps(record))
    (root / f"{backend}_results.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "check"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--backend", choices=("cpp", "dsl"), default="cpp")
    args = parser.parse_args()
    if args.action == "generate":
        generate(args.directory)
    else:
        check(args.directory, args.backend)


if __name__ == "__main__":
    main()
