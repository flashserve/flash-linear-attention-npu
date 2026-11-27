#!/usr/bin/env bash
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
out=$(mkdir -p "$1" && cd "$1" && pwd)
site=$(python -c 'import sysconfig;print(sysconfig.get_path("purelib"))')
triton_lib=$(python -c 'import pathlib,triton;print(pathlib.Path(triton.__file__).resolve().parent/"backends/ascend/bishengir/lib/meta_op.aiv.c220.O0.bc")')
cmake -S "$src" -B "$out/build" -DTRITON_METAOP_LIBRARY="$triton_lib"
cmake --build "$out/build" -j2
mkdir -p "$2/manual_tiling"
cp "$out/build/libvdn_pipeline_device.so" "$2/"
cp "$src"/tiling/*.bin "$2/manual_tiling/"
