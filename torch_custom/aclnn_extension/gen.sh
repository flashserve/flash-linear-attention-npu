#!/bin/bash

# the first parameter is the yaml file
YAML_FILE="$1"
DERIVATIVES_YAML_FILE="npu_custom_derivatives.yaml"

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd $CDIR

# check if the file exists
if [ ! -f "$YAML_FILE" ]; then
    echo "Error: yaml file $YAML_FILE does not exit"
    exit 1
fi

# get the torch version
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")

IFS='.' read -ra version_parts <<< "$PYTORCH_VERSION"
PYTORCH_VERSION_DIR="v${version_parts[0]}r${version_parts[1]}"

export PYTORCH_VERSION="$PYTORCH_VERSION"
export PYTORCH_CUSTOM_DERIVATIVES_PATH="${CDIR}/op_plugin/config/${PYTORCH_VERSION_DIR}/derivatives.yaml"
export ACLNN_EXTENSION_PATH="${CDIR}"
export ACLNN_EXTENSION_SWITCH="TRUE"

ATRN_DIR="$CDIR/torch_npu/csrc/aten" 
if [ ! -d "${ATRN_DIR}" ]; then
    mkdir -p "${ATRN_DIR}"
fi

UTILS_DIR="$CDIR/torch_npu/utils" 
if [ ! -d "${UTILS_DIR}" ]; then
    mkdir -p "${UTILS_DIR}"
fi
#################### op-plugin torchnpugen ####################

OUTPUT_DIR="$CDIR/op_plugin/config/$PYTORCH_VERSION_DIR"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

OPAPI_OUTPUT_DIR="$CDIR/op_plugin/ops/opapi"
if [ ! -d "$OPAPI_OUTPUT_DIR" ]; then
    mkdir -p "$OPAPI_OUTPUT_DIR"
fi

python3 -m torchnpugen.gen_op_plugin_functions \
  --version="$PYTORCH_VERSION" \
  --output_dir="$OUTPUT_DIR/" \
  --source_yaml="$CDIR/$YAML_FILE"

# check if the second parameter is passed
if [ -n "$DERIVATIVES_YAML_FILE" ]; then
    python3 -m torchnpugen.gen_derivatives \
      --version="$PYTORCH_VERSION" \
      --output_dir="$OUTPUT_DIR/" \
      --source_yaml="$CDIR/$DERIVATIVES_YAML_FILE"
fi

python3 -m torchnpugen.gen_op_backend  \
  --version="$PYTORCH_VERSION" \
  --output_dir="$CDIR/op_plugin/" \
  --source_yaml="$OUTPUT_DIR/op_plugin_functions.yaml" \
  --deprecate_yaml="$CDIR/deprecated.yaml"

python3 -m torchnpugen.struct.gen_struct_opapi \
  --output_dir="$OPAPI_OUTPUT_DIR/" \
  --native_yaml="$OUTPUT_DIR/op_plugin_functions.yaml" \
  --struct_yaml="$CDIR/$YAML_FILE"


#################### torch_npu torchnpugen ####################

python3 -m torchnpugen.gen_backend_stubs  \
  --output_dir="$CDIR/torch_npu/csrc/aten" \
  --source_yaml="./test_native_functions.yaml" \
  --impl_path="$CDIR/torch_npu/csrc/aten" \
  --op_plugin_impl_path="$CDIR/op_plugin/ops" \
  --op_plugin_yaml_path="$CDIR/op_plugin/config/$PYTORCH_VERSION_DIR/op_plugin_functions.yaml"
