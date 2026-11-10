#!/bin/bash
# Install experimental OPP only into an explicitly named, isolated empty root.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# Avoid executing an installed stable wheel's .pth hook during installation.
exec python3 -S "${script_dir}/install_experimental.py" "$@"
