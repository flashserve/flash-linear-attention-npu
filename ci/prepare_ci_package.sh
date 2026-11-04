#!/usr/bin/env bash
# Build/install the package selected by the PR change plan. The caller sources
# CI_PACKAGE_ENV_FILE, then FLA_NPU_ENV before starting each test process.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
exec python3 ci/prepare_ci_package.py
