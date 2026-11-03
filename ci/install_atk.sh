#!/usr/bin/env bash
# Image-build only: install Ascend ATK from an explicitly configured, pinned wheel.
set -euo pipefail

if [[ -z "${ATK_WHL_URL:-}" ]]; then
    if [[ -n "${ATK_WHL_SHA256:-}" ]]; then
        echo "[CI][ATK][ERROR] ATK_WHL_SHA256 requires ATK_WHL_URL." >&2
        exit 2
    fi
    echo "[CI][ATK] ATK_WHL_URL is unset: this image has no ATK. Before enabling ATK CI, rebuild with a controlled wheel URL and SHA256 for this architecture."
    exit 0
fi
if [[ "$ATK_WHL_URL" != https://* || "$ATK_WHL_URL" == *$'\n'* || "$ATK_WHL_URL" == *$'\r'* ]]; then
    echo "[CI][ATK][ERROR] ATK_WHL_URL must be a controlled HTTPS wheel URL." >&2
    exit 2
fi
if [[ ! "${ATK_WHL_SHA256:-}" =~ ^[a-fA-F0-9]{64}$ ]]; then
    echo "[CI][ATK][ERROR] A 64-character ATK_WHL_SHA256 is required for the configured wheel." >&2
    exit 2
fi

wheel_name="$(python3 - "$ATK_WHL_URL" <<'PY'
import re
import sys
from urllib.parse import unquote, urlsplit

url = urlsplit(sys.argv[1])
name = unquote(url.path.rsplit("/", 1)[-1])
if not re.fullmatch(r"[Aa][Tt][Kk]-[A-Za-z0-9_.+!-]+\.whl", name):
    raise SystemExit("[CI][ATK][ERROR] URL must name an Ascend ATK wheel (atk-<version>-<tags>.whl).")
print(name)
PY
)"
download_dir="$(mktemp -d)"
trap 'rm -rf -- "$download_dir"' EXIT
wheel_path="${download_dir}/${wheel_name}"
curl -fL --retry 3 --connect-timeout 15 --max-time 900 -o "$wheel_path" "$ATK_WHL_URL"
printf '%s  %s\n' "$ATK_WHL_SHA256" "$wheel_path" | sha256sum -c -

python3 - "$wheel_path" "${REQUIRED_ATK_VERSION:-26.8.8}" <<'PY'
import email
import sys
import zipfile
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version
from pathlib import Path

path, minimum = Path(sys.argv[1]), Version(sys.argv[2])
name, version, _, tags = parse_wheel_filename(path.name)
if name.lower() != "atk" or version < minimum:
    raise SystemExit(f"[CI][ATK][ERROR] Expected ATK >= {minimum}; wheel declares {name} {version}.")
if not tags.intersection(sys_tags()):
    raise SystemExit(f"[CI][ATK][ERROR] Wheel is incompatible with this Python/CPU architecture: {path.name}.")
with zipfile.ZipFile(path) as archive:
    metadata_paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    if len(metadata_paths) != 1:
        raise SystemExit("[CI][ATK][ERROR] Wheel must have exactly one package METADATA.")
    metadata = email.message_from_bytes(archive.read(metadata_paths[0]))
    if metadata.get("Name", "").lower() != "atk" or Version(metadata.get("Version", "0")) != version:
        raise SystemExit("[CI][ATK][ERROR] Wheel filename and ATK package METADATA disagree.")
    if "Ascend" not in metadata.get("Summary", ""):
        raise SystemExit("[CI][ATK][ERROR] Wheel metadata does not identify the Ascend API Toolkit.")
print(f"[CI][ATK] Verified Ascend ATK {version} for this Python/CPU architecture.")
PY

python3 -m pip install --no-cache-dir "$wheel_path"
python3 - "${REQUIRED_ATK_VERSION:-26.8.8}" <<'PY'
import sys
from importlib.metadata import distribution
from packaging.version import Version

installed = distribution("atk")
if Version(installed.version) < Version(sys.argv[1]):
    raise SystemExit("[CI][ATK][ERROR] Installed ATK is below REQUIRED_ATK_VERSION.")
if not any(entry.group == "console_scripts" and entry.name == "atk" for entry in installed.entry_points):
    raise SystemExit("[CI][ATK][ERROR] Installed ATK does not expose its required atk command.")
print(f"[CI][ATK] Installed ATK {installed.version}; CANN/torch_npu are checked by the CI runtime.")
PY
