#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

third_party_dir="${CI_THIRD_PARTY_DIR:-$repo_dir/third_party}"
seed_dir="${CI_THIRD_PARTY_SEED_DIR:-/opt/fla-ci/third_party}"
meta_dir="$third_party_dir/.ci-cache"
signature_file="$meta_dir/build-inputs.sha256"
signature_detail_file="$meta_dir/build-inputs.files"

mkdir -p "$third_party_dir" "$meta_dir"

hash_file_list() {
    local list_file="$1"
    local path

    : >"$list_file"
    printf '%s\n' "ci-cache-signature-v1" >>"$list_file"
    printf 'cann=%s\n' "${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-unknown}}" >>"$list_file"
    if [[ -f /usr/local/Ascend/ascend-toolkit/latest/version.info ]]; then
        sha256sum /usr/local/Ascend/ascend-toolkit/latest/version.info >>"$list_file"
    fi

    local roots=(
        build.sh
        CMakeLists.txt
        cmake
        ci
        scripts/package
        scripts/util
        scripts/ci
        requirements.txt
        tests/requirements.txt
        torch_extension
        torch_custom
    )

    {
        for path in "${roots[@]}"; do
            if [[ -f "$path" ]]; then
                printf '%s\0' "$path"
            elif [[ -d "$path" ]]; then
                find "$path" -type f -print0
            fi
        done
    } | LC_ALL=C sort -z | while IFS= read -r -d '' path; do
        sha256sum "$path"
    done >>"$list_file"
}

seed_third_party() {
    if [[ "${CI_SEED_THIRD_PARTY:-true}" != "true" || ! -d "$seed_dir" ]]; then
        return
    fi

    echo "[CI] Seeding third_party cache from $seed_dir"
    cp -an "$seed_dir"/. "$third_party_dir"/
}

remove_known_partial_cache() {
    if [[ -d "$third_party_dir/catlass" && ! -f "$third_party_dir/catlass/include/catlass/catlass.hpp" ]]; then
        echo "[CI][WARN] Removing incomplete catlass cache."
        rm -rf "$third_party_dir/catlass"
    fi
    if [[ -d "$third_party_dir/opbase" && ! -d "$third_party_dir/opbase/.git" ]]; then
        echo "[CI][WARN] Removing suspicious opbase cache."
        rm -rf "$third_party_dir/opbase"
    fi
}

tmp_detail="$(mktemp)"
trap 'rm -f "$tmp_detail"' EXIT
hash_file_list "$tmp_detail"
current_signature="$(sha256sum "$tmp_detail" | awk '{print $1}')"
previous_signature=""
if [[ -f "$signature_file" ]]; then
    previous_signature="$(cat "$signature_file")"
fi

if [[ "${CI_FORCE_CLEAN_CACHE:-false}" == "true" || "$current_signature" != "$previous_signature" ]]; then
    echo "[CI] Build inputs changed; cleaning build outputs and third_party cache."
    rm -rf build build_out output
    find "$third_party_dir" -mindepth 1 -maxdepth 1 ! -name .ci-cache -exec rm -rf {} +
    mkdir -p "$meta_dir"
    cp "$tmp_detail" "$signature_detail_file"
    printf '%s\n' "$current_signature" >"$signature_file"
else
    echo "[CI] Build inputs unchanged; keeping third_party cache."
fi

seed_third_party
remove_known_partial_cache
