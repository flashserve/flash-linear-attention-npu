#!/usr/bin/env python3
"""Publish all configured NPU platforms for the architectures in a manifest.

An ARM manifest needs every platform's ARM wheel; an x86 manifest needs every
platform's x86 wheel. Builds on separate machines can publish independently.
Daily builds go only to the OBS nightly prefixes. Release builds first sync OBS
and then create/update one GitHub Release. Re-running is safe: OBS uses a size
based sync and GitHub replaces same-name assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


from npu_platforms import ARCH_TO_WHEEL, SOC_TO_PACKAGE

DEFAULT_BUCKET = "ascend-artifcat-packages"
DEFAULT_ENDPOINT = "https://obs.cn-north-4.myhuaweicloud.com"
DEFAULT_GH_REPO = "flashserve/flash-linear-attention-npu"
PREFIX_ROOT = "pypi/packages/ascend/repos/pypi"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_branch(value: str) -> str:
    value = value.removeprefix("heads-").strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unknown"


def publish_destinations(build_type: str) -> list[str]:
    """Return the only external destinations allowed for this build type."""
    if build_type == "daily":
        return ["obs-nightly"]
    if build_type == "release":
        return ["obs-release", "github-release"]
    raise ValueError(f"不支持的包类型: {build_type}")


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"缺少环境变量 {name}")
    return value


def validate_build(manifest_path: Path) -> tuple[dict, list[dict]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取 manifest {manifest_path}: {exc}") from exc

    out_dir = manifest_path.parent
    records: dict[tuple[str, str], dict] = {}
    errors: list[str] = []
    arches = manifest.get("arches") or []
    if (not isinstance(arches, list) or not arches
            or any(arch not in ARCH_TO_WHEEL for arch in arches)
            or len(set(arches)) != len(arches)):
        raise RuntimeError("发布检查失败: manifest.arches 必须明确指定 arm 或 x86")
    expected = {(soc, arch) for soc in SOC_TO_PACKAGE for arch in arches}
    if manifest.get("interrupted"):
        errors.append("构建被中断")
    for record in manifest.get("results") or []:
        key = (record.get("soc", ""), record.get("arch", ""))
        if key in records:
            errors.append(f"重复组合: {key[0]}/{key[1]}")
            continue
        records[key] = record

    missing = sorted(expected - set(records))
    extra = sorted(set(records) - expected)
    if missing:
        errors.append("缺少组合: " + ", ".join(f"{s}/{a}" for s, a in missing))
    if extra:
        errors.append("意外组合: " + ", ".join(f"{s}/{a}" for s, a in extra))

    wheels: list[dict] = []
    for soc, arch in sorted(expected):
        record = records.get((soc, arch))
        if record is None:
            continue
        if record.get("status") != "success":
            errors.append(f"{soc}/{arch} 未成功: {record.get('error', '')}")
            continue
        names = record.get("wheels") or []
        if len(names) != 1:
            errors.append(f"{soc}/{arch} 应有且仅有一个 wheel，实际 {len(names)}")
            continue
        name = str(names[0])
        if Path(name).name != name or "/" in name or "\\" in name:
            errors.append(f"wheel 文件名必须是单个文件名: {name}")
            continue
        path = out_dir / "wheels" / f"{soc}-{arch}" / name
        package = SOC_TO_PACKAGE[soc]
        platform = ARCH_TO_WHEEL[arch]
        if (not name.startswith(package + "-")
                or not name.endswith(f"_{platform}.whl")):
            errors.append(f"{soc}/{arch} wheel 名称与档位/架构不匹配: {name}")
            continue
        if not path.is_file():
            errors.append(f"wheel 文件不存在: {path}")
            continue
        wheels.append({
            "soc": soc,
            "arch": arch,
            "package": package,
            "name": name,
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        })

    if errors:
        raise RuntimeError("全 NPU 平台发布检查失败:\n- " + "\n- ".join(errors))
    if len(wheels) != len(expected):
        raise RuntimeError(f"全 NPU 平台发布检查失败: 期望 {len(expected)} 个 wheel，实际 {len(wheels)}")
    return manifest, wheels


def validate_identity(args: argparse.Namespace, manifest: dict,
                      wheels: list[dict]) -> None:
    """Prevent publish-only retries from relabelling daily wheels as releases."""
    branch = manifest.get("branch") or manifest.get("ref", "")
    if safe_branch(branch) != safe_branch(args.branch):
        raise RuntimeError(f"manifest 分支 {branch} 与发布分支 {args.branch} 不一致")
    if manifest.get("build_type") != args.build_type:
        raise RuntimeError("manifest.build_type 与发布包类型不一致")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", manifest.get("head_sha", "")):
        raise RuntimeError("manifest 缺少完整源码 commit SHA")
    versions = {item["name"].split("-")[1] for item in wheels}
    if len(versions) != 1:
        raise RuntimeError("各 NPU 平台 wheel 版本不一致")
    if args.build_type == "release":
        version = args.release_version.removeprefix("v")
        if versions != {version} or manifest.get("release_version") != version:
            raise RuntimeError(f"正式 wheel 版本与 --release-version={version} 不一致")
    else:
        from daily_version import version_for, verify_wheel_version
        expected = version_for(args.branch, manifest.get("daily_date", ""), manifest["head_sha"])
        if versions != {expected} or manifest.get("package_version") != expected:
            raise RuntimeError(f"daily 版本与日期/分支/commit 不一致: expected={expected}")
        for item in wheels:
            verify_wheel_version(Path(item["path"]), expected)


def stage_by_package(out_dir: Path, wheels: list[dict]) -> dict[str, Path]:
    root = out_dir / "publish-staging"
    if root.exists():
        shutil.rmtree(root)
    staged: dict[str, Path] = {}
    for item in wheels:
        package = item["package"]
        target_dir = root / package
        target_dir.mkdir(parents=True, exist_ok=True)
        source = Path(item["path"])
        target = target_dir / item["name"]
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copy2(source, target)
        staged[package] = target_dir
    return staged


def mirror_project_name(package: str) -> str:
    """Return the PEP 503-style project directory used by the mirror."""
    return re.sub(r"[-_.]+", "-", package).lower()


def obs_prefix(build_type: str, package: str) -> str:
    project = mirror_project_name(package)
    if build_type == "daily":
        return f"{PREFIX_ROOT}/nightly/{project}"
    return f"{PREFIX_ROOT}/{project}"


def publish_obs(args: argparse.Namespace, staged: dict[str, Path],
                report: dict) -> None:
    if not args.dry_run:
        require_env("OBS_ACCESS_KEY")
        require_env("OBS_SECRET_KEY")

    obs_results: list[dict] = []
    targets = [(package, latest) for package in sorted(staged)
               for latest in ((False, True) if args.build_type == "daily" else (False,))]
    for package, latest in targets:
        prefix = obs_prefix(args.build_type, package) + ("/latest" if latest else "")
        suffix = "-latest" if latest else ""
        obs_report = args.manifest.parent / f"obs-{args.build_type}-{package}{suffix}.json"
        command = [
            sys.executable, str(args.obs_script),
            "--src", str(staged[package]),
            "--bucket", args.obs_bucket,
            "--endpoint", args.obs_endpoint,
            "--prefix", prefix,
            "--patterns", "*.whl",
        ]
        if args.dry_run:
            command.append("--local-only")
        else:
            command += ["--sync", "--report", str(obs_report)]
            if args.build_type == "daily" and os.getenv("FLA_PKG_OBS_FORCE") == "1":
                command.append("--force")
        print("[publish] OBS:", " ".join(command[:-1] + [command[-1]]), flush=True)
        completed = subprocess.run(command, check=False)
        obs_results.append({"package": package, "prefix": prefix,
                            "exit_code": completed.returncode,
                            "report": str(obs_report) if not args.dry_run else ""})
        if completed.returncode != 0:
            report["obs"] = obs_results
            raise RuntimeError(f"OBS 发布失败: {package}，exit={completed.returncode}")
    report["obs"] = obs_results


def release_body(args: argparse.Namespace, manifest: dict,
                 wheels: list[dict]) -> str:
    lines = [
        "flash-linear-attention-npu 正式发布", "",
        f"- 分支: `{args.branch}`",
        f"- 版本: `{args.release_version}`",
        f"- commit: `{manifest.get('head_sha', '')[:12]}`",
        "",
    ]
    for arch in sorted({item["arch"] for item in wheels}):
        lines += [f"<!-- fla-npu-arch:{arch} -->", f"### {arch}",
                  f"commit: `{manifest.get('head_sha', '')[:12]}`", "",
                  "| SOC | 架构 | wheel | SHA256 |",
                  "| --- | --- | --- | --- |"]
        for item in wheels:
            if item["arch"] == arch:
                lines.append(
                    f"| {item['soc']} | {item['arch']} | `{item['name']}` | "
                    f"`{item['sha256']}` |")
        lines += [f"<!-- /fla-npu-arch:{arch} -->", ""]
    return "\n".join(lines)


def merge_release_body(existing: str, incoming: str) -> str:
    """Each builder updates its architecture section and retains the other one."""
    if not existing.strip():
        return incoming
    body = existing
    for arch in ARCH_TO_WHEEL:
        pattern = (rf"<!-- fla-npu-arch:{arch} -->.*?"
                   rf"<!-- /fla-npu-arch:{arch} -->")
        section = re.search(pattern, incoming, flags=re.DOTALL)
        if section:
            if re.search(pattern, body, flags=re.DOTALL):
                body = re.sub(pattern, lambda _: section.group(0), body, flags=re.DOTALL)
            else:
                body = body.rstrip() + "\n\n" + section.group(0) + "\n"
    return body


def promote_complete_release(client, release: dict, version: str) -> bool:
    """Promote only a complete newer formal version, across both host arches."""
    candidate = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version.removeprefix("v"))
    if candidate is None:
        return False
    expected = {f"{package}-{version.removeprefix('v')}-py3-none-manylinux_2_34_{machine}.whl"
                for package in SOC_TO_PACKAGE.values() for machine in ARCH_TO_WHEEL.values()}
    uploaded = {asset["name"] for asset in release.get("assets", [])
                if asset.get("state") == "uploaded"}
    if not expected <= uploaded:
        return False
    current = client.find_latest_release()
    if current:
        if current["id"] == release["id"]:
            return False
        old = re.fullmatch(r"(?:release-)?v?(\d+)\.(\d+)\.(\d+)(?:-release)?", current["tag_name"])
        if old is None or tuple(map(int, old.groups())) >= tuple(map(int, candidate.groups())):
            return False
    client.mark_latest(release)
    return True


def publish_github(args: argparse.Namespace, manifest: dict,
                   wheels: list[dict], report: dict) -> None:
    tag = args.gh_tag or f"{safe_branch(args.branch)}-release"
    title = args.gh_title or f"flash-linear-attention-npu {args.release_version}"
    if args.dry_run:
        print(f"[publish] GitHub dry-run: repo={args.gh_repo} tag={tag} "
              f"assets={len(wheels)}")
        report["github"] = {"tag": tag, "assets": [w["name"] for w in wheels],
                            "dry_run": True}
        return

    token = require_env("GITHUB_TOKEN")
    from make_packages import GitHubRelease

    client = GitHubRelease(args.gh_repo, token, args.gh_api, prerelease=False)
    custom_body = getattr(args, "gh_body_file", None)
    incoming_body = (custom_body.read_text(encoding="utf-8") if custom_body
                     else release_body(args, manifest, wheels))
    if not incoming_body.strip():
        raise RuntimeError("GitHub Release 正文不能为空")
    release = client.find_release(tag)
    if release is None:
        release = client.ensure_release(tag, title, manifest["head_sha"], incoming_body)
    uploaded: list[str] = []
    for item in wheels:
        path = Path(item["path"])
        client.upload_asset(release, item["name"], path)
        uploaded.append(item["name"])
        print(f"[publish] GitHub asset: {item['name']}", flush=True)
    latest = client.find_release(tag) or release
    body = (incoming_body if custom_body else
            merge_release_body(latest.get("body") or "", incoming_body))
    client.update_body(latest, body, title)
    promoted = promote_complete_release(client, latest, args.release_version)
    if promoted:
        print(f"[publish] GitHub Latest: {tag}", flush=True)
    report["github"] = {
        "tag": tag,
        "url": release.get("html_url", ""),
        "assets": uploaded,
        "dry_run": False,
        "promoted_latest": promoted,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 manifest 发布所选架构下的全部 NPU 平台包")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--build-type", choices=("daily", "release"), required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--release-version", default="")
    parser.add_argument("--obs-script", type=Path,
                        default=Path(__file__).with_name("obs_publish.py"))
    parser.add_argument("--obs-bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--obs-endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--gh-repo", default=DEFAULT_GH_REPO)
    parser.add_argument("--gh-api", default="https://api.github.com")
    parser.add_argument("--gh-tag", default="")
    parser.add_argument("--gh-title", default="")
    parser.add_argument("--gh-body-file", type=Path,
                        help="release 专用：使用指定 UTF-8 正文，保留已有附件")
    parser.add_argument("--dry-run", action="store_true",
                        help="只验证全 NPU 平台包并本地扫描 OBS staging，不联网、不上传")
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.obs_script = args.obs_script.resolve()
    if args.build_type == "release" and not args.release_version:
        parser.error("release 必须指定 --release-version")
    if args.build_type == "daily" and args.release_version:
        parser.error("daily 不能指定 --release-version")
    if args.build_type == "daily" and (args.gh_tag or args.gh_title or args.gh_body_file):
        parser.error("daily 禁止指定 GitHub Release 参数；每日包只能发布到 OBS nightly")
    if args.gh_body_file and (not args.gh_body_file.is_file()
                             or not args.gh_body_file.read_text(encoding="utf-8").strip()):
        parser.error("--gh-body-file 必须为非空 UTF-8 文件")
    if not args.obs_script.is_file():
        parser.error(f"OBS 发布脚本不存在: {args.obs_script}")
    return args


def main() -> int:
    args = parse_args()
    try:
        manifest, wheels = validate_build(args.manifest)
        validate_identity(args, manifest, wheels)
    except Exception as exc:  # noqa: BLE001 - user-facing publish gate
        print(f"[publish][ERROR] {exc}", file=sys.stderr)
        return 1
    staged = stage_by_package(args.manifest.parent, wheels)
    report = {
        "schema": "fla-npu-publish-report-v1",
        "build_type": args.build_type,
        "branch": args.branch,
        "release_version": args.release_version,
        "head_sha": manifest.get("head_sha", ""),
        "daily_date": manifest.get("daily_date", ""),
        "package_version": manifest.get("package_version", ""),
        "wheels": wheels,
        "dry_run": args.dry_run,
        "destinations": publish_destinations(args.build_type),
    }
    report_path = args.manifest.parent / "publish-manifest.json"
    try:
        if args.build_type == "daily":
            publish_obs(args, staged, report)
        else:
            publish_obs(args, staged, report)
            publish_github(args, manifest, wheels, report)
        report["status"] = "success"
        return 0
    except Exception as exc:  # noqa: BLE001 - persist partial publisher state
        report["status"] = "failed"
        report["error"] = str(exc)
        print(f"[publish][ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"[publish] report: {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
