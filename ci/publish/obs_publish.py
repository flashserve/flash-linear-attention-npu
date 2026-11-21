#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OBS upload/sync utility used by the daily publisher.

All callers must provide an explicit --prefix. --sync enables uploads;
destructive operations additionally require --yes. Daily callers target
nightly project prefixes, never GitHub Releases. See README.md for the
supported entrypoints, environment credentials and recovery commands.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BUCKET = "ascend-artifcat-packages"
DEFAULT_ENDPOINT = "https://obs.cn-north-4.myhuaweicloud.com"
DEFAULT_PATTERNS = "*.whl,*.json,*.tar.gz"

MULTIPART_THRESHOLD = 4 * 1024**3  # >4GiB 用 uploadFile（分片/断点续传）
MAX_DELETE_DEFAULT = 200           # 单次删除安全上限
PAGE_SIZE = 1000

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


# ------------------------------------------------------------------ types


@dataclass
class Item:
    path: Path
    key: str = ""
    size: int = 0

    def __post_init__(self) -> None:
        if not self.size:
            self.size = self.path.stat().st_size


@dataclass
class Result:
    op: str                      # upload / verify / delete
    key: str
    status: str = "pending"      # uploaded|skipped|deleted|verified|failed|mismatch|absent
    http_status: int = 0
    elapsed: float = 0.0
    remote_size: int = -1
    error: str = ""


@dataclass
class Session:
    mode: str
    bucket: str
    prefix: str
    endpoint: str
    src: str = ""
    started_at: str = field(default_factory=lambda: now_iso())
    finished_at: str = ""
    local_files: int = 0
    remote_files: int = 0
    results: list[Result] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["results"] = [r.__dict__ for r in self.results]
        return d


# ---------------------------------------------------------------- helpers


def human_size(n: int) -> str:
    v = float(max(0, n))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if v < 1024.0 or unit == "GiB":
            return f"{int(v)} B" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024.0
    return f"{v:.1f} TiB"


def mask(secret: str | None) -> str:
    if not secret:
        return "<unset>"
    s = str(secret)
    return "*" * min(len(s), 6) if len(s) <= 6 else f"{s[:4]}...{s[-2:]}(len={len(s)})"


def mask_secret(secret: str | None) -> str:
    """SK 只暴露长度：连首尾字符都不该出现在日志/截图里。"""
    return "<unset>" if not secret else f"*** (len={len(secret)})"


def fingerprint(secret: str | None) -> str:
    """SK 的 sha256 前 8 位：用于确认"环境变量是否已换成新值"，可安全外发。

    SK 是 40 位高熵随机串，无法从哈希反推；所以这个指纹既能比对，又不泄露密钥。
    """
    if not secret:
        return "<unset>"
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:8]


# ------------------------------------------------------- credentials (env)


def _reg_targets() -> list[tuple[str, Any, str, int]]:
    """(来源标签, hive, 子键, 访问掩码)；非 Windows 返回空列表。"""
    if sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:  # 理论上不会发生
        return []
    wow64 = getattr(winreg, "KEY_WOW64_64KEY", 0)
    return [
        ("user env (registry)", winreg.HKEY_CURRENT_USER, "Environment", winreg.KEY_READ),
        ("machine env (registry)", winreg.HKEY_LOCAL_MACHINE,
         r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
         winreg.KEY_READ | wow64),
    ]


def _query_reg_value(hive: Any, subkey: str, name: str, access: int) -> str:
    """读注册表里的环境变量；名字大小写不敏感（Windows 语义）。读不到返回 ""。"""
    import winreg

    try:
        with winreg.OpenKey(hive, subkey, 0, access) as key:
            try:
                val, _ = winreg.QueryValueEx(key, name)
                return str(val)
            except FileNotFoundError:
                pass
            # 兜底：遍历所有值做大小写不敏感匹配（同时兼容 REG_SZ / REG_EXPAND_SZ）
            i = 0
            while True:
                try:
                    vname, vdata, _ = winreg.EnumValue(key, i)
                except OSError:
                    return ""
                if vname.lower() == name.lower():
                    return str(vdata)
                i += 1
    except OSError:
        return ""


def resolve_credential(name: str) -> tuple[str, str]:
    """按 process → 用户环境变量 → 系统环境变量 的顺序查找，返回 (值, 来源)。

    只读、无副作用；不打印值本身。这样即使变量是在"系统属性"里后设的、
    当前终端比它更早启动，也能直接读到（不必重开终端）。
    """
    val = os.environ.get(name, "")
    if val.strip():
        return val.strip(), "process env"

    for label, hive, subkey, access in _reg_targets():
        raw = _query_reg_value(hive, subkey, name, access)
        if raw.strip():
            # REG_EXPAND_SZ 里可能有 %USERPROFILE% 之类，展开一次
            return os.path.expandvars(raw).strip(), label
    return "", ""


def resolve_credentials() -> tuple[str, str, str, str]:
    """返回 (ak, sk, ak_source, sk_source)。仅从环境读取，绝不接受命令行传参。"""
    ak, ak_src = resolve_credential("OBS_ACCESS_KEY")
    sk, sk_src = resolve_credential("OBS_SECRET_KEY")
    return ak, sk, ak_src, sk_src


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resp_status(resp) -> tuple[int, str]:
    """OBS SDK 多数错误返回带 status 的 resp，少数直接抛异常（由调用方兜住）。"""
    if resp is None:
        return -1, "no response object"
    status = int(getattr(resp, "status", -1))
    msg = getattr(resp, "errorMessage", None) or getattr(resp, "reason", "") or ""
    code = getattr(resp, "errorCode", None)
    if code:
        msg = f"{code}: {msg}"
    return status, str(msg)


def is_ok(status: int) -> bool:
    """只有 2xx 算成功；SDK 的 -1 表示本地异常（DNS/TLS/超时），绝不能当成功。"""
    return 0 <= status < 300


def quiet_obs_logger() -> None:
    """抑制 SDK 的 'OBS_LOGGER'（默认把完整 traceback 打到 stderr，干扰输出）。"""
    lg = logging.getLogger("OBS_LOGGER")
    lg.propagate = False
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.CRITICAL)


def build_client(endpoint: str, ak: str, sk: str, timeout_s: int = 30, verbose: bool = False):
    from obs import ObsClient  # 延迟导入：--local-only 不需要装 SDK

    if not verbose:
        quiet_obs_logger()
    cli = ObsClient(
        access_key_id=ak,
        secret_access_key=sk,
        server=endpoint,
        timeout=timeout_s,
        max_retry_count=0,  # 重试由本脚本控制，便于打印每次尝试
    )
    if not verbose:
        quiet_obs_logger()  # __init__ 可能重置 logger 配置
    return cli


def validate_prefix(prefix: str) -> str:
    """删除/写入的护栏：前缀必须有意义，不能是空、不能含通配符。"""
    p = (prefix or "").strip("/")
    if not p:
        raise ValueError("prefix is empty")
    if any(ch in p for ch in "*?[]"):
        raise ValueError(f"prefix must not contain wildcards: {prefix!r}")
    if len([s for s in p.split("/") if s]) < 3:
        raise ValueError(f"prefix too short, refuse to operate on it: {prefix!r} (need >=3 segments)")
    return p


def key_scopes(items: list[Item]) -> list[str]:
    """从本地文件名推导 <project>-<version> 前缀，用于限定 prune 的范围。

    wheel/变体文件名形如 flash_linear_attention_npu_a2-0.dev20261118+main.abcdef0-cp312-...whl，
    前两段就是 name 和 version（PEP 440 里 version 不含 '-'），所以取 tokens[1]。
    """
    scopes: set[str] = set()
    for it in items:
        parts = it.key.split("-")
        if len(parts) >= 2 and parts[1]:
            scopes.add(f"{parts[0]}-{parts[1]}")
    return sorted(scopes)


# ---------------------------------------------------------------- scanning


def scan_local(src: Path, patterns: list[str], recursive: bool) -> list[Item]:
    if not src.exists():
        raise FileNotFoundError(f"src not found: {src}")
    it = src.rglob("*") if recursive else src.glob("*")
    out: list[Item] = []
    for p in sorted(it):
        if p.is_file() and any(fnmatch.fnmatch(p.name, pat) for pat in patterns):
            out.append(Item(path=p))
    return out


def assign_keys(items: list[Item], src: Path, keep_dirs: bool, recursive: bool) -> list[Item] | None:
    """给每个文件算 object key；默认按文件名平铺（目标目录是扁平结构）。"""
    for it in items:
        it.key = (
            str(it.path.relative_to(src)).replace(os.sep, "/")
            if keep_dirs and recursive
            else it.path.name
        )
    seen: dict[str, Path] = {}
    for it in items:
        if it.key in seen:
            print(f"[ERROR] key collision: '{it.key}' from both\n"
                  f"    {seen[it.key]}\n    {it.path}\n"
                  "    -> rename one, narrow --patterns, or use --keep-dirs")
            return None
        seen[it.key] = it.path
    return items


# ---------------------------------------------------------------- remote ops


def list_all(client, bucket: str, prefix: str, page_size: int = PAGE_SIZE,
             max_total: int = 200000) -> tuple[int, str, dict[str, int]]:
    """分页列举 prefix 下所有对象，返回 (status, msg, {key: size})。纯读。"""
    found: dict[str, int] = {}
    marker = None
    while True:
        try:
            resp = client.listObjects(bucket, prefix=prefix, max_keys=page_size, marker=marker)
            status, msg = resp_status(resp)
        except Exception as exc:
            return -1, f"{type(exc).__name__}: {exc}", found
        if not is_ok(status):
            return status, msg, found

        body = getattr(resp, "body", None)
        contents = list(getattr(body, "contents", None) or [])
        for obj in contents:
            key = getattr(obj, "key", "")
            if key:
                found[key] = int(getattr(obj, "size", 0) or 0)

        truncated = getattr(body, "isTruncated", None)
        if isinstance(truncated, str):
            truncated = truncated.lower() == "true"
        next_marker = getattr(body, "nextMarker", None)
        if not truncated or not contents or len(found) >= max_total:
            return 200, "", found
        marker = next_marker or getattr(contents[-1], "key", None)
        if not marker:
            return 200, "", found


def upload_one(client, bucket: str, key: str, path: Path, retries: int) -> Result:
    res = Result(op="upload", key=key)
    t0 = time.time()
    last = ""
    for attempt in range(1, max(1, retries) + 1):
        try:
            if path.stat().st_size > MULTIPART_THRESHOLD:
                resp = client.uploadFile(bucket, key, str(path), taskNum=4, enableCheckpoint=True)
            else:
                resp = client.putFile(bucket, key, str(path))
            status, msg = resp_status(resp)
        except Exception as exc:
            status, msg = -1, f"{type(exc).__name__}: {exc}"

        res.http_status = status
        if is_ok(status):
            res.status, res.elapsed = "uploaded", round(time.time() - t0, 2)
            return res

        last = f"HTTP {status} {msg}".strip()
        if attempt < retries:
            backoff = min(2 ** (attempt - 1), 8)
            print(f"    retry {attempt}/{retries - 1} in {backoff}s  ({last})")
            time.sleep(backoff)

    res.status, res.error, res.elapsed = "failed", last, round(time.time() - t0, 2)
    return res


def verify_one(client, bucket: str, key: str, local_size: int) -> Result:
    res = Result(op="verify", key=key)
    try:
        resp = client.getObjectMetadata(bucket, key)
        status, msg = resp_status(resp)
    except Exception as exc:
        status, msg = -1, f"{type(exc).__name__}: {exc}"
    res.http_status = status
    if not is_ok(status):
        res.status, res.error = "failed", f"HEAD failed: HTTP {status} {msg}".strip()
        return res

    size = -1
    body = getattr(resp, "body", None)
    for attr in ("contentLength", "content_length"):
        val = getattr(body, attr, None) if body is not None else None
        if val is not None:
            try:
                size = int(val)
            except (TypeError, ValueError):
                size = -1
            break
    res.remote_size = size
    if size >= 0 and size != local_size:
        res.status = "mismatch"
        res.error = f"remote size {size} != local size {local_size}"
    else:
        res.status = "verified"
    return res


def delete_one(client, bucket: str, key: str, retries: int) -> Result:
    """按 key 删除。OBS 对不存在的 key 返回 204，天然幂等。"""
    res = Result(op="delete", key=key)
    t0 = time.time()
    last = ""
    for attempt in range(1, max(1, retries) + 1):
        try:
            resp = client.deleteObject(bucket, key)
            status, msg = resp_status(resp)
        except Exception as exc:
            status, msg = -1, f"{type(exc).__name__}: {exc}"
        res.http_status = status
        if is_ok(status):
            res.status, res.elapsed = "deleted", round(time.time() - t0, 2)
            return res
        last = f"HTTP {status} {msg}".strip()
        if attempt < retries:
            backoff = min(2 ** (attempt - 1), 8)
            print(f"    retry {attempt}/{retries - 1} in {backoff}s  ({last})")
            time.sleep(backoff)
    res.status, res.error, res.elapsed = "failed", last, round(time.time() - t0, 2)
    return res


# ---------------------------------------------------------------- selectors


def resolve_delete_targets(remote: dict[str, int], prefix: str, args) -> tuple[list[tuple[str, int]], str]:
    """把 --delete-keys / --delete-glob / --delete-version 解析成受护栏约束的候选列表。"""
    base = prefix + "/"
    picked: dict[str, int] = {}

    def add(key: str) -> str | None:
        if not key.startswith(base):
            return f"key outside prefix, refused: {key}"
        if key == base and not args.include_folders:
            return f"folder marker skipped: {key} (add --include-folders)"
        if key.endswith("/") and not args.include_folders:
            return f"folder marker skipped: {key} (add --include-folders)"
        picked[key] = remote.get(key, -1)
        return None

    errors: list[str] = []
    for raw in args.delete_keys or []:
        key = raw if "/" in raw else base + raw
        if err := add(key):
            errors.append(err)

    for pat in args.delete_glob or []:
        for key in sorted(remote):
            name = key[len(base):] if key.startswith(base) else key
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(key, pat):
                if err := add(key):
                    errors.append(err)

    for ver in args.delete_version or []:
        matched = 0
        for key in sorted(remote):
            name = key[len(base):] if key.startswith(base) else key
            parts = name.split("-")
            if len(parts) >= 2 and parts[1] == ver:
                matched += 1
                if err := add(key):
                    errors.append(err)
        if not matched:
            errors.append(f"no object matches version {ver!r} under prefix")

    return sorted(picked.items()), "; ".join(errors)


# --------------------------------------------------------------------- cli


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="obs_publish.py",
        description="Publish / sync / delete objects on Huawei OBS. Read-only unless told otherwise.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Pass --prefix <OBS project prefix> to every command below.\ndelete recipes:\n"
            "  py -3 obs_publish.py --delete-glob \"flash_linear_attention_npu_a2-0.dev20261118+main.abcdef0-*\"            # preview\n"
            "  py -3 obs_publish.py --delete-glob \"flash_linear_attention_npu_a2-0.dev20261118+main.abcdef0-*\" --yes      # delete\n"
            "  py -3 obs_publish.py --delete-keys a.whl b.json --yes --verify\n"
            "  py -3 obs_publish.py --delete-version 0.dev20261118+main.abcdef0 --yes\n"
            "sync recipes:\n"
            "  py -3 obs_publish.py --src .\\dist\\variants --sync                 # upload diff only\n"
            "  py -3 obs_publish.py --src .\\dist\\variants --sync --prune         # preview prune\n"
            "  py -3 obs_publish.py --src .\\dist\\variants --sync --prune --yes   # prune + upload\n"
        ),
    )
    p.add_argument("--src", default=".", help="local dir (default: .), ignored in delete mode")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--prefix", required=True, help="explicit OBS project prefix")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p.add_argument("--patterns", default=DEFAULT_PATTERNS, help=f"comma-separated glob (default: {DEFAULT_PATTERNS})")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--keep-dirs", action="store_true", help="keep relative dirs in key (default: flatten)")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--page-size", type=int, default=PAGE_SIZE)
    p.add_argument("--local-only", action="store_true", help="scan local only; no network, no credentials")
    p.add_argument("--check-creds", action="store_true",
                   help="only report where AK/SK come from (no network call)")
    p.add_argument("--check-perm", action="store_true",
                   help="probe permissions on --prefix without uploading anything")
    p.add_argument("--write-probe", action="store_true",
                   help="with --check-perm: also probe PutObject/DeleteObject via "
                        "multipart-init (creates no object, but may leave a 0-byte "
                        "unfinished upload session if abort is denied)")
    p.add_argument("--verbose", action="store_true", help="keep OBS SDK internal logging")
    p.add_argument("--report", default="", help="write JSON report here")
    p.add_argument("--yes", action="store_true", help="confirm destructive operations (delete/prune)")
    p.add_argument("--list", action="store_true",
                   help="read-only inventory: print remote objects and diff against --src")

    g_write = p.add_argument_group("write")
    g_write.add_argument("--upload", action="store_true", help="actually upload (default: dry-run)")
    g_write.add_argument("--mkdir", action="store_true", help="also put a 0-byte '<prefix>/' marker")
    g_write.add_argument("--verify", action="store_true", help="HEAD every key afterwards")

    g_sync = p.add_argument_group("sync")
    g_sync.add_argument("--sync", action="store_true", help="upload only missing/size-differs keys")
    g_sync.add_argument("--force", action="store_true", help="with --sync: upload even if size matches")
    g_sync.add_argument("--prune", action="store_true",
                        help="with --sync: delete remote keys not in local set (needs --yes)")
    g_sync.add_argument("--prune-scope", default="",
                        help="glob limiting prune range (default: derived from local <name>-<version>)")
    g_sync.add_argument("--allow-empty", action="store_true",
                        help="with --sync --prune: allow empty local set (dangerous)")

    g_del = p.add_argument_group("delete")
    g_del.add_argument("--delete-keys", nargs="+", metavar="KEY",
                       help="delete exact keys (basename or full key)")
    g_del.add_argument("--delete-glob", nargs="+", metavar="PATTERN",
                       help="delete keys whose name matches glob")
    g_del.add_argument("--delete-version", nargs="+", metavar="VER",
                       help="delete all keys of version VER under prefix")
    g_del.add_argument("--include-folders", action="store_true",
                       help="also allow deleting 'xxx/' folder markers")
    g_del.add_argument("--max-delete", type=int, default=MAX_DELETE_DEFAULT,
                       help=f"refuse if candidates exceed this (default: {MAX_DELETE_DEFAULT})")
    return p.parse_args(argv)


def header(args) -> None:
    print("=" * 78)
    print("obs_publish  |  read-only unless --upload / --yes")
    print("=" * 78)
    print(f"endpoint : {args.endpoint}")
    print(f"bucket   : {args.bucket}")
    print(f"prefix   : {args.prefix.strip('/')}")
    print()


def connect(args) -> tuple[Any, str, int]:
    """校验凭证并建连；返回 (client, prefix, exit_code)。exit_code!=0 表示应直接退出。

    凭证来源（只读，按优先级）：
        1. 进程环境变量      —— CI 的 env: 块、PowerShell 里 $env:X='...'
        2. 用户环境变量      —— 注册表 HKCU\\Environment（系统属性里"用户变量"）
        3. 系统环境变量      —— 注册表 HKLM\\...\\Environment（"系统变量"）
    不接受 --ak/--sk 这类命令行参数；也不读配置文件，避免口令落到命令行与历史里。
    """
    ak, sk, ak_src, sk_src = resolve_credentials()

    print(f"[creds] AK: {mask(ak)}  from {ak_src or 'NOT FOUND'}")
    print(f"[creds] SK: {mask_secret(sk)}  from {sk_src or 'NOT FOUND'}")
    if ak and ak_src.endswith("(registry)"):
        print("[creds] note: read from Windows registry env directly; "
              "no need to re-open the terminal")

    if not ak or not sk:
        if sys.platform == "win32":
            print("\n[ERROR] credentials not found. set them **without putting them on the "
                  "command line**:\n"
                  "  A) GUI (recommended): Win+R -> sysdm.cpl -> Advanced -> Environment "
                  "Variables -> User variables -> New\n"
                  "       OBS_ACCESS_KEY / OBS_SECRET_KEY\n"
                  "  B) PowerShell prompt (typed input is NOT recorded in PSReadLine history):\n"
                  "       $ak = Read-Host 'AK'\n"
                  "       $sk = Read-Host 'SK' -AsSecureString\n"
                  "       [Environment]::SetEnvironmentVariable('OBS_ACCESS_KEY',$ak,'User')\n"
                  "       [Environment]::SetEnvironmentVariable('OBS_SECRET_KEY',"
                  "(New-Object System.Management.Automation.PSCredential('x',$sk))"
                  ".GetNetworkCredential().Password,'User')\n"
                  "    (this script reads the registry directly, so no terminal restart needed)")
        else:
            print("\n[ERROR] credentials not found in this process environment.\n"
                  "Inject OBS_ACCESS_KEY and OBS_SECRET_KEY into the service, scheduler, "
                  "or SSH process that runs this command. Do not place their values in "
                  "scripts, config files, command arguments, or shell history.")
        return None, "", EXIT_USAGE
    try:
        return build_client(args.endpoint, ak, sk, args.timeout, args.verbose), "", EXIT_OK
    except ImportError:
        command = ("py -3 -m pip install esdk-obs-python" if sys.platform == "win32"
                   else "python3 -m pip install esdk-obs-python")
        print(f"\n[ERROR] esdk-obs-python not installed:  {command}")
        return None, "", EXIT_USAGE


def _explain_403(msg: str) -> None:
    """把 403 的两种常见成因分开说清楚，避免"权限不足 / AK-SK 不配对"混淆。"""
    low = msg.lower()
    if "signaturedoesnotmatch" in low:
        print("        -> AK/SK DO NOT MATCH: they are not a pair, or the SK was rotated.")
        print("           check the IAM console: My Credentials -> Access Keys "
              "(AK=20 chars, SK=40 chars, SK is shown only once)")
    elif "invalidaccesskeyid" in low:
        print("        -> AK not recognized: wrong key, or it was deleted")
    elif "accessdenied" in low or "forbidden" in low:
        print("        -> permission denied: the AK/SK lacks the required OBS policy")
    elif "getaddrinfo" in low or "gaierror" in low or "name or service" in low:
        print("        -> endpoint not resolvable: check network / DNS / proxy / --endpoint")
    else:
        print("        -> bucket must live in the endpoint's region (cn-north-4); "
              "or wrong bucket/endpoint")


def precheck(client, args, prefix: str, require_list: bool) -> tuple[dict[str, int], bool, int]:
    """headBucket + listObjects，返回 (remote, listing_ok, exit_code)。

    - headBucket 失败只告警：HEAD 请求 403 既可能是缺 obs:bucket:HeadBucket，
      也可能是签名问题，不该挡住一个只有 PutObject 权限的最小化凭证。
    - listObjects 失败：require_list=True（sync/delete/prune）时致命；
      发布模式下只告警并跳过"覆盖预检"。
    """
    print("\n[check] headBucket ...", end=" ", flush=True)
    try:
        status, msg = resp_status(client.headBucket(args.bucket))
    except Exception as exc:
        status, msg = -1, f"{type(exc).__name__}: {exc}"
    if is_ok(status):
        print(f"OK ({status})")
    else:
        print(f"WARN ({status}) {msg}")
        _explain_403(msg)
        print("        -> continuing: the authoritative test is listObjects below")

    print("[check] listObjects(prefix) ...", end=" ", flush=True)
    status, msg, remote = list_all(client, args.bucket, prefix + "/", args.page_size)
    if is_ok(status):
        print(f"OK ({status}), {len(remote)} object(s) under prefix")
        return remote, True, EXIT_OK

    print(f"FAILED ({status}) {msg}")
    _explain_403(msg)
    if require_list:
        print("        -> this mode (sync/delete/prune) needs listing, aborting")
        return {}, False, EXIT_FAIL
    print("        -> continuing without overwrite pre-check "
          "(upload needs only PutObject)")
    return {}, False, EXIT_OK


# ------------------------------------------------------------------- modes


def perm_probe(client, args, prefix: str) -> int:
    """检查某前缀的权限，**不创建任何对象**。

    只读探测：headBucket / getBucketAcl / getBucketPolicy / listObjects / HEAD object。

    写探测默认关闭（--write-probe 打开），因为 OBS 没有 dry-run API，只能用两个
    "不落地对象"的旁路办法：
        * deleteObject 一个刻意不存在的 key —— 204=有 DeleteObject 权限，
          403=没有；key 不存在，什么都不会被删（幂等）。
        * initiateMultipartUpload（需要 PutObject 权限，但不产生对象）+ abort。
          **注意**：若策略没给 AbortMultipartUpload，abort 会 403，会残留一个
          不含任何 part（0 字节）的未完成会话，只能由桶 owner 或生命周期规则清理。
    """
    rows: list[tuple[str, str, str]] = []   # (op, verdict, detail)
    sig_error = ""
    leftover_mpu = ""

    def probe(op: str, fn):
        nonlocal sig_error
        try:
            resp = fn()
            status, msg = resp_status(resp)
        except Exception as exc:
            status, msg = -1, f"{type(exc).__name__}: {exc}"
        if "signaturedoesnotmatch" in msg.lower():
            verdict = "AUTH FAILED"
            sig_error = msg
        elif "invalidaccesskeyid" in msg.lower():
            verdict = "AUTH FAILED"
            sig_error = msg
        elif is_ok(status):
            verdict = "ALLOW"
        elif status in (401, 403):
            # HEAD 响应没有 body，OBS 不会带 errorCode，无法区分"签名错"和"没权限"
            verdict = "AMBIGUOUS" if op.startswith("HEAD") and ":" not in msg else "DENY"
        elif status == 404:
            verdict = "NOT FOUND"
        else:
            verdict = "UNKNOWN"
        rows.append((op, verdict, f"{status} {msg}".strip()))
        print(f"  {op:34s} {verdict:11s} {status:>4}  {msg}")
        return status, resp

    print(f"\n[perm] target prefix: {prefix}/")
    print("[perm] read-only probes:")
    probe("HEAD bucket", lambda: client.headBucket(args.bucket))
    probe("GET bucket ACL", lambda: client.getBucketAcl(args.bucket))
    probe("GET bucket policy", lambda: client.getBucketPolicy(args.bucket))

    st, resp = probe("LIST prefix", lambda: client.listObjects(
        args.bucket, prefix=prefix + "/", max_keys=20))
    objects, folders = [], []
    if is_ok(st) and resp is not None:
        objects = list(getattr(getattr(resp, "body", None), "contents", None) or [])
        folders = [o for o in objects if str(getattr(o, "key", "")).endswith("/")]
        files = [o for o in objects if not str(getattr(o, "key", "")).endswith("/")]
        print(f"       -> {len(files)} file(s), {len(folders)} folder marker(s)")
        for o in objects[:10]:
            key = str(getattr(o, "key", ""))
            kind = "DIR " if key.endswith("/") else "FILE"
            print(f"          [{kind}] {key}  {getattr(o, 'size', 0)}")
        if files:
            target = files[0]
            probe(f"HEAD object ({str(target.key).split('/')[-1][:34]})",
                  lambda: client.getObjectMetadata(args.bucket, target.key))
        elif folders:
            print("       -> only a 0-byte folder marker: the prefix EXISTS but holds "
                  "no file,\n          so GetObject cannot be tested from here")
        else:
            print("       -> prefix is empty (exists but has no object, "
                  "or you lack ListBucket permission)")

    # 没有对象也要能区分"前缀级策略"和"完全没列举权限"：再列一次上一级
    parent = prefix.rsplit("/", 1)[0] if "/" in prefix else ""
    if parent:
        probe(f"LIST parent ({parent.split('/')[-1][:20]})",
              lambda: client.listObjects(args.bucket, prefix=parent + "/", max_keys=5))

    if not sig_error and args.write_probe:
        print("[perm] write probes (--write-probe; no object is created):")
        ghost = f"{prefix}/__perm_probe_do_not_create__"
        probe("DELETE nonexistent key (no-op)", lambda: client.deleteObject(args.bucket, ghost))

        test_key = f"{prefix}/__perm_probe_multipart__"
        st, resp = probe("INIT multipart upload (needs PutObject)",
                         lambda: client.initiateMultipartUpload(args.bucket, test_key))
        if is_ok(st) and resp is not None:
            body = getattr(resp, "body", None)
            upload_id = str(getattr(body, "uploadId", "") or
                            getattr(body, "upload_id", "") or "")
            if not upload_id:
                leftover_mpu = test_key
                print("       -> WARN: init 200 but no uploadId; an incomplete upload "
                      f"may remain at {test_key}")
            else:
                ast, _ = probe("ABORT multipart upload (cleanup)",
                               lambda: client.abortMultipartUpload(args.bucket, test_key,
                                                                   upload_id))
                if is_ok(ast):
                    print("       -> cleaned up: no object created, session aborted")
                else:
                    leftover_mpu = test_key
                    print("       -> WARN: abort was DENIED, so an incomplete upload "
                          "session remains at")
                    print(f"          {test_key}")
                    print("          It holds 0 bytes (no part uploaded), but only the "
                          "bucket owner can abort it;")
                    print("          or a lifecycle rule 'AbortIncompleteMultipartUpload' "
                          "will collect it.")
    elif not sig_error:
        print("[perm] write probes skipped (add --write-probe to test PutObject/"
              "DeleteObject)")

    print("\n[perm] summary")
    for op, verdict, _ in rows:
        print(f"  {verdict:11s} {op}")

    if sig_error:
        print("\n[perm] RESULT: identity INVALID - the request was rejected at "
              "authentication,\n"
              "        so NOTHING about this directory's permissions can be concluded.\n"
              "        Fix AK/SK (must be one pair from the same IAM user) and re-run.")
        return EXIT_USAGE

    list_rows = [r for r in rows if r[0].startswith("LIST")]
    if list_rows and all(r[1] == "DENY" for r in list_rows):
        print("\n[perm] RESULT: identity valid, but no ListBucket permission -> "
              "cannot confirm the prefix exists")
    else:
        print("\n[perm] RESULT: identity valid (signature accepted)")
    return EXIT_OK


def run_list(args, prefix: str, remote: dict[str, int], items: list[Item],
             session: Session) -> int:
    """只读清单：打印远端对象，并与本地目录做差异对比（不写任何东西）。"""
    base = prefix + "/"
    total = sum(remote.values())
    print(f"\n[list] obs://{args.bucket}/{prefix}/   {len(remote)} object(s), "
          f"{human_size(total)}")
    for key in sorted(remote):
        print(f"  {human_size(remote[key]):>10}  {key}")

    if not items:
        print("\n[list] no --src given, nothing to diff")
        return EXIT_OK

    local = {base + it.key: it for it in items}
    ok = missing = diff = 0
    print(f"\n[diff] local {len(items)} file(s) vs remote {len(remote)} object(s)")
    for key, it in sorted(local.items()):
        if key not in remote:
            missing += 1
            print(f"  MISSING    {human_size(it.size):>10}  {it.key}")
        elif remote[key] == it.size:
            ok += 1
            print(f"  OK         {human_size(it.size):>10}  {it.key}")
        else:
            diff += 1
            print(f"  SIZE-DIFF  remote {human_size(remote[key])} != local "
                  f"{human_size(it.size)}  {it.key}")
    extra = [k for k in remote if k not in local and not k.endswith("/")]
    for key in sorted(extra):
        print(f"  EXTRA      {human_size(remote[key]):>10}  {key}")

    print(f"\n[diff] ok={ok} missing={missing} size-diff={diff} extra={len(extra)}")
    if missing or diff:
        print("[diff] not in sync: run with --upload (or --sync) to fix")
        return EXIT_FAIL
    if extra:
        print("[diff] identical set for local files; extras exist remotely "
              "(use --sync --prune to preview removal)")
    return EXIT_OK


def run_delete(args, client, prefix: str, remote: dict[str, int], session: Session) -> int:
    candidates, errs = resolve_delete_targets(remote, prefix, args)
    if errs:
        print(f"\n[WARN] {errs}")

    if not candidates:
        print("\n[delete] nothing matched; nothing to do")
        return EXIT_OK

    total = sum(sz for _, sz in candidates if sz >= 0)
    print(f"\n[delete] {len(candidates)} candidate(s), {human_size(total)}")
    for key, sz in candidates:
        print(f"    {key}  ({human_size(sz) if sz >= 0 else 'size unknown'})")

    if len(candidates) > args.max_delete:
        print(f"\n[REFUSE] {len(candidates)} > --max-delete {args.max_delete}; "
              "narrow the selector or raise --max-delete explicitly")
        return EXIT_USAGE

    if not args.yes:
        sel: list[str] = []
        for flag, vals in (("--delete-keys", args.delete_keys),
                           ("--delete-glob", args.delete_glob),
                           ("--delete-version", args.delete_version)):
            if vals:
                sel.append(flag)
                sel.extend(f'"{v}"' if " " in v else v for v in vals)
        print("\n[dry-run] nothing deleted. re-run with --yes to confirm:")
        print(f"    py -3 {Path(__file__).name} --prefix \"{prefix}\" {' '.join(sel)} --yes")
        return EXIT_OK

    print(f"\n[delete] executing ...")
    ok = bad = 0
    for i, (key, _) in enumerate(candidates, 1):
        print(f"  [{i}/{len(candidates)}] {key} ", end="", flush=True)
        res = delete_one(client, args.bucket, key, args.retries)
        session.results.append(res)
        if res.status == "deleted":
            ok += 1
            print(f"DELETED {res.http_status} {res.elapsed}s")
        else:
            bad += 1
            print(f"FAILED {res.error}")
    print(f"[delete] deleted={ok} failed={bad}")

    if args.verify:
        print("\n[verify] re-list to confirm absence ...")
        status, msg, after = list_all(client, args.bucket, prefix + "/", args.page_size)
        if not is_ok(status):
            print(f"    cannot verify: HTTP {status} {msg}")
            return EXIT_FAIL
        still = [key for key, _ in candidates if key in after]
        for key, _ in candidates:
            gone = key not in after
            session.results.append(Result(op="verify", key=key,
                                          status="absent" if gone else "failed",
                                          error="" if gone else "still present"))
        if still:
            print(f"    STILL PRESENT ({len(still)}):")
            for key in still:
                print(f"      {key}")
            bad += len(still)
        else:
            print(f"    all {len(candidates)} key(s) gone")

    return EXIT_OK if bad == 0 else EXIT_FAIL


def run_sync(args, client, prefix: str, remote: dict[str, int], items: list[Item],
             session: Session) -> int:
    base = prefix + "/"
    remote_by_key = remote
    to_upload: list[Item] = []
    skipped = 0
    for it in items:
        full = base + it.key
        if full in remote_by_key and remote_by_key[full] == it.size and not args.force:
            skipped += 1
            session.results.append(Result(op="upload", key=full, status="skipped",
                                          remote_size=remote_by_key[full]))
            continue
        to_upload.append(it)

    scopes = [args.prune_scope] if args.prune_scope else [f"{s}*" for s in key_scopes(items)]
    to_prune: list[tuple[str, int]] = []
    out_of_scope = 0
    if args.prune:
        local_full = {base + it.key for it in items}
        for key in sorted(remote_by_key):
            if key in local_full or key.endswith("/"):
                continue
            name = key[len(base):] if key.startswith(base) else key
            if any(fnmatch.fnmatch(name, sc) for sc in scopes):
                to_prune.append((key, remote_by_key[key]))
            else:
                out_of_scope += 1

    print(f"\n[sync] local={len(items)}  remote={len(remote_by_key)}")
    print(f"       upload={len(to_upload)}  unchanged={skipped}"
          + (f"  prune={len(to_prune)}" if args.prune else ""))
    if args.prune:
        print(f"       prune scope: {', '.join(scopes) or '-'}"
              + (f"   (ignored out-of-scope stale: {out_of_scope})" if out_of_scope else ""))

    exit_code = EXIT_OK

    if to_upload:
        print(f"\n[upload] {len(to_upload)} file(s) ...")
        for i, it in enumerate(to_upload, 1):
            full = base + it.key
            print(f"  [{i}/{len(to_upload)}] {it.key} ({human_size(it.size)}) ", end="", flush=True)
            res = upload_one(client, args.bucket, full, it.path, args.retries)
            session.results.append(res)
            if res.status == "uploaded":
                print(f"OK {res.http_status} {res.elapsed}s")
            else:
                print(f"FAILED {res.error}")
                exit_code = EXIT_FAIL
    elif skipped:
        print("\n[upload] nothing to upload (all sizes match; use --force to re-upload)")

    if not args.prune:
        if args.verify:
            exit_code |= verify_items(client, args, base, items, session)
        return exit_code

    # ---- prune
    if not items and not args.allow_empty:
        print("\n[REFUSE] local set is empty; refusing to prune (use --allow-empty if you really mean it)")
        return EXIT_FAIL
    if not to_prune:
        print("\n[prune] nothing to remove")
        if args.verify:
            exit_code |= verify_items(client, args, base, items, session)
        return exit_code

    total = sum(sz for _, sz in to_prune)
    print(f"\n[prune] {len(to_prune)} stale object(s), {human_size(total)}:")
    for key, sz in to_prune:
        print(f"    {key}  ({human_size(sz)})")
    if len(to_prune) > args.max_delete:
        print(f"\n[REFUSE] {len(to_prune)} > --max-delete {args.max_delete}")
        return EXIT_USAGE
    if not args.yes:
        print("\n[dry-run] nothing pruned. re-run with --yes:")
        print(f"    py -3 {Path(__file__).name} --src \"{args.src}\" --sync --prune --yes")
        return exit_code

    print("\n[prune] executing ...")
    bad = 0
    for i, (key, _) in enumerate(to_prune, 1):
        print(f"  [{i}/{len(to_prune)}] {key} ", end="", flush=True)
        res = delete_one(client, args.bucket, key, args.retries)
        session.results.append(res)
        if res.status == "deleted":
            print(f"DELETED {res.http_status} {res.elapsed}s")
        else:
            bad += 1
            print(f"FAILED {res.error}")
    print(f"[prune] deleted={len(to_prune) - bad} failed={bad}")
    if bad:
        exit_code = EXIT_FAIL

    if args.verify:
        exit_code |= verify_items(client, args, base, items, session)
    return exit_code


def verify_items(client, args, base: str, items: list[Item], session: Session) -> int:
    print("\n[verify] HEAD every uploaded key ...")
    ok = bad = 0
    for it in items:
        full = base + it.key
        res = verify_one(client, args.bucket, full, it.size)
        session.results.append(res)
        if res.status == "verified":
            ok += 1
            print(f"    OK   {it.key} ({human_size(res.remote_size)})")
        else:
            bad += 1
            print(f"    BAD  {it.key}: {res.error or res.status}")
    print(f"[verify] ok={ok} bad={bad}")
    return EXIT_OK if bad == 0 else EXIT_FAIL


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    header(args)

    delete_mode = bool(args.delete_keys or args.delete_glob or args.delete_version)

    try:
        prefix = validate_prefix(args.prefix)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return EXIT_USAGE

    session = Session(
        mode="delete" if delete_mode else ("sync" if args.sync else "publish"),
        bucket=args.bucket, prefix=prefix, endpoint=args.endpoint, src=str(args.src),
    )

    # ---- phase 0: --check-creds 只报凭证来源，不扫本地、不联网
    if args.check_creds:
        ak, sk, ak_src, sk_src = resolve_credentials()
        print(f"[creds] AK: {mask(ak)}  from {ak_src or 'NOT FOUND'}")
        print(f"[creds] SK: {mask_secret(sk)}  from {sk_src or 'NOT FOUND'}")
        print(f"[creds] SK fingerprint: {fingerprint(sk)}  "
              "(sha256/8 - use it to confirm the env var really changed)")
        print("[creds] note: OBS authentication carries ONLY the AK; there is no "
              "user-name/account field in the request.")
        if ak and sk:
            print("\n[ok] both credentials resolved from the environment; "
                  "nothing needs to be typed on the command line")
            return EXIT_OK
        if sys.platform == "win32":
            print("\n[ERROR] missing credentials.\n"
                  "  A) GUI: Win+R -> sysdm.cpl -> Advanced -> Environment Variables -> "
                  "User variables -> New  (OBS_ACCESS_KEY / OBS_SECRET_KEY)\n"
                  "  B) PowerShell prompt: use Read-Host (input is not written to history)\n"
                  "     see the header of this file for the exact 4 lines")
        else:
            print("\n[ERROR] OBS_ACCESS_KEY / OBS_SECRET_KEY are missing from this "
                  "process environment. Inject them into the service, scheduler, "
                  "or SSH process; do not store them in scripts or command arguments.")
        return EXIT_USAGE

    # ---- phase 1: local scan (delete / check-perm 不需要)
    items: list[Item] = []
    if not delete_mode and not args.check_perm:
        try:
            items = scan_local(Path(args.src).expanduser(),
                               [s.strip() for s in args.patterns.split(",") if s.strip()],
                               args.recursive)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            return EXIT_USAGE
        if assign_keys(items, Path(args.src).expanduser(), args.keep_dirs, args.recursive) is None:
            return EXIT_USAGE
        session.local_files = len(items)
        print(f"[local] {len(items)} file(s), {human_size(sum(i.size for i in items))}")
        for it in items:
            print(f"    {it.key:<70} {human_size(it.size):>10}")

    if args.local_only:
        if delete_mode:
            print("\n[ERROR] --local-only cannot be combined with --delete-*")
            return EXIT_USAGE
        print("\n[local-only] no network call, no credentials needed. done.")
        _report(args.report, session)
        return EXIT_OK

    # ---- phase 2/3: connect + read-only self-check
    client, _, rc = connect(args)
    if rc:
        return rc
    try:
        if args.check_perm:
            return perm_probe(client, args, prefix)

        remote, listing_ok, rc = precheck(
            client, args, prefix, require_list=delete_mode or args.sync or args.list)
        if rc:
            return rc
        session.remote_files = len(remote)

        if args.list:
            return run_list(args, prefix, remote, items, session)

        if delete_mode:
            return run_delete(args, client, prefix, remote, session)

        if args.sync:
            return run_sync(args, client, prefix, remote, items, session)

        # ---- phase 4: publish plan
        base = prefix + "/"
        will_overwrite, changed = [], []
        for it in items:
            full = base + it.key
            if full in remote:
                will_overwrite.append(full)
                if remote[full] != it.size:
                    changed.append((full, remote[full], it.size))
        if will_overwrite:
            print(f"\n[plan] {len(will_overwrite)}/{len(items)} key(s) already exist -> OVERWRITE")
            for full, rsz, lsz in changed[:10]:
                print(f"    {full}\n        remote {human_size(rsz)} != local {human_size(lsz)}")
        elif listing_ok:
            print(f"\n[plan] all {len(items)} key(s) are new")
        else:
            print(f"\n[plan] overwrite pre-check unavailable (listing failed); "
                  f"{len(items)} key(s) would be uploaded blindly")

        if not args.upload:
            print("\n[dry-run] nothing written. re-run with --upload:")
            print(f"    py -3 {Path(__file__).name} --src \"{args.src}\" --upload")
            return EXIT_OK

        if args.mkdir:
            key = base
            if key in remote:
                print(f"\n[mkdir] '{key}' already exists, skip")
            else:
                try:
                    status, msg = resp_status(client.putContent(args.bucket, key, b""))
                except Exception as exc:
                    status, msg = -1, f"{type(exc).__name__}: {exc}"
                print(f"\n[mkdir] '{key}' " + (f"created ({status})" if is_ok(status)
                                               else f"FAILED ({status}) {msg}"))

        print(f"\n[upload] {len(items)} file(s) -> obs://{args.bucket}/{prefix}/")
        exit_code, ok = EXIT_OK, 0
        for i, it in enumerate(items, 1):
            full = base + it.key
            print(f"  [{i}/{len(items)}] {it.key} ({human_size(it.size)}) ", end="", flush=True)
            res = upload_one(client, args.bucket, full, it.path, args.retries)
            session.results.append(res)
            if res.status == "uploaded":
                ok += 1
                print(f"OK {res.http_status} {res.elapsed}s")
            else:
                print(f"FAILED {res.error}")
                exit_code = EXIT_FAIL
        print(f"[upload] ok={ok} failed={len(items) - ok}")

        if args.verify and ok:
            exit_code |= verify_items(client, args, base, items, session)

        print(f"\n[done] obs://{args.bucket}/{prefix}/")
        print("[note] mirror pages are cached (Cache-Control max-age=1296000 / 15 days);"
              " upload success != immediate public visibility")
        return exit_code

    except KeyboardInterrupt:
        print("\n[abort] interrupted; already-uploaded/deleted objects are NOT rolled back")
        return EXIT_FAIL
    finally:
        try:
            client.close()
        except Exception:
            pass
        session.finished_at = now_iso()
        _report(args.report, session)


def _report(path: str, session: Session) -> None:
    if not path:
        return
    try:
        Path(path).write_text(json.dumps(session.to_dict(), indent=2, ensure_ascii=False),
                              encoding="utf-8")
        print(f"\n[report] written to {path}")
    except OSError as exc:
        print(f"\n[report] failed to write {path}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
