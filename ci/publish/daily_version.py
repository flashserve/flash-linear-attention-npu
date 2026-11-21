"""One daily version contract for planning, building and publishing."""
from __future__ import annotations

import ast
import base64
import csv
import datetime as dt
import email.parser
import hashlib
import io
import os
import re
import zipfile
from pathlib import Path

TIMEZONE = dt.timezone(dt.timedelta(hours=8), name="Asia/Shanghai")


def validate_date(value: str) -> str:
    if not re.fullmatch(r"[0-9]{8}", value):
        raise ValueError("daily 日期必须为 YYYYMMDD")
    dt.datetime.strptime(value, "%Y%m%d")
    return value


def today() -> str:
    return dt.datetime.now(TIMEZONE).strftime("%Y%m%d")


def version_for(branch: str, date: str, sha: str) -> str:
    validate_date(date)
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("daily 需要完整的 40 位 commit SHA")
    if branch == "main":
        return f"0.dev{date}+main.{sha[:7]}"
    match = re.fullmatch(r"v([0-9]+(?:\.[0-9]+)+)", branch)
    if not match:
        raise ValueError(f"不支持的 daily 分支: {branch}")
    return f"{match[1]}.dev{date}+{sha[:7]}"


def legacy_versions(branch: str, date: str, sha: str) -> tuple[str, ...]:
    """Known earlier spellings for the exact same daily source commit."""
    version_for(branch, date, sha)
    stem = f'0.dev{date}+main.' if branch == 'main' else f'{branch[1:]}.dev{date}+'
    return (stem + 'g' + sha[:7], stem + 'g' + sha[-8:])


def wheel_name(package: str, branch: str, date: str, sha: str, arch: str) -> str:
    return f"{package}-{version_for(branch, date, sha)}-py3-none-manylinux_2_34_{arch}.whl"


def wheel_identity(name: str, package: str) -> tuple[str, str, str] | None:
    """Strictly identify only wheels belonging to the new daily contract."""
    match = re.fullmatch(
        re.escape(package) + r"-(0|[0-9]+(?:\.[0-9]+)+)\.dev([0-9]{8})"
        r"\+(main\.)?(?:[0-9a-f]{7}|g[0-9a-f]{7,8})-py3-none-manylinux_2_34_(aarch64|x86_64)\.whl",
        name,
    )
    if not match:
        return None
    base, date, main, arch = match.groups()
    if (base == "0") != bool(main):
        return None
    validate_date(date)
    return ("main" if main else f"v{base}", date, arch)


def prepare_version_hook(repo: Path) -> None:
    """Inject the packaging override after every clean checkout, before building.

    The source operator code stays at the pinned commit. This small packaging
    hook affects both the wheel filename and its internally generated METADATA.
    A source reset removes the hook; every daily build installs it again.
    """
    path = repo / "scripts/fla_npu_artifacts.py"
    source = path.read_text(encoding="utf-8")
    marker = "# fla-npu-publish: daily version override v3 sha7 no-g"
    if marker in source:
        return
    tree = ast.parse(source)
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "get_package_version"]
    if len(functions) != 1:
        raise RuntimeError(f"源码版本接口已变化，拒绝注入: {path}")
    body = functions[0].body
    start = 1 if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str) else 0
    line = body[start].lineno - 1
    lines = source.splitlines(keepends=True)
    hook = (
        f"    {marker}\n"
        "    explicit = os.getenv('FLA_NPU_PACKAGE_VERSION', '').strip()\n"
        "    if explicit and os.getenv('FLA_PKG_BUILD_TYPE') == 'daily':\n"
        "        import re\n"
        "        if not re.fullmatch(r'(0|[0-9]+(?:\\.[0-9]+)+)\\.dev[0-9]{8}\\+(?:main\\.)?[0-9a-f]{7}', explicit):\n"
        "            raise ValueError('Invalid daily package version: ' + explicit)\n"
        "        return explicit\n"
    )
    updated = "".join(lines[:line]) + hook + "".join(lines[line:])
    ast.parse(updated)
    path.write_text(updated, encoding="utf-8")


def verify_wheel_version(path: Path, expected: str) -> None:
    if path.name.split("-")[1] != expected:
        raise RuntimeError(f"wheel 文件名版本不匹配: {path.name}, expected={expected}")
    with zipfile.ZipFile(path) as archive:
        metadata = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise RuntimeError(f"wheel METADATA 不唯一: {path}")
        message = email.parser.Parser().parsestr(archive.read(metadata[0]).decode('utf-8'))
        if message.get('Version') != expected:
            raise RuntimeError(f"wheel 内部 METADATA 版本不匹配: {path}")


def repair_wheel_version(source: Path, expected: str) -> Path:
    """Repackage metadata and RECORD without recompiling or changing payloads."""
    with source.with_suffix('.repair.lock').open('a') as lock:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        return _repair_wheel_locked(source, expected)


def _repair_wheel_locked(source: Path, expected: str) -> Path:
    original = source.name.split('-')[1]
    verify_wheel_version(source, original)
    if original == expected:
        return source
    target = source.with_name(source.name.replace('-' + original + '-', '-' + expected + '-', 1))
    if target.exists():
        verify_wheel_version(target, expected)
        return target
    temporary = target.with_suffix(f'.{os.getpid()}.whl.tmp')
    records = []
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(temporary, 'w') as after:
        # Reuse stored payload RECORD entries; no binary comparison or download.
        original_record = next(n for n in before.namelist() if n.endswith('.dist-info/RECORD'))
        stored = {row[0]: (row[1], row[2]) for row in csv.reader(io.StringIO(before.read(original_record).decode('utf-8'))) if len(row) == 3}
        for info in before.infolist():
            if info.filename.endswith(('.dist-info/RECORD', '.dist-info/RECORD.jws', '.dist-info/RECORD.p7s')):
                continue
            old_path = info.filename
            name = old_path.replace('-' + original + '.dist-info/', '-' + expected + '.dist-info/')
            name = name.replace('-' + original + '.data/', '-' + expected + '.data/')
            data = before.read(info.filename)
            if name.endswith('.dist-info/METADATA'):
                text = data.decode('utf-8')
                text, count = re.subn(r'(?m)^Version: [^\r\n]+', 'Version: ' + expected, text)
                if count != 1:
                    raise RuntimeError('METADATA Version header must be unique')
                data = text.encode('utf-8')
            info.filename = name
            after.writestr(info, data)
            if not name.endswith('.dist-info/METADATA') and old_path in stored:
                digest, size = stored[old_path]
            else:
                digest = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                size = str(len(data))
            records.append((name, digest, size))
        info_dir = next(name.rsplit('/', 1)[0] for name, _, _ in records if name.endswith('.dist-info/METADATA'))
        record_name = info_dir + '/RECORD'
        stream = io.StringIO(newline='')
        csv.writer(stream, lineterminator='\n').writerows(records + [(record_name, '', '')])
        after.writestr(record_name, stream.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    temporary.replace(target)
    verify_wheel_version(target, expected)
    return target


def migrate_manifest(manifest_path: Path) -> bool:
    """Migrate known previous daily labels; preserve all compiled payloads."""
    import json
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('build_type') != 'daily':
        return False
    branch, date, sha = manifest['branch'], manifest.get('daily_date', ''), manifest['head_sha']
    expected = version_for(branch, date, sha)
    legacy = manifest.get('package_version')
    if manifest.get('package_version') == expected:
        return False
    if legacy not in legacy_versions(branch, date, sha):
        raise RuntimeError('拒绝迁移未知 daily 命名格式')
    changes = []
    for result in manifest.get('results', []):
        if result.get('status') != 'success':
            continue
        if len(result.get('wheels', [])) != 1:
            raise RuntimeError('迁移要求每组合恰好一个 wheel')
        old = result['wheels'][0]
        if old.split('-')[1] != legacy:
            raise RuntimeError('迁移源与 manifest 版本不符')
        source = manifest_path.parent / 'wheels' / f"{result['soc']}-{result['arch']}" / old
        target = repair_wheel_version(source, expected)
        result['wheels'] = [target.name]
        changes.append({'old': old, 'new': target.name, 'soc': result['soc'], 'arch': result['arch']})
    manifest['package_version'] = expected
    manifest['version_scheme'] = 'daily-date-sha7-no-g-v2'
    manifest['renamed_wheels'] = changes
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(manifest_path)
    print(f'[version] migrated {len(changes)} wheel(s): {legacy} -> {expected}', flush=True)
    return True
