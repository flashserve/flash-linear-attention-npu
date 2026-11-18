import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from daily_version import version_for, wheel_name, wheel_identity, prepare_version_hook, verify_wheel_version, repair_wheel_version, migrate_manifest
import daily_plan
import publish_latest
import verify_daily_public as public
import obs_publish


class VersionTests(unittest.TestCase):
    def test_main_is_independent_and_uses_first_seven(self):
        sha = 'abcdef01' + '9' * 24 + '00123456'
        self.assertEqual(version_for('main', '20261102', sha), '0.dev20261102+main.abcdef0')
        self.assertEqual(version_for('v26.10.0', '20261103', sha), '26.10.0.dev20261103+abcdef0')

    def test_requested_filename_has_only_seven_commit_characters(self):
        sha = 'ea56c7ae033494c74926edda1e1ad13bf39b66ad'
        self.assertEqual(wheel_name('flash_linear_attention_npu_a5', 'v26.10.0', '20261103', sha, 'x86_64'),
                         'flash_linear_attention_npu_a5-26.10.0.dev20261103+ea56c7a-py3-none-manylinux_2_34_x86_64.whl')
        self.assertEqual(wheel_identity('flash_linear_attention_npu_a5-26.10.0.dev20261103+gea56c7a-py3-none-manylinux_2_34_x86_64.whl', 'flash_linear_attention_npu_a5'),
                         ('v26.10.0', '20261103', 'x86_64'))
        self.assertIsNone(wheel_identity('flash_linear_attention_npu_a5-26.10.0.dev20261103+ea56c7ae-py3-none-manylinux_2_34_x86_64.whl', 'flash_linear_attention_npu_a5'))

    def test_invalid_identity_is_rejected(self):
        for branch, day, sha in [('main', '20260230', 'a' * 40), ('bad', '20261102', 'a' * 40),
                                 ('main', '20261102', 'a' * 8)]:
            with self.assertRaises(ValueError):
                version_for(branch, day, sha)

    def test_filename_and_metadata_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            version = version_for('main', '20261102', 'a' * 40)
            path = Path(tmp) / wheel_name('flash_linear_attention_npu_a2', 'main', '20261102', 'a' * 40, 'aarch64')
            with zipfile.ZipFile(path, 'w') as wheel:
                wheel.writestr(f'flash_linear_attention_npu_a2-{version}.dist-info/METADATA',
                               f'Name: flash-linear-attention-npu-a2\nVersion: {version}\n')
            verify_wheel_version(path, version)
            with self.assertRaises(RuntimeError):
                verify_wheel_version(path, '0.dev20261103+main.aaaaaaa')

    def test_hook_updates_packaging_version_and_preserves_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / 'scripts').mkdir()
            path = repo / 'scripts/fla_npu_artifacts.py'
            path.write_text('import os\nfrom pathlib import Path\ndef get_package_version(repo_root: Path) -> str:\n    """Version hook."""\n    return "26.10.0"\n')
            prepare_version_hook(repo)
            original = path.read_text()
            prepare_version_hook(repo)
            self.assertEqual(path.read_text(), original)
            spec = importlib.util.spec_from_file_location('test_artifacts', path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            with patch.dict(os.environ, {'FLA_PKG_BUILD_TYPE': 'daily', 'FLA_NPU_PACKAGE_VERSION': '0.dev20261102+main.aaaaaaa'}):
                self.assertEqual(mod.get_package_version(repo), '0.dev20261102+main.aaaaaaa')
            with patch.dict(os.environ, {'FLA_PKG_BUILD_TYPE': 'release', 'FLA_NPU_PACKAGE_VERSION': '0.dev20261102+main.aaaaaaa'}):
                self.assertEqual(mod.get_package_version(repo), '26.10.0')

    def test_repackage_updates_metadata_record_and_preserves_binary(self):
        import base64
        import csv
        import hashlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            old = '0.dev20261102+main.g12345678'
            new = '0.dev20261102+main.abcdef0'
            source = Path(tmp) / f'flash_linear_attention_npu_a2-{old}-py3-none-manylinux_2_34_aarch64.whl'
            prefix = f'flash_linear_attention_npu_a2-{old}.dist-info/'
            binary = b'unchanged-compiled-payload\x00\xff'
            with zipfile.ZipFile(source, 'w') as archive:
                archive.writestr(prefix + 'METADATA', f'Name: flash-linear-attention-npu-a2\nVersion: {old}\n')
                archive.writestr(prefix + 'WHEEL', 'Wheel-Version: 1.0\n')
                archive.writestr(prefix + 'RECORD', 'old record')
                archive.writestr('fla_npu/lib.so', binary)
            target = repair_wheel_version(source, new)
            self.assertTrue(source.exists())
            verify_wheel_version(target, new)
            with zipfile.ZipFile(target) as archive:
                self.assertEqual(archive.read('fla_npu/lib.so'), binary)
                self.assertFalse(any(old in name for name in archive.namelist()))
                record = next(name for name in archive.namelist() if name.endswith('/RECORD'))
                for name, digest, size in csv.reader(io.StringIO(archive.read(record).decode())):
                    if name == record:
                        continue
                    data = archive.read(name)
                    self.assertEqual(str(len(data)), size)
                    self.assertEqual('sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode(), digest)

    def test_manual_plan_uses_actual_branch_head(self):
        with patch.object(daily_plan, 'has_formal_release', return_value=(False, '')), \
                patch.object(daily_plan, 'request_json', return_value={'commit': {'sha': 'c' * 40}}), \
                patch.object(daily_plan, 'resolve_commit') as scheduled:
            plan = daily_plan.build_plan(dt.date.today(), 1, ['main'], 'a/b', 'https://api.test', '', latest=True)
        scheduled.assert_not_called()
        self.assertEqual(plan['branches'][0]['sha'], 'c' * 40)
        self.assertEqual(plan['resolution'], 'branch-head')

    def test_completed_g_prefix_manifest_is_relabelled_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            old_version = '26.10.0.dev20261102+gabcdef0'
            name = f'flash_linear_attention_npu_a5-{old_version}-py3-none-manylinux_2_34_x86_64.whl'
            path = out / 'wheels/ascend950-x86' / name
            path.parent.mkdir(parents=True)
            info = f'flash_linear_attention_npu_a5-{old_version}.dist-info/'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr(info + 'METADATA', f'Name: flash-linear-attention-npu-a5\nVersion: {old_version}\n')
                archive.writestr(info + 'RECORD', '')
            manifest = out / 'manifest.json'
            manifest.write_text(json.dumps(dict(build_type='daily', branch='v26.10.0', daily_date='20261102',
                                                head_sha='abcdef0' + '1' * 33, package_version=old_version,
                                                results=[dict(status='success', soc='ascend950', arch='x86', wheels=[name])])))
            self.assertTrue(migrate_manifest(manifest))
            renamed = json.loads(manifest.read_text())
            self.assertEqual(renamed['package_version'], '26.10.0.dev20261102+abcdef0')
            verify_wheel_version(path.parent / renamed['results'][0]['wheels'][0], renamed['package_version'])
            self.assertFalse(migrate_manifest(manifest))


class LatestTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(schema=daily_plan.SCHEMA, date='20261102', cutoff_utc='2026-11-02T08:00:00Z',
                         branches=[dict(branch='main', sha='a' * 40), dict(branch='v26.10.0', sha='b' * 40)])
        self.rows = public.expected_rows(self.plan)
        self.row = self.rows[0]
        self.package = self.row['package']
        self.prefix = f'{publish_latest.PREFIX}/{self.package.replace("_", "-")}/latest'

    def name(self, branch='main', day='20261102', sha='a' * 40, arch='aarch64'):
        return wheel_name(self.package, branch, day, sha, arch)

    def test_only_same_branch_same_arch_can_be_pruned(self):
        old = self.prefix + '/' + self.name(day='20261101')
        remote = {self.prefix + '/' + name: 1 for name in [self.name(), self.name(day='20261101'),
                  self.name(arch='x86_64', day='20261101'), self.name(branch='v26.10.0', day='20261101'),
                  'unrelated.whl', 'nested/' + self.name(day='20261101')]}
        self.assertEqual(publish_latest.stale_keys(remote, self.prefix, [self.row]), [old])

    def test_newer_daily_blocks_cleanup(self):
        remote = {self.prefix + '/' + self.name(day='20261103'): 1}
        with self.assertRaisesRegex(RuntimeError, '更晚日期'):
            publish_latest.stale_keys(remote, self.prefix, [self.row])

    def test_missing_new_public_wheel_never_deletes_old(self):
        with patch.object(publish_latest, 'fetch', return_value=(404, b'')), \
                patch.object(obs_publish, 'delete_one') as delete, \
                patch.object(obs_publish, 'list_all') as listing:
            self.assertIsNone(publish_latest.promote_one(object(), 'bucket', self.row))
        delete.assert_not_called()
        listing.assert_not_called()

    def test_new_public_wheel_must_also_exist_in_obs_before_delete(self):
        remote = {self.prefix + '/' + self.name(day='20261101'): 1}
        with patch.object(publish_latest, 'fetch', return_value=(206, b'a')), \
                patch.object(obs_publish, 'list_all', return_value=(200, '', remote)), \
                patch.object(obs_publish, 'delete_one') as delete:
            self.assertIsNone(publish_latest.promote_one(object(), 'bucket', self.row))
        delete.assert_not_called()

    def test_successful_slot_cleans_old_even_if_other_slots_missing(self):
        from types import SimpleNamespace
        old = self.prefix + '/' + self.name(day='20261101')
        remote = {self.prefix + '/' + self.name(): 1, old: 1}
        def listing(client, bucket, prefix):
            if prefix == self.prefix + '/':
                return 200, '', remote
            return 200, '', {prefix + self.name(): 1}
        with patch.object(publish_latest, 'fetch', return_value=(206, b'a')), \
                patch.object(obs_publish, 'list_all', side_effect=listing), \
                patch.object(obs_publish, 'delete_one', return_value=SimpleNamespace(status='deleted')) as delete:
            self.assertEqual(publish_latest.promote_one(object(), 'bucket', self.row), [old])
        self.assertEqual(delete.call_count, 1)
        self.assertEqual(delete.call_args.args[2], old)

    def test_same_day_old_plan_cannot_overwrite_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.json'
            path.write_text(json.dumps(dict(self.plan, cutoff_utc='2026-11-02T09:00:00Z')))
            with self.assertRaisesRegex(RuntimeError, '早于'):
                publish_latest.check_plan(self.plan, 'arm', path)

    def test_wrong_sha_tail_is_removed_only_from_latest_and_preserved_in_history(self):
        from types import SimpleNamespace
        old_name = self.name().replace('main.aaaaaaa-', 'main.gaaaaaaaa-')
        history_prefix = self.prefix.removesuffix('/latest')
        wrong_latest = self.prefix + '/' + old_name
        wrong_history = history_prefix + '/' + old_name
        unrelated_history = history_prefix + '/' + self.name(day='20261101')
        def listing(client, bucket, prefix):
            if prefix == self.prefix + '/':
                return 200, '', {self.prefix + '/' + self.name(): 1, wrong_latest: 1}
            return 200, '', {history_prefix + '/' + self.name(): 1, wrong_history: 1, unrelated_history: 1}
        with patch.object(publish_latest, 'fetch', return_value=(206, b'a')), \
                patch.object(obs_publish, 'list_all', side_effect=listing), \
                patch.object(obs_publish, 'delete_one', return_value=SimpleNamespace(status='deleted')) as delete:
            self.assertEqual(set(publish_latest.promote_one(object(), 'bucket', self.row)), {wrong_latest})
        self.assertEqual({call.args[2] for call in delete.call_args_list}, {wrong_latest})

    def test_cleanup_rejects_any_candidate_outside_latest(self):
        history_prefix = self.prefix.removesuffix('/latest')
        history_key = history_prefix + '/' + self.name(day='20261101')
        def listing(client, bucket, prefix):
            return 200, '', {prefix + self.name(): 1}
        with patch.object(publish_latest, 'fetch', return_value=(206, b'a')), \
                patch.object(obs_publish, 'list_all', side_effect=listing), \
                patch.object(publish_latest, 'stale_keys', return_value=[history_key]), \
                patch.object(obs_publish, 'delete_one') as delete:
            with self.assertRaisesRegex(RuntimeError, 'latest 目录以外'):
                publish_latest.promote_one(object(), 'bucket', self.row)
        delete.assert_not_called()

    def test_same_day_older_commit_removed_only_for_matching_branch_and_arch(self):
        old = self.prefix + '/' + self.name(sha='b' * 40)
        remote = {self.prefix + '/' + name: 1 for name in [self.name(), self.name(sha='b' * 40),
                  self.name(arch='x86_64', sha='b' * 40), self.name(branch='v26.10.0', sha='b' * 40)]}
        self.assertEqual(publish_latest.stale_keys(remote, self.prefix, [self.row]), [old])

    def test_rebuild_requires_new_reports_for_both_locations(self):
        import time
        with tempfile.TemporaryDirectory() as tmp, patch.object(publish_latest, 'BASE', Path(tmp)):
            out = Path(tmp) / 'out/daily' / self.row['branch']
            out.mkdir(parents=True)
            started = time.time() - 1
            self.assertFalse(publish_latest.uploaded_this_run(self.row, started))
            for suffix in ('', '-latest'):
                report = out / f"obs-daily-{self.package}-{self.row['arch']}-incremental{suffix}.json"
                key = f"{publish_latest.PREFIX}/{self.package.replace('_', '-')}/" + ('latest/' if suffix else '') + self.name()
                report.write_text(json.dumps({'results': [{'key': key, 'status': 'uploaded'}]}))
                if not suffix:
                    self.assertFalse(publish_latest.uploaded_this_run(self.row, started))
            self.assertTrue(publish_latest.uploaded_this_run(self.row, started))
            os.utime(report, (started - 10, started - 10))
            self.assertFalse(publish_latest.uploaded_this_run(self.row, started))

    def test_verify_checks_date_sha_branch_and_all_24_locations(self):
        self.assertEqual(len(self.rows), 12)
        self.assertEqual(public.match_filename(self.row, {self.name(day='20261101')}), '')
        self.assertEqual(public.match_filename(self.row, {self.name(sha='b' * 40)}), '')
        def directory(base, package, latest=False):
            return 200, {wheel_name(package, row['branch'], row['date'], row['sha'], row['wheel_arch'])
                         for row in self.rows if row['package'] == package}
        with patch.object(public, 'directory_files', side_effect=directory), \
                patch.object(public, 'fetch', return_value=(206, b'a')):
            results = public.check_once(self.plan, 'https://mirror.test')
        self.assertEqual(len(results), 24)
        self.assertTrue(all(row['visible'] for row in results))

    def test_retired_branch_keeps_last_daily_without_blocking_active_branch(self):
        plan = dict(self.plan, branches=[self.plan['branches'][0]])
        def directory(base, package, latest=False):
            return 200, {wheel_name(package, row['branch'], row['date'], row['sha'], row['wheel_arch'])
                         for row in self.rows if row['package'] == package}
        with patch.object(public, 'directory_files', side_effect=directory), \
                patch.object(public, 'fetch', return_value=(206, b'a')):
            self.assertTrue(all(row['visible'] for row in public.check_once(plan, 'https://mirror.test')))

    def test_same_slot_old_latest_blocks_public_success_until_sync(self):
        def directory(base, package, latest=False):
            names = {wheel_name(package, row['branch'], row['date'], row['sha'], row['wheel_arch'])
                     for row in self.rows if row['package'] == package}
            if latest:
                names.add(wheel_name(package, 'main', '20261101', 'a' * 40, 'aarch64'))
            return 200, names
        with patch.object(public, 'directory_files', side_effect=directory), \
                patch.object(public, 'fetch', return_value=(206, b'a')):
            results = public.check_once(self.plan, 'https://mirror.test')
        self.assertEqual(sum(row['visible'] for row in results), 21)


if __name__ == '__main__':
    unittest.main()
