"""Exercise portable entrypoints without Docker, NPU, network or real secrets."""
import base64
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import daily_plan
import obs_publish

ROOT = Path(__file__).resolve().parent


def isolated_env(**updates):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('FLA_', 'OBS_', 'GITHUB_'))
           and 'proxy' not in key.lower()}
    env.update(updates)
    return env


@unittest.skipUnless(shutil.which('bash') and os.name == 'posix', 'Linux entrypoint tests')
class CredentialTests(unittest.TestCase):
    def call(self, env, mode=''):
        script = '''set -euo pipefail
BASE=$1
source "$BASE/publish_env.sh"
load_publish_env "$2"
"$3" -c 'import os,json; print(json.dumps({k:v for k,v in os.environ.items() if k.startswith(("OBS_", "GITHUB_", "FLA_PKG_GH_TOKEN"))}))'
'''
        return subprocess.run(['bash','-c',script,'bash',str(ROOT),mode,sys.executable],
                              env=isolated_env(**env),capture_output=True,text=True)

    def test_process_credentials_work_without_a_file(self):
        result = self.call({'OBS_ACCESS_KEY':'fixture-ak','OBS_SECRET_KEY':'fixture-sk'})
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['OBS_SECRET_KEY'],'fixture-sk')

    def test_legacy_encoded_config_is_loaded_then_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'credentials.env'
            encoded=base64.b64encode(b'fixture-token').decode()
            config.write_text(f'OBS_ACCESS_KEY=fixture-ak\nGITHUB_TOKEN_B64={encoded}\n')
            result=self.call({'FLA_PKG_ENV_FILE':str(config)})
        self.assertEqual(result.returncode,0,result.stderr)
        values=json.loads(result.stdout)
        self.assertEqual(values['GITHUB_TOKEN'],'fixture-token')
        self.assertNotIn('GITHUB_TOKEN_B64',values)

    def test_watcher_removes_github_credentials(self):
        result=self.call({'GITHUB_TOKEN':'fixture-token','FLA_PKG_GH_TOKEN':'fixture-old',
                          'GITHUB_TOKEN_B64':base64.b64encode(b'fixture-encoded').decode(),
                          'OBS_ACCESS_KEY':'fixture-ak'},'obs-only')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout),{'OBS_ACCESS_KEY':'fixture-ak'})

    def test_explicit_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=self.call({'FLA_PKG_ENV_FILE':str(Path(tmp)/'missing')})
        self.assertNotEqual(result.returncode,0)
        self.assertIn('unreadable',result.stderr)

    def test_invalid_encoding_fails_without_printing_value(self):
        result=self.call({'OBS_SECRET_KEY_B64':'fixture-invalid-***'})
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn('fixture-invalid',result.stdout+result.stderr)


@unittest.skipUnless(shutil.which('bash') and os.name == 'posix', 'Linux entrypoint tests')
class EntrypointTests(unittest.TestCase):
    def run_daily(self, build_rc):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'publish'
            shutil.copytree(ROOT,root,ignore=shutil.ignore_patterns('__pycache__','.venv','out','repos'))
            (root/'.publish.env').write_text(
                'OBS_ACCESS_KEY=fixture-ak\nOBS_SECRET_KEY=fixture-sk\nGITHUB_TOKEN=fixture-token\n')
            (root/'run_compile.sh').write_text(f'''#!/bin/bash
set -eu
[[ -z "${{GITHUB_TOKEN:-}}" && -z "${{GITHUB_TOKEN_B64:-}}" ]]
[[ "$OBS_ACCESS_KEY" == fixture-ak ]]
printf '%s\\n' "$@" > build-arguments.txt
exit {build_rc}
''')
            (root/'run_compile.sh').chmod(0o755)
            (root/'watch_latest.sh').chmod(0o755)
            (root/'publish_latest.py').write_text('''import os,sys
from pathlib import Path
if '--check-plan' not in sys.argv:
    assert not os.getenv('GITHUB_TOKEN')
    assert not os.getenv('GITHUB_TOKEN_B64')
    assert os.environ['OBS_SECRET_KEY']=='fixture-sk'
    Path('watcher-ok').touch()
''')
            (root/'verify_daily_public.py').write_text("from pathlib import Path\nPath('public-ok').touch()\n")
            day=dt.datetime.now(daily_plan.TIMEZONE).strftime('%Y%m%d')
            plan=root/'plan.json'
            plan.write_text(json.dumps({'schema':daily_plan.SCHEMA,'date':day,
                'branches':[{'branch':'main','sha':'a'*40}]}))
            result=subprocess.run(['bash',str(root/'run_daily.sh'),'--plan',str(plan)],
                env=isolated_env(FLA_PKG_PYTHON=sys.executable),capture_output=True,text=True)
            self.assertEqual(result.returncode,build_rc,(root/'logs/cron-daily.log').read_text())
            self.assertTrue((root/'watcher-ok').exists())
            self.assertEqual((root/'public-ok').exists(),build_rc==0)
            self.assertIn('main='+'a'*40,(root/'build-arguments.txt').read_text())

    def test_pinned_daily_clears_token_and_runs_public_verifier(self):
        self.run_daily(0)

    def test_failed_build_does_not_report_public_success(self):
        self.run_daily(1)

    def test_build_only_plan_needs_no_credentials_or_proxy(self):
        result=subprocess.run(['bash',str(ROOT/'run_compile.sh'),'--branch','main',
            '--branch-sha','main='+'a'*40,'--build-only','--plan'],
            env=isolated_env(FLA_PKG_PYTHON=sys.executable),capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('publish=0',result.stdout)


class DockerConfigTests(unittest.TestCase):
    def test_image_overrides_apply_to_both_architectures(self):
        for arch,variable in [('arm','FLA_PKG_DOCKER_IMAGE'),('x86','FLA_PKG_X86_DOCKER_IMAGE')]:
            result=subprocess.check_output([sys.executable,str(ROOT/'docker_config.py'),arch],
                env=isolated_env(**{variable:'test-ci:immutable'}),text=True)
            self.assertEqual(result.strip(),'test-ci:immutable')

    def test_raw_obs_cli_requires_explicit_destination(self):
        with self.assertRaises(SystemExit) as error:
            obs_publish.parse_args(['--upload'])
        self.assertEqual(error.exception.code,2)


if __name__ == '__main__':
    unittest.main()
