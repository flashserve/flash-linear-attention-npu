#!/usr/bin/env python3
"""Run the frozen Prepare cases through either ATK backend and check reports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import openpyxl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('npu', 'pyaclnn'), required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=200)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    op = Path(__file__).resolve().parents[1]
    cases = json.loads((op / 'atk_chunk_kda_fwd_prepare.json').read_text())
    if not 0 <= args.start < args.end <= len(cases):
        parser.error('invalid case range')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for case in cases:
        # Empty regex matches every exception in ATK. These are positive cases.
        if case.get('expected_error_msg') == '':
            case['expected_error_msg'] = None
        if args.backend == 'pyaclnn':
            case['aclnn_api_type'] = 'aclnn_chunk_kda_fwd_prepare'
    config = out / 'cases.json'
    config.write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    env = os.environ.copy()
    if args.backend == 'pyaclnn':
        import fla_npu
        symbols = ('aclnnChunkKdaFwdPrepareGetWorkspaceSize', 'aclnnChunkKdaFwdPrepare')
        for lib in fla_npu.load_ascendc_opapi_libraries():
            if all(hasattr(lib, s) for s in symbols):
                env.update(os.environ)
                env['ATK_CUSTOM_OPP_PATH'] = str(lib._name)
                break
        else:
            raise RuntimeError('No selected FLA vendor exports both Prepare ACLNN stages')
    command = [sys.executable, '-m', 'atk', 'node', '--backend', args.backend, '--devices', str(args.device),
               '-o', str(out / 'accuracy'), 'node', '--backend', 'cpu', 'task',
               '-c', str(config), '--task', 'accuracy', '-p', str(op / 'executor_chunk_kda_fwd_prepare.py')]
    if args.backend == 'pyaclnn':
        command += ['-p', str(op / 'aclnn_executor.py')]
    command += ['-s', str(args.start), '-e', str(args.end), '-sp', '-to', '120']
    (out / 'command.json').write_text(json.dumps({'argv': command,
        'library': env.get('ATK_CUSTOM_OPP_PATH'), 'backend': args.backend}, indent=2))
    result = subprocess.run(command, cwd=op, env=env)
    reports = list((out / 'accuracy').rglob('*.xlsx'))
    if len(reports) != 1:
        raise RuntimeError(f'Expected one precision report; got {reports}')
    wb = openpyxl.load_workbook(reports[0], read_only=True, data_only=True)
    rows = list(wb['summary'].values)
    records = [dict(zip(rows[0], row)) for row in rows[1:]]
    detail_rows = list(wb['statistic'].values)
    details = [dict(zip(detail_rows[0], row)) for row in detail_rows[1:]]
    wb.close()
    (out / 'summary.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    expected = args.end - args.start
    ids = [r['编号'] for r in details]
    if sorted(ids) != sorted(case['id'] for case in cases[args.start:args.end]):
        raise RuntimeError(f'Incomplete or duplicate case IDs: {ids}')
    if result.returncode or not records or not all(
        int(r['总用例数']) == expected and int(r['通过用例个数']) == expected
        and int(r['执行失败用例个数']) == 0 and r['精度是否达标'] == 'Pass'
        for r in records
    ):
        raise RuntimeError(f'ATK accuracy failed: exit={result.returncode}, summary={records}')
    print(f'{args.backend}: {expected}/{expected} accuracy PASS; {reports[0]}')


if __name__ == '__main__':
    main()
