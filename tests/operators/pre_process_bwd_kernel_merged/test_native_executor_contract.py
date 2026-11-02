"""Observe actual installed CANN executor first errors and real CP propagation.

Safe failures: null executor, or null workspace with a valid executor and
workspaceSize=1. No allocator exhaustion, invalid memory, or kernel execution.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest

from test_host_fault_contract import compile_checked


class NativeExecutorContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build = Path(os.environ['DFX_BUILD_DIR'])
        cann = Path(os.environ['ASCEND_HOME_PATH'])
        root = Path(__file__).resolve().parents[3]
        commands = json.loads((build/'compile_commands.json').read_text())
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        objects, hashes = [], {}
        for index, name in enumerate(('aclnn_pre_process_bwd_kernel_merged.cpp', 'pre_process_bwd_kernel_merged.cpp')):
            entry = next(row for row in commands if row['file'].endswith('/op_api/'+name))
            raw = shlex.split(entry['command'])
            flags, position = [raw[0]], 1
            while position < len(raw):
                if raw[position] in ('-o', '-c'):
                    position += 2
                else:
                    flags.append(raw[position])
                    position += 1
            output = Path(cls.temp.name)/f'source{index}.o'
            compile_checked(flags+['-O0','-g','-fPIC','-c',entry['file'],'-o',str(output)], cwd=entry['directory'])
            objects.append(str(output))
            hashes[name] = hashlib.sha256(Path(entry['file']).read_bytes()).hexdigest()
        probe = Path(cls.temp.name)/'native.o'
        compile_checked(flags+['-O0','-g','-fPIC',f'-I{root}','-c',
            str(Path(__file__).with_name('native_executor_probe.cpp')),'-o',str(probe)], cwd=entry['directory'])
        library = Path(cls.temp.name)/'cp_native_executor.so'
        compile_checked([raw[0],'-shared','-Wl,-Bsymbolic-functions','-Wl,-z,defs', *objects, str(probe),
            f'-L{cann}/lib64',f'-Wl,-rpath,{cann}/lib64','-lnnopbase','-lopapi',
            '-lerror_manager','-lunified_dlog','-lascendalog','-lgraph','-lgraph_base','-lmetadef','-ldl',
            '-o',str(library)])
        cls.acl = ctypes.CDLL('libascendcl.so')
        cls.opapi = ctypes.CDLL('libopapi.so')
        cls.custom = ctypes.CDLL(str(build/'libcust_opapi.so'))
        cls.acl.aclInit.argtypes, cls.acl.aclInit.restype = [ctypes.c_char_p], ctypes.c_int
        initialized = cls.acl.aclInit(None)
        if initialized not in (0,100002):
            raise RuntimeError(f'aclInit failed: {initialized}')
        device = int(os.environ['DFX_DEVICE'])
        if cls.acl.aclrtSetDevice(device):
            raise RuntimeError('aclrtSetDevice failed')
        cls.addClassCleanup(cls.acl.aclrtResetDevice,device)
        cls.probe = ctypes.CDLL(str(library))
        owner = cls.probe.cp_native_executor_owner
        owner.restype = ctypes.c_char_p
        cls.owner = (owner() or b'').decode()
        if 'libnnopbase.so' not in cls.owner:
            raise AssertionError('CommonOpExecutorRun is not native nnopbase: '+cls.owner)
        cls.native_hash = hashlib.sha256(Path(cls.owner).read_bytes()).hexdigest()
        cls.source_hash = hashes
        cls.call = cls.probe.cp_native_executor_case
        cls.call.argtypes, cls.call.restype = [ctypes.c_int,ctypes.c_int],ctypes.c_int
        cls.recent = cls.acl.aclGetRecentErrMsg
        cls.recent.argtypes, cls.recent.restype = [],ctypes.c_char_p
        cls.evidence = []

    def test_native_first_error_and_l2_preservation(self):
        for scenario, parameter in ((0,'executor'),(1,'workspace')):
            for through in (0,1):
                with self.subTest(parameter=parameter, through_production_l2=bool(through)):
                    self.recent()
                    status = self.call(scenario,through)
                    message = (self.recent() or b'').decode(errors='replace')
                    row = dict(scenario=scenario, parameter=parameter, through_production_l2=bool(through),
                               status=status,message=message)
                    self.evidence.append(row)
                    print('NATIVE_EXECUTOR_EVIDENCE '+json.dumps(row,ensure_ascii=False),flush=True)
                    self.assertEqual(status,561000 if through else 161001,row)
                    self.assertIn(parameter,message)
                    self.assertTrue('nullptr' in message or 'NULL' in message or 'null' in message,row)
                    self.assertEqual(re.findall(r'\bE[A-Z0-9]\d{4}\b',message),['EZ1001'],row)
                    self.assertNotIn('CP_UT_SIMULATED_LOWER',message,row)

    @classmethod
    def tearDownClass(cls):
        output = os.environ.get('CP_NATIVE_EVIDENCE')
        if output:
            Path(output).write_text(json.dumps(dict(scope='native null-executor/null-workspace failures only; no wrappers',
                native_library=cls.owner,native_sha256=cls.native_hash,source_sha256=cls.source_hash,
                cases=cls.evidence),ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    unittest.main()
