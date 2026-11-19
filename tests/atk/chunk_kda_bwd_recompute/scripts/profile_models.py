import argparse,ctypes,json,sys,os
from pathlib import Path
import torch,torch_npu
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root/'tests/atk/chunk_kda_bwd_recompute'))
from reference import build_inputs
from fla_npu.ops.ascendc import chunk_kda_bwd_recompute
p=argparse.ArgumentParser();p.add_argument('--case',choices=['F','V'],required=True);p.add_argument('--round',type=int,default=0);p.add_argument('--output',type=Path,default=Path('profile_metadata'));a=p.parse_args()
torch.set_num_threads(4);torch.npu.set_device(0)
acl=ctypes.CDLL('libascendcl.so');acl.aclrtGetDeviceInfo.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.POINTER(ctypes.c_int64)]
cores=ctypes.c_int64();status=acl.aclrtGetDeviceInfo(0,102,ctypes.byref(cores));assert status==0 and cores.value>0
T=11264 if a.case=='F' else 16384
s=dict(dtype='bf16',B=1,HK=16,HV=128,T=T,K=128,V=128,chunk_size=64,g_dtype='bf16',beta_dtype='bf16',use_gate=True,use_exp2=True,has_dt_bias=True,lower_bound=-5.,seed=20261119)
if a.case=='V':s.update(variable=True,lengths=[1024]*16)
cpu=build_inputs(s,torch.device('cpu'));d={k:v.to('npu') if isinstance(v,torch.Tensor) else v for k,v in cpu.items()};del cpu
cu=torch.tensor(d['cu_seqlens'],dtype=torch.int64) if 'cu_seqlens' in d else None
ci=torch.tensor(d['chunk_indices'],dtype=torch.int64) if 'chunk_indices' in d else None
torch.npu.synchronize();baseline=torch.npu.memory_allocated();torch.npu.reset_peak_memory_stats()
for i in range(120):
    out=chunk_kda_bwd_recompute(d['q'],d['k'],d['v'],d['g'],d['beta'],d['A'],64,A_log=d['A_log'],dt_bias=d['dt_bias'],cu_seqlens=cu,chunk_indices=ci,use_gate_in_kernel=True,use_exp2=True,lower_bound=-5.)
    torch.npu.synchronize()
    del out
peak=torch.npu.max_memory_allocated();required_output=1*128*T*128*(4+2*4)
record=dict(case=a.case,round=a.round,spec=s,queried_cube_cores=cores.value,HEAD_MODE='GVA',CG=4,warmup=20,measured=100,required_output_bytes=required_output,input_allocated_bytes=baseline,peak_framework_allocated_bytes=peak,framework_extra_over_input_and_output=peak-baseline-required_output,user_workspace_bytes=cores.value*2*4*32768,device=torch.npu.get_device_name(0),visible_devices=os.getenv('ASCEND_RT_VISIBLE_DEVICES'))
output=a.output;output.mkdir(parents=True,exist_ok=True)
(output/f'public_perf_{a.case}_{a.round}.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
