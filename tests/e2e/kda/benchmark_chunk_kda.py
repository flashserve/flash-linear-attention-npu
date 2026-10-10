#!/usr/bin/env python3
"""Performance benchmark for chunk_kda_fwd, matching ATK Case 250 shape."""
import sys, time, json
from pathlib import Path

# 本脚本位于 <repo>/tests/e2e/kda/，据此定位仓库的 torch_custom
_torch_custom = Path(__file__).resolve().parents[3] / 'torch_custom'
if str(_torch_custom) not in sys.path:
    sys.path.insert(0, str(_torch_custom))

import torch, torch_npu
from fla_npu.ops.ascendc import chunk_kda_fwd

torch.npu.set_device(0)
device = torch.device("npu:0")

CASE_250 = {'B':1,'T':1024,'H':96,'K':128,'V':128,'chunk':64,'q_s':0.05,'g_s':0.125,'b_s':0.35}

def run_case(params, warmup=5, runs=20):
    B,T,H,K,V,ch = params['B'],params['T'],params['H'],params['K'],params['V'],params['chunk']
    sc = K**-0.5
    torch.manual_seed(20260812)
    q = (torch.randn(B,T,H,K)*params['q_s']).to(torch.bfloat16).to(device)
    k = (torch.randn(B,T,H,K)*params['q_s']).to(torch.bfloat16).to(device)
    v = (torch.randn(B,T,H,V)*params['q_s']).to(torch.bfloat16).to(device)
    g = (torch.randn(B,T,H,K)*params['g_s']).to(torch.float32).to(device)
    bt = (torch.randn(B,T,H)*params['b_s']).to(torch.float32).to(device)
    
    for _ in range(warmup):
        chunk_kda_fwd(q=q,k=k,v=v,g=g,beta=bt,scale=sc,chunk_size=ch,layout="BSND",safe_gate=True,lower_bound=-5.0)
    torch.npu.synchronize()
    
    ts = []
    for _ in range(runs):
        torch.npu.synchronize()
        t0 = time.perf_counter_ns()
        chunk_kda_fwd(q=q,k=k,v=v,g=g,beta=bt,scale=sc,chunk_size=ch,layout="BSND",safe_gate=True,lower_bound=-5.0)
        torch.npu.synchronize()
        ts.append((time.perf_counter_ns()-t0)/1e3)
    
    st = ts[3:]  # skip first 3 for stabilization
    av = sum(st)/len(st)
    sd = (sum((t-av)**2 for t in st)/len(st))**0.5
    return {'avg': round(av,2), 'min': round(min(st),2), 'max': round(max(st),2), 'std': round(sd,2)}

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--case', default='250')
    args = p.parse_args()
    
    cases = {'250': CASE_250, 'K3': {**CASE_250, 'H':8, 'q_s':0.1, 'g_s':0.01, 'b_s':0.1}}
    params = cases.get(args.case, CASE_250)
    
    r = run_case(params)
    print(f"Case {args.case}: B={params['B']} T={params['T']} H={params['H']} K={params['K']} V={params['V']}")
    print(f"  Avg:{r['avg']:.2f}us Min:{r['min']:.2f}us Max:{r['max']:.2f}us Std:{r['std']:.2f}us")
    
    if args.case == '250':
        ref = 1323.7
        print(f"  Benchmark: {ref}us | Improvement: {(1-r['avg']/ref)*100:.1f}%")
    print(f"  Std%: {r['std']/r['avg']*100:.1f}%")
    
    out = str(Path(__file__).resolve().parent / f'benchmark_case_{args.case}.json')
    with open(out, 'w') as f:
        json.dump({'case': args.case, 'results': r}, f, indent=2)
    print(f"  Saved to {out}")

if __name__ == '__main__':
    main()
