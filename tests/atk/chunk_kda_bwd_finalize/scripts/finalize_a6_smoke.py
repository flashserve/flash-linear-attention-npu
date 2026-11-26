# Copyright (c) 2026 Tianjin University, Ltd.
"""A6 development smoke: nonzero FP64 reference, seven outputs, two launches.

Default mode reports errors without claiming accuracy acceptance. --rtol/--atol
must be supplied together if using externally confirmed mixed tolerances.
"""
import argparse
import math
import os
from pathlib import Path
import time

os.environ.setdefault('TBE_PARALLEL_COMPILE_ENABLE', '0')
os.environ.setdefault('PARALLEL_COMPILE', '0')
import torch
from a6_reference import finalize_chunk

NAMES = ('dq', 'dk', 'dv', 'd_beta', 'd_g', 'd_a_log', 'd_dt_bias')


def build_case(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    b, heads, tokens, dim = args.batch, args.heads, args.tokens, 128
    if args.packed and b != 1:
        raise ValueError('packed smoke requires --batch=1')
    shape = (b, heads, tokens, dim)
    bf16 = torch.bfloat16
    gate_type = bf16 if args.raw_bf16 else torch.float32
    beta_type = bf16 if args.beta_bf16 else torch.float32
    x = {name: (torch.randn(shape)*.1).to(bf16) for name in ('q','k','v','v_new','dv_scan')}
    x.update(gk=torch.zeros(shape), raw_g=(torch.randn(shape)*.1-.5).to(gate_type),
             beta=(torch.rand(b,heads,tokens)*.2+.1).to(beta_type),
             a_log=torch.full((heads,),-1.), dt_bias=torch.randn(heads,dim)*.1,
             d_aqk=torch.randn(b,heads,tokens,64)*.01, dq_raw=torch.randn(shape)*.1)
    kwargs = dict(scale=dim**-.5, lower_bound=-.1, output_layout=(
        'TND' if args.token_major else 'NTD') if args.packed else ('BSND' if args.token_major else 'BNSD'))
    for name in ('q','k'):
        if args.rstd:
            rstd = x[name].float().square().sum(-1).clamp_min(1e-8).rsqrt()
            x[name] = (x[name].float()*rstd[...,None]).to(bf16)
            kwargs[f'{name}_rstd'] = rstd.squeeze(0) if args.packed else rstd
    work = []
    if args.packed:
        cu = [0,0,min(33,tokens),tokens]
        indices = []
        for seq in range(len(cu)-1):
            for local, start in enumerate(range(cu[seq],cu[seq+1],64)):
                work.append((0,start,min(64,cu[seq+1]-start),len(work)))
                indices += [seq,local]
        kwargs.update(cu_seqlens=cu,chunk_indices=indices)
        state_shape = (len(work),heads,dim,dim)
    else:
        chunks = (tokens+63)//64
        for batch in range(b):
            for local, start in enumerate(range(0,tokens,64)):
                work.append((batch,start,min(64,tokens-start),local))
        state_shape = (b,chunks,heads,dim,dim)
    x['h'],x['dh'] = [(torch.randn(state_shape)*.1).to(bf16) for _ in range(2)]
    x['akk'] = torch.zeros(b,heads,tokens,64,dtype=bf16)
    expected = [torch.zeros(shape,dtype=torch.float64) for _ in range(3)]
    expected += [torch.zeros(b,heads,tokens,dtype=torch.float64),torch.zeros(shape,dtype=torch.float64),
                 torch.zeros(heads,dtype=torch.float64),torch.zeros(heads,dim,dtype=torch.float64)]
    for batch,start,rows,state in work:
        for head in range(heads):
            select = (batch,head,slice(start,start+rows))
            raw = x['raw_g'][select].double()
            natural = (kwargs['lower_bound']*torch.sigmoid(x['a_log'][head].double().exp()*(raw+x['dt_bias'][head].double()))).cumsum(0)
            x['gk'][select] = (natural/math.log(2)).float()
            # Consistent saved triangular inverse, quantized once to BF16.
            k = x['k'][select].double()
            kernel = torch.eye(rows,dtype=torch.float64)
            for i in range(rows):
                kernel[i,:i] = x['beta'][batch,head,start+i].double()*(k[i]*k[:i]*(natural[i]-natural[:i]).exp()).sum(-1)
            x['akk'][batch,head,start:start+rows,:rows] = torch.linalg.inv(kernel).to(bf16)
            h_index = (state,head) if args.packed else (batch,state,head)
            ref = finalize_chunk(*(x[n][select] for n in ('q','k','v','gk','raw_g')),
                x['beta'][batch,head,start:start+rows],x['a_log'][head],x['dt_bias'][head],
                x['akk'][batch,head,start:start+rows,:rows],x['v_new'][select],x['h'][h_index],x['dh'][h_index],
                x['dv_scan'][select],x['d_aqk'][batch,head,start:start+rows,:rows],x['dq_raw'][select],
                kwargs['scale'],kwargs['lower_bound'],
                None if not args.rstd else kwargs['q_rstd'][(head,slice(start,start+rows)) if args.packed else select[:2]+(select[2],)],
                None if not args.rstd else kwargs['k_rstd'][(head,slice(start,start+rows)) if args.packed else select[:2]+(select[2],)])
            for i in (0,1,2,4): expected[i][select] = ref[i]
            expected[3][batch,head,start:start+rows] = ref[3]
            expected[5][head] += ref[5]
            expected[6][head] += ref[6]
    if args.packed:
        for name in list(x):
            if name not in ('h','dh','a_log','dt_bias'): x[name]=x[name].squeeze(0)
        for i in range(5): expected[i]=expected[i].squeeze(0)
    if args.token_major:
        for i in range(5):
            first = 0 if args.packed else 1
            expected[i]=expected[i].transpose(first,first+1).contiguous()
    return x,kwargs,expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag,default in (('batch',1),('heads',2),('tokens',64),('seed',42)):
        parser.add_argument('--'+flag,type=int,default=default)
    for flag in ('packed','rstd','token-major','raw-bf16','beta-bf16','cpu-only'):
        parser.add_argument('--'+flag,action='store_true')
    parser.add_argument('--rtol',type=float)
    parser.add_argument('--atol',type=float)
    parser.add_argument('--dump',default='/tmp/finalize_a6_smoke.pt')
    args=parser.parse_args()
    if min(args.batch,args.heads,args.tokens)<=0: parser.error('B/H/T must be positive')
    if (args.rtol is None)!=(args.atol is None): parser.error('provide both --rtol and --atol')
    print('[CPU] computing FP64 reference',flush=True)
    inputs,kwargs,reference=build_case(args)
    for tensor in reference: assert torch.isfinite(tensor).all()
    actual=[]
    if not args.cpu_only:
        import torch_npu  # Explicit device runner; CPU reference needs no NPU package.
        from fla_npu.ops.ascendc import npu_chunk_kda_bwd_finalize
        torch.npu.config.allow_internal_format=False
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.set_device(0)
        device_inputs={n:x.npu() for n,x in inputs.items()}
        device_kwargs={n:x.npu() if isinstance(x,torch.Tensor) else x for n,x in kwargs.items()}
        torch.npu.synchronize()
        for launch in range(2):
            print(f'[NPU] launch {launch+1}/2; waiting for camodel',flush=True)
            start=time.perf_counter()
            outputs=npu_chunk_kda_bwd_finalize(**device_inputs,**device_kwargs)
            torch.npu.synchronize()
            assert len(outputs)==7
            actual.append([x.detach().cpu() for x in outputs])
            print(f'[NPU] completed in {time.perf_counter()-start:.1f}s',flush=True)
    torch.save(dict(inputs=inputs,kwargs=kwargs,reference=dict(zip(NAMES,reference)),actual=actual),args.dump)
    for i,name in enumerate(NAMES):
        if args.cpu_only: continue
        got,want=actual[0][i].double(),reference[i]
        assert got.shape==want.shape and torch.isfinite(got).all(),name
        assert torch.equal(actual[0][i],actual[1][i]),f'{name}: repeated launch differs'
        delta=got-want
        print(f'[CMP] {name:10s} shape={tuple(got.shape)} max_abs={delta.abs().max():.6g} '
              f'rmse={delta.square().mean().sqrt():.6g} rel_l2={delta.norm()/want.norm().clamp_min(1e-12):.6g}',flush=True)
        if args.rtol is not None:
            torch.testing.assert_close(got,want,rtol=args.rtol,atol=args.atol,equal_nan=False)
    print(f'[DONE] {"CPU-only" if args.cpu_only else "diagnostic smoke"}; saved to {args.dump}',flush=True)


if __name__=='__main__':
    main()
