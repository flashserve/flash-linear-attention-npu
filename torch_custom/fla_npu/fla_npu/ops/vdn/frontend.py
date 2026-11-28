"""VDN from per-frame means and projected text inputs (inference, FP32 Newton)."""
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
import sys
import zlib

import torch
import torch.nn.functional as F

from . import _manual
from .forward import _default_frame_tile, vdn_forward


@lru_cache(maxsize=128)
def _host_tables(frames, heads, width, length, soc):
    path=Path(__file__).with_name("manual_tiling") / (soc+"-frontend.json.z")
    if path.exists():
        presets=json.loads(zlib.decompress(path.read_bytes()))
        keys=[f"down-{frames}-{width}",f"up-{frames}-{heads}"]
        if length:
            keys.extend((f"a-{heads}-{length}",f"b-{heads}-{length}"))
        if all(key in presets for key in keys):
            return [presets[keys[0]],presets[keys[1]],"".join(presets[k] for k in keys[2:])]
    # OPP stores the active SoC in process-global state. Isolate it from model execution.
    worker = Path(__file__).with_name("_frontend_tiling.py")
    result = subprocess.run([sys.executable,str(worker),*map(str,(frames,heads,width,length,soc))],
                            capture_output=True,text=True)
    if result.returncode:
        return None
    return json.loads(result.stdout.strip().splitlines()[-1])


@lru_cache(maxsize=16)
def _device_tables(device, stream, frames, heads, width, length, soc):
    tables = _host_tables(frames,heads,width,length,soc)
    if tables is None:
        return None
    return tuple(torch.frombuffer(bytearray.fromhex(t),dtype=torch.uint8).to(device)
                 if t else torch.empty(0,device=device,dtype=torch.uint8) for t in tables)


@lru_cache(maxsize=32)
def _device_soc(device):
    soc = torch.npu.get_device_name(device)
    if soc in ("Ascend910B2", "Ascend910B3") and (Path(__file__).with_name("manual_tiling") / (soc+"-frontend.json.z")).is_file():
        return soc
    return None


def _text_state(k,v,beta,steps,scale):
    key = F.normalize(F.silu(k),dim=-1,eps=1e-6).transpose(0,1).contiguous()
    value = (F.silu(v).transpose(0,1)*beta.unsqueeze(-1)).contiguous()
    k32 = key.float()
    gram = (k32*beta.float().unsqueeze(-1)).transpose(1,2) @ k32
    b = (value.transpose(1,2) @ key).float()
    eye = torch.eye(128,device=k.device,dtype=torch.float32)
    m = (gram+gram.transpose(1,2))*0.5+eye
    x = (eye*scale).expand_as(m)
    two_eye = eye*2
    for _ in range(steps):
        x = x @ (two_eye-m@x)
    return (b@x)*0.5


def vdn_forward_from_features(q,k,v,beta,bounds,*,frame_mean,down_weight,up_weight,
                              dt_bias,a_log,text_k=None,text_v=None,text_beta=None,
                              short_conv=None,frame_size=None,frame_tile=None,value_first=False):
    """Use local-head alpha weights; text K/V are [L,H,128], beta [L,H] or [L,H,1], post-sigmoid."""
    if torch.is_grad_enabled():
        raise ValueError("vdn_forward_from_features requires an inference or no_grad context")
    f,h,s,d=q.shape
    if (d!=128 or frame_mean.ndim!=2 or frame_mean.shape[0]!=f or down_weight.ndim!=2
        or down_weight.shape[1]!=frame_mean.shape[1] or up_weight.shape!=(h*d,down_weight.shape[0])
        or dt_bias.numel()!=h*d or a_log.shape!=(h,)):
        raise ValueError("inconsistent frame-mean or local-head alpha weight shape")
    inputs=(frame_mean,down_weight,up_weight,dt_bias,a_log)
    if any(x.device!=q.device or x.dtype not in (torch.float32,torch.bfloat16) for x in inputs):
        raise ValueError("alpha inputs must be FP32 or BF16 on the video device")
    text=(text_k,text_v,text_beta)
    if any(t is not None for t in text):
        if any(t is None or t.dtype!=torch.bfloat16 or t.device!=q.device for t in text):
            raise ValueError("text K, V and beta must be supplied together in BF16 on the video device")
        if (text_k.ndim!=3 or text_k.shape[1:]!=(h,d) or text_v.shape!=text_k.shape or
            tuple(text_beta.shape) not in ((text_k.shape[0],h),(text_k.shape[0],h,1))):
            raise ValueError("text input shapes must be [L,H,128], [L,H,128] and [L,H] or [L,H,1]")
    length=0 if text_k is None else text_k.shape[0]
    scale=2.0/(length+2)
    steps=math.ceil(math.log2(math.log(1e-7)/math.log1p(-scale))) if length else 0
    if frame_tile is not None and (type(frame_tile) is not int or frame_tile<1 or frame_tile & (frame_tile-1)):
        raise ValueError("frame_tile must be a positive power of two")
    if torch.npu.matmul.allow_hf32 or torch.npu.conv.allow_hf32:
        raise ValueError("the FP32 VDN frontend requires HF32 disabled")
    kw=dict(short_conv=short_conv,frame_size=frame_size,frame_tile=frame_tile or _default_frame_tile(q,"newton_schulz"),
            fuse_silu=True,fuse_qk_l2norm=True,delta_rule="newton_schulz",pad_tail=True,value_first=value_first)
    with torch.npu.device(q.device),torch.autocast(device_type=q.device.type,enabled=False):
        mean,wd,wu,bias,alog=inputs
        bias=bias.reshape(-1)
        if length:
            tk,tv=text_k,text_v
            tb=text_beta.reshape(length,h).transpose(0,1)
        else:
            tk=tv=tb=q.new_empty(0)
        tables=None
        if (_manual._HAS_KERNEL and down_weight.shape[0]==128 and 64<=f<=96 and 4<=h<=16
            and s==405 and 0<mean.shape[1]<=65536 and length<=f*s and not torch.npu.is_current_stream_capturing()
            and not torch.npu.matmul.allow_hf32 and torch.are_deterministic_algorithms_enabled()):
            soc=_device_soc(q.device)
            if soc is not None:
                tables=_device_tables(q.device,torch.npu.current_stream(q.device).stream_id,f,h,mean.shape[1],length,soc)
        if length and tables is not None:
            if any(t.stride(-1)!=1 or t.stride(1)<128 or t.stride(0)<(h-1)*t.stride(1)+128 or t.data_ptr()%32 for t in (tk,tv)):
                tables=None
            if any(min(t.stride())<1 or sum((n-1)*d for n,d in zip(t.shape,t.stride()))>=2**31 for t in (tk,tv,tb)):
                tables=None
        if tables is not None:
            import torch_npu
            operands = (*inputs, tk, tv, tb) if length else inputs
            if any(torch_npu.get_npu_format(t) not in (0, 2) for t in operands):
                tables = None
        if (tables is not None and all(t.is_contiguous() for t in (mean,wd,wu,bias,alog))
                and all(t.data_ptr()%32==0 for t in (mean,wd,wu,bias))):
            alpha=torch.empty((f,h,d),dtype=torch.float32,device=q.device)
            state=alpha.new_empty((h,d,d)) if length else None
            extra=(mean,wd,wu,bias,alog,tk,tv,tb,*tables,mean.shape[1],length,steps)
            output=_manual.try_forward(q,k,v,beta,alpha,bounds,state,frontend=extra,**kw)
            if output is not None:
                return output
        mean,wd,wu,bias,alog=(x.float() for x in (mean,wd,wu,bias,alog))
        delta=F.linear(F.linear(mean,wd),wu)+bias
        alpha=torch.exp(-alog.exp().view(1,h,1)*F.softplus(delta.reshape(f,h,d)))
        state=_text_state(tk,tv,tb,steps,scale) if length else None
        return vdn_forward(q,k,v,beta,alpha,bounds,state,**kw)
