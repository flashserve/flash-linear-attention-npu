"""输入张量的生命周期回归：stable 适配层"消费"栈引用之后，调用方手里的张量必须完好。

boxed kernel 的契约（``csrc/include/torch_stable_abi/torch/csrc/stable/library.h:69``）
要求 kernel 自己"偷走"输入的栈引用。手写入口不消费 -> 每次调用漏一份引用
（910B3 实测：2000 次 fresh 调用涨 191 MiB，conc32 服务涨到 8.2 GiB 后 OOM）；
消费过头 -> 调用方手里的张量提前析构，表现为数值损坏或 allocator 崩溃。
两种都只能靠"调用之后再碰一碰这些张量"抓出来，所以这个脚本做四件事：

1. 调用返回后，输入张量逐位不变（既没被提前释放，也没被算子改写）；
2. 调用方仍能原地写这些张量，写完再调用，算子看得见新值；
3. 每轮新建输入、调用后立刻 del，200 轮后 allocated 不涨（漏引用会让这些张量
   在 del 之后仍然活着）；
4. 复原输入后重复调用，结果逐位一致（引用记账没有累积漂移）。

``npu_recurrent_gated_delta_rule`` / ``npu_recurrent_kda`` 是手写入口（本次修复的
对象），``npu_causal_conv1d_update`` 走宏，作为对照。

    ASCEND_RT_VISIBLE_DEVICES=<n> FLA_NPU_STABLE_LIB=<so> PYTHONPATH=<env> \
        python tests/stable_abi/test_input_lifetime.py
"""
from __future__ import annotations

import gc
import os
import sys

import torch
import torch_npu  # noqa: F401

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fla_npu.ops.ascendc import (  # noqa: E402
    npu_causal_conv1d_update as conv1d_update,
    npu_recurrent_gated_delta_rule as gdr_public,
    npu_recurrent_kda as kda_public,
)
from regression_mutation_contract import gdr_inputs, kda_inputs  # noqa: E402


ROUNDS = 200
# 漏引用时这 200 轮会留下约 20 MiB 的活张量（每轮 ~100 KiB：q/k/v 各 32 KiB
# 加 beta/asl/idx 各自的分配块）。正常路径的漂移只有几十 KiB。
MEMORY_LIMIT_MIB = 8.0


def allocated_mib() -> float:
    return torch.npu.memory_allocated() / float(1 << 20)


def settle() -> None:
    gc.collect()
    torch.npu.synchronize()


def bitwise_same(a, b) -> bool:
    """逐位比较，不用 torch.equal：NaN payload 也必须一致。"""
    return bool(torch.equal(a.contiguous().view(torch.uint8),
                            b.contiguous().view(torch.uint8)))


def same_result(a, b) -> bool:
    if isinstance(a, (tuple, list)):
        return (len(a) == len(b)
                and all(same_result(x, y) for x, y in zip(a, b)))
    if a is None or b is None:
        return a is None and b is None
    return bitwise_same(a, b)


def check_entry(label, build, call, mutated=(0,)):
    """调用返回后输入仍然完好；复原输入后同一调用逐位复现。

    ``mutated`` 列出算子按契约原地改写的参数下标（如 recurrent 的 state），
    其余参数在调用后必须逐位不变。
    """
    args, kwargs = build()
    keep = [i for i in range(len(args)) if i not in mutated]
    keep_ref = [args[i].clone() for i in keep]
    mut_ref = [args[i].clone() for i in mutated]

    out1 = call(args, kwargs)
    torch.npu.synchronize()
    for i, ref in zip(keep, keep_ref):
        assert bitwise_same(args[i], ref), (
            f"{label}: input #{i} is not intact after the call (values changed)")
    print(f"PASS {label}: inputs unchanged after the call")

    # 写一次：既证明张量还活着，也证明算子看得见调用方的新值。
    args[keep[0]].fill_(0.5)
    assert bitwise_same(args[keep[0]],
                        torch.full_like(args[keep[0]], 0.5)), (
        f"{label}: input #{keep[0]} is not writable after the call")
    args[keep[0]].copy_(keep_ref[0])
    print(f"PASS {label}: inputs writable after the call")

    for i, ref in zip(keep, keep_ref):
        args[i].copy_(ref)
    for i, ref in zip(mutated, mut_ref):
        args[i].copy_(ref)
    out2 = call(args, kwargs)
    torch.npu.synchronize()
    assert same_result(out1, out2), (
        f"{label}: a repeat call on restored inputs is not bitwise identical")
    print(f"PASS {label}: repeat call on restored inputs is bitwise identical")


def check_no_retained_inputs(label, build, call):
    """fresh 输入 + 调用后立刻 del：漏引用会让它们活到进程结束。"""
    for _ in range(5):
        args, kwargs = build()
        call(args, kwargs)
    settle()
    base = allocated_mib()
    for _ in range(ROUNDS):
        args, kwargs = build()
        call(args, kwargs)
        del args, kwargs
    settle()
    growth = allocated_mib() - base
    assert growth < MEMORY_LIMIT_MIB, (
        f"{label}: {ROUNDS} fresh calls grew allocated by {growth:.2f} MiB "
        f"(limit {MEMORY_LIMIT_MIB}); the adapter is retaining inputs")
    print(f"PASS {label}: {ROUNDS} fresh calls grew allocated by "
          f"{growth:+.2f} MiB")


def build_gdr():
    query, key, value, state, kwargs = gdr_inputs()
    return (query, key, value, state), kwargs


def call_gdr(args, kwargs):
    return gdr_public(*args, **kwargs)


def build_kda():
    q, k, v, g, beta, state, kwargs = kda_inputs()
    return (q, k, v, g, beta, state), kwargs


def call_kda(args, kwargs):
    return kda_public(*args, **kwargs)


def build_conv1d_update():
    def ramp(count, start, shape):
        return (torch.arange(count, dtype=torch.float32) + start).reshape(
            shape).to(torch.bfloat16).npu()

    x = ramp(2 * 16, 1.0, (2, 16))
    weight = ramp(4 * 16, 101.0, (4, 16))
    bias = ramp(16, 201.0, (16,))
    conv_state = ramp(3 * 3 * 16, 301.0, (3, 3, 16))
    indices = torch.tensor([1, 2], dtype=torch.int32, device="npu")
    return ((x, conv_state, weight, bias),
            dict(activation="silu", conv_state_indices=indices))


def call_conv1d_update(args, kwargs):
    return conv1d_update(*args, **kwargs)


def main() -> int:
    torch.npu.set_device(0)
    torch.manual_seed(20260916)
    # (label, build, call, mutated-index)  —— 手写入口在前，宏路径作对照。
    entries = [
        ("recurrent_gated_delta_rule", build_gdr, call_gdr, (3,)),
        ("recurrent_kda", build_kda, call_kda, (5,)),
        # conv1d_update 的 out= 默认写回 x，所以 x 也按原地参数检查。
        ("causal_conv1d_update", build_conv1d_update, call_conv1d_update,
         (0, 1)),
    ]
    for label, build, call, mutated in entries:
        check_entry(label, build, call, mutated)
        check_no_retained_inputs(label, build, call)
    print("ALL PASS: input lifetime regression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
