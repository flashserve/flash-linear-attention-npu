"""Exercise the public C ABI, including output views and rejected descriptors."""

import argparse
import ctypes as C

import torch
import torch_npu  # noqa: F401

from check_layout_equivalence import launch, load_executor
import _layout_abi_probe as rt


TENSORS = ("q", "k", "v", "g", "beta", "A", "d_o", "initial_state", "dht",
           "q_rstd", "k_rstd", "beta_raw", "a_log", "dt_bias")


def arguments(ctx, inputs, outputs, spec, layout, fake_shapes=None):
    fake_shapes = fake_shapes or {}

    def tensor(name, value):
        if name not in fake_shapes:
            return ctx.tensor(value, name, acl_format_override=2,
                              storage_shape_override=value.shape if value is not None else None)
        # Descriptor-only negative tests: never launch these synthetic shapes.
        shape = fake_shapes[name]
        strides, stride = [], 1
        for dim in reversed(shape):
            strides.insert(0, stride)
            stride *= dim
        dims = (C.c_int64 * len(shape))(*shape)
        steps = (C.c_int64 * len(shape))(*strides)
        ptr = ctx.runtime.acl_create_tensor(
            dims, len(shape), rt.dtype_to_acl(value.dtype), steps, 0, 2,
            dims, len(shape), C.c_void_p(rt.storage_data_ptr(value)))
        if not ptr:
            raise RuntimeError(f"aclCreateTensor failed for synthetic {name}")

        class Descriptor:
            def destroy(self):
                ctx.runtime.acl_destroy_tensor(ptr)

        ctx.resources.append(Descriptor())
        return C.c_void_p(ptr)

    return [*[tensor(name, inputs.get(name)) for name in TENSORS],
            ctx.int_array(inputs["cu_seqlens"]), ctx.int_array(inputs["chunk_indices"]),
            C.c_char_p(layout.encode()), C.c_double(inputs["scale"]), C.c_int64(64),
            C.c_bool(spec.get("use_exp2", True)), C.c_bool(False), C.c_bool(False),
            C.c_bool(False), C.c_bool(False),
            *[tensor(name, value) for name, value in zip(
                ("dq", "dk", "dv", "dbeta", "dg", "dh0", "dalog", "ddtbias"), outputs)]]


def outputs_like(inputs):
    return tuple(torch.empty_like(inputs[name]) if inputs.get(name) is not None else None
                 for name in ("q", "k", "v", "beta", "g", "initial_state", "a_log", "dt_bias"))


def reject(name, inputs, outputs, spec, layout, *, fake_shapes=None, error_parameter=None):
    runtime = rt.runtime()
    ctx = rt._CallContext(runtime, inputs["q"].device)
    try:
        fn = runtime.symbol("aclnnChunkGatedDeltaRuleBwdGetWorkspaceSize")
        fn.restype = C.c_int
        workspace, executor = C.c_uint64(), C.c_void_p()
        code = fn(*arguments(ctx, inputs, outputs, spec, layout, fake_shapes),
                  C.byref(workspace), C.byref(executor))
        if code != 161002:
            raise AssertionError(f"{name}: expected 161002, got {code}")
        if error_parameter is not None:
            message = runtime._aclnn_error("aclnnChunkGatedDeltaRuleBwdGetWorkspaceSize", code)
            if error_parameter not in message:
                raise AssertionError(f"{name}: missing parameter context: {message}")
        print(f"negative {name}: PASS ({code})", flush=True)
    finally:
        ctx.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-views", action="store_true",
                        help="Verify the existing A2/A3 noncontiguous output contract")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    executor = load_executor()
    spec = dict(B=2, HK=2, HV=6, T=65, with_state=True, with_dht=True,
                use_exp2=True, seed=20261107)
    source = executor.build_inputs(spec, torch.device(f"npu:{args.device}"))
    for sequence_major in (False, True):
        layout = "BSND" if sequence_major else "BNSD"
        expected = launch(executor, source, spec, sequence_major)
        inputs = dict(source)
        for name in ("q", "k", "v"):
            inputs[name] = executor._public_layout(inputs[name], sequence_major)
        for name in ("g", "beta", "d_o"):
            inputs[name] = executor._public_layout(inputs[name], True)
        outputs = outputs_like(inputs)
        rt.call_aclnn("aclnnChunkGatedDeltaRuleBwd",
                      lambda ctx: arguments(ctx, inputs, outputs, spec, layout), outputs)
        torch.npu.synchronize()
        for actual, reference in zip(outputs, expected):
            if reference is not None:
                assert torch.equal(actual.cpu().view(torch.uint8), reference.cpu().view(torch.uint8))
        print(f"public C ABI {layout}: PASS", flush=True)

        if args.output_views:
            viewed = list(outputs)
            for index in (0, 1, 2, 5):
                value = outputs[index]
                storage = torch.full((value.numel() * 2 + 17,), float("nan"),
                                     dtype=value.dtype, device=value.device)
                viewed[index] = storage[9:9 + value.numel() * 2:2].view(value.shape)
            rt.call_aclnn("aclnnChunkGatedDeltaRuleBwd",
                          lambda ctx: arguments(ctx, inputs, viewed, spec, layout), tuple(viewed))
            torch.npu.synchronize()
            for actual, reference in zip(viewed, expected):
                if reference is not None:
                    assert torch.equal(actual.cpu().contiguous().view(torch.uint8),
                                       reference.cpu().view(torch.uint8))
            print(f"output views with storage offset {layout}: PASS", flush=True)

        reject("unknown layout", inputs, outputs, spec, "invalid")
        for name in ("q", "k", "v", "g", "beta", "A", "d_o"):
            invalid = dict(inputs)
            invalid[name] = inputs[name].float() if name not in ("g", "beta") else inputs[name].half()
            reject(f"{name} dtype", invalid, outputs, spec, layout)
        bad_outputs = list(outputs)
        bad_outputs[0] = outputs[0].transpose(1, 2).contiguous()
        reject("dq output layout", inputs, bad_outputs, spec, layout)
        invalid = dict(inputs, initial_state=None, dht=None,
                       cu_seqlens=[0, 65], chunk_indices=None)
        metadata_outputs = list(outputs)
        metadata_outputs[5] = None
        reject("unpaired metadata", invalid, metadata_outputs, spec, layout)
        invalid["chunk_indices"] = [0, 1, 0, 0]
        reject("noncanonical chunks", invalid, metadata_outputs, spec, layout)
        if sequence_major:
            reject("DMA row gap overflow", inputs, outputs, spec, layout,
                   fake_shapes={"q": (1, 1, 16777217, 128), "v": (1, 1, 16777217, 128)},
                   error_parameter="q/k/dq/dk.HK")
            reject("byte address overflow", inputs, outputs, spec, layout,
                   fake_shapes={"q": (1 << 54, 1, 2, 128), "v": (1 << 54, 1, 2, 128)},
                   error_parameter="v/dv byte address")


if __name__ == "__main__":
    main()
