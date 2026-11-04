"""ATK's native pyaclnn adapter for the Prepare C API (test code only)."""
from __future__ import annotations

import ctypes

import torch
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat, nnopbase

from executor_chunk_kda_fwd_prepare import (
    OUTPUT_DTYPES, OUTPUT_MASKS, _as_bool, _derive_spec, _direct_inputs,
    _expected_output_shapes,
)


@register("aclnn_chunk_kda_fwd_prepare")
class PrepareAclnnApi(AclnnBaseApi):
    """Map Python test inputs to the public header's 33 arguments.

    ATK owns descriptor cleanup, workspace, stream and both ACLNN calls.
    This adapter never calls the FLA Python wrapper or Stable launcher.
    """

    def init_by_input_data(self, input_data):
        values = input_data.kwargs
        spec = _derive_spec(values, int(self.task_result.case_config.id))
        inputs = _direct_inputs(values, high_precision=False)
        self.keepalive = []

        def tensor(value):
            if value is None:
                return ctypes.c_void_p()
            value = value.to(f"npu:{self.backend.device_id}").contiguous()
            self.keepalive.append(value)
            return nnopbase.create_acl_tensor(value, AclFormat.ACL_FORMAT_ND)

        def int_array(value):
            if value is None:
                return ctypes.c_void_p()
            return nnopbase.create_x_list([ctypes.c_int64(v) for v in value])

        indices = inputs.chunk_indices
        if inputs.cu_seqlens and not indices:
            indices = tuple(
                x for seq, (start, end) in enumerate(zip(inputs.cu_seqlens, inputs.cu_seqlens[1:]))
                for chunk in range((end - start + 63) // 64) for x in (seq, chunk)
            )
        args = [tensor(t) for t in (inputs.q, inputs.k, inputs.v, inputs.g,
                                   inputs.beta, inputs.a_log, inputs.dt_bias)]
        args += [int_array(inputs.cu_seqlens), int_array(indices),
                 ctypes.c_char_p(str(spec['layout']).encode()),
                 ctypes.c_double(float(spec['scale'])), ctypes.c_int64(int(spec['chunk_size'])),
                 ctypes.c_double(float(spec['epsilon']))]
        args += [ctypes.c_bool(_as_bool(spec[name])) for name in (
            'use_qk_l2norm_in_kernel', 'use_gate_in_kernel', 'use_beta_sigmoid_in_kernel',
            'allow_neg_eigval', 'safe_gate')]
        args += [ctypes.c_double(float(spec['lower_bound'])),
                 ctypes.c_bool(_as_bool(spec['use_exp2']))]
        outputs = []
        self.output_tensors = []
        for enabled, shape, dtype in zip(OUTPUT_MASKS[spec['backward_mode']],
                                         _expected_output_shapes(spec), OUTPUT_DTYPES):
            if not enabled:
                args.append(ctypes.c_void_p())
                continue
            value = torch.empty(shape, dtype=dtype, device=f'npu:{self.backend.device_id}')
            descriptor = tensor(value)
            self.output_tensors.append(value)
            outputs.append(descriptor)
            args.append(descriptor)
        return args, outputs

    def after_call(self, output_packages):
        # ATK synchronizes its ACL stream before this hook. Retain all buffers
        # until descriptors are destroyed; copy the actual DUT tensors to CPU.
        del output_packages
        return tuple(t.cpu() for t in self.output_tensors)
