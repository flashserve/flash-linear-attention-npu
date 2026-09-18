// Single translation unit for every Stable-ABI adapter.
//
// torch/csrc/stable/tensor_inl.h defines non-inline member functions (e.g.
// `Tensor::scalar_type()`), so including the stable headers from more than one
// TU fails at link time with "multiple definition of
// torch::stable::Tensor::scalar_type() const".  All adapters therefore live in
// this one file, one `stable_<family>.cpp` per group of operators.
#include "stable_recurrent_gdr.cpp"
#include "stable_recurrent_kda.cpp"
#include "stable_fast_gelu.cpp"
#include "stable_kda.cpp"
#include "stable_chunk.cpp"
#include "stable_gdn.cpp"
#include "stable_conv1d.cpp"
#include "stable_fwd_h.cpp"

// Build stamp: the md5 of the adapter sources this library was compiled from,
// injected by csrc/build_stable.py.  fla_npu/ops/ascendc/_stable.py
// reads it (through ctypes, no torch needed) and refuses to run against a
// library built from different sources than the glue it was imported with, so
// a stale .so cannot silently drive kernels with old schemas or old stack
// indices.
#ifndef FLA_STABLE_SOURCE_HASH
#define FLA_STABLE_SOURCE_HASH "unknown"
#endif
// The library is built with -fvisibility=hidden, so the stamp has to ask for
// default visibility explicitly -- otherwise ctypes cannot find the symbol and
// the check would silently pass for every artifact.
extern "C" __attribute__((visibility("default")))
const char* fla_npu_stable_source_hash() {
  return FLA_STABLE_SOURCE_HASH;
}

// Whether this library hands its launches to torch_npu's task queue.  The
// Python glue asks this before it reads the stream: a queue-ordered launch may
// use torch_npu's non-flushing accessor, an inline launch may not, and asking
// the library instead of re-deriving the answer in Python is what keeps the two
// sides from disagreeing about it.
extern "C" __attribute__((visibility("default")))
int32_t fla_npu_stable_queue_enqueue_available() {
  return fla_npu_stable::Runtime::instance().enqueue_enabled() ? 1 : 0;
}

// The stream the most recent operator call on the calling thread launched on
// (see note_launch_stream).  The multi-stream regression reads it back after
// every call: it is the only way to see *which* stream a call used, and the
// value is per thread, which is exactly the property that has to hold when vLLM
// interleaves workers.
extern "C" __attribute__((visibility("default")))
int64_t fla_npu_stable_last_launch_stream() {
  return fla_npu_stable::stable::t_last_launch_stream;
}

// Exactly one library-definition block and one implementation block per
// namespace per TU: the macros expand to a fixed static-init symbol name, so a
// second block for the same namespace would be a redefinition.  The codegen
// phase therefore collects every adapter's schema/impl into these two lists.
STABLE_TORCH_LIBRARY(fla_npu_stable, m) {
  m.def(kSchemaRecurrentGdr);
  m.def(kSchemaRecurrentKda);
  m.def(kSchema_npu_fast_gelu_custom);
  m.def(kSchema_npu_fast_gelu_custom_backward);
  m.def(kSchema_kda_gate_cumsum);
  m.def(kSchema_chunk_kda_bwd_intra);
  m.def(kSchema_chunk_kda_bwd_recompute);
  m.def(kSchema_chunk_kda_fwd);
  m.def(kSchema_chunk_kda_fwd_finalize);
  m.def(kSchema_chunk_bwd_dv_local);
  m.def(kSchema_chunk_local_cumsum);
  m.def(kSchema_chunk_scaled_dot_kkt);
  m.def(kSchema_chunk_bwd_dqkwg);
  m.def(kSchema_prepare_wy_repr_bwd_da);
  m.def(kSchema_prepare_wy_repr_bwd_full);
  m.def(kSchema_prepare_wy_repr_bwd);
  m.def(kSchema_recompute_w_u_fwd);
  m.def(kSchema_causal_conv1d_bwd);
  m.def(kSchema_chunk_fwd_o);
  m.def(kSchema_chunk_gdn_bwd_intra);
  m.def(kSchema_causal_conv1d);
  m.def(kSchema_causal_conv1d_fn);
  m.def(kSchema_causal_conv1d_update);
  m.def(kSchema_chunk_fwd_h);
  m.def(kSchema_chunk_gated_delta_rule_fwd_h);
  m.def(kSchema_chunk_gated_delta_rule_bwd_dhu);
  m.def(kSchema_chunk_gated_delta_rule_fwd);
  m.def(kSchema_solve_tri);
  m.def(kSchema_chunk_gated_delta_rule_fwd_prepare);
  m.def(kSchema_chunk_gated_delta_rule_bwd_finalize);
  m.def(kSchema_chunk_kda_bwd);
  m.def(kSchema_chunk_gated_delta_rule_bwd);
#ifndef FLA_STABLE_NO_DEBUG_PROBE
  m.def("_stream_probe(int device_index) -> (int, int)");
#endif
}

STABLE_TORCH_LIBRARY_IMPL(fla_npu_stable, CompositeExplicitAutograd, m) {
  m.impl("npu_recurrent_gated_delta_rule", &boxed_recurrent_gated_delta_rule);
  m.impl("npu_recurrent_kda", &boxed_recurrent_kda);
  m.impl("npu_fast_gelu_custom",
         &fla_npu_stable::stable::boxed_adapter<run_npu_fast_gelu_custom>);
  m.impl("npu_fast_gelu_custom_backward",
         &fla_npu_stable::stable::boxed_adapter<
             run_npu_fast_gelu_custom_backward>);
  m.impl("npu_kda_gate_cumsum",
         &fla_npu_stable::stable::boxed_adapter<run_npu_kda_gate_cumsum>);
  m.impl("npu_chunk_kda_bwd_intra",
         &fla_npu_stable::stable::boxed_adapter<run_npu_chunk_kda_bwd_intra>);
  m.impl("npu_chunk_kda_bwd_recompute",
         &fla_npu_stable::stable::boxed_adapter<
             run_npu_chunk_kda_bwd_recompute>);
  m.impl("npu_chunk_kda_fwd",
         &fla_npu_stable::stable::boxed_adapter<run_npu_chunk_kda_fwd>);
  m.impl("npu_chunk_kda_fwd_finalize",
         &fla_npu_stable::stable::boxed_adapter<
             run_npu_chunk_kda_fwd_finalize>);
  m.impl("npu_chunk_bwd_dv_local",
         &fla_npu_stable::stable::boxed_adapter<run_npu_chunk_bwd_dv_local>);
  m.impl("npu_chunk_local_cumsum",
         &fla_npu_stable::stable::boxed_adapter<run_npu_chunk_local_cumsum>);
  m.impl("npu_chunk_scaled_dot_kkt",
         &fla_npu_stable::stable::boxed_adapter<run_npu_chunk_scaled_dot_kkt>);
  m.impl("npu_chunk_bwd_dqkwg",
         &fla_npu_stable::stable::boxed_adapter<run_npu_chunk_bwd_dqkwg>);
  m.impl("npu_prepare_wy_repr_bwd_da",
         &fla_npu_stable::stable::boxed_adapter<run_npu_prepare_wy_repr_bwd_da>);
  m.impl(
      "npu_prepare_wy_repr_bwd_full",
      &fla_npu_stable::stable::boxed_adapter<run_npu_prepare_wy_repr_bwd_full>);
  m.impl("npu_prepare_wy_repr_bwd",
         &fla_npu_stable::stable::boxed_adapter<run_npu_prepare_wy_repr_bwd>);
  m.impl("npu_recompute_w_u_fwd",
         &fla_npu_stable::stable::boxed_adapter<run_npu_recompute_w_u_fwd>);
  m.impl("npu_causal_conv1d_bwd",
         &fla_npu_stable::stable::boxed_adapter<run_npu_causal_conv1d_bwd>);
  m.impl("npu_chunk_fwd_o",
         &fla_npu_stable::stable::boxed_adapter<run_npu_chunk_fwd_o>);
  m.impl("npu_chunk_gdn_bwd_intra",
         &fla_npu_stable::stable::boxed_adapter<run_npu_chunk_gdn_bwd_intra>);
  m.impl("npu_causal_conv1d_fn",
         &fla_npu_stable::stable::boxed_adapter<run_npu_causal_conv1d_fn>);
  m.impl("npu_causal_conv1d_update",
         &fla_npu_stable::stable::boxed_adapter<run_npu_causal_conv1d_update>);
  m.impl("npu_causal_conv1d",
         &fla_npu_stable::stable::boxed_adapter<run_npu_causal_conv1d>);
  m.impl("npu_chunk_fwd_h",
         &fla_npu_stable::stable::boxed_adapter<run_npu_chunk_fwd_h>);
  m.impl(
      "npu_chunk_gated_delta_rule_fwd_h",
      &fla_npu_stable::stable::boxed_adapter<
          run_npu_chunk_gated_delta_rule_fwd_h>);
  m.impl(
      "npu_chunk_gated_delta_rule_bwd_dhu",
      &fla_npu_stable::stable::boxed_adapter<
          run_npu_chunk_gated_delta_rule_bwd_dhu>);
  m.impl(
      "npu_chunk_gated_delta_rule_fwd",
      &fla_npu_stable::stable::boxed_adapter<
          run_npu_chunk_gated_delta_rule_fwd>);
  m.impl("npu_solve_tri",
         &fla_npu_stable::stable::boxed_adapter<run_npu_solve_tri>);
  m.impl(
      "npu_chunk_gated_delta_rule_fwd_prepare",
      &fla_npu_stable::stable::boxed_adapter<
          run_npu_chunk_gated_delta_rule_fwd_prepare>);
  m.impl(
      "npu_chunk_gated_delta_rule_bwd_finalize",
      &fla_npu_stable::stable::boxed_adapter<
          run_npu_chunk_gated_delta_rule_bwd_finalize>);
  m.impl("npu_chunk_kda_bwd",
         &fla_npu_stable::stable::boxed_adapter<run_npu_chunk_kda_bwd>);
  m.impl(
      "npu_chunk_gated_delta_rule_bwd",
      &fla_npu_stable::stable::boxed_adapter<
          run_npu_chunk_gated_delta_rule_bwd>);
#ifndef FLA_STABLE_NO_DEBUG_PROBE
  m.impl("_stream_probe", &boxed_stream_probe);
#endif
}
