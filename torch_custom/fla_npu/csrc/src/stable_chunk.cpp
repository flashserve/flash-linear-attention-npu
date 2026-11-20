// Stable-ABI adapters for the chunked GDN backward path.
//
// Every operator here is the same three steps as fast_gelu -- allocate the
// outputs, hand the aclnn argument list to FLA_STABLE_EXEC, return -- so the
// only per-operator knowledge is the shape rule of each output.  Two details
// are worth pointing at:
//
//   * `scale` is `double` on some aclnn entry points and `float` on others
//     (the schema always spells it `float`, which only constrains the Python
//     binding); the cast at each call site follows the operator's own
//     prototype.
//   * chunk_local_cumsum's `output_dtype` is a `char*` argument, so it travels
//     as an int code plus the name table below.  The table order is what
//     _stable.py's _char_code table must match.
//
// Included by stable_ops.cpp (single TU); registration lives there.

// Owns the chunk-level helpers both families use: npu_prepare_wy_repr,
// npu_prepare_wy_repr_bwd, npu_prepare_wy_repr_bwd_da,
// npu_prepare_wy_repr_bwd_full, npu_chunk_scaled_dot_kkt,
// npu_chunk_local_cumsum, npu_chunk_bwd_dqkwg, npu_chunk_bwd_dv_local,
// npu_recompute_w_u_fwd, npu_solve_tri.

#include "stable/at_facade.h"
#include "stable/boxed.h"
#include "stable/exec.h"

#include <cstdint>
#include <optional>
#include <tuple>

namespace {

using torch::stable::Tensor;
using fla_npu_stable::stable::TensorMeta;
using fla_npu_stable::stable::at_shim::kBFloat16;
using fla_npu_stable::stable::at_shim::kFloat;
using fla_npu_stable::stable::allocate_like;
using fla_npu_stable::stable::allocate_sizes;
using fla_npu_stable::stable::cstr;
using fla_npu_stable::stable::int_array;
using fla_npu_stable::stable::meta_of;
using fla_npu_stable::stable::optional_tensor;
using fla_npu_stable::stable::out_tensor;
using fla_npu_stable::stable::scalar;
using fla_npu_stable::stable::size_of;
using fla_npu_stable::stable::tensor;


// ---------------------------------------------------------------------------
// npu_chunk_bwd_dv_local
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_bwd_dv_local =
    "npu_chunk_bwd_dv_local(Tensor q, Tensor k, Tensor d_o, Tensor g, "
    "Tensor? g_gamma, Tensor? A, Tensor? cu_seqlens, Tensor? chunk_indices, "
    "float scale, int chunk_size, int stream) -> Tensor";

Tensor run_npu_chunk_bwd_dv_local(Tensor q, Tensor k, Tensor d_o, Tensor g,
                                  std::optional<Tensor> g_gamma,
                                  std::optional<Tensor> A,
                                  std::optional<Tensor> cu_seqlens,
                                  std::optional<Tensor> chunk_indices,
                                  double scale, int64_t chunk_size,
                                  int64_t stream) {
  const TensorMeta d_o_meta = meta_of(d_o);
  Tensor out = allocate_like(d_o_meta);
  FLA_STABLE_EXEC("aclnnChunkBwdDvLocal", d_o_meta, stream, tensor(meta_of(q)),
                  tensor(meta_of(k)), tensor(d_o_meta), tensor(meta_of(g)),
                  optional_tensor(g_gamma), optional_tensor(A),
                  int_array(cu_seqlens), int_array(chunk_indices),
                  scalar(scale), scalar(chunk_size), out_tensor(meta_of(out)));
  return out;
}

// ---------------------------------------------------------------------------
// npu_chunk_local_cumsum
// ---------------------------------------------------------------------------

constexpr const char* kChunkLocalCumsumOutputDtypeNames[] = {"float32",
                                                             "bfloat16"};

constexpr const char* kSchema_chunk_local_cumsum =
    "npu_chunk_local_cumsum(Tensor g, Tensor? cu_seqlens, "
    "Tensor? chunk_indices_out, int chunk_size, bool reverse, float scale, "
    "bool head_first, int output_dtype, int stream) -> Tensor";

Tensor run_npu_chunk_local_cumsum(Tensor g, std::optional<Tensor> cu_seqlens,
                                  std::optional<Tensor> chunk_indices_out,
                                  int64_t chunk_size, bool reverse, double scale,
                                  bool head_first, int64_t output_dtype,
                                  int64_t stream) {
  const TensorMeta g_meta = meta_of(g);
  Tensor out = allocate_sizes(g_meta.sizes,
                              output_dtype == 0 ? kFloat : kBFloat16, g_meta);
  FLA_STABLE_EXEC("aclnnChunkLocalCumsum", g_meta, stream, tensor(g_meta),
                  int_array(cu_seqlens), int_array(chunk_indices_out),
                  scalar(chunk_size), scalar(reverse), scalar(scale),
                  scalar(head_first),
                  cstr(kChunkLocalCumsumOutputDtypeNames, output_dtype),
                  out_tensor(meta_of(out)));
  return out;
}

// ---------------------------------------------------------------------------
// npu_chunk_scaled_dot_kkt
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_scaled_dot_kkt =
    "npu_chunk_scaled_dot_kkt(Tensor k, Tensor g, Tensor beta, "
    "Tensor? cu_seqlens, Tensor? chunk_indices, int chunk_size, int stream) "
    "-> Tensor";

Tensor run_npu_chunk_scaled_dot_kkt(Tensor k, Tensor g, Tensor beta,
                                    std::optional<Tensor> cu_seqlens,
                                    std::optional<Tensor> chunk_indices,
                                    int64_t chunk_size, int64_t stream) {
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta g_meta = meta_of(g);
  Tensor out = allocate_sizes(
      {size_of(k_meta, 0), size_of(g_meta, 1), size_of(k_meta, 2), chunk_size},
      kFloat, k_meta);
  FLA_STABLE_EXEC("aclnnChunkScaledDotKkt", k_meta, stream, tensor(k_meta),
                  tensor(g_meta), tensor(meta_of(beta)),
                  int_array(cu_seqlens), int_array(chunk_indices),
                  scalar(chunk_size), out_tensor(meta_of(out)));
  return out;
}

// ---------------------------------------------------------------------------
// npu_chunk_bwd_dqkwg
// ---------------------------------------------------------------------------

constexpr const char* kSchema_chunk_bwd_dqkwg =
    "npu_chunk_bwd_dqkwg(Tensor q, Tensor k, Tensor v, Tensor g, Tensor h, "
    "Tensor dox, Tensor dh, Tensor dv, Tensor? cu_seqlens, "
    "Tensor? chunk_indices, Tensor? w, Tensor? g_gamma, float scale, "
    "int chunk_size, bool use_exp2, bool transpose_state_layout, int stream) "
    "-> (Tensor, Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor, Tensor> run_npu_chunk_bwd_dqkwg(
    Tensor q, Tensor k, Tensor v, Tensor g, Tensor h, Tensor dox, Tensor dh,
    Tensor dv, std::optional<Tensor> cu_seqlens,
    std::optional<Tensor> chunk_indices, std::optional<Tensor> w,
    std::optional<Tensor> g_gamma, double scale, int64_t chunk_size,
    bool use_exp2, bool transpose_state_layout, int64_t stream) {
  const TensorMeta q_meta = meta_of(q);
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta v_meta = meta_of(v);
  const TensorMeta g_meta = meta_of(g);
  Tensor out_dq = allocate_like(q_meta);
  Tensor out_dk = allocate_like(k_meta);
  Tensor out_dw = allocate_sizes(
      {size_of(q_meta, 0), size_of(v_meta, 1), size_of(q_meta, 2),
       size_of(q_meta, 3)},
      q_meta.scalar_type, q_meta);
  Tensor out_dg = allocate_like(g_meta);
  FLA_STABLE_EXEC(
      "aclnnChunkBwdDqkwg", q_meta, stream,
      tensor(q_meta), tensor(k_meta), tensor(v_meta), tensor(g_meta),
      tensor(meta_of(h)), tensor(meta_of(dox)), tensor(meta_of(dh)),
      tensor(meta_of(dv)), int_array(cu_seqlens), int_array(chunk_indices),
      optional_tensor(w), optional_tensor(g_gamma),
      scalar(static_cast<float>(scale)), scalar(chunk_size), scalar(use_exp2),
      scalar(transpose_state_layout), out_tensor(meta_of(out_dq)),
      out_tensor(meta_of(out_dk)), out_tensor(meta_of(out_dw)),
      out_tensor(meta_of(out_dg)));
  return std::make_tuple(out_dq, out_dk, out_dw, out_dg);
}

// ---------------------------------------------------------------------------
// npu_prepare_wy_repr_bwd_da / _full / npu_prepare_wy_repr_bwd
// ---------------------------------------------------------------------------

constexpr const char* kSchema_prepare_wy_repr_bwd_da =
    "npu_prepare_wy_repr_bwd_da(Tensor k, Tensor v, Tensor beta, Tensor A, "
    "Tensor dw, Tensor du, Tensor g, Tensor? cu_seqlens, "
    "Tensor? chunk_indices, int chunk_size, int stream) -> Tensor";

Tensor run_npu_prepare_wy_repr_bwd_da(Tensor k, Tensor v, Tensor beta, Tensor A,
                                      Tensor dw, Tensor du, Tensor g,
                                      std::optional<Tensor> cu_seqlens,
                                      std::optional<Tensor> chunk_indices,
                                      int64_t chunk_size, int64_t stream) {
  const TensorMeta A_meta = meta_of(A);
  Tensor out = allocate_like(A_meta);
  FLA_STABLE_EXEC("aclnnPrepareWyReprBwdDa", A_meta, stream,
                  tensor(meta_of(k)), tensor(meta_of(v)), tensor(meta_of(beta)),
                  tensor(A_meta), tensor(meta_of(dw)), tensor(meta_of(du)),
                  tensor(meta_of(g)), int_array(cu_seqlens),
                  int_array(chunk_indices), scalar(chunk_size),
                  out_tensor(meta_of(out)));
  return out;
}

constexpr const char* kSchema_prepare_wy_repr_bwd_full =
    "npu_prepare_wy_repr_bwd_full(Tensor k, Tensor v, Tensor beta, Tensor A, "
    "Tensor dA, Tensor dw, Tensor du, Tensor g, Tensor? cu_seqlens, "
    "Tensor? chunk_indices, int chunk_size, int stream) "
    "-> (Tensor, Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor, Tensor> run_npu_prepare_wy_repr_bwd_full(
    Tensor k, Tensor v, Tensor beta, Tensor A, Tensor dA, Tensor dw, Tensor du,
    Tensor g, std::optional<Tensor> cu_seqlens,
    std::optional<Tensor> chunk_indices, int64_t chunk_size, int64_t stream) {
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta v_meta = meta_of(v);
  const TensorMeta beta_meta = meta_of(beta);
  const TensorMeta g_meta = meta_of(g);
  Tensor out_dk = allocate_like(k_meta);
  Tensor out_dv = allocate_like(v_meta);
  Tensor out_dbeta = allocate_like(beta_meta);
  Tensor out_dg = allocate_like(g_meta);
  FLA_STABLE_EXEC(
      "aclnnPrepareWyReprBwdFull", k_meta, stream,
      tensor(k_meta), tensor(v_meta), tensor(beta_meta), tensor(meta_of(A)),
      tensor(meta_of(dA)), tensor(meta_of(dw)), tensor(meta_of(du)),
      tensor(g_meta), int_array(cu_seqlens), int_array(chunk_indices),
      scalar(chunk_size), out_tensor(meta_of(out_dk)),
      out_tensor(meta_of(out_dv)), out_tensor(meta_of(out_dbeta)),
      out_tensor(meta_of(out_dg)));
  return std::make_tuple(out_dk, out_dv, out_dbeta, out_dg);
}

constexpr const char* kSchema_prepare_wy_repr_bwd =
    "npu_prepare_wy_repr_bwd(Tensor k, Tensor v, Tensor beta, Tensor A, "
    "Tensor dw, Tensor du, Tensor g, Tensor? cu_seqlens, "
    "Tensor? chunk_indices, int chunk_size, int stream) "
    "-> (Tensor, Tensor, Tensor, Tensor)";

std::tuple<Tensor, Tensor, Tensor, Tensor> run_npu_prepare_wy_repr_bwd(
    Tensor k, Tensor v, Tensor beta, Tensor A, Tensor dw, Tensor du, Tensor g,
    std::optional<Tensor> cu_seqlens, std::optional<Tensor> chunk_indices,
    int64_t chunk_size, int64_t stream) {
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta v_meta = meta_of(v);
  const TensorMeta beta_meta = meta_of(beta);
  const TensorMeta g_meta = meta_of(g);
  Tensor out_dk = allocate_like(k_meta);
  Tensor out_dv = allocate_like(v_meta);
  Tensor out_dbeta = allocate_like(beta_meta);
  Tensor out_dg = allocate_like(g_meta);
  FLA_STABLE_EXEC(
      "aclnnPrepareWyReprBwd", k_meta, stream, tensor(k_meta), tensor(v_meta),
      tensor(beta_meta), tensor(meta_of(A)), tensor(meta_of(dw)),
      tensor(meta_of(du)), tensor(g_meta), int_array(cu_seqlens),
      int_array(chunk_indices), scalar(chunk_size), out_tensor(meta_of(out_dk)),
      out_tensor(meta_of(out_dv)), out_tensor(meta_of(out_dbeta)),
      out_tensor(meta_of(out_dg)));
  return std::make_tuple(out_dk, out_dv, out_dbeta, out_dg);
}

// ---------------------------------------------------------------------------
// npu_recompute_w_u_fwd
// ---------------------------------------------------------------------------

constexpr const char* kSchema_recompute_w_u_fwd =
    "npu_recompute_w_u_fwd(Tensor k, Tensor v, Tensor beta, Tensor A, "
    "Tensor? g, Tensor? gk, Tensor? cu_seqlens, Tensor? chunk_indices, "
    "int chunk_size, int stream) -> (Tensor, Tensor)";

std::tuple<Tensor, Tensor> run_npu_recompute_w_u_fwd(
    Tensor k, Tensor v, Tensor beta, Tensor A, std::optional<Tensor> g,
    std::optional<Tensor> gk, std::optional<Tensor> cu_seqlens,
    std::optional<Tensor> chunk_indices, int64_t chunk_size, int64_t stream) {
  const TensorMeta k_meta = meta_of(k);
  const TensorMeta v_meta = meta_of(v);
  Tensor out_w = allocate_sizes(
      {size_of(v_meta, 0), size_of(v_meta, 1), size_of(v_meta, 2),
       size_of(k_meta, 3)},
      k_meta.scalar_type, k_meta);
  Tensor out_u = allocate_like(v_meta);
  FLA_STABLE_EXEC("aclnnRecomputeWUFwd", k_meta, stream, tensor(k_meta),
                  tensor(v_meta), tensor(meta_of(beta)), tensor(meta_of(A)),
                  optional_tensor(g), optional_tensor(gk),
                  int_array(cu_seqlens), int_array(chunk_indices),
                  scalar(chunk_size), out_tensor(meta_of(out_w)),
                  out_tensor(meta_of(out_u)));
  return std::make_tuple(out_w, out_u);
}

// ---------------------------------------------------------------------------
// npu_solve_tri
// ---------------------------------------------------------------------------

// The kernel takes the layout as a `char*`.  Its lowercase spelling is what
// the reference sends, so the table keeps it; a caller passing anything else
// gets an error from `cstr` rather than a string handed straight to aclnn.
constexpr const char* kSolveTriLayoutNames[] = {"bsnd", "bnsd", "tnd", "ntd"};

constexpr const char* kSchema_solve_tri =
    "npu_solve_tri(Tensor x, Tensor? cu_seqlens, Tensor? chunk_indices, "
    "int layout, int stream) -> Tensor";

Tensor run_npu_solve_tri(Tensor x, std::optional<Tensor> cu_seqlens,
                         std::optional<Tensor> chunk_indices, int64_t layout,
                         int64_t stream) {
  const TensorMeta x_meta = meta_of(x);
  Tensor out = allocate_sizes(x_meta.sizes, x_meta.scalar_type, x_meta);
  FLA_STABLE_EXEC("aclnnSolveTri", x_meta, stream, tensor(x_meta),
                  int_array(cu_seqlens), int_array(chunk_indices),
                  cstr(kSolveTriLayoutNames, layout), out_tensor(meta_of(out)));
  return out;
}

}  // namespace
