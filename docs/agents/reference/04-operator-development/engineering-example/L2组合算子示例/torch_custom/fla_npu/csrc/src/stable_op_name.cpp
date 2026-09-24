/**
 * 示例文件（形态 B）：torch_custom/fla_npu/csrc/src/stable_op_name.cpp
 *
 * 场景：该算子已发布 V1（`aclnnOpName`），现在要新增 V2（`aclnnOpNameV2`）。
 * 本文件是 `stable_op_name.cpp` 的**增量**：schema 不变，只在 `run_` 里加一条分支。
 *
 * 注意事项：
 *   1. 一个 schema ↔ 一个 `run_`：V1/V2 都由同一个 `kSchema_op_name` 进入，
 *      在 `run_` 里按参数/形状选择下发哪条 aclnn（仓内真实例子：`stable_chunk_kda_fwd.cpp`
 *      对 `aclnnChunkKdaFwdV2` / `aclnnChunkKdaFwd` 的分支）。
 *   2. 场景判据要只依据文档化的支持范围（dtype/layout/chunk_size/连续性等），并且与
 *      `aclnn_op_name_v2.h` 头文件注释、`docs/api.md` 的支持范围逐条一致。
 *   3. 显式打开的新开关不在 V2 范围内时，Python wrapper 先拒绝（给出可读报错），
 *      `run_` 侧不要"悄悄回落到 V1"——回落只发生在开关保持默认值时。
 *   4. V1 的形参列表、顺序、含义一律不动；V2 的新增输出用可选槽位表达（`std::optional<Tensor>`）。
 *   5. FLA_STABLE_EXEC 的实参顺序必须与对应 aclnn 头文件一致（V1 对 V1 头、V2 对 V2 头）。
 */

// ... 文件头部：schema（不变）与 run_ 签名（不变） ...

  // 场景选择：命中 V2 的组合场景走 aclnnOpNameV2，其余回落签名未变的 aclnnOpName。
  const bool useV2 = op_name_supports_v2(x_meta, layout, chunk_size, tail_mode);
  if (useV2) {
    FLA_STABLE_EXEC("aclnnOpNameV2", x_meta, stream, tensor(x_meta),
                    tensor(meta_of(g)), optional_tensor(a_log),
                    optional_tensor(initial_state), int_array(cu_seqlens),
                    int_array(chunk_indices), cstr(kOpNameLayoutNames, layout),
                    scalar(scale), scalar(chunk_size), scalar(epsilon),
                    scalar(tail_mode), out_tensor(meta_of(y)),
                    optional_tensor(state), optional_tensor(x_norm),
                    optional_tensor(tail));
  } else {
    FLA_STABLE_EXEC("aclnnOpName", x_meta, stream, tensor(x_meta),
                    tensor(meta_of(g)), optional_tensor(a_log),
                    optional_tensor(initial_state), int_array(cu_seqlens),
                    int_array(chunk_indices), cstr(kOpNameLayoutNames, layout),
                    scalar(scale), scalar(chunk_size), scalar(epsilon),
                    out_tensor(meta_of(y)), optional_tensor(state),
                    optional_tensor(x_norm));
  }
  return {y, state, x_norm, tail};
