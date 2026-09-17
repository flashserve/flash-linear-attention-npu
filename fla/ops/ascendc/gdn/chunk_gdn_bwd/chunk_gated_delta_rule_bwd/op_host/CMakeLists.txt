add_op_to_compiled_list()
set(chunk_gated_delta_rule_bwd_depends
    "fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gdn_bwd_intra;fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h;fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu;fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_finalize"
    CACHE STRING "Kernel source dependencies for chunk_gated_delta_rule_bwd" FORCE)

add_modules_sources(OPTYPE chunk_gated_delta_rule_bwd ACLNNTYPE aclnn_exclude)
