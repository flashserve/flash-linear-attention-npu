# -----------------------------------------------------------------------------------------------------------
# chunk_kda_fwd_fused(chunk32 大融合,issue #829)
# kernel 对编译 flag 敏感:
#   以下 OPTIONS 与手工编译(9.3.0 ccec)逐项对应,缺/多都可能改变 codegen
#   或数值行为;修改后必须重跑逐位对齐验证。
#   注意:不要加 -mllvm -cce-vf-stack-size,当前实现不需要该选项。
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_fwd_fused_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kda_fwd_fused ACLNNTYPE aclnn_exclude)

add_ops_compile_options(
    OP_NAME ChunkKdaFwdFused
    OPTIONS --cce-auto-sync=off
            -mllvm -cce-aicore-function-stack-size=0x4000
            -mllvm -cce-aicore-addr-transform=true
            -mllvm -cce-aicore-or-combine=false
            -mllvm -instcombine-code-sinking=false
            -Xclang -fcce-vf-vl=256
            -mllvm -cce-aicore-jump-expand=true
            -mllvm -cce-aicore-mask-opt=false
            --cce-simd-vf-fusion=false
            -DENABLE_MOVVP -DPERF_API_NORM
            -mllvm -cce-aicore-dcci-before-kernel-end=false
            -mllvm -cce-aicore-dcci-insert-for-scalar=false
)
