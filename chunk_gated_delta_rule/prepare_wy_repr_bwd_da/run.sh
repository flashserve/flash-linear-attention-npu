export ASCEND_RT_VISIBLE_DEVICES=1

clear

op=prepare_wy_repr_bwd_da

rm -rf build build_out

bash build.sh --pkg --ops=$op

build_out/cann-*.run

# bash build.sh --run_example prepare_wy_repr_bwd_da eager cust > /data/yzq/OP_LOG/DA_Example.log

python /data/yzq/ops-transformer_GDN/chunk_gated_delta_rule/prepare_wy_repr_bwd_da/test/test_da.py > /data/yzq/OP_LOG/DA.log