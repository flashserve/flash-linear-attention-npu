# RecomputeWUFwdExperimental

This is a compile-only fixture, not an implementation of the stable
`recompute_w_u_fwd` computation. It retains a distinct operator type, kernel
entry, `opFile` and aclnn exports. The kernel copies one eight-element FP32
tile to exercise compilation and includes `experimental/_common/build_probe.hpp`
to exercise the isolated common-header path. Do not run it as a recompute op.
The stable operator source is unchanged.

Build and install validation does not require running the operator:

```bash
bash build.sh --pkg --experimental --soc=ascend950 \
  --ops=recompute_w_u_fwd_experimental -j4
./build_out/fla_npu_linux-x86_64.run \
  --install-path=/tmp/fla-experimental-recompute-opp --quiet
```

Check that the installed OPP contains only `recompute_w_u_fwd_experimental`
kernel/config/dynamic/proto files, the independent common header, and the
`aclnnRecomputeWUFwdExperimental*` exports. Check the stable build separately
with `--ops=recompute_w_u_fwd` and without `--experimental`. No operator
execution or numerical validation is part of this fixture.
