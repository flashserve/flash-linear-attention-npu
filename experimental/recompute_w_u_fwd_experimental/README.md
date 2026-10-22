# RecomputeWUFwdExperimental

This operator is a build and packaging fixture for the experimental directory.
It is derived from the stable `recompute_w_u_fwd` implementation on upstream
`main` at `ed987cdf5c3083633542f76ce34c1e1a62aecefa`. The operator type,
kernel entry, `opFile`, aclnn exports, and source filenames have distinct
experimental names. The stable operator source is unchanged.

Build and install validation does not require running the operator:

```bash
bash build.sh --pkg --experimental --soc=ascend950 \
  --ops=recompute_w_u_fwd_experimental -j4
./build_out/fla_npu_linux-x86_64.run \
  --install-path=/tmp/fla-experimental-recompute-opp --quiet
```

The installed OPP must contain `recompute_w_u_fwd_experimental` and must not
contain the stable `recompute_w_u_fwd` operator. The stable build is checked
separately with `--ops=recompute_w_u_fwd` and without `--experimental`.
