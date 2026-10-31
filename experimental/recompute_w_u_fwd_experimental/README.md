# RecomputeWUFwdExperimental

This is the full experimental port of the stable `recompute_w_u_fwd` source from
upstream `main@ed987cdf5c3083633542f76ce34c1e1a62aecefa`, restored after
the temporary compile-only reduction. Its CANN operator type, kernel entry,
`opFile`, aclnn exports, and source filenames have distinct experimental names.
The stable operator source is unchanged.

Build and install into an isolated OPP root:

```bash
bash build.sh --pkg --experimental --soc=ascend950 \
  --ops=recompute_w_u_fwd_experimental -j4
./build_out/fla_npu_linux-x86_64.run \
  --install-path=/tmp/fla-experimental-recompute-opp --quiet
```

The installed OPP must contain `recompute_w_u_fwd_experimental` and must not
contain the stable `recompute_w_u_fwd` operator. The stable build is checked
separately with `--ops=recompute_w_u_fwd` and without `--experimental`.

On an Ascend 950 with CANN initialized, the isolated runtime smoke test uses
only the experimental ACLNN exports and checks both outputs against an identity
matrix case:

```bash
bash experimental/recompute_w_u_fwd_experimental/tests/run_aclnn_smoke.sh \
  /tmp/fla-experimental-recompute-opp 0 128 64
bash experimental/recompute_w_u_fwd_experimental/tests/run_aclnn_smoke.sh \
  /tmp/fla-experimental-recompute-opp 0 256 128
```

`bash ci/run_experimental_validation.sh` runs discovery tests, build, isolated
install, package/source audit, and both ACLNN smoke cases. This is not a full
ATK accuracy, performance, A2/A3, or GitHub NPU CI result. Do not overlay this
run package onto the stable wheel/OPP while it shares the stable vendor name.
