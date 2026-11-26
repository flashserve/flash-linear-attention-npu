"""Run model-shaped KDA forward and backward on one A2 or A5 device."""
import argparse
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import monotonic
import traceback

from daily_plan import daily_plan


def execute(case, cu, width, device):
    import torch
    from examples import chunk_kda as example
    from reference import reference

    args = example.parse_args([])
    args.batch, args.tokens, args.query_heads = case["B"], case["T"], case["H"]
    args.value_heads, args.key_dim, args.value_dim = args.query_heads, width, width
    args.seed = case["tensor_seed"]
    tensors = example.make_inputs(args, torch.bfloat16, torch.device("cpu"))
    grad = torch.randn(tensors["v"].shape, generator=torch.Generator().manual_seed(args.seed + 1))
    grad = grad.bfloat16()
    q, k, v, raw, beta = [tensors[x].to(device).transpose(1, 2).contiguous()
                          for x in ("q", "k", "v", "g", "beta")]
    a, bias = tensors["A_log"].to(device), tensors["dt_bias"].to(device)
    qhat, khat = torch.empty_like(q), torch.empty_like(k)
    qr, kr = [torch.empty(q.shape[:-1], dtype=torch.float32, device=device) for _ in range(2)]
    common = dict(cu_seqlens=cu, safe_gate=True, lower_bound=case["lower_bound"],
                  use_gate_in_kernel=True, A_log=a, use_exp2=True, disable_recompute=True)
    o, final, gk, aqk, akk, w, _, qg, kg, vnew, h, _ = example.chunk_kda_fwd(
        q, k, v, raw, beta, width ** -0.5, 64, layout="BNSD", dt_bias=bias,
        output_final_state=True, use_qk_l2norm_in_kernel=True, q_hat_out=qhat,
        k_hat_out=khat, q_rstd_out=qr, k_rstd_out=kr, **common)
    dq, dk, dv, dbeta, dg, *_ = example.chunk_kda_bwd(
        qhat, khat, tensors["v"].to(device), tensors["beta"].to(device),
        gk, aqk, akk, w, qg, kg, vnew, h, grad.to(device), width ** -0.5,
        q_rstd=qr, k_rstd=kr, layout="BSND", raw_g=tensors["g"].to(device),
        chunk_size=64, dt_bias=bias.reshape(args.query_heads, width), **common)
    torch.npu.synchronize()
    actual = {name: value.cpu() for name, value in
              dict(o=o, final_state=final, dq=dq, dk=dk, dv=dv, dbeta=dbeta, dg=dg).items()}
    expected = reference(tensors, grad, cu, case["lower_bound"])
    metrics = {}
    for name, value in actual.items():
        if value.shape != expected[name].shape:
            raise ValueError(f"{name}: got {value.shape}, expected {expected[name].shape}")
        thresholds = example.tolerance_for(args, "o" if name == "final_state" else name)
        passed, detail = example.accuracy_metric(name, value, expected[name], *thresholds)
        metrics[name] = {"passed": passed, "detail": detail}
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("a2", "a5"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--case", action="append", help="case ID; repeat to run a subset")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    import torch_npu  # noqa: F401

    torch.set_num_threads(args.cpu_threads)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    chip = torch.npu.get_device_name(device)
    if ("910B" if args.platform == "a2" else "950") not in chip:
        parser.error(f"--platform {args.platform} does not match {chip}")
    day = args.date or datetime.now(timezone(timedelta(hours=8))).date()
    catalog = json.loads(Path(__file__).with_name("cases.json").read_text())
    plan = daily_plan(catalog, day)
    ids = {c["id"] for c in plan["cases"]}
    if args.case and set(args.case) - ids:
        parser.error(f"unknown case IDs: {sorted(set(args.case) - ids)}")
    cases = [c for c in plan["cases"] if not args.case or c["id"] in args.case]
    report = {"date": day.isoformat(), "platform": args.platform, "device": chip,
              "wheel": version(f"flash-linear-attention-npu-{args.platform}"),
              "safe_gate": True, "planned": len(cases), "passed": False, "cases": []}
    args.report.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.report.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        tmp.replace(args.report)

    save()
    for case in cases:
        print(f"Starting {case['id']} lower_bound={case['lower_bound']}", flush=True)
        started = monotonic()
        record = dict(case)
        try:
            cu = plan["cu_seqlens"].get(case.get("group"))
            record["metrics"] = execute(case, cu, plan["head_dim"], device)
            record["passed"] = all(m["passed"] for m in record["metrics"].values())
        except Exception:
            record.update(passed=False, error=traceback.format_exc())
        record["seconds"] = round(monotonic() - started, 2)
        report["cases"].append(record)
        save()
        print(json.dumps(record), flush=True)
    report["passed"] = all(c["passed"] for c in report["cases"])
    save()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
