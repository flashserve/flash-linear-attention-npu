import csv
import sys
import os
import glob
import json


# 分别统计 da 算子与 full 算子
OP_NAME_DA = "PrepareWyReprBwdDa"
OP_NAME_FULL = "PrepareWyReprBwdFull"


def parse_input_shapes(shapes_str):
    parts = shapes_str.split(";")
    is_varlen = len(parts) >= 9
    k_shape = parts[0].split(",")
    v_shape = parts[1].split(",")
    a_shape = parts[3].split(",")
    if len(k_shape) == 4 and len(v_shape) == 4 and len(a_shape) == 4:
        B = k_shape[0]
        HK = k_shape[1]
        T = k_shape[2]
        K = k_shape[3]
        HV = v_shape[1]
        V = v_shape[3]
        chunk_size = a_shape[3]
        return B, HK, HV, T, K, V, chunk_size, is_varlen
    return None


def find_csv(prof_dir):
    candidates = glob.glob(os.path.join(prof_dir, "**/mindstudio_profiler_output/op_summary_*.csv"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(prof_dir, "op_summary_*.csv"), recursive=True)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_case_index(json_path):
    index = {}
    if not json_path or not os.path.isfile(json_path):
        return index
    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    for case in cases:
        if not case.get("enabled", True):
            continue
        varlen = case.get("varlen", False)
        key = (
            str(case["B"]), str(case["query_head"]), str(case["value_head"]),
            str(case["T"]), str(case["Kdim"]), str(case["Vdim"]),
            str(case["chunk_size"]), str(varlen),
        )
        index[key] = {
            "name": case.get("name", ""),
            "dtype": case.get("dtype", ""),
            "gtype": case.get("gtype", ""),
            "varlen": varlen,
            "mean_len": case.get("mean_len", ""),
        }
    return index


def collect_op_rows(csv_path, op_name, case_index):
    """收集指定算子的耗时行，返回 dict: key -> row info"""
    rows = {}
    if not os.path.isfile(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_type = row.get("OP Type", "").strip()
            if op_type != op_name:
                continue
            duration = row.get("Task Duration(us)", "").strip()
            input_shapes = row.get("Input Shapes", "").strip().strip('"')
            parsed = parse_input_shapes(input_shapes)
            if parsed is None:
                continue
            B, HK, HV, T, K, V, chunk_size, is_varlen = parsed
            key = (B, HK, HV, T, K, V, chunk_size, str(is_varlen))
            case_info = case_index.get(key, {})
            rows[key] = {
                "name": case_info.get("name", ""),
                "B": B, "HK": HK, "HV": HV, "T": T,
                "K": K, "V": V, "chunk_size": chunk_size,
                "dtype": case_info.get("dtype", ""),
                "gtype": case_info.get("gtype", ""),
                "varlen": case_info.get("varlen", is_varlen),
                "mean_len": case_info.get("mean_len", ""),
                "duration_us": duration,
            }
    return rows


def safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def gen_report(csv_path, output_path=None, json_path=None):
    case_index = load_case_index(json_path)

    # 分别收集 da 与 full 算子的耗时
    da_rows = collect_op_rows(csv_path, OP_NAME_DA, case_index)
    full_rows = collect_op_rows(csv_path, OP_NAME_FULL, case_index)

    if not da_rows and not full_rows:
        print(f"[WARN] No {OP_NAME_DA} or {OP_NAME_FULL} rows found in {csv_path}")
        return

    # 统一以 JSON case 顺序为准 (case_index 是 dict，保持插入顺序)
    ordered_keys = list(case_index.keys())

    # ---- 第一部分: da 算子耗时 ----
    if da_rows:
        header = f"{'name':<16} {'B':>4} {'HK':>4} {'HV':>4} {'T':>8} {'K':>4} {'V':>4} {'chunk':>5} {'dtype':>6} {'gtype':>6} {'varlen':>6} {'mean_len':>8} {'Duration(us)':>14}"
        sep = "-" * len(header)
        print(f"\n{'='*len(header)}")
        print(f"  {OP_NAME_DA} Performance Report")
        print(f"{'='*len(header)}")
        print(header)
        print(sep)
        for key in ordered_keys:
            if key not in da_rows:
                continue
            r = da_rows[key]
            varlen_str = str(r['varlen']) if r['varlen'] != "" else ""
            mean_len_str = str(r['mean_len']) if r['mean_len'] != "" else ""
            print(f"{r['name']:<16} {r['B']:>4} {r['HK']:>4} {r['HV']:>4} {r['T']:>8} {r['K']:>4} {r['V']:>4} {r['chunk_size']:>5} {r['dtype']:>6} {r['gtype']:>6} {varlen_str:>6} {mean_len_str:>8} {r['duration_us']:>14}")
        print(sep)
        print(f"Total: {len(da_rows)} cases")

    # ---- 第二部分: full 算子耗时 ----
    if full_rows:
        header = f"{'name':<16} {'B':>4} {'HK':>4} {'HV':>4} {'T':>8} {'K':>4} {'V':>4} {'chunk':>5} {'dtype':>6} {'gtype':>6} {'varlen':>6} {'mean_len':>8} {'Duration(us)':>14}"
        sep = "-" * len(header)
        print(f"\n{'='*len(header)}")
        print(f"  {OP_NAME_FULL} Performance Report")
        print(f"{'='*len(header)}")
        print(header)
        print(sep)
        for key in ordered_keys:
            if key not in full_rows:
                continue
            r = full_rows[key]
            varlen_str = str(r['varlen']) if r['varlen'] != "" else ""
            mean_len_str = str(r['mean_len']) if r['mean_len'] != "" else ""
            print(f"{r['name']:<16} {r['B']:>4} {r['HK']:>4} {r['HV']:>4} {r['T']:>8} {r['K']:>4} {r['V']:>4} {r['chunk_size']:>5} {r['dtype']:>6} {r['gtype']:>6} {varlen_str:>6} {mean_len_str:>8} {r['duration_us']:>14}")
        print(sep)
        print(f"Total: {len(full_rows)} cases")

    # ---- 第三部分: da + full 加合耗时 (同一 shape) ----
    # 按 JSON case 顺序构建 sum_rows
    sum_rows = {}
    for key in ordered_keys:
        da_info = da_rows.get(key, {})
        full_info = full_rows.get(key, {})
        if not da_info and not full_info:
            continue
        base = da_info or full_info
        da_us = safe_float(da_info.get('duration_us', 0))
        full_us = safe_float(full_info.get('duration_us', 0))
        sum_rows[key] = {
            "name": base.get("name", ""),
            "B": base.get("B", ""), "HK": base.get("HK", ""), "HV": base.get("HV", ""),
            "T": base.get("T", ""), "K": base.get("K", ""), "V": base.get("V", ""),
            "chunk_size": base.get("chunk_size", ""),
            "dtype": base.get("dtype", ""), "gtype": base.get("gtype", ""),
            "varlen": base.get("varlen", ""), "mean_len": base.get("mean_len", ""),
            "da_us": da_us,
            "full_us": full_us,
            "sum_us": da_us + full_us,
        }

    if sum_rows:
        header = f"{'name':<16} {'B':>4} {'HK':>4} {'HV':>4} {'T':>8} {'K':>4} {'V':>4} {'chunk':>5} {'dtype':>6} {'gtype':>6} {'varlen':>6} {'mean_len':>8} {'da(us)':>12} {'full(us)':>12} {'sum(us)':>12}"
        sep = "-" * len(header)
        print(f"\n{'='*len(header)}")
        print(f"  {OP_NAME_DA} + {OP_NAME_FULL} Combined Performance Report")
        print(f"{'='*len(header)}")
        print(header)
        print(sep)
        for key in ordered_keys:
            if key not in sum_rows:
                continue
            r = sum_rows[key]
            varlen_str = str(r['varlen']) if r['varlen'] != "" else ""
            mean_len_str = str(r['mean_len']) if r['mean_len'] != "" else ""
            print(f"{r['name']:<16} {r['B']:>4} {r['HK']:>4} {r['HV']:>4} {r['T']:>8} {r['K']:>4} {r['V']:>4} {r['chunk_size']:>5} {r['dtype']:>6} {r['gtype']:>6} {varlen_str:>6} {mean_len_str:>8} {r['da_us']:>12.2f} {r['full_us']:>12.2f} {r['sum_us']:>12.2f}")
        print(sep)
        print(f"Total: {len(sum_rows)} cases")

    # ---- CSV 输出 ----
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("name,B,HK,HV,T,K,V,chunk_size,dtype,gtype,varlen,mean_len,da_us,full_us,sum_us\n")
            for key in ordered_keys:
                if key not in sum_rows:
                    continue
                r = sum_rows[key]
                f.write(f"{r['name']},{r['B']},{r['HK']},{r['HV']},{r['T']},{r['K']},{r['V']},{r['chunk_size']},{r['dtype']},{r['gtype']},{r['varlen']},{r['mean_len']},{r['da_us']:.2f},{r['full_us']:.2f},{r['sum_us']:.2f}\n")
        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate performance report from msprof op_summary CSV (da + full)")
    parser.add_argument("csv_path", nargs="?", default=None, help="Path to op_summary CSV; if omitted, auto-detect from prof_dir")
    parser.add_argument("--prof-dir", type=str, default=None, help="msprof output directory for auto-detect")
    parser.add_argument("--json", type=str, default=None, help="JSON case file for case info enrichment")
    parser.add_argument("--output", type=str, default=None, help="Output CSV report path")
    args = parser.parse_args()

    csv_path = args.csv_path
    if not csv_path:
        prof_dir = args.prof_dir
        if not prof_dir:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            prof_dir = os.path.join(script_dir, "prof_output")
        csv_path = find_csv(prof_dir)
        if not csv_path:
            print(f"[ERROR] No op_summary CSV found in {prof_dir}")
            sys.exit(1)

    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    gen_report(csv_path, args.output, args.json)
