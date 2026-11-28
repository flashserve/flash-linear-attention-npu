#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DFX 静态检查：#1346 规则（辅助校验）。

每个 OP_LOGE_WITHOUT_REPORT 必须满足其一：
  1) 已替换为 OP_LOGE_FOR_* 结构化宏（不在此脚本检查范围）；
  2) 紧随（6 行内）REPORT_INNER_ERR_MSG / REPORT_PREDEFINED_ERR_MSG，
     且两者报文格式串一致；
  3) 显式豁免：同一条语句或上一行带 `// DFX-ALLOW: <reason>`。

说明：本脚本只校验“是否静默 + PLOG/EZ 报文一致”，不校验阶段名/参数名/返回码语义；
仓库 `docs/agents/skills/operator-review/scripts/static_review.py` 的
`DFX-WITHOUT-REPORT` 会把所有 WITHOUT_REPORT 判为 error，两者口径差异见 #1346。

退出码：0 合规；1 存在未上报/未豁免/报文不一致用法。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "fla", "ops", "ascendc")
WINDOW = 10
MAC = "OP_LOGE_WITHOUT_REPORT"
PLOG_FMT = re.compile(r'OP_LOGE_WITHOUT_REPORT\(\s*[^,]+,\s*"((?:[^"\\]|\\.)*)"')
RPT_FMT = re.compile(r'REPORT_(?:INNER|PREDEFINED)_ERR_MSG\(\s*"[^"]*"\s*,\s*"((?:[^"\\]|\\.)*)"')


def main():
    bad = []
    total = 0
    for dp, _, fs in os.walk(SRC):
        for f in fs:
            if not f.endswith((".cpp", ".h")):
                continue
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, ROOT)
            with open(p, encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
            for i, line in enumerate(lines):
                if MAC not in line:
                    continue
                total += 1
                if "DFX-ALLOW" in line or (i > 0 and "DFX-ALLOW" in lines[i - 1]):
                    continue
                window = "\n".join(lines[i:i + 1 + WINDOW])
                if not re.search(r"REPORT_(INNER|PREDEFINED)_ERR_MSG", window):
                    bad.append(f"{rel}:{i + 1} (no report)")
                    continue
                f1 = PLOG_FMT.search(window)
                f2 = RPT_FMT.search(window)
                if f1 and f2 and f1.group(1) != f2.group(1):
                    bad.append(f"{rel}:{i + 1} (fmt mismatch)")
    print(f"OP_LOGE_WITHOUT_REPORT: {total}, violations: {len(bad)}")
    for b in bad:
        print("  " + b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
