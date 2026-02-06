import os
import re
import sys
import difflib
from collections import Counter

ROOT = r"C:\\Users\\OEM\\proyectos_gito\\test2\\original_performance_takehome"
ASM_1615 = os.path.join(ROOT, "investigaciones", "logs", "kernel_asm_1615.txt")

ENGINE_ORDER = ["flow", "load", "store", "alu", "valu", "debug"]

LINE_RE = re.compile(r"^\s*\d+:")
SEG_RE = re.compile(r"^\s*(\w+):\((.*)\)\s*$")


def read_logical_lines(path):
    logical = []
    cur = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if LINE_RE.match(line):
                if cur is not None:
                    logical.append(cur)
                cur = line
            else:
                if cur is None:
                    continue
                cur += " " + line.strip()
        if cur is not None:
            logical.append(cur)
    return logical


def normalize_asm_lines(path):
    lines = []
    for line in read_logical_lines(path):
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        idx = parts[0].strip()
        slots = [s.strip() for s in parts[1].split("|")]
        buckets = {k: [] for k in ENGINE_ORDER}
        for slot in slots:
            m = SEG_RE.match(slot)
            if not m:
                continue
            eng = m.group(1)
            body = m.group(2)
            if eng not in buckets:
                buckets[eng] = []
            buckets[eng].append(body)
        segs = []
        for eng in ENGINE_ORDER:
            if buckets.get(eng):
                for body in buckets[eng]:
                    segs.append(f"{eng}:({body})")
        lines.append(f"{idx}: " + " | ".join(segs))
    return lines


def dump_current_kernel():
    sys.path.insert(0, ROOT)
    from perf_takehome import KernelBuilder
    kb = KernelBuilder()
    kb.build_kernel(10, 2 ** 11 - 1, 256, 16)
    lines = []
    for i, instr in enumerate(kb.instrs):
        segs = []
        for eng in ENGINE_ORDER:
            if eng not in instr:
                continue
            for slot in instr[eng]:
                segs.append(f"{eng}:({slot})")
        lines.append(f"{i}: " + " | ".join(segs))
    return lines


def summarize(lines):
    counts = Counter()
    for line in lines:
        if ":" not in line:
            continue
        _, rest = line.split(":", 1)
        for eng in ENGINE_ORDER:
            if f"{eng}:(" in rest:
                counts[eng] += rest.count(f"{eng}:(")
    return counts


def write(path, lines):
    with open(path, "w", encoding="ascii", errors="ignore") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    asm_1615_lines = normalize_asm_lines(ASM_1615)
    cur_lines = dump_current_kernel()

    out_1615 = os.path.join(ROOT, "investigaciones", "analysis", "asm_1615_norm.txt")
    out_cur = os.path.join(ROOT, "investigaciones", "analysis", "asm_current_norm.txt")
    write(out_1615, asm_1615_lines)
    write(out_cur, cur_lines)

    diff = list(difflib.unified_diff(
        asm_1615_lines,
        cur_lines,
        fromfile="asm_1615",
        tofile="asm_current",
        n=3,
    ))

    summary_1615 = summarize(asm_1615_lines)
    summary_cur = summarize(cur_lines)

    report = []
    report.append("# ASM bundle diff (1615 vs current)\n")
    report.append(f"bundles_1615: {len(asm_1615_lines)}")
    report.append(f"bundles_current: {len(cur_lines)}\n")
    report.append("## Slot counts (by engine)\n")
    report.append("engine | asm_1615 | current")
    report.append("------ | -------- | -------")
    for eng in ENGINE_ORDER:
        report.append(f"{eng} | {summary_1615.get(eng,0)} | {summary_cur.get(eng,0)}")
    report.append("\n## First 200 diff lines\n")
    report.extend(diff[:200])

    out_report = os.path.join(ROOT, "investigaciones", "analysis", "asm_diff.md")
    with open(out_report, "w", encoding="ascii", errors="ignore") as f:
        for line in report:
            f.write(line + "\n")

    print(out_report)


if __name__ == "__main__":
    main()