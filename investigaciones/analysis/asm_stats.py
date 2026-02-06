import os
import re
from collections import Counter

ROOT = r"C:\\Users\\OEM\\proyectos_gito\\test2\\original_performance_takehome"
ASM_PATH = os.path.join(ROOT, "investigaciones", "logs", "kernel_asm_1615.txt")

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


def count_asm_stats(path):
    load_offset = 0
    load_scalar = 0
    vload = 0
    flow_vselect = 0
    valu_hist = Counter()

    for line in read_logical_lines(path):
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        slots = [s.strip() for s in parts[1].split("|")]
        vcount = 0
        for slot in slots:
            m = SEG_RE.match(slot)
            if not m:
                continue
            eng = m.group(1)
            body = m.group(2)
            if eng == "valu":
                vcount += 1
            if eng == "load":
                op = body.strip().lstrip("'\"")
                if op.startswith("load_offset"):
                    load_offset += 1
                elif op.startswith("load"):
                    load_scalar += 1
                elif op.startswith("vload"):
                    vload += 1
            if eng == "flow":
                op = body.strip().lstrip("'\"")
                if op.startswith("vselect"):
                    flow_vselect += 1
        valu_hist[vcount] += 1

    return {
        "load_offset": load_offset,
        "load": load_scalar,
        "vload": vload,
        "flow_vselect": flow_vselect,
        "valu_hist": valu_hist,
    }


def count_current_kernel():
    import sys
    sys.path.insert(0, ROOT)
    from perf_takehome import KernelBuilder
    kb = KernelBuilder()
    kb.build_kernel(10, 2 ** 11 - 1, 256, 16)
    instrs = kb.instrs

    load_offset = 0
    load_scalar = 0
    vload = 0
    flow_vselect = 0
    valu_hist = Counter()

    for instr in instrs:
        vslots = len(instr.get("valu", [])) if isinstance(instr, dict) else 0
        valu_hist[vslots] += 1
        for engine, slots in instr.items():
            if engine == "load":
                for slot in slots:
                    if slot[0] == "load_offset":
                        load_offset += 1
                    elif slot[0] == "load":
                        load_scalar += 1
                    elif slot[0] == "vload":
                        vload += 1
            elif engine == "flow":
                for slot in slots:
                    if slot[0] == "vselect":
                        flow_vselect += 1

    return {
        "load_offset": load_offset,
        "load": load_scalar,
        "vload": vload,
        "flow_vselect": flow_vselect,
        "valu_hist": valu_hist,
    }


def fmt_hist(hist, max_k=7):
    keys = list(range(0, max_k + 1))
    return ", ".join([f"{k}:{hist.get(k, 0)}" for k in keys])


def main():
    asm = count_asm_stats(ASM_PATH)
    cur = count_current_kernel()

    print("# ASM 1615 vs current kernel (10/16/256)\n")
    print("metric | asm_1615 | current")
    print("------ | -------- | -------")
    print(f"load_offset | {asm['load_offset']} | {cur['load_offset']}")
    print(f"load (scalar) | {asm['load']} | {cur['load']}")
    print(f"vload | {asm['vload']} | {cur['vload']}")
    print(f"flow:vselect | {asm['flow_vselect']} | {cur['flow_vselect']}")
    print("\n# VALU slots per bundle histogram\n")
    print("type | asm_1615 | current")
    print("---- | -------- | -------")
    print(f"hist | {fmt_hist(asm['valu_hist'])} | {fmt_hist(cur['valu_hist'])}")


if __name__ == "__main__":
    main()
