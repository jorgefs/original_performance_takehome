import importlib
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


def build_kernel(module_name):
    mod = importlib.import_module(module_name)
    kb = mod.KernelBuilder()
    kb.build_kernel(10, 2 ** (10 + 1) - 1, 256, 16)
    return kb.instrs


def count_stats(instrs):
    load_offset = 0
    load_scalar = 0
    vload = 0
    flow_vselect = 0
    valu_hist = Counter()

    for instr in instrs:
        # valu histogram
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


def format_hist(hist, max_k=7):
    keys = list(range(0, max_k + 1))
    return ", ".join([f"{k}:{hist.get(k, 0)}" for k in keys])


def main():
    base = count_stats(build_kernel("perf_takehome_1694"))
    new = count_stats(build_kernel("perf_takehome"))

    print("# Kernel stats (benchmark 10/16/256)\n")
    print("metric | before (perf_takehome_1694) | after (perf_takehome)\n")
    print("------ | --------------------------- | ----------------------\n")
    print(f"load_offset | {base['load_offset']} | {new['load_offset']}")
    print(f"load (scalar) | {base['load']} | {new['load']}")
    print(f"vload | {base['vload']} | {new['vload']}")
    print(f"flow:vselect | {base['flow_vselect']} | {new['flow_vselect']}")
    print("\n# VALU slots per bundle histogram\n")
    print("type | hist (0..7)\n")
    print("---- | -----------\n")
    print(f"before | {format_hist(base['valu_hist'])}")
    print(f"after | {format_hist(new['valu_hist'])}")


if __name__ == "__main__":
    main()
