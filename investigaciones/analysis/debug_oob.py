import os
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from tests.frozen_problem import Machine as BaseMachine, build_mem_image, Tree, Input, N_CORES
from perf_takehome import KernelBuilder

class DebugMachine(BaseMachine):
    def store(self, core, *slot):
        match slot:
            case ("store", addr, src):
                addr_val = core.scratch[addr]
                if addr_val < 0 or addr_val >= len(self.mem):
                    print(f"OOB store addr={addr_val} pc={core.pc}")
                    raise IndexError
                self.mem_write[addr_val] = core.scratch[src]
            case ("vstore", addr, src):
                addr_val = core.scratch[addr]
                for vi in range(8):
                    a = addr_val + vi
                    if a < 0 or a >= len(self.mem):
                        print(f"OOB vstore addr={a} pc={core.pc} base={addr_val} vi={vi}")
                        raise IndexError
                    self.mem_write[a] = core.scratch[src + vi]
            case _:
                return super().store(core, *slot)


def main():
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)
    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)
    m = DebugMachine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    m.enable_pause = False
    m.enable_debug = False
    try:
        m.run()
    except Exception:
        traceback.print_exc()
        print("pc", m.cores[0].pc)
        if m.cores[0].pc < len(kb.instrs):
            instr = kb.instrs[m.cores[0].pc]
            print("instr", instr)
            # Print scratch names for any ints in the first slot
            if "alu" in instr:
                slot = instr["alu"][0]
                names = []
                for val in slot[1:]:
                    if isinstance(val, int) and val in kb.debug_info().scratch_map:
                        names.append((val, kb.debug_info().scratch_map[val][0]))
                print("scratch_names", names)
            start = max(0, m.cores[0].pc - 20)
            end = min(len(kb.instrs), m.cores[0].pc + 5)
            print("context:")
            for i in range(start, end):
                print(i, kb.instrs[i])
        eb = kb.scratch.get("extra_base")
        if eb is not None:
            print("extra_base addr", eb, "value", m.cores[0].scratch[eb])
            print("mem[7]", m.mem[7])
            print("mem_len", len(m.mem))
        tvb = kb.scratch.get("tmp_vals_base")
        if tvb is not None:
            print("tmp_vals_base addr", tvb, "value", m.cores[0].scratch[tvb])


if __name__ == "__main__":
    main()
