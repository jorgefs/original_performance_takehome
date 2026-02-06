"""
Medir cuántos ciclos toma init vs body.
"""

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Agregar medición de init cycles
# Buscar donde se emite el pause después de init

old_pause = '''            # Pause to sync with first yield
            init.append(("flow", ("pause",)))

            # Pack and emit initialization
            init_instrs = self.build(init, vliw=True)
            self.instrs.extend(init_instrs)'''

new_pause = '''            # Pause to sync with first yield
            init.append(("flow", ("pause",)))

            # Pack and emit initialization
            init_instrs = self.build(init, vliw=True)
            self.instrs.extend(init_instrs)
            print(f"INIT: {len(init_instrs)} bundles (cycles)")'''

if old_pause in content:
    content = content.replace(old_pause, new_pause)
    print("Added init measurement")

# Agregar medición de body
old_body_build = '''            body_instrs = self.build(body, vliw=True)
            self.instrs.extend(body_instrs)
            # Unconditional pause to sync with second yield from reference_kernel2
            self.instrs.append({"flow": [("pause",)]})'''

new_body_build = '''            body_instrs = self.build(body, vliw=True)
            self.instrs.extend(body_instrs)
            print(f"BODY: {len(body_instrs)} bundles (cycles)")
            # Unconditional pause to sync with second yield from reference_kernel2
            self.instrs.append({"flow": [("pause",)]})'''

if old_body_build in content:
    content = content.replace(old_body_build, new_body_build)
    print("Added body measurement")

with open('perf_takehome_measure.py', 'w', encoding='utf-8') as f:
    f.write(content)

import subprocess
result = subprocess.run(
    ['python', 'perf_takehome_measure.py', 'Tests.test_kernel_cycles'],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
print(output)
