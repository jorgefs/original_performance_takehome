# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is Anthropic's original performance engineering take-home challenge. The goal is to optimize a kernel that performs parallel tree traversal with hashing on a custom VLIW SIMD architecture simulator.

## Commands

```bash
# Run all tests (correctness + performance)
python perf_takehome.py

# Run a specific test
python perf_takehome.py Tests.test_kernel_cycles

# Generate a trace file for debugging (opens in Perfetto)
python perf_takehome.py Tests.test_kernel_trace
# Then in another terminal:
python watch_trace.py

# Run submission tests to see which performance thresholds you pass
python tests/submission_tests.py
```

## Architecture

### Simulated Machine (`problem.py`)
- **VLIW (Very Large Instruction Word)**: Cores execute multiple "slots" per cycle in parallel across different engines
- **SIMD**: Vector operations on VLEN=8 elements
- **Engines and slot limits per cycle**: `alu: 12`, `valu: 6`, `load: 2`, `store: 2`, `flow: 1`
- **Scratch space**: 1536 words of fast memory (serves as registers/cache)
- **Memory**: Main memory for tree values and input/output

### Key Instructions
- Scalar: `load`, `store`, `const`, ALU ops (`+`, `-`, `*`, `^`, `&`, `|`, `<<`, `>>`, etc.)
- Vector: `vload`, `vstore`, `vbroadcast`, vector ALU ops, `multiply_add`, `vselect`
- Control: `select`, `add_imm`, `cond_jump`, `halt`, `pause`

### Kernel (`perf_takehome.py`)
- `KernelBuilder`: Generates instruction sequences for the simulator
- `build_kernel()`: The main optimization target - implements parallel tree traversal
- `build()`: VLIW instruction scheduler that packs slots into bundles respecting dependencies
- Hash function: 6-stage computation defined in `HASH_STAGES`

### The Algorithm
Each batch element traverses a binary tree for multiple rounds:
1. Load node value at current index
2. Compute `val = myhash(val ^ node_val)`
3. Branch left if val is even, right if odd
4. Wrap to root when reaching leaves

### Performance Target
- Baseline: 147,734 cycles
- Benchmark parameters: `forest_height=10`, `rounds=16`, `batch_size=256`
- Current best known performances listed in README range from ~2164 down to ~1363 cycles

## Debugging

The trace visualization (Perfetto) shows:
- Per-engine slot utilization over time
- Scratch memory access patterns
- Use `debug` engine for assertions: `("compare", addr, key)` and `("vcompare", addr, keys)`
- Match `pause` instructions with `yield` statements in `reference_kernel2` for intermediate checks
