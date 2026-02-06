import math
from collections import defaultdict, Counter
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from problem import Tree, Input, myhash


def depth_from_idx(idx):
    # idx is 0-based; depth = floor(log2(idx+1))
    return int(math.log2(idx + 1)) if idx >= 0 else 0


def run_stats(forest_height, rounds=16, batch_size=256, seed=123):
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)

    # stats[depth] -> list of per-round unique counts
    unique_counts = defaultdict(list)
    # stats[depth] -> Counter(bucket_size -> occurrences)
    bucket_hist = defaultdict(Counter)

    for r in range(rounds):
        # bucket by depth and idx
        buckets = defaultdict(list)
        for i, idx in enumerate(inp.indices):
            d = depth_from_idx(idx)
            buckets[(d, idx)].append(i)
        # summarize per depth
        depth_to_nodes = defaultdict(list)
        for (d, idx), paths in buckets.items():
            depth_to_nodes[d].append((idx, paths))
        for d, nodes in depth_to_nodes.items():
            unique_counts[d].append(len(nodes))
            for _idx, paths in nodes:
                bucket_hist[d][len(paths)] += 1

        # advance one round
        for i in range(batch_size):
            idx = inp.indices[i]
            val = inp.values[i]
            node_val = forest.values[idx]
            val = myhash(val ^ node_val)
            idx = 2 * idx + (1 if val % 2 == 0 else 2)
            if idx >= len(forest.values):
                idx = 0
            inp.values[i] = val
            inp.indices[i] = idx

    return unique_counts, bucket_hist


def fmt_hist(counter, max_bins=8):
    # show top buckets by size (largest count)
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    shown = items[:max_bins]
    parts = [f"{size}:{count}" for size, count in shown]
    if len(items) > max_bins:
        parts.append("...")
    return ", ".join(parts) if parts else "-"


def summarize(forest_height):
    unique_counts, bucket_hist = run_stats(forest_height)
    lines = []
    lines.append(f"# Bucket stats (forest_height={forest_height}, rounds=16, batch=256)\n")
    lines.append("depth | U_d(avg) | U_d(min..max) | collision(avg) | bucket_hist (size:count)\n")
    lines.append("----- | -------- | ------------- | -------------- | --------------------------\n")
    for d in range(forest_height + 1):
        if d not in unique_counts:
            continue
        counts = unique_counts[d]
        avg = sum(counts) / len(counts)
        mn = min(counts)
        mx = max(counts)
        collision = 1.0 - (avg / 256.0)
        hist = fmt_hist(bucket_hist[d])
        lines.append(f"{d} | {avg:.1f} | {mn}..{mx} | {collision:.3f} | {hist}\n")
    return "".join(lines)


def main():
    for h in (8, 9, 10):
        print(summarize(h))


if __name__ == "__main__":
    main()
