import argparse
import math
import random
from collections import Counter, defaultdict, OrderedDict

from problem import Tree, Input, build_mem_image, VLEN


def depth_sequence(forest_height: int, rounds: int) -> list[int]:
    if rounds <= forest_height + 1:
        return list(range(rounds))
    seq = list(range(forest_height + 1))
    tail = rounds - len(seq)
    seq.extend(range(tail))
    return seq


def lru_touch(cache: OrderedDict, key: int, max_size: int) -> bool:
    hit = key in cache
    if hit:
        cache.move_to_end(key)
    else:
        cache[key] = True
        if len(cache) > max_size:
            cache.popitem(last=False)
    return hit


def fmt_hist(counter: Counter, limit: int | None = None) -> str:
    items = sorted(counter.items())
    if limit is not None and len(items) > limit:
        shown = items[:limit]
        rest = len(items) - limit
        return " ".join(f"{k}:{v}" for k, v in shown) + f" ...(+{rest} more)"
    return " ".join(f"{k}:{v}" for k, v in items)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forest-height", type=int, default=10)
    p.add_argument("--rounds", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--sims", type=int, default=5000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--hist-limit", type=int, default=None)
    args = p.parse_args()

    random.seed(args.seed)
    forest = Tree.generate(args.forest_height)
    n_nodes = len(forest.values)
    depth_seq = depth_sequence(args.forest_height, args.rounds)

    depths = sorted(set(depth_seq))
    per_depth = {}
    for d in depths:
        per_depth[d] = {
            "occ": 0,
            "sum_u": 0,
            "min_u": None,
            "max_u": None,
            "sum_uchild": 0,
            "bucket_hist": Counter(),
            "max_bucket": 0,
            "idx_counts": Counter(),
            "idx_occ_counts": Counter(),
            "persist_num": Counter(),
            "persist_den": Counter(),
            "uniq_per_vec_hist": Counter(),
            "sum_uval": 0,
            "min_uval": None,
            "max_uval": None,
            "val_counts": Counter(),
            "dup_ratio_sum": 0.0,
        }

    cache_sizes = [4, 8, 16, 32]
    lru_caches = {d: {k: OrderedDict() for k in cache_sizes} for d in depths}
    lru_avoided = {d: {k: 0 for k in cache_sizes} for d in depths}
    lru_total = {d: 0 for d in depths}

    spec_sizes = [4, 8, 16, 32]
    spec_avoided = {d: {k: 0 for k in spec_sizes} for d in depths}
    spec_extra = {d: {k: 0 for k in spec_sizes} for d in depths}

    mask = 0xFFFFFFFF

    def hash32(a: int) -> int:
        t1 = (a + 0x7ED55D16) & mask
        t3 = (a << 12) & mask
        a = (t1 + t3) & mask
        t1 = (a ^ 0xC761C23C) & mask
        t3 = (a >> 19) & mask
        a = (t1 ^ t3) & mask
        t1 = (a + 0x165667B1) & mask
        t3 = (a << 5) & mask
        a = (t1 + t3) & mask
        t1 = (a + 0xD3A2646C) & mask
        t3 = (a << 9) & mask
        a = (t1 ^ t3) & mask
        t1 = (a + 0xFD7046C5) & mask
        t3 = (a << 3) & mask
        a = (t1 + t3) & mask
        t1 = (a ^ 0xB55A4F09) & mask
        t3 = (a >> 16) & mask
        a = (t1 ^ t3) & mask
        return a

    for sim in range(args.sims):
        random.seed(args.seed + sim + 1)
        inp = Input.generate(forest, args.batch_size, args.rounds)
        mem = build_mem_image(forest, inp)
        inp_indices_p = mem[5]
        inp_values_p = mem[6]
        forest_values_p = mem[4]
        counts_arr = [0] * n_nodes

        for h in range(args.rounds):
            depth = depth_seq[h]
            stats = per_depth[depth]
            idxs = mem[inp_indices_p : inp_indices_p + args.batch_size]
            # Count idxs with an array to reduce overhead.
            touched = []
            for idx in idxs:
                c = counts_arr[idx]
                if c == 0:
                    touched.append(idx)
                counts_arr[idx] = c + 1
            unique_idxs = touched
            u = len(touched)
            stats["occ"] += 1
            stats["sum_u"] += u
            stats["min_u"] = u if stats["min_u"] is None else min(stats["min_u"], u)
            stats["max_u"] = u if stats["max_u"] is None else max(stats["max_u"], u)

            for idx in unique_idxs:
                c = counts_arr[idx]
                stats["bucket_hist"][c] += 1
                if c > stats["max_bucket"]:
                    stats["max_bucket"] = c
                stats["idx_counts"][idx] += c
                stats["idx_occ_counts"][idx] += 1

            # Unique per vector (natural order)
            for v in range(0, args.batch_size, VLEN):
                vec = idxs[v : v + VLEN]
                stats["uniq_per_vec_hist"][len(set(vec))] += 1

            # U_child
            child_set = set()
            for idx in unique_idxs:
                left = 2 * idx + 1
                right = 2 * idx + 2
                if left < n_nodes:
                    child_set.add(left)
                else:
                    child_set.add(0)
                if right < n_nodes:
                    child_set.add(right)
                else:
                    child_set.add(0)
            stats["sum_uchild"] += len(child_set)

            # U_val
            val_set = set(forest.values[idx] for idx in unique_idxs)
            uval = len(val_set)
            stats["sum_uval"] += uval
            stats["min_uval"] = uval if stats["min_uval"] is None else min(stats["min_uval"], uval)
            stats["max_uval"] = uval if stats["max_uval"] is None else max(stats["max_uval"], uval)
            stats["dup_ratio_sum"] += 1.0 - (uval / u if u else 0.0)
            # Count value occurrences by path
            for idx in unique_idxs:
                c = counts_arr[idx]
                stats["val_counts"][forest.values[idx]] += c

            # LRU cache (unique idxs per occurrence)
            lru_total[depth] += u
            for k in cache_sizes:
                cache = lru_caches[depth][k]
                for idx in unique_idxs:
                    if lru_touch(cache, idx, k):
                        lru_avoided[depth][k] += 1

            # Prepare for persistence/speculation
            parent_counts = defaultdict(int)
            child_counts = defaultdict(lambda: defaultdict(int))

            # Advance one round
            new_idxs = [0] * args.batch_size
            for i in range(args.batch_size):
                idx = mem[inp_indices_p + i]
                val = mem[inp_values_p + i]
                node_val = mem[forest_values_p + idx]
                val = hash32(val ^ node_val)
                new_idx = 2 * idx + (1 if val % 2 == 0 else 2)
                if new_idx >= n_nodes:
                    new_idx = 0
                mem[inp_values_p + i] = val
                mem[inp_indices_p + i] = new_idx
                new_idxs[i] = new_idx
                parent_counts[idx] += 1
                child_counts[idx][new_idx] += 1

            # Persistence per parent idx
            for idx, total in parent_counts.items():
                max_child = max(child_counts[idx].values())
                stats["persist_num"][idx] += max_child
                stats["persist_den"][idx] += total

            # Speculation (top-M parents) with next depth
            if h + 1 < args.rounds:
                next_unique = set(new_idxs)
                # Top-M parents by count
                sorted_parents = sorted(parent_counts.items(), key=lambda kv: kv[1], reverse=True)
                for m in spec_sizes:
                    top_parents = sorted_parents[:m]
                    child_set = set()
                    for pidx, _ in top_parents:
                        left = 2 * pidx + 1
                        right = 2 * pidx + 2
                        if left >= n_nodes:
                            left = 0
                        if right >= n_nodes:
                            right = 0
                        child_set.add(left)
                        child_set.add(right)
                    spec_extra[depth][m] += len(child_set)
                    spec_avoided[depth][m] += len(child_set & next_unique)

            # Reset counts for next round.
            for idx in touched:
                counts_arr[idx] = 0

    # Phase A output
    print("PHASE A: Frontier stats")
    print(
        "depth | occ | U_idx avg(min..max) | collision% avg | max_bucket | U_child avg | top_idx_counts | uniq_per_vec_hist"
    )
    for d in depths:
        s = per_depth[d]
        occ = s["occ"]
        u_avg = s["sum_u"] / occ if occ else 0.0
        u_min = s["min_u"] if s["min_u"] is not None else 0
        u_max = s["max_u"] if s["max_u"] is not None else 0
        coll = 1.0 - (u_avg / args.batch_size if args.batch_size else 0.0)
        uchild_avg = s["sum_uchild"] / occ if occ else 0.0
        top_idx = s["idx_counts"].most_common(10)
        top_idx_str = " ".join(f"{idx}:{cnt}" for idx, cnt in top_idx)
        uniq_hist = fmt_hist(s["uniq_per_vec_hist"], limit=args.hist_limit)
        print(
            f"{d:>5} | {occ:>3} | {u_avg:>6.1f} ({u_min}..{u_max}) |"
            f" {coll*100:>6.2f}% | {s['max_bucket']:>9} | {uchild_avg:>10.1f} |"
            f" {top_idx_str} | {uniq_hist}"
        )

    print("\nPHASE A: Bucket size histograms")
    for d in depths:
        hist = fmt_hist(per_depth[d]["bucket_hist"], limit=args.hist_limit)
        print(f"depth {d}: {hist}")

    # Phase B output
    print("\nPHASE B: Value duplication stats")
    print("depth | U_val avg(min..max) | dup_ratio_val avg | top_values")
    for d in depths:
        s = per_depth[d]
        occ = s["occ"]
        uval_avg = s["sum_uval"] / occ if occ else 0.0
        uval_min = s["min_uval"] if s["min_uval"] is not None else 0
        uval_max = s["max_uval"] if s["max_uval"] is not None else 0
        dup_avg = s["dup_ratio_sum"] / occ if occ else 0.0
        top_vals = s["val_counts"].most_common(5)
        top_vals_str = " ".join(f"{val}:{cnt}" for val, cnt in top_vals)
        print(
            f"{d:>5} | {uval_avg:>6.1f} ({uval_min}..{uval_max}) |"
            f" {dup_avg*100:>6.2f}% | {top_vals_str}"
        )

    # Phase C output
    print("\nPHASE C: Forest pattern stats")
    parent_count = (n_nodes - 1) // 2
    pattern_counts = Counter()
    pair_left = Counter()
    pair_right = Counter()
    for i in range(parent_count):
        p = forest.values[i]
        l = forest.values[2 * i + 1]
        r = forest.values[2 * i + 2]
        pattern_counts[(p, l, r)] += 1
        pair_left[(p, l)] += 1
        pair_right[(p, r)] += 1
    total = sum(pattern_counts.values())
    unique_patterns = len(pattern_counts)
    dup_ratio = 1.0 - (unique_patterns / total if total else 0.0)
    entropy = 0.0
    for cnt in pattern_counts.values():
        p = cnt / total
        entropy -= p * math.log2(p)
    print(f"patterns_total: {total}")
    print(f"patterns_unique: {unique_patterns}")
    print(f"patterns_dup_ratio: {dup_ratio*100:.2f}%")
    print(f"patterns_entropy_bits: {entropy:.3f}")
    print("top20 patterns (parent,left,right -> count):")
    for (p, l, r), cnt in pattern_counts.most_common(20):
        print(f"  ({p}, {l}, {r}) -> {cnt}")
    print("top10 parent-left pairs:")
    for (p, l), cnt in pair_left.most_common(10):
        print(f"  ({p}, {l}) -> {cnt}")
    print("top10 parent-right pairs:")
    for (p, r), cnt in pair_right.most_common(10):
        print(f"  ({p}, {r}) -> {cnt}")

    # Phase D output
    print("\nPHASE D: Cache simulation (LRU and TopK by depth)")
    print("depth | K | loads_total | loads_avoided_lru | loads_avoided_topk")
    for d in depths:
        total_loads = lru_total[d]
        occ_counts = per_depth[d]["idx_occ_counts"]
        for k in cache_sizes:
            topk = set(idx for idx, _ in occ_counts.most_common(k))
            avoided_topk = sum(occ_counts[idx] for idx in topk)
            print(
                f"{d:>5} | {k:>2} | {total_loads:>11} | {lru_avoided[d][k]:>17} | {avoided_topk:>18}"
            )

    print("\nPHASE D: Speculate children for top-M parents (per depth)")
    print("depth | M | child_loads_extra | next_loads_avoided | net")
    for d in depths:
        for m in spec_sizes:
            extra = spec_extra[d][m]
            avoided = spec_avoided[d][m]
            print(f"{d:>5} | {m:>2} | {extra:>17} | {avoided:>18} | {avoided - extra:>4}")

    # Phase A persistence
    print("\nPHASE A: Persistence for top-10 idx per depth")
    print("depth | idx:count:persist")
    for d in depths:
        s = per_depth[d]
        top_idx = s["idx_counts"].most_common(10)
        parts = []
        for idx, cnt in top_idx:
            den = s["persist_den"][idx]
            num = s["persist_num"][idx]
            persist = (num / den) if den else 0.0
            parts.append(f"{idx}:{cnt}:{persist:.3f}")
        print(f"{d:>5} | " + " ".join(parts))


if __name__ == "__main__":
    main()
