# Fix chunk>1 by allocating v_tmp3 per vector instead of shared

with open('perf_takehome.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace v_tmp3_shared with v_tmp3 array (only need CHUNK count, not UNROLL_MAIN)
old_tmp3 = '''            v_tmp3_shared = self.alloc_scratch("v_tmp3_shared", length=VLEN)'''

new_tmp3 = '''            CHUNK_SIZE = 2  # For chunk=2
            v_tmp3 = [
                self.alloc_scratch(f"v_tmp3_{u}", length=VLEN) for u in range(CHUNK_SIZE)
            ]'''

content = content.replace(old_tmp3, new_tmp3)

# 2. Update depth 2 code to use v_tmp3[u] instead of v_tmp3_shared
old_depth2 = '''                elif depth == 2:
                    for u in range(count):
                        body.append(("valu", ("&", v_tmp1_l[u], v_idx_l[u], v_one)))  # b0
                        body.append(("valu", ("&", v_tmp2_l[u], v_idx_l[u], v_two)))  # b1 inverted
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp3_shared,
                                    v_tmp1_l[u],
                                    v_level2[3],
                                    v_level2[2],
                                ),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp1_l[u],
                                    v_tmp1_l[u],
                                    v_level2[1],
                                    v_level2[0],
                                ),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp1_l[u],
                                    v_tmp2_l[u],
                                    v_tmp1_l[u],
                                    v_tmp3_shared,
                                ),
                            )
                        )
                        body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_tmp1_l[u])))'''

new_depth2 = '''                elif depth == 2:
                    for u in range(count):
                        body.append(("valu", ("&", v_tmp1_l[u], v_idx_l[u], v_one)))  # b0
                        body.append(("valu", ("&", v_tmp2_l[u], v_idx_l[u], v_two)))  # b1 inverted
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp3[u % CHUNK_SIZE],  # Use modulo for reuse
                                    v_tmp1_l[u],
                                    v_level2[3],
                                    v_level2[2],
                                ),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp1_l[u],
                                    v_tmp1_l[u],
                                    v_level2[1],
                                    v_level2[0],
                                ),
                            )
                        )
                        body.append(
                            (
                                "flow",
                                (
                                    "vselect",
                                    v_tmp1_l[u],
                                    v_tmp2_l[u],
                                    v_tmp1_l[u],
                                    v_tmp3[u % CHUNK_SIZE],
                                ),
                            )
                        )
                        body.append(("valu", ("^", v_val_l[u], v_val_l[u], v_tmp1_l[u])))'''

content = content.replace(old_depth2, new_depth2)

# 3. Change chunk from 1 to 2
content = content.replace('chunk = 1', 'chunk = 2')

# 4. Fix starts tuples - with chunk=2, we need half the starts (every other start)
# Original starts cover all 32 vectors individually, but with chunk=2 each start processes 2 vectors
old_starts_deep = '''                    if depth > 2:
                        starts = (
                            0, 16, 4, 20, 8, 24, 12, 28, 2, 18, 6, 22, 10, 26, 14, 30,
                            1, 17, 5, 21, 9, 25, 13, 29, 3, 19, 7, 23, 11, 27, 15, 31,
                        )
                    else:
                        starts = (
                            0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
                            1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31,
                        )'''

# For chunk=2: use only even indices as starts (16 starts * 2 vectors each = 32 vectors)
new_starts_deep = '''                    if depth > 2:
                        # chunk=2: 16 starts, each processes 2 vectors
                        starts = (
                            0, 16, 4, 20, 8, 24, 12, 28, 2, 18, 6, 22, 10, 26, 14, 30,
                        )
                    else:
                        starts = (
                            0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30,
                        )'''

content = content.replace(old_starts_deep, new_starts_deep)

with open('perf_takehome_chunk2.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Created: perf_takehome_chunk2.py')
print('Changes:')
print('  - v_tmp3_shared -> v_tmp3[u] per vector')
print('  - chunk = 2')
print('  - starts reduced to 16 (only even indices)')
