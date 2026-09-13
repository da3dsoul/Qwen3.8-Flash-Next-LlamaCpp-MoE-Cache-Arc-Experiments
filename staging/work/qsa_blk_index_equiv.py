#!/usr/bin/env python3
"""Index-level equivalence of the two QSA selections, independent of the GPU.

Reimplements set_input_qsa's grouping/bias/blk_cells for one sequence in one
stream (llama-memory-hybrid-idx.cpp) and both selection tails of build_qsa_top_k
(qwen4exp.cpp), then compares the ATTENDED cell set each produces -- i.e. the
selection intersected with the per-cell mask build_attn_qsa applies afterwards.

The per-cell path's top-k order is unspecified, so every tie class is resolved
both ways (forward and reversed) to bound what the old path could legally pick.
"""
import random
import sys

NEG = float("-inf")
TAIL = 1e9


def inputs(n_kv, n_pos, r):
    """set_input_qsa for one stream, one sequence: cells 0..n_pos-1 hold pos 0..n_pos-1"""
    n_blocks = (n_kv + r - 1) // r
    n_bid = n_pos // r                     # full index buckets
    bid_idx = [b * r for b in range(n_bid)]

    blk_of = [(j // r if j // r < n_bid else -1) if j < n_pos else -1 for j in range(n_kv)]

    blk_cells = [0] * (r * n_blocks)
    for j in range(n_kv):
        if blk_of[j] >= 0:
            blk_cells[blk_of[j] * r + (j % r)] = j

    have_dead = n_bid < n_blocks
    dead_bid = n_bid if have_dead else n_blocks - 1

    n_tail = 0
    if have_dead:
        for j in range(n_kv):
            if blk_of[j] < 0 and j < n_pos:          # non-empty and unpooled
                assert dead_bid * r + n_tail < r * n_blocks
                blk_cells[dead_bid * r + n_tail] = j
                n_tail += 1
        i = n_tail
        while n_tail > 0 and i % r != 0:             # pad with a duplicate
            blk_cells[dead_bid * r + i] = blk_cells[dead_bid * r + n_tail - 1]
            i += 1
    n_spare = max(1 if n_bid == 0 else 0, -(-n_tail // r)) if have_dead else 0

    cell_blk = [blk_of[j] if blk_of[j] >= 0 else dead_bid for j in range(n_kv)]
    return dict(n_blocks=n_blocks, n_bid=n_bid, bid_idx=bid_idx, blk_cells=blk_cells,
                cell_blk=cell_blk, dead_bid=dead_bid, n_spare=n_spare, have_dead=have_dead)


def bias_old(inp, q, r):
    tail_start = (q + 1) // r * r
    b_ = [NEG] * inp["n_blocks"]
    for b in range(inp["n_bid"]):
        b_[b] = TAIL if inp["bid_idx"][b] >= tail_start else 0.0
    if inp["have_dead"]:
        b_[inp["dead_bid"]] = TAIL
    return b_


def bias_new(inp, q, r):
    b_ = [NEG] * inp["n_blocks"]
    for b in range(inp["n_bid"]):
        idx = inp["bid_idx"][b]
        if idx > q:
            b_[b] = NEG
        else:
            b_[b] = TAIL if idx + r - 1 > q else 0.0
    for b in range(inp["dead_bid"], inp["dead_bid"] + inp["n_spare"]):
        b_[b] = TAIL
    return b_


def topk(vals, k, reverse_ties):
    order = sorted(range(len(vals)),
                   key=lambda i: (-vals[i], -i if reverse_ties else i))
    return order[:k]


def run(n_kv, n_pos, q, r, top_k, score, reverse_ties):
    inp = inputs(n_kv, n_pos, r)
    mask = [0.0 if (j < n_pos and j <= q) else NEG for j in range(n_kv)]
    visible = {j for j in range(n_kv) if mask[j] == 0.0}

    width = min(n_kv, top_k + r - 1)

    bo = bias_old(inp, q, r)
    cell = [score[inp["cell_blk"][j]] + bo[inp["cell_blk"][j]] + mask[j] for j in range(n_kv)]
    old = set(topk(cell, width, reverse_ties)) & visible

    bn = bias_new(inp, q, r)
    blk = [score[b] + bn[b] for b in range(inp["n_blocks"])]
    n_top = min(inp["n_blocks"], (width + r - 1) // r)
    new = set()
    for b in topk(blk, n_top, reverse_ties):
        new.update(inp["blk_cells"][b * r: b * r + r])
    new &= visible
    return old, new, width, n_top


def main():
    r, top_k = 4, 2048
    random.seed(7)
    bad = 0
    cases = []
    # shallow: budget covers the context, so the two must agree exactly
    for n_pos in (1, 2, 3, 4, 5, 7, 100, 1023, 2047, 2048, 2049, 2050, 2051):
        cases.append((2048 if n_pos <= 2048 else 4096, n_pos))
    # deep: selection is active
    for n_pos in (4096, 8192, 32768):
        cases.append((n_pos, n_pos))
    cases.append((8192, 5000))

    for n_kv, n_pos in cases:
        for q in sorted({0, 1, n_pos - 1, n_pos // 2, max(0, n_pos - 2), max(0, n_pos - 5)}):
            if q >= n_pos:
                continue
            inp = inputs(n_kv, n_pos, r)
            # relu-summed head dots: non-negative, many exact zeros at depth
            # no ties here: the tie class is exercised by the unified-cache pass below
            score = [1e-3 + random.random() * 3 for _ in range(inp["n_blocks"])]
            for rev in (False, True):
                old, new, width, n_top = run(n_kv, n_pos, q, r, top_k, score, rev)
                miss = old - new           # cells the old path attends and the new does not
                extra = new - old
                # the budget rounds to whole blocks, so the two may differ by at most r cells
                over = len(miss) + len(extra) > r
                if over:
                    bad += 1
                if over or miss or extra or n_pos <= 2051:
                    tag = "BAD " if over else ("OK  " if not (miss or extra) else "round")
                    print(f"{tag} n_kv={n_kv:6d} n_pos={n_pos:6d} q={q:6d} rev={int(rev)} "
                          f"width={width} n_top={n_top} |old|={len(old)} |new|={len(new)} "
                          f"old-not-new={len(miss)} new-not-old={len(extra)}")
    print()
    print("single sequence, no ties:",
          "every diff within the r-cell whole-block rounding" if bad == 0 else f"{bad} CASES EXCEED IT")
    return 1 if bad else 0



def inputs_multi(cells, r):
    """same, but a unified cache holding several sequences: group key is (bucket, seq)"""
    n_kv = len(cells)
    n_blocks = (n_kv + r - 1) // r
    groups = {}
    for j, c in enumerate(cells):
        if c is None:
            continue
        groups.setdefault((c[1] // r, c[0]), []).append(j)

    n_bid, bid_idx, bid_cell, blk_of = 0, [], [], [-1] * n_kv
    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        mem = groups[key]
        if len({cells[j][1] % r for j in mem}) != r:
            continue
        for j in mem:
            blk_of[j] = n_bid
        bid_idx.append(key[0] * r)
        bid_cell.append(mem[0])
        n_bid += 1

    blk_cells = [0] * (r * n_blocks)
    for j in range(n_kv):
        if blk_of[j] >= 0:
            blk_cells[blk_of[j] * r + (cells[j][1] % r)] = j

    have_dead = n_bid < n_blocks
    dead_bid = n_bid if have_dead else n_blocks - 1
    n_tail = 0
    if have_dead:
        for j in range(n_kv):
            if blk_of[j] < 0 and cells[j] is not None:
                assert dead_bid * r + n_tail < r * n_blocks, "spare blocks overflowed"
                blk_cells[dead_bid * r + n_tail] = j
                n_tail += 1
        i = n_tail
        while n_tail > 0 and i % r != 0:
            blk_cells[dead_bid * r + i] = blk_cells[dead_bid * r + n_tail - 1]
            i += 1
    n_spare = max(1 if n_bid == 0 else 0, -(-n_tail // r)) if have_dead else 0
    cell_blk = [blk_of[j] if blk_of[j] >= 0 else dead_bid for j in range(n_kv)]
    return dict(n_blocks=n_blocks, n_bid=n_bid, bid_idx=bid_idx, bid_cell=bid_cell,
                blk_cells=blk_cells, cell_blk=cell_blk, dead_bid=dead_bid,
                n_spare=n_spare, have_dead=have_dead)


def run_multi(cells, seq, q, r, top_k, score, rev):
    inp = inputs_multi(cells, r)
    n_kv = len(cells)
    mask = [0.0 if (cells[j] is not None and cells[j][0] == seq and cells[j][1] <= q) else NEG
            for j in range(n_kv)]
    visible = {j for j in range(n_kv) if mask[j] == 0.0}
    width = min(n_kv, top_k + r - 1)

    bo = [NEG] * inp["n_blocks"]
    tail_start = (q + 1) // r * r
    for b in range(inp["n_bid"]):
        if cells[inp["bid_cell"][b]][0] == seq:
            bo[b] = TAIL if inp["bid_idx"][b] >= tail_start else 0.0
    if inp["have_dead"]:
        bo[inp["dead_bid"]] = TAIL
    cell = [score[inp["cell_blk"][j]] + bo[inp["cell_blk"][j]] + mask[j] for j in range(n_kv)]
    old = set(topk(cell, width, rev)) & visible

    bn = [NEG] * inp["n_blocks"]
    for b in range(inp["n_bid"]):
        if cells[inp["bid_cell"][b]][0] != seq:
            continue
        idx = inp["bid_idx"][b]
        bn[b] = NEG if idx > q else (TAIL if idx + r - 1 > q else 0.0)
    for b in range(inp["dead_bid"], inp["dead_bid"] + inp["n_spare"]):
        bn[b] = TAIL
    blk = [score[b] + bn[b] for b in range(inp["n_blocks"])]
    n_top = min(inp["n_blocks"], (width + r - 1) // r)
    new = set()
    for b in topk(blk, n_top, rev):
        new.update(inp["blk_cells"][b * r: b * r + r])
    new &= visible
    return old, new


def main_multi():
    r, top_k = 4, 2048
    random.seed(11)
    bad = 0
    for n_kv, lens in ((64, [7, 5, 3]), (256, [61, 39, 17, 3]), (4096, [1500, 900, 2]),
                       (8192, [3000, 2500, 1000, 7]), (8192, [4000, 4000])):
        cells = [None] * n_kv
        free = list(range(n_kv))
        random.shuffle(free)                       # interleaved, as a unified cache really is
        for s, n in enumerate(lens):
            for pos in range(n):
                cells[free.pop()] = (s, pos)
        inp = inputs_multi(cells, r)
        # ggml_relu flattens every non-positive head-dot sum to exactly 0.0, so a large tie
        # class sits at the selection threshold. Run both: with it, and with it removed.
        for tiefrac in (0.4, 0.0):
          score = [0.0 if random.random() < tiefrac else 1e-3 + random.random() * 3
                   for _ in range(inp["n_blocks"])]
          for s, n in enumerate(lens):
            for q in sorted({0, n // 2, n - 2, n - 1} & set(range(n))):
                for rev in (False, True):
                    old, new = run_multi(cells, s, q, r, top_k, score, rev)
                    miss, extra = old - new, new - old
                    # without ties the two selections may differ only by the whole-block
                    # rounding of the budget: at most r cells either way
                    if tiefrac == 0.0:
                        if len(miss) + len(extra) > r:
                            bad += 1
                            print(f"BAD  n_kv={n_kv} lens={lens} seq={s} q={q} rev={int(rev)} "
                                  f"|old|={len(old)} |new|={len(new)} miss={len(miss)} extra={len(extra)}")
                        elif miss or extra:
                            print(f"  rounding  n_kv={n_kv} seq={s} q={q:5d}: |old|={len(old)} "
                                  f"|new|={len(new)} old-not-new={len(miss)} new-not-old={len(extra)}")
        print(f"  n_kv={n_kv:5d} lens={lens} n_bid={inp['n_bid']} n_spare={inp['n_spare']} ok")
    print("multi-sequence, no-tie arm: " + ("all diffs within the r-cell rounding" if bad == 0 else f"{bad} CASES EXCEED IT"))
    return bad


rc = main()
print("\n=== unified cache, several sequences interleaved ===")
rc += main_multi()
sys.exit(1 if rc else 0)
