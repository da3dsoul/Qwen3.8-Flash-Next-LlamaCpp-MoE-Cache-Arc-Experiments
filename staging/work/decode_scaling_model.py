#!/usr/bin/env python3
"""Decode-cost-vs-depth model for Qwen3.8-Flash-Next on Arc Pro B70.

Every input is measured on this box; nothing is cited secondhand.
  FA_US / IDX_US / IDX_CACHED_US
      test-backend-ops perf, production decode shape, SYCL0
      logs/decode-scaling/{fa-depth-curve,qsa-indexer-depth-curve,
                           qsa-indexer-cached}.log
  BENCH
      llama-bench -p 0 -n 32 -d ... -r 2 on the recommended config
      logs/decode-scaling/llama-bench-depth-curve.log

Question: what must the decode-vs-depth SLOPE become to hold <=5% slowdown at
300K and <=25% at 600K relative to short context, and which components have to
change to get there.
"""
N_QSA = 12          # QSA/full-attention layers; a decode token pays each once
REF   = 8192        # short-context reference depth (measured)

# --- measured, us/call (one QSA layer) -------------------------------------
FA_US = {2048: 55.47, 4096: 100.25, 8192: 178.97, 16384: 398.66, 32768: 807.59,
         65536: 1614.66, 118016: 2913.43, 200704: 4917.78, 307200: 7488.28,
         614400: 14964.10}
IDX_US = {4096: 108.37, 8192: 152.35, 16384: 243.64, 32768: 432.72, 65536: 827.27,
          118016: 1528.56, 200704: 2647.11, 307200: 4043.51, 614400: 8011.00}
# same chain with the pooled block keys taken as a ready input (what an
# incremental pooled-key cache would leave); filled from the measurement
IDX_CACHED_US = {4096: 55.29, 8192: 64.34, 16384: 76.35, 32768: 109.09,
                 65536: 148.57, 118016: 217.51, 200704: 355.40, 307200: 534.41,
                 614400: 1000.96}

# block-granularity top-k (this tree's build_qsa_top_k), same instrument, same day
IDX_BLK_US = {4096: 90.68, 8192: 130.56, 16384: 209.20, 32768: 369.10, 65536: 731.05,
              118016: 1385.75, 200704: 2430.65, 307200: 3677.91, 614400: 7277.08}
# ... and the same on top of the pooled-key cache: the combination 5.1 + 5.3 would give
IDX_CACHED_BLK_US = {4096: 42.48, 8192: 45.63, 16384: 46.72, 32768: 52.41, 65536: 60.44,
                     118016: 78.45, 200704: 118.19, 307200: 148.53, 614400: 236.26}

# --- measured end to end, tok/s -------------------------------------------
BENCH = {0: 26.48, 2048: 26.03, 8192: 24.63, 32768: 19.43, 65536: 15.26,
         118016: 11.22}

DEPTHS = [8192, 32768, 65536, 118016, 200704, 307200, 614400]


def ms(us_per_call):
    return N_QSA * us_per_call / 1000.0


def interp(tbl, D):
    if D in tbl:
        return tbl[D]
    ks = sorted(tbl)
    if D < ks[0]:
        return tbl[ks[0]] * D / ks[0]
    if D > ks[-1]:                                    # linear-regime slope
        a, b = ks[-2], ks[-1]
        m = (tbl[b] - tbl[a]) / (b - a)
        return tbl[b] + m * (D - b)
    for a, b in zip(ks, ks[1:]):
        if a <= D <= b:
            return tbl[a] + (tbl[b] - tbl[a]) * (D - a) / (b - a)


def floors():
    """implied depth-independent remainder at every benchmarked depth"""
    out = {}
    for D, ts in BENCH.items():
        if D < 4096:
            continue
        out[D] = 1000.0 / ts - ms(interp(FA_US, D)) - ms(interp(IDX_US, D))
    return out


FA_SPARSE_MS = ms(FA_US[2048])   # bounded to width 2051 -> n_kv ~2304, flat


def scenarios(floor):
    s = {
        "today (dense FA + full indexer)":
            lambda D: ms(interp(FA_US, D)) + ms(interp(IDX_US, D)),
        "sparse FA only (doc 10 Design B)":
            lambda D: FA_SPARSE_MS + ms(interp(IDX_US, D)),
    }
    if IDX_CACHED_US:
        s["indexer pooled-key cache only"] = \
            lambda D: ms(interp(FA_US, D)) + ms(interp(IDX_CACHED_US, D))
        s["both"] = \
            lambda D: FA_SPARSE_MS + ms(interp(IDX_CACHED_US, D))
        # block-level top-k alone, measured rather than modelled (it is implemented)
        s["block top-k only (measured)"] = \
            lambda D: ms(interp(FA_US, D)) + ms(interp(IDX_BLK_US, D))
        s["both + block top-k (measured)"] = \
            lambda D: FA_SPARSE_MS + ms(interp(IDX_CACHED_BLK_US, D))
        # + an f16 pooled cache, which halves the 132 B score-matmul read of what is left
        s["  ... + f16 pool (est.)"] = \
            lambda D: FA_SPARSE_MS + ms(interp(IDX_CACHED_BLK_US, D)) * 0.75
        # ... and fusing the residual chain's ~9 small ops. The cached arm runs at
        # only 122-131 GB/s against the 285 GB/s the streaming pool achieves, on
        # 4x-smaller tensors -- i.e. it is launch/latency-bound, so a fused kernel
        # is worth ~2x. Speculative: the only unmeasured factor in this table.
        s["  ... + fused residual chain (est.)"] = \
            lambda D: FA_SPARSE_MS + ms(interp(IDX_CACHED_BLK_US, D)) * 0.75 / 2.0
    return s


if __name__ == "__main__":
    f = floors()
    floor = sum(f.values()) / len(f)
    print("implied floor by depth (ms/token): " +
          ", ".join(f"d{D}={v:.2f}" for D, v in sorted(f.items())))
    print(f"floor used: {floor:.2f} ms/token  (spread {min(f.values()):.2f}-{max(f.values()):.2f})\n")

    print("model vs measured, ms/token:")
    for D, ts in sorted(BENCH.items()):
        if D < 4096:
            continue
        pred = floor + ms(interp(FA_US, D)) + ms(interp(IDX_US, D))
        real = 1000.0 / ts
        print(f"  d{D:<7} measured {real:7.2f}  model {pred:7.2f}  ({(pred/real-1)*100:+.1f}%)")
    print()

    hdr = "".join(f"{D//1000}K".rjust(18) for D in DEPTHS)
    print(" " * 40 + hdr)
    for name, fn in scenarios(floor).items():
        base = floor + fn(REF)
        cells = []
        for D in DEPTHS:
            v = floor + fn(D)
            cells.append(f"{1000/v:5.1f}t/s{(v/base-1)*100:+7.1f}%")
        print(f"{name:<40}" + "".join(c.rjust(18) for c in cells))
    print("\nbar: <=+5% at 300K, <=+25% at 600K, vs the 8K column")
