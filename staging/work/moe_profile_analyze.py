#!/usr/bin/env python3
# Doc 14 section 6/10: compute h(f) (decode hit-rate at residency fraction f)
# from a pair of (prefill-only, prefill+decode-combined) profile CSVs, and
# report prefill's own top-10%-share for the flatness check (section 9 item 3).
import sys
import csv
from collections import defaultdict


def load(path):
    counts = defaultdict(int)
    with open(path) as f:
        for row in csv.reader(f):
            tensor, eid, cnt = row[0], int(row[1]), int(row[2])
            counts[(tensor, eid)] += cnt
    return counts


def h_curve(counts, fractions):
    total = sum(counts.values())
    if total == 0:
        return {f: float("nan") for f in fractions}, 0
    ranked = sorted(counts.values(), reverse=True)
    n = len(ranked)
    cum = 0
    cum_by_rank = []
    for c in ranked:
        cum += c
        cum_by_rank.append(cum)
    out = {}
    for f in fractions:
        k = max(1, int(round(f * n)))
        k = min(k, n)
        out[f] = cum_by_rank[k - 1] / total
    return out, total


def top10_share(counts):
    total = sum(counts.values())
    ranked = sorted(counts.values(), reverse=True)
    n = len(ranked)
    k = max(1, int(round(0.10 * n)))
    return sum(ranked[:k]) / total if total else float("nan")


def decode_only(prefill_path, combined_path):
    prefill = load(prefill_path)
    combined = load(combined_path)
    decode = {}
    keys = set(prefill) | set(combined)
    for k in keys:
        d = combined.get(k, 0) - prefill.get(k, 0)
        if d > 0:
            decode[k] = d
    return prefill, combined, decode


def report(label, prefill_path, combined_path):
    prefill, combined, decode = decode_only(prefill_path, combined_path)
    fractions = [0.10, 0.20, 0.33, 0.50, 0.67, 0.80, 1.00]
    print(f"=== {label} ===")
    print(f"  distinct (tensor,expert) slabs seen: prefill={len(prefill)} combined={len(combined)} decode-only={len(decode)} (of 73728 total)")
    print(f"  prefill top-10% share of prefill routing: {top10_share(prefill):.3f}")
    h, total_decode = h_curve(decode, fractions)
    print(f"  total decode-only routed events: {total_decode}")
    for f in fractions:
        print(f"  h({f:.2f}) = {h[f]:.4f}")
    return decode


if __name__ == "__main__":
    args = sys.argv[1:]
    # pairs of (label, prefill_csv, combined_csv)
    for i in range(0, len(args), 3):
        label, prefill_csv, combined_csv = args[i], args[i + 1], args[i + 2]
        report(label, prefill_csv, combined_csv)
