# Fit ms/token = floor + slope*depth per arm of a llama-bench depth curve A/B.
import re, sys, collections

def parse(path):
    arms = collections.defaultdict(dict)
    arm = None
    for line in open(path):
        m = re.match(r'### arm=(\S+)', line)
        if m:
            arm = m.group(1)
            continue
        m = re.search(r'tg32(?: @ d(\d+))?\s*\|\s*([\d.]+) ± ([\d.]+)', line)
        if m and arm:
            d = int(m.group(1) or 0)
            arms[arm].setdefault(d, []).append((float(m.group(2)), float(m.group(3))))
    return arms

def fit(pts):
    # least squares over depth >= 8192
    xs = [d for d, v in pts if d >= 8192]
    ys = [1000.0/v for d, v in pts if d >= 8192]
    n = len(xs)
    if n < 2:
        return None, None
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = sum((x-mx)**2 for x in xs)
    slope = num/den
    return my - slope*mx, slope

for path in sys.argv[1:]:
    print("==", path)
    arms = parse(path)
    agg = collections.defaultdict(list)
    for arm, ds in arms.items():
        if arm == 'warm':
            continue
        for d, vs in ds.items():
            for v, e in vs:
                agg[(arm, d)].append(v)
    per_arm = collections.defaultdict(dict)
    for (arm, d), vs in sorted(agg.items()):
        per_arm[arm][d] = sum(vs)/len(vs)
    for arm in sorted(per_arm):
        pts = sorted(per_arm[arm].items())
        f, s = fit(pts)
        print(" arm=%-6s" % arm, " ".join("d%d=%.2f" % (d, v) for d, v in pts))
        if f:
            print("        floor=%.3f ms/token  slope=%.6f ms per 1k depth" % (f, s*1000))
    if 'pool' in per_arm and 'plain' in per_arm:
        fp, sp = fit(sorted(per_arm['pool'].items()))
        fq, sq = fit(sorted(per_arm['plain'].items()))
        print("  slope plain/pool = %.2fx   (removed %.6f ms per 1k depth)" % (sq/sp, (sq-sp)*1000))
        for d in (307200, 614400):
            mp = fp + sp*d
            mq = fq + sq*d
            print("  d=%d : pool %.1f ms (%.2f t/s)  plain %.1f ms (%.2f t/s)  %.2fx" %
                  (d, mp, 1000/mp, mq, 1000/mq, mq/mp))
        base = fp + sp*8192
        for d in (307200, 614400):
            print("  pool slowdown vs 8K at d=%d : %+.1f%%" % (d, 100*((fp+sp*d)/base - 1)))
