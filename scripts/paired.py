"""Paired bootstrap of closed-loop success differences vs the official solver."""
import json, sys, glob
import numpy as np
res = {}
for f in sys.argv[1:]:
    res.update(json.load(open(f)))
ref_name = [k for k in res if k.startswith("official")][0]
ref = np.array(res[ref_name]["success"])
rng = np.random.default_rng(0)
print("%-46s %5s %7s %9s %18s %9s" % ("method", "n", "succ%", "lat ms", "diff vs official", "speedup"))
out = {}
for k, v in res.items():
    s = np.array(v["success"]); n = min(len(s), len(ref))
    d = s[:n].astype(float) - ref[:n]
    bs = [d[rng.integers(0, n, n)].mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out[k] = {"n": n, "success": float(s.mean()), "latency_ms": v["latency_ms_mean"], "p95_ms": v["latency_ms_p95"],
              "diff": float(d.mean()), "ci95": [float(lo), float(hi)], "speedup": res[ref_name]["latency_ms_mean"] / v["latency_ms_mean"]}
    print("%-46s %5d %7.1f %9.1f %+6.1f [%+5.1f,%+5.1f] %8.1fx" % (k, n, 100 * s.mean(), v["latency_ms_mean"], 100 * d.mean(), 100 * lo, 100 * hi, out[k]["speedup"]))
json.dump(out, open("results/paired_summary_%d.json" % len(ref), "w"), indent=1)
