"""Summarise e9 real-time runs: success within 50 / 75 env steps of physical time, and paired
bootstrap differences against a reference config. Usage: e9_report.py <ref_cfg> <json> [...]"""
import json, sys
import numpy as np
ref_name, files = sys.argv[1], sys.argv[2:]
res = {}
for f in files:
    res.update(json.load(open(f)))
rng = np.random.default_rng(0)
ref = np.array(res[ref_name]["first_success"])
out = {}
print("%-36s %4s %7s %7s %8s   %s" % ("config", "n", "succ@50", "succ@75", "t_succ", "diff@50 vs ref [95% CI]"))
for k, v in res.items():
    f = np.array(v["first_success"]); n = min(len(f), len(ref))
    d = (f[:n] < 50).astype(float) - (ref[:n] < 50)
    bs = [d[rng.integers(0, n, n)].mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    T = v.get("T", 75)
    ts = float(f[f < T].mean()) if (f < T).any() else float("nan")
    out[k] = {"n": int(len(f)), "succ50": float((f < 50).mean()), "succ75": float((f < 75).mean()), "mean_first_success_step": ts,
              "diff50_vs_ref": float(d.mean()), "ci95": [float(lo), float(hi)]}
    print("%-36s %4d %7.1f %7.1f %8.1f   %+5.1f [%+5.1f, %+5.1f]" % (k, len(f), 100 * out[k]["succ50"], 100 * out[k]["succ75"], ts, 100 * d.mean(), 100 * lo, 100 * hi))
json.dump({"ref": ref_name, "rows": out}, open("results/e9_summary.json", "w"), indent=1)
