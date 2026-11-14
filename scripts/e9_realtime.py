"""E9 (issue #7): closed-loop Push-T under a real-time clock with injected planning delay.
Usage: e9_realtime.py <n_extra> <tag> <cfg> [<cfg> ...]
cfg = method:m:d:iters:cost[:tp:radius:angle]   (m in blocks of 5 env steps, d in env steps,
cost in last|sum|min)"""
import json, sys, time
import numpy as np, torch
from flash_lewm.core import *
from flash_lewm.realtime import simulate

model = load_model()
n_extra, tag, cfgs = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
rows = official_eval_rows(50, seed=42)
if n_extra:
    rows = np.concatenate([rows, np.setdiff1d(official_eval_rows(n_extra, seed=7), rows)])
pairs = load_pairs(rows)
a_mean, a_std = action_stats()
path = f"results/e9_realtime_{tag}.json"
try:
    res = json.load(open(path))
except Exception:
    res = {}
for cfg in cfgs:
    p = cfg.split(":")
    method, m, d, iters = p[0], int(p[1]), int(p[2]), int(p[3])
    cost = p[4] if len(p) > 4 else "last"
    perturb = (int(p[5]), float(p[6]), float(p[7])) if len(p) > 5 else None
    t0 = time.time()
    first, cost = simulate(model, pairs, a_mean, a_std, method, m=m, d=d, iters=iters, perturb=perturb, cost=cost,
                           log=lambda s: print(s, flush=True))
    res[cfg] = {"n": len(rows), "first_success": first.tolist(), "mean_model_cost": cost,
                "succ50": float(np.mean(first < 50)), "succ75": float(np.mean(first < 75)),
                "mean_time_to_success": float(first[first < 75].mean()) if (first < 75).any() else None}
    print(f"{cfg:34s} n={len(rows)} succ@50={100*res[cfg]['succ50']:5.1f}% succ@75={100*res[cfg]['succ75']:5.1f}%  cost={cost:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(path, "w"))
