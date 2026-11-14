"""E9 (issue #7): closed-loop Push-T under a real-time clock with injected planning delay.
Usage: e9_realtime.py <n_extra> <tag> <cfg> [<cfg> ...]
cfg = method:m:d:iters:cost[:scenario]   (m in blocks of 5 env steps, d in env steps,
cost in last|sum|min, scenario in base|pert|kick|drift|switch|far)"""
import json, sys, time
import numpy as np, torch
from flash_lewm.core import *
from flash_lewm.realtime import simulate

model = load_model()
n_extra, tag, cfgs = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
# scenario presets (6th cfg field): name -> (simulate kwargs, goal offset, time budget)
SCN = {
    "base":   ({}, 25, 75),
    "pert":   ({"perturb": (12, 12.0, 0.15)}, 25, 75),      # one push at step 12
    "kick":   ({"kick": (5, 5.0)}, 25, 75),                   # 5 px random push every 5 steps
    "drift":  ({"drift": 0.6}, 25, 75),                       # block slides 0.6 px / step
    "switch": ({"switch": 8}, 25, 75),                        # goal changes at step 8 to the +50 state
    "far":    ({}, 50, 100),                                  # goal 50 expert steps away
}
_pairs = {}
def get_pairs(off, need2):
    key = (off, need2)
    if key not in _pairs:
        gmax = 50 if need2 else off
        rows = official_eval_rows(50, seed=42, goal_offset=gmax)
        if n_extra:
            rows = np.concatenate([rows, np.setdiff1d(official_eval_rows(n_extra, seed=7, goal_offset=gmax), rows)])
        p = load_pairs(rows, goal_offset=off)
        if need2:
            q = load_pairs(rows, goal_offset=50)
            p["goal2"], p["goal2_state"] = q["goal"], q["goal_state"]
        _pairs[key] = p
    return _pairs[key]
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
    scn = p[5] if len(p) > 5 else "base"
    kw, off, T = SCN[scn]
    pairs = get_pairs(off, "switch" in kw)
    n_ep = len(pairs["state"])
    t0 = time.time()
    first, mc = simulate(model, pairs, a_mean, a_std, method, m=m, d=d, iters=iters, cost=cost, T=T,
                         log=lambda s: print(s, flush=True), **kw)
    res[cfg] = {"n": n_ep, "T": T, "first_success": first.tolist(), "mean_model_cost": mc,
                "succ50": float(np.mean(first < 50)), "succ75": float(np.mean(first < 75)), "succT": float(np.mean(first < T)),
                "mean_time_to_success": float(first[first < T].mean()) if (first < T).any() else None}
    print(f"{cfg:34s} n={n_ep} succ@50={100*res[cfg]['succ50']:5.1f}% succ@75={100*res[cfg]['succ75']:5.1f}% succ@T={100*res[cfg]['succT']:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(path, "w"))
