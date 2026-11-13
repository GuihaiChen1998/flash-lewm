"""E5: closed-loop Push-T success and plan latency. Episodes: the official 50 (seed 42) plus
extra rows drawn with other seeds. Usage: e5_closedloop.py <n_extra> <tag> <method-substring|...>"""
import contextlib, io, json, sys, time, types
import numpy as np, torch
from flash_lewm.core import *
from flash_lewm.search import *
from flash_lewm.closedloop import *

dev = "cuda"
model = load_model()
ps = PrefixScorer(model)
n_extra = int(sys.argv[1]); tag = sys.argv[2]; only = sys.argv[3].split("|")
rows = official_eval_rows(50, seed=42)
if n_extra:
    extra = official_eval_rows(n_extra, seed=7)
    rows = np.concatenate([rows, np.setdiff1d(extra, rows)])
pairs = load_pairs(rows)
a_mean, a_std = action_stats()
env = make_env()

from stable_worldmodel.solver.cem import CEMSolver
import gymnasium as gym
solver = CEMSolver(model, batch_size=1, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device=dev, seed=42)
solver.configure(action_space=gym.spaces.Box(-1, 1, shape=(1, 2)), n_envs=1, config=types.SimpleNamespace(horizon=1, action_block=25))
GEN = torch.Generator(device=dev)

def official(px, goal):
    info = {"pixels": preprocess(px[None]).cpu()[:, None], "goal": preprocess(goal[None]).cpu()[:, None], "action": torch.zeros(1, 1, 2)}
    solver.torch_gen = GEN
    with contextlib.redirect_stdout(io.StringIO()):
        out = solver.solve(info)
    return {"plan": out["actions"][0, 0].to(dev), "cost": out["costs"][0]}
official.needs_pixels = True

def wide(z, zg, n, k, it, ret="best", **kw):
    kv = ps.prefill(z, torch.zeros(1, 0, 10, device=dev))
    return cem(model, z, zg, n=n, topk=k, iters=it, gen=GEN, ret=ret, score_fn=lambda p: ps.cost(kv, p.view(1, -1, 5, 10), zg)[0], **kw)

M = {
 "official (re-encode every iter) 300x30 mean": official,
 "cached-z cem 300x30 mean": lambda z, zg: cem(model, z, zg, gen=GEN),
 "cached-z cem 300x30 best": lambda z, zg: cem(model, z, zg, gen=GEN, ret="best"),
 "cached-z cem 300x15 mean": lambda z, zg: cem(model, z, zg, iters=15, gen=GEN),
 "cached-z cem 300x10 mean": lambda z, zg: cem(model, z, zg, iters=10, gen=GEN),
 "cached-z cem 300x5 mean": lambda z, zg: cem(model, z, zg, iters=5, gen=GEN),
 "cached-z cem 300x3 mean": lambda z, zg: cem(model, z, zg, iters=3, gen=GEN),
 "wide 3000x5 k30 mean": lambda z, zg: wide(z, zg, 3000, 30, 5, "mean"),
 "wide 3000x3 k30 mean": lambda z, zg: wide(z, zg, 3000, 30, 3, "mean"),
 "wide 1000x10 k30 mean": lambda z, zg: wide(z, zg, 1000, 30, 10, "mean"),
 "tree L15 s1 m20 uniform sync": lambda z, zg: tree_search(model, z, zg, gen=GEN, adaptive=False),
 "tree L15 s1 m20 adaptive sync": lambda z, zg: tree_search(model, z, zg, gen=GEN, adaptive=True),
 "tree L15 s1 m20 adaptive ASYNC": lambda z, zg: tree_search(model, z, zg, gen=GEN, adaptive=True, mode="async"),
 "tree L15 s1 m20 adaptive sync 3000ev": lambda z, zg: tree_search(model, z, zg, gen=GEN, adaptive=True, max_evals=3000),
 "islands L5 s1 m60 adaptive": lambda z, zg: tree_search(model, z, zg, gen=GEN, adaptive=True, refine_prefix=True, L=5, m=60),
}
_flash = {}
def flash(key, **kw):
    def f(z, zg):
        if key not in _flash:
            _flash[key] = FlashCEM(model, **kw)
        return _flash[key](z, zg, gen=GEN)
    return f
M.update({
 "flash 300x30 mean": flash("a", n=300, topk=30, iters=30),
 "flash 300x30 best": flash("b", n=300, topk=30, iters=30, ret="best"),
 "flash 300x20 mean": flash("c", n=300, topk=30, iters=20),
 "flash 300x15 mean": flash("d", n=300, topk=30, iters=15),
 "flash 300x10 mean": flash("e", n=300, topk=30, iters=10),
 "flash 1000x20 k30 mean": flash("f", n=1000, topk=30, iters=20),
})
try:
    res = json.load(open(f"results/e5_closedloop_{tag}.json"))
except Exception:
    res = {}
for name, f in M.items():
    if not any(o in name for o in only):
        continue
    GEN.manual_seed(42)
    for _ in range(3):
        run_episode(env, model, f, {k: v[0] for k, v in pairs.items()}, a_mean, a_std)   # warm-up / compile
    GEN.manual_seed(42)
    succ, lat, dist, cost = [], [], [], []
    for i in range(len(rows)):
        r = run_episode(env, model, f, {k: v[i] for k, v in pairs.items()}, a_mean, a_std)
        succ.append(r["success"]); lat += r["latency"]; dist.append(r["final_dist"]); cost.append(r["cost"][0])
    s = np.array(succ)
    res[name] = {"n": len(rows), "success_all": float(s.mean()), "success_official50": float(s[:50].mean()),
                 "latency_ms_mean": float(np.mean(lat) * 1e3), "latency_ms_p50": float(np.percentile(lat, 50) * 1e3),
                 "latency_ms_p95": float(np.percentile(lat, 95) * 1e3), "final_dist_median": float(np.median(dist)),
                 "success": [int(x) for x in s]}
    print(f"{name:46s} n={len(rows)} succ={100*s.mean():5.1f}%  off50={100*s[:50].mean():5.1f}%  lat mean={np.mean(lat)*1e3:7.1f} p95={np.percentile(lat,95)*1e3:7.1f} ms  dist={np.median(dist):6.1f}", flush=True)
    json.dump(res, open(f"results/e5_closedloop_{tag}.json", "w"), indent=1)
