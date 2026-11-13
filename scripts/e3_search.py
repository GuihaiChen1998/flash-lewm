"""E3/E4 (offline): search quality vs wall-clock on fixed (observation, goal) pairs.
Quality = model cost of the returned plan (lower is better), compared pairwise with the
cached-latent official CEM (N=300, T=30, mean output)."""
import json, sys, time
import numpy as np, torch
from flash_lewm.core import *
from flash_lewm.search import *

dev = "cuda"
model = load_model()
ps = PrefixScorer(model)
n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0"])]
rows = official_eval_rows(n_pairs)
pairs = load_pairs(rows)
zs = encode_pixels(model, np.concatenate([pairs["pixels"], pairs["goal"]]))
Z, ZG = zs[:n_pairs], zs[n_pairs:]

def G(seed):
    return torch.Generator(device=dev).manual_seed(seed)

def wide(z, zg, g, n, k, it, ret="best", cache0=True, **kw):
    if cache0:   # latent-token K/V shared by every candidate of every iteration (exact)
        kv = ps.prefill(z, torch.zeros(1, 0, 10, device=dev))
        return cem(model, z, zg, n=n, topk=k, iters=it, gen=g, ret=ret,
                   score_fn=lambda p: ps.cost(kv, p.view(1, -1, 5, 10), zg)[0], **kw)
    return cem(model, z, zg, n=n, topk=k, iters=it, gen=g, ret=ret, **kw)

M = {
 "cem300x30_mean (official alg, cached z)": lambda z, zg, g: cem(model, z, zg, gen=g),
 "cem300x30_best": lambda z, zg, g: cem(model, z, zg, gen=g, ret="best"),
 "cem300x15_mean": lambda z, zg, g: cem(model, z, zg, iters=15, gen=g),
 "cem300x10_mean": lambda z, zg, g: cem(model, z, zg, iters=10, gen=g),
 "cem300x5_mean": lambda z, zg, g: cem(model, z, zg, iters=5, gen=g),
 "cem300x3_mean": lambda z, zg, g: cem(model, z, zg, iters=3, gen=g),
 "cem300x10_best": lambda z, zg, g: cem(model, z, zg, iters=10, gen=g, ret="best"),
 "cem300x5_best": lambda z, zg, g: cem(model, z, zg, iters=5, gen=g, ret="best"),
 "wide1000x10_k100_mean": lambda z, zg, g: wide(z, zg, g, 1000, 100, 10, "mean"),
 "wide1000x10_k30_mean": lambda z, zg, g: wide(z, zg, g, 1000, 30, 10, "mean"),
 "wide3000x5_k300_mean": lambda z, zg, g: wide(z, zg, g, 3000, 300, 5, "mean"),
 "wide3000x5_k100_mean": lambda z, zg, g: wide(z, zg, g, 3000, 100, 5, "mean"),
 "wide3000x5_k30_mean": lambda z, zg, g: wide(z, zg, g, 3000, 30, 5, "mean"),
 "wide3000x3_k30_mean": lambda z, zg, g: wide(z, zg, g, 3000, 30, 3, "mean"),
 "wide3000x8_k30_mean": lambda z, zg, g: wide(z, zg, g, 3000, 30, 8, "mean"),
 "wide3000x5_k30_best": lambda z, zg, g: wide(z, zg, g, 3000, 30, 5, "best"),
 "wide10000x3_k100_mean": lambda z, zg, g: wide(z, zg, g, 10000, 100, 3, "mean"),
 "wide10000x4_k50_mean": lambda z, zg, g: wide(z, zg, g, 10000, 50, 4, "mean"),
 "tree L15 s1 m20 uniform sync dense": lambda z, zg, g: tree_search(model, z, zg, gen=g, adaptive=False),
 "tree L15 s1 m20 adaptive sync dense": lambda z, zg, g: tree_search(model, z, zg, gen=g, adaptive=True),
 "tree L15 s1 m20 adaptive sync kv-cache": lambda z, zg, g: tree_search(model, z, zg, gen=g, adaptive=True, use_cache=True),
 "tree L15 s1 m20 adaptive ASYNC dense": lambda z, zg, g: tree_search(model, z, zg, gen=g, adaptive=True, mode="async"),
 "tree L15 s1 m20 adaptive ASYNC kv-cache": lambda z, zg, g: tree_search(model, z, zg, gen=g, adaptive=True, mode="async", use_cache=True),
 "tree L15 s1 m20 adaptive no-root": lambda z, zg, g: tree_search(model, z, zg, gen=g, adaptive=True, root_frac=0.0),
 "tree L15 s2 m20 adaptive sync dense": lambda z, zg, g: tree_search(model, z, zg, gen=g, adaptive=True, s=2),
 "tree L30 s1 m10 adaptive sync dense": lambda z, zg, g: tree_search(model, z, zg, gen=g, adaptive=True, L=30, m=10),
 "tree L5 s1 m60 adaptive sync dense": lambda z, zg, g: tree_search(model, z, zg, gen=g, adaptive=True, L=5, m=60),
 "tree L15 s1 m200 adaptive kv-cache (wide)": lambda z, zg, g: tree_search(model, z, zg, gen=g, adaptive=True, m=200, max_evals=15000, use_cache=True),
 "islands L15 s1 m20 (prefix refit) adaptive": lambda z, zg, g: tree_search(model, z, zg, gen=g, adaptive=True, refine_prefix=True),
 "islands L5 s1 m60 (prefix refit) adaptive": lambda z, zg, g: tree_search(model, z, zg, gen=g, adaptive=True, refine_prefix=True, L=5, m=60),
}
only = sys.argv[3].split("|") if len(sys.argv) > 3 else None
for f in list(M.values())[:3]:
    f(Z[0], ZG[0], G(0)); f(Z[0], ZG[0], G(0))
out = {}
for name, f in M.items():
    if only and not any(o in name for o in only):
        continue
    f(Z[0], ZG[0], G(0)); f(Z[0], ZG[0], G(0))
    C = np.zeros((len(seeds), n_pairs)); T = np.zeros_like(C); E = np.zeros_like(C)
    for si, sd in enumerate(seeds):
        for i in range(n_pairs):
            r = f(Z[i], ZG[i], G(1000 * sd + i))
            C[si, i], T[si, i], E[si, i] = r["cost"], r["time"], r["n_evals"]
    out[name] = {"cost": C.tolist(), "time_ms": float(T.mean() * 1e3), "time_p95_ms": float(np.percentile(T, 95) * 1e3), "evals": float(E.mean())}
    print(f"{name:48s} t={T.mean()*1e3:7.1f} ms  evals={E.mean():7.0f}  median cost={np.median(C):.4f}", flush=True)
ref = np.array(out[list(out)[0]]["cost"])
print("\n%-48s %8s %7s %9s %9s %8s" % ("method", "ms", "evals", "med.ratio", "geo.ratio", "win%"))
for name, o in out.items():
    c = np.array(o["cost"]); ratio = c / ref
    o["median_ratio_vs_ref"] = float(np.median(ratio)); o["geomean_ratio_vs_ref"] = float(np.exp(np.log(ratio).mean()))
    o["win_rate_vs_ref"] = float((c < ref).mean())
    print("%-48s %8.1f %7.0f %9.3f %9.3f %8.1f" % (name, o["time_ms"], o["evals"], o["median_ratio_vs_ref"], o["geomean_ratio_vs_ref"], 100 * o["win_rate_vs_ref"]))
json.dump(out, open(f"results/e3_search_{n_pairs}.json", "w"))
