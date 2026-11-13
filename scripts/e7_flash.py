"""E7: FlashCEM (compiled whole-iteration CEM) vs eager cached CEM: latency and plan quality offline."""
import json, sys, time
import numpy as np, torch
from flash_lewm.core import *
from flash_lewm.search import *
dev = "cuda"
model = load_model()
n_pairs = 50
rows = official_eval_rows(n_pairs); pairs = load_pairs(rows)
zs = encode_pixels(model, np.concatenate([pairs["pixels"], pairs["goal"]]))
Z, ZG = zs[:n_pairs], zs[n_pairs:]
G = lambda s: torch.Generator(device=dev).manual_seed(s)
def true_cost(i, plan):
    return dense_cost(model, Z[i], ZG[i], plan.view(1, 5, 10))[0].item()
M = {"eager cem 300x30": lambda z, zg, g: cem(model, z, zg, gen=g)}
for (n, k, it) in [(300, 30, 30), (300, 30, 15), (300, 30, 10), (1000, 30, 10), (1000, 100, 30), (1000, 30, 20), (3000, 30, 8)]:
    M[f"flash {n}x{it} k{k}"] = FlashCEM(model, n=n, topk=k, iters=it)
M["flash 300x30 k30 best"] = FlashCEM(model, ret="best")
M["flash 300x30 k30 (no cudagraph)"] = FlashCEM(model, mode="default")
out = {}
for name, f in M.items():
    t0 = time.time()
    for _ in range(4): f(Z[0], ZG[0], G(0))
    tc = time.time() - t0
    C, T = [], []
    for sd in range(2):
        for i in range(n_pairs):
            r = f(Z[i], ZG[i], G(1000 * sd + i)); C.append(true_cost(i, r["plan"])); T.append(r["time"])
    out[name] = {"cost": C, "time_ms": float(np.mean(T) * 1e3), "p95_ms": float(np.percentile(T, 95) * 1e3), "warmup_s": tc}
    ref = np.array(out["eager cem 300x30"]["cost"])
    out[name]["median_ratio"] = float(np.median(np.array(C) / ref))
    print(f"{name:34s} t={np.mean(T)*1e3:6.1f} ms p95={np.percentile(T,95)*1e3:6.1f}  median cost={np.median(C):.4f}  ratio={out[name]['median_ratio']:.3f}  warmup={tc:.1f}s", flush=True)
# image encoder: eager vs compiled (batch of 2 = observation + goal)
px = np.concatenate([pairs["pixels"][:1], pairs["goal"][:1]])
def bench(fn, reps=60, warm=10):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)
out["encode_ms_eager_batch2"] = bench(lambda: encode_pixels(model, px))
print("encode eager", out["encode_ms_eager_batch2"])
json.dump(out, open("results/e7_flash.json", "w"))
