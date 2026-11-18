"""E12: compiled FlashCEM latency vs sample count and iteration count (for the 50 Hz study, where one control
step is 20 ms). Same timing protocol as e11 (median over n_pairs, includes encoding observation + goal).
Usage: e12_flash_iters.py <tag> [n_pairs=10]"""
import json, sys, time
import numpy as np, torch
from flash_lewm.core import load_model, official_eval_rows, load_pairs, encode_pixels, sync
from flash_lewm.search import FlashCEM
tag = sys.argv[1]; npairs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
model = load_model(); pairs = load_pairs(official_eval_rows(50)[:npairs]); dev = "cuda"
res = {}
for n, it in [(300, 30), (300, 20), (300, 15), (300, 10), (300, 5), (100, 10), (100, 5)]:
    fl = FlashCEM(model, n=n, topk=30, iters=it)
    def f(i):
        zz = encode_pixels(model, np.concatenate([pairs["pixels"][i:i + 1], pairs["goal"][i:i + 1]])); fl(zz[0], zz[1], torch.Generator(device=dev).manual_seed(0))
    for _ in range(3): f(0)
    ts = []
    for i in range(npairs):
        sync(); t = time.perf_counter(); f(i); sync(); ts.append(time.perf_counter() - t)
    res[f"flash cem {n}x{it} (compiled)"] = float(np.median(ts) * 1e3); print(n, it, res[f"flash cem {n}x{it} (compiled)"], flush=True)
json.dump(res, open(f"results/e12_flash_iters_{tag}.json", "w"), indent=1)
