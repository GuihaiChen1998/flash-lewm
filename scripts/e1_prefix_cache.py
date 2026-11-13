"""E1: exactness and throughput of shared-prefix K/V caching on fixed candidate trees,
plus the latency-vs-batch-size curve that decides whether asynchronous small batches can pay off."""
import json, time, types
import numpy as np, torch
from flash_lewm.core import *

dev = "cuda"
model = load_model()
ps = PrefixScorer(model)
rows = official_eval_rows(50)[:4]
pairs = load_pairs(rows)
zs = encode_pixels(model, np.concatenate([pairs["pixels"], pairs["goal"]]))
Z, ZG = zs[:4], zs[4:]
g = torch.Generator(device=dev).manual_seed(0)

def bench(fn, reps=60, warm=10):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3), float(np.percentile(ts, 95) * 1e3)

res = {}
# 1) dense_cost == official get_cost on identical candidates
cand = torch.randn(300, 5, 10, generator=g, device=dev)
info = {"pixels": preprocess(pairs["pixels"][:1]).cpu()[:, None, None].expand(1, 300, 1, 3, 224, 224),
        "goal": preprocess(pairs["goal"][:1]).cpu()[:, None, None].expand(1, 300, 1, 3, 224, 224),
        "action": torch.zeros(1, 300, 1, 2)}
with torch.inference_mode():
    c_off = model.get_cost(dict(info), cand.view(1, 300, 1, 50))[0]
c_den = dense_cost(model, Z[0], ZG[0], cand)
res["dense_vs_official_get_cost"] = {"max_abs": (c_off - c_den).abs().max().item(),
    "max_rel": ((c_off - c_den).abs() / c_off.abs()).max().item(),
    "top30_overlap": len(set(c_off.topk(30, largest=False).indices.tolist()) & set(c_den.topk(30, largest=False).indices.tolist())) / 30}

# 2) tree: exactness + timing
tree = []
for (L, K, s) in [(1, 300, 0), (15, 20, 1), (30, 10, 1), (15, 20, 2), (15, 20, 3), (5, 60, 1),
                  (1, 3000, 0), (15, 200, 1), (15, 200, 2), (15, 200, 3), (150, 20, 1),
                  (1, 30000, 0), (15, 2000, 1), (15, 2000, 3)]:
    errs, ov, rho = [], [], []
    for e in range(4):
        pre = torch.randn(L, s, 10, generator=g, device=dev)
        suf = torch.randn(L, K, 5 - s, 10, generator=g, device=dev)
        full = torch.cat([pre[:, None].expand(L, K, s, 10), suf], 2).reshape(L * K, 5, 10)
        cd = dense_cost(model, Z[e], ZG[e], full)
        cc = ps.cost(ps.prefill(Z[e], pre), suf, ZG[e]).reshape(-1)
        errs.append(((cd - cc).abs() / cd.abs()).max().item())
        k = min(30, L * K // 10)
        ov.append(len(set(cd.topk(k, largest=False).indices.tolist()) & set(cc.topk(k, largest=False).indices.tolist())) / k)
        ra, rb = cd.argsort().argsort().float(), cc.argsort().argsort().float()
        rho.append(torch.corrcoef(torch.stack([ra, rb]))[0, 1].item())
    z, zg = Z[0], ZG[0]
    t_dense = bench(lambda: dense_cost(model, z, zg, full))
    t_cache = bench(lambda: ps.cost(ps.prefill(z, pre), suf, zg))
    kv = ps.prefill(z, pre)
    t_append = bench(lambda: ps.cost(kv, suf, zg))
    emb = z.view(1, 1, -1).expand(L * K, 1, -1)
    t_ae = bench(lambda: model.action_encoder(full, return_last_only=True, latent=emb))
    t_ae_c = bench(lambda: ps.append(kv, suf))
    tok_dense, tok_cache = L * K * 6, L * (1 + s) + L * K * (5 - s)
    tree.append({"L": L, "K": K, "s": s, "N": L * K, "max_rel_cost_err": max(errs), "topk_overlap": float(np.mean(ov)),
                 "spearman": float(np.mean(rho)), "token_positions_saved": 1 - tok_cache / tok_dense,
                 "dense_ms": t_dense, "prefill+append_ms": t_cache, "append_only_ms": t_append,
                 "action_encoder_dense_ms": t_ae, "action_encoder_cached_ms": t_ae_c})
    print(tree[-1], flush=True)
res["tree"] = tree

# 3) latency vs batch size (dense scorer, one observation)
curve = []
json.dump(res, open("results/e1_prefix_cache.json", "w"), indent=1)
for n in [10, 30, 100, 300, 1000, 3000, 10000, 30000]:
    c = torch.randn(n, 5, 10, generator=g, device=dev)
    curve.append({"N": n, "ms": bench(lambda: dense_cost(model, Z[0], ZG[0], c), reps=40)[0]})
    try:   # SDPA under bf16 fails for very large batch x heads ("invalid configuration argument")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            curve[-1]["ms_bf16"] = bench(lambda: dense_cost(model, Z[0], ZG[0], c), reps=40)[0] if n <= 10000 else None
    except Exception as e:
        curve[-1]["ms_bf16"] = None
    curve[-1]["us_per_candidate"] = curve[-1]["ms"] * 1e3 / n
    print(curve[-1], flush=True)
res["latency_vs_batch"] = curve
res["image_encode_ms"] = {"batch1": bench(lambda: encode_pixels(model, pairs["pixels"][:1]))[0],
                          "batch2": bench(lambda: encode_pixels(model, np.concatenate([pairs["pixels"][:1], pairs["goal"][:1]])))[0]}
print(res["dense_vs_official_get_cost"], res["image_encode_ms"])
json.dump(res, open("results/e1_prefix_cache.json", "w"), indent=1)
