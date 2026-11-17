"""E6: attack the fixed per-call overhead (the real bottleneck once latents are cached)."""
import json, time, sys
import numpy as np, torch
from flash_lewm.core import *

dev = "cuda"
model = load_model()
rows = official_eval_rows(50)[:2]
pairs = load_pairs(rows)
zs = encode_pixels(model, np.concatenate([pairs["pixels"], pairs["goal"]]))
z, zg = zs[0].clone(), zs[2].clone()
g = torch.Generator(device=dev).manual_seed(0)
cand = torch.randn(300, 5, 10, generator=g, device=dev)

def bench(fn, reps=100, warm=20):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)

class Scorer(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, z, zg, plans):
        n = plans.shape[0]
        emb = z.view(1, 1, -1).expand(n, 1, -1)
        act = self.m.action_encoder(plans, return_last_only=True, latent=emb)
        pred = self.m.predict(emb, act)
        return (pred[:, 0] - zg.view(1, -1)).pow(2).sum(-1)

sc = Scorer(model).eval()
res = {}
with torch.inference_mode():
    ref = sc(z, zg, cand)
    res["eager_ms"] = bench(lambda: sc(z, zg, cand))
print(res, flush=True)
# profile where eager time goes
with torch.inference_mode():
    t = {}
    emb = z.view(1, 1, -1).expand(300, 1, -1)
    t["action_encoder"] = bench(lambda: model.action_encoder(cand, return_last_only=True, latent=emb))
    act = model.action_encoder(cand, return_last_only=True, latent=emb)
    t["predictor"] = bench(lambda: model.predictor(emb, act))
    pr = model.predictor(emb, act)
    t["pred_proj"] = bench(lambda: model.pred_proj(pr[:, 0]))
    res["eager_parts_ms"] = t
print(res, flush=True)
for mode in ["default", "reduce-overhead", "max-autotune-no-cudagraphs"]:
    try:
        t0 = time.time()
        cs = torch.compile(sc, mode=mode, dynamic=False)
        with torch.no_grad():
            out = cs(z, zg, cand); out = cs(z, zg, cand)
            err = ((out - ref).abs() / ref.abs()).max().item()
            ms = bench(lambda: cs(z, zg, cand))
        res[f"compile[{mode}]"] = {"ms": ms, "max_rel_err": err, "compile_s": time.time() - t0}
    except Exception as e:
        res[f"compile[{mode}]"] = {"error": repr(e)[:300]}
    print(mode, res[f"compile[{mode}]"], flush=True)
try:
    with torch.no_grad():
        tr = torch.jit.trace(sc, (z, zg, cand), check_trace=False)
        tr = torch.jit.freeze(tr)
        out = tr(z, zg, cand); out = tr(z, zg, cand)
        res["jit_trace"] = {"ms": bench(lambda: tr(z, zg, cand)), "max_rel_err": ((out - ref).abs() / ref.abs()).max().item()}
except Exception as e:
    res["jit_trace"] = {"error": repr(e)[:300]}
print(res["jit_trace"], flush=True)
json.dump(res, open("results/e6_overhead.json", "w"), indent=1)
