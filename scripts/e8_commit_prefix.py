"""E8 (issue #7): (a) is scoring with one shared committed prefix exact and faster, with both
paths compiled to the same degree; (b) how many CEM iterations does the smaller suffix search need."""
import json, time
import numpy as np, torch
from flash_lewm.core import *
from flash_lewm.realtime import cem_clamped, H, A

dev = "cuda"
model = load_model()
ps = PrefixScorer(model)
ae = model.action_encoder.impl
rows = official_eval_rows(50); pairs = load_pairs(rows)
zs = encode_pixels(model, np.concatenate([pairs["pixels"], pairs["goal"]]))
Z, ZG = zs[:50].clone(), zs[50:].clone()
g = torch.Generator(device=dev).manual_seed(0)

def bench(fn, reps=200, warm=30):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)

class Dense(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, z, zg, plans):
        n = plans.shape[0]
        emb = z.view(1, 1, -1).expand(n, 1, -1)
        act = self.m.action_encoder(plans, return_last_only=True, latent=emb)
        return (self.m.predict(emb, act)[:, 0] - zg.view(1, -1)).pow(2).sum(-1)

class Suffix(torch.nn.Module):
    """Scores suffixes against the cached K/V of [latent token, committed prefix]."""
    def __init__(self, m, tp, ts):
        super().__init__(); self.m, self.tp, self.ts = m, tp, ts
        self.register_buffer("pos", ae._sinusoidal_pos_encoding(tp + ts, 192, dev, torch.float32)[:, tp:])
        # additive causal mask (finite, so the compiled kernel needs no -inf constant)
        self.register_buffer("mask", (1.0 - torch.ones(ts, ts, device=dev).tril()) * -1e30)
    def forward(self, z, zg, K, V, suf):
        n = suf.shape[0]
        x = ae.embed(ae.token_proj(suf)) + self.pos
        layers = ae.temporal_mixer.layers
        for li, blk in enumerate(layers):
            at = blk.attn; h = at.heads
            q, k, v = at.to_qkv(at.norm(blk.norm1(x))).chunk(3, dim=-1)
            q, k, v = (t.view(n, self.ts, h, -1).transpose(1, 2) for t in (q, k, v))
            last = li == len(layers) - 1
            if last:
                q = q[:, :, -1:]
            lp = torch.einsum("nhtd,hpd->nhtp", q, K[li]) * at.scale
            ls = torch.einsum("nhtd,nhsd->nhts", q, k) * at.scale
            if not last:   # the last token may attend to every suffix token
                ls = ls + self.mask
            w = torch.cat([lp, ls], -1).softmax(-1)
            o = torch.einsum("nhtp,hpd->nhtd", w[..., :self.tp], V[li]) + torch.einsum("nhts,nhsd->nhtd", w[..., self.tp:], v)
            o = o.transpose(1, 2).reshape(n, o.shape[2], -1)
            x = (x[:, -1:] if last else x) + at.to_out(o)
            x = x + blk.mlp(blk.norm2(x))
        p = ae.out_norm(ae.temporal_mixer.norm(x))[:, -1:]
        emb = z.view(1, 1, -1).expand(n, 1, -1)
        return (self.m.predict(emb, p)[:, 0] - zg.view(1, -1)).pow(2).sum(-1)

res = {"compute": [], "convergence": {}}
z, zg = Z[0], ZG[0]
for n in [300, 1000, 3000]:
    cand = torch.randn(n, 5, 10, generator=g, device=dev)
    dense = Dense(model).eval()
    cd = torch.compile(dense, mode="reduce-overhead", dynamic=False)
    with torch.no_grad():
        ref = dense(z, zg, cand)
        t_eager = bench(lambda: dense(z, zg, cand))
        cd(z, zg, cand); cd(z, zg, cand)
        t_dense = bench(lambda: cd(z, zg, cand))
    for m in [1, 2, 3]:
        pre = cand[0, :m]
        full = torch.cat([pre[None].expand(n, m, 10), cand[:, m:]], 1).contiguous()
        kv = ps.prefill(z, pre[None])["kv"]
        K = torch.stack([k[0] for k, _ in kv]).clone(); V = torch.stack([v[0] for _, v in kv]).clone()
        suf = cand[:, m:].contiguous()
        sm = Suffix(model, 1 + m, 5 - m).eval()
        cs = torch.compile(sm, mode="reduce-overhead", dynamic=False)
        with torch.no_grad():
            r_d = dense(z, zg, full); r_s = sm(z, zg, K, V, suf)
            err = ((r_d - r_s).abs() / r_d.abs()).max().item()
            same_top = len(set(r_d.topk(30, largest=False).indices.tolist()) & set(r_s.topk(30, largest=False).indices.tolist())) / 30
            t_se = bench(lambda: sm(z, zg, K, V, suf))
            cs(z, zg, K, V, suf); cs(z, zg, K, V, suf)
            t_sc = bench(lambda: cs(z, zg, K, V, suf))
            t_pre = bench(lambda: ps.prefill(z, pre[None]), reps=50, warm=10)
        row = {"N": n, "m": m, "max_rel_err": err, "top30_overlap": same_top, "dense_eager_ms": t_eager, "dense_compiled_ms": t_dense,
               "suffix_eager_ms": t_se, "suffix_compiled_ms": t_sc, "prefill_once_ms": t_pre}
        res["compute"].append(row); print(row, flush=True)
json.dump(res, open("results/e8_commit_prefix.json", "w"), indent=1)

# (b) iterations needed when the first m blocks are fixed (prefix taken from a converged plan)
G = lambda s: torch.Generator(device=dev).manual_seed(s)
ref_plan, _ = cem_clamped(model, Z, ZG, gen=G(1))
for m in [0, 1, 2, 3]:
    pre = None if m == 0 else ref_plan[:, :m * A]
    final = {}
    for it in [3, 5, 8, 10, 15, 20, 30]:
        cs_ = []
        for sd in range(3):
            _, c = cem_clamped(model, Z, ZG, prefix=pre, iters=it, gen=G(10 + sd))
            cs_.append(c.cpu().numpy())
        final[it] = np.stack(cs_)
    res["convergence"][m] = {it: float(np.median(final[it] / final[30])) for it in final}
    res["convergence"][m]["median_cost_at_30"] = float(np.median(final[30]))
    print("free dims", 50 - 10 * m, res["convergence"][m], flush=True)
json.dump(res, open("results/e8_commit_prefix.json", "w"), indent=1)
