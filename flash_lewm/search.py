"""Planners operating on cached latents. Every planner returns a dict with
plan [H,10], cost of that plan under the model, n_evals, time, and an anytime trace
[(elapsed_s, n_evals, best_scored_cost)].
"""
import time

import torch

from .core import PrefixScorer, dense_cost, dense_cost_multi, sync

import os as _os
H, A = 5, 5 * int(_os.environ.get("FLASH_ADIM", "2"))     # 5 blocks x (5 steps x action dim)


def _now():
    sync()
    return time.perf_counter()


@torch.inference_mode()
def cem(model, z, zg, n=300, topk=30, iters=30, gen=None, ret="mean", deadline=None,
        keep_elites=0, score_fn=None, min_std=0.0):
    """Cross-entropy method. With ret='mean' and keep_elites=0 this follows the sampling / update
    rule of stable_worldmodel.CEMSolver exactly (first candidate forced to the mean, std update).
    ret='best' returns the best scored candidate instead of the final mean.
    deadline: stop starting new iterations after this many seconds.
    """
    dev = z.device
    score = score_fn or (lambda p: dense_cost(model, z, zg, p))
    t0 = _now()
    mean = torch.zeros(H * A, device=dev)
    std = torch.ones(H * A, device=dev)
    best_c, best_p, evals, trace, elites = float("inf"), None, 0, [], None
    for _ in range(iters):
        cand = torch.randn(n, H * A, generator=gen, device=dev) * std + mean
        cand[0] = mean
        if keep_elites and elites is not None:
            cand[1:1 + keep_elites] = elites[:keep_elites]
        c = score(cand.view(n, H, A))
        evals += n
        v, i = torch.topk(c, topk, largest=False)
        elites = cand[i]
        if v[0].item() < best_c:
            best_c, best_p = v[0].item(), elites[0].clone()
        mean = elites.mean(0)
        std = elites.std(0).clamp_min(min_std)
        el = _now() - t0
        trace.append((el, evals, best_c))
        if deadline is not None and el >= deadline:
            break
    if ret == "mean":
        plan = mean
        cost = score(plan.view(1, H, A))[0].item()
        evals += 1
    else:
        plan, cost = best_p, best_c
    return {"plan": plan.view(H, A), "cost": cost, "n_evals": evals, "time": _now() - t0, "trace": trace}


@torch.inference_mode()
def cem_multi(model, z, zg, n=300, topk=30, iters=30, gen=None, ret="mean"):
    """CEM batched over E environments in a single forward per iteration. z, zg: [E,192]."""
    dev, e = z.device, z.shape[0]
    t0 = _now()
    mean = torch.zeros(e, H * A, device=dev)
    std = torch.ones(e, H * A, device=dev)
    best_c = torch.full((e,), float("inf"), device=dev)
    best_p = torch.zeros(e, H * A, device=dev)
    for _ in range(iters):
        cand = torch.randn(e, n, H * A, generator=gen, device=dev) * std[:, None] + mean[:, None]
        cand[:, 0] = mean
        c = dense_cost_multi(model, z, zg, cand.view(e, n, H, A))
        v, i = torch.topk(c, topk, dim=1, largest=False)
        elites = torch.gather(cand, 1, i[..., None].expand(-1, -1, H * A))
        upd = v[:, 0] < best_c
        best_c = torch.where(upd, v[:, 0], best_c)
        best_p = torch.where(upd[:, None], elites[:, 0], best_p)
        mean, std = elites.mean(1), elites.std(1)
    if ret == "mean":
        plan = mean
        cost = dense_cost_multi(model, z, zg, plan.view(e, 1, H, A))[:, 0]
    else:
        plan, cost = best_p, best_c
    return {"plan": plan.view(e, H, A), "cost": cost, "time": _now() - t0}


@torch.inference_mode()
def tree_search(model, z, zg, L=15, s=1, m=20, topk_frac=0.25, max_evals=9000, root_frac=0.2,
                gen=None, deadline=None, adaptive=True, mode="sync", refine_prefix=False,
                use_cache=False):
    """Shared-prefix branch search from issue #6 (Algorithm 1).

    L branches each fix the first s action blocks P_j ~ N(0, I) and run their own elite-fit
    search over the suffix. A root branch (root_frac of the budget) runs ordinary CEM over full
    plans, so the first block can still change. The best scored plan is returned.

    adaptive:      soft budget allocation from each branch's elite mean in the previous round
                   (all branches are compared after a full round; allocations are quantised to
                   {m/2, m, 2m, 4m} and every branch keeps at least m/2).
    mode:          "sync"  - all branches of a round are scored in one batched call;
                   "async" - every branch batch is its own GPU call and the branch updates as
                             soon as it returns. On one GPU this is what an asynchronous
                             scheduler does when it does not wait to merge ready work.
    refine_prefix: improvement over the issue - each branch also refits its prefix from its
                   elites, so a branch is a local full-plan distribution, not a frozen prefix.
    use_cache:     score through PrefixScorer (exact K/V sharing) instead of the dense scorer.
    """
    dev = z.device
    ps = PrefixScorer(model) if use_cache else None
    t0 = _now()
    Ts = H - s
    pre = torch.randn(L, s * A, generator=gen, device=dev)
    pre_std = torch.ones(L, s * A, device=dev)
    mu = torch.zeros(L, Ts * A, device=dev)
    sd = torch.ones(L, Ts * A, device=dev)
    r_mu = torch.zeros(H * A, device=dev)
    r_sd = torch.ones(H * A, device=dev)
    n_root = int(round(root_frac * L * m / (1 - root_frac))) if root_frac > 0 else 0
    cache = ps.prefill(z, pre.view(L, s, A)) if (use_cache and not refine_prefix) else None
    best = {"c": float("inf"), "p": None}
    evals, trace = 0, []
    alloc = [m] * L

    def sample(idx, k):
        b = len(idx)
        suf = torch.randn(b, k, Ts * A, generator=gen, device=dev) * sd[idx, None] + mu[idx, None]
        suf[:, 0] = mu[idx]
        if refine_prefix:
            p = torch.randn(b, k, s * A, generator=gen, device=dev) * pre_std[idx, None] + pre[idx, None]
            p[:, 0] = pre[idx]
        else:
            p = pre[idx, None].expand(b, k, -1)
        return torch.cat([p, suf], -1)

    def update(idx, full, c):
        k = c.shape[1]
        ne = max(2, int(round(topk_frac * k)))
        v, i = torch.topk(c, ne, dim=1, largest=False)
        el = torch.gather(full, 1, i[..., None].expand(-1, -1, H * A))
        mu[idx] = el[..., s * A:].mean(1)
        sd[idx] = el[..., s * A:].std(1)
        if refine_prefix:
            pre[idx] = el[..., :s * A].mean(1)
            pre_std[idx] = el[..., :s * A].std(1)
        j = v[:, 0].argmin()
        if v[j, 0].item() < best["c"]:
            best["c"], best["p"] = v[j, 0].item(), el[j, 0].clone()
        return v.mean(1)

    def score_group(idx, full):
        b, k = full.shape[:2]
        if use_cache and not refine_prefix:
            sub = {"kv": [(kk[idx], vv[idx]) for kk, vv in cache["kv"]], "s": s, "L": b, "z": z}
            return ps.cost(sub, full[..., s * A:].reshape(b, k, Ts, A), zg)
        return dense_cost(model, z, zg, full.reshape(b * k, H, A)).view(b, k)

    while evals < max_evals:
        q = torch.zeros(L, device=dev)
        groups = {}
        for j, k in enumerate(alloc):
            groups.setdefault(k, []).append(j)
        root = None
        if n_root > 0:
            root = torch.randn(n_root, H * A, generator=gen, device=dev) * r_sd + r_mu
            root[0] = r_mu
        if mode == "async":
            if root is not None:
                rc = dense_cost(model, z, zg, root.view(n_root, H, A))
            for j in range(L):
                idx = torch.tensor([j], device=dev)
                full = sample(idx, alloc[j])
                q[j] = update(idx, full, score_group(idx, full))[0]
                evals += alloc[j]
        else:
            idxs = [torch.tensor(v, device=dev) for v in groups.values()]
            fulls = [sample(ix, k) for ix, k in zip(idxs, groups.keys())]
            if use_cache and not refine_prefix:
                cs = [score_group(ix, f) for ix, f in zip(idxs, fulls)]
                if root is not None:
                    rc = dense_cost(model, z, zg, root.view(n_root, H, A))
            else:  # one flat batched call for every branch and the root
                flat = [f.reshape(-1, H * A) for f in fulls] + ([root] if root is not None else [])
                call = dense_cost(model, z, zg, torch.cat(flat).view(-1, H, A))
                cs, o = [], 0
                for f in fulls:
                    nn_ = f.shape[0] * f.shape[1]
                    cs.append(call[o:o + nn_].view(f.shape[0], f.shape[1]))
                    o += nn_
                if root is not None:
                    rc = call[o:]
            for ix, f, c in zip(idxs, fulls, cs):
                q[ix] = update(ix, f, c)
                evals += c.numel()
        if root is not None:
            evals += n_root
            v, i = torch.topk(rc, max(2, n_root // 10), largest=False)
            r_mu, r_sd = root[i].mean(0), root[i].std(0)
            if v[0].item() < best["c"]:
                best["c"], best["p"] = v[0].item(), root[i[0]].clone()
        if adaptive:
            w = torch.softmax(-(q - q.min()) / (q.std() + 1e-8), 0)
            raw = (m // 2) + w * (L * m - L * (m // 2))
            levels = torch.tensor([m // 2, m, 2 * m, 4 * m], device=dev, dtype=raw.dtype)
            alloc = levels[(raw[:, None] - levels[None]).abs().argmin(1)].long().tolist()
        el = _now() - t0
        trace.append((el, evals, best["c"]))
        if deadline is not None and el >= deadline:
            break
    return {"plan": best["p"].view(H, A), "cost": best["c"], "n_evals": evals, "time": _now() - t0, "trace": trace}


class _CEMStep(torch.nn.Module):
    """One whole CEM iteration (sample -> score -> top-k -> refit) as a single graph."""

    def __init__(self, model, topk, cost="last"):
        super().__init__()
        self.m, self.topk, self.cost = model, topk, cost

    def forward(self, z, zg, mean, std, noise):
        cand = torch.cat([mean[None], noise[1:] * std + mean], 0)
        n = cand.shape[0]
        emb = z.view(1, 1, -1).expand(n, 1, -1)
        if getattr(self.m, "is_lewm", False):     # original LeWM: autoregressive block rollout
            from .lewm import lewm_preds_raw
            c = (lewm_preds_raw(self.m, z.view(1, -1), cand.view(1, n, H, A))[0] - zg.view(1, 1, -1)).pow(2).sum(-1)
            c = c[:, -1] if self.cost == "last" else c.min(-1).values
        elif self.cost == "last":
            act = self.m.action_encoder(cand.view(n, H, A), return_last_only=True, latent=emb)
            pred = self.m.predict(emb, act)
            c = (pred[:, 0] - zg.view(1, -1)).pow(2).sum(-1)
        else:   # "min": closest endpoint over all prefix lengths (see realtime.plan_cost)
            act = self.m.action_encoder(cand.view(n, H, A), return_last_only=False, latent=emb)
            pred = self.m.predict(emb, act)
            c = (pred - zg.view(1, 1, -1)).pow(2).sum(-1).min(-1).values
        v, i = torch.topk(c, self.topk, largest=False)
        el = cand[i]
        return el.mean(0), el.std(0), v[0], el[0], v.mean()


class FlashCEM:
    """Latency-oriented CEM: latents encoded once, the full CEM iteration compiled into one
    static graph (torch.compile, CUDA graphs), all noise drawn in one call. The update rule is
    the same as stable_worldmodel.CEMSolver, so it is a drop-in replacement of the official
    solver for one environment. The first call compiles (several seconds); later calls reuse it.
    """

    def __init__(self, model, n=300, topk=30, iters=30, ret="mean", mode="reduce-overhead", cost="last"):
        self.n, self.iters, self.ret = n, iters, ret
        step = _CEMStep(model, topk, cost).eval()
        self.step = torch.compile(step, mode=mode, dynamic=False) if mode else step

    def __call__(self, z, zg, gen=None):
        dev = z.device
        t0 = _now()
        with torch.no_grad():
            z, zg = z.detach().clone(), zg.detach().clone()
            noise = torch.randn(self.iters, self.n, H * A, generator=gen, device=dev)
            mean = torch.zeros(H * A, device=dev)
            std = torch.ones(H * A, device=dev)
            best_c = torch.full((), float("inf"), device=dev)
            best_p = mean
            for t in range(self.iters):
                mean, std, c0, p0, cm = self.step(z, zg, mean, std, noise[t])
                mean, std, cm = mean.clone(), std.clone(), cm.clone()
                if self.ret == "best":
                    upd = c0 < best_c
                    best_c = torch.where(upd, c0, best_c)
                    best_p = torch.where(upd, p0, best_p)
            if self.ret == "best":
                plan, cost = best_p, best_c.item()
            else:
                plan, cost = mean, cm.item()   # elite-mean cost of the last iteration, as the official solver logs
        return {"plan": plan.view(H, A), "cost": cost, "n_evals": self.n * self.iters, "time": _now() - t0, "trace": []}
