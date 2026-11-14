"""Issue #7: commitment-prefix planning under a real-time clock.

Time unit = one environment step. A plan is H=5 blocks of B=5 env actions (A=10 dims per block).
While a plan is being computed the world keeps moving: either the robot holds still (synchronous
planning) or it keeps executing actions it has already committed to (asynchronous planning).

Planning latency is injected as an integer number of env steps `d`, so the protocol is a
deterministic discrete-event simulation; all methods use the same CEM budget.
"""
import numpy as np
import torch

from .closedloop import make_env, set_episode
from .core import dense_cost_multi, encode_pixels

H, A, B = 5, 10, 5


@torch.inference_mode()
def plan_cost(model, z, zg, plans, cost="last"):
    """z, zg: [E,192]; plans: [E,N,H,A] -> [E,N].

    last  official objective: distance to the goal after the whole plan (terminal cost only)
    sum   mean over the endpoints of every prefix length 1..H. The causal prefix encoder returns
          all prefix states in the same pass, so this costs H predictor tokens instead of one.
          It rewards reaching the goal early and staying there, which makes the objective
          consistent under receding-horizon replanning.
    min   best endpoint over prefix lengths (reach the goal at any time within the plan)
    """
    if cost == "last":
        return dense_cost_multi(model, z, zg, plans)
    e, n, h = plans.shape[:3]
    emb = z.view(e, 1, 1, -1).expand(e, n, 1, -1).reshape(e * n, 1, -1)
    act = model.action_encoder(plans.reshape(e * n, h, -1), return_last_only=False, latent=emb)
    pred = model.predict(emb, act)                                   # [E*N, H, D]
    c = (pred.view(e, n, h, -1) - zg.view(e, 1, 1, -1)).pow(2).sum(-1)
    return c.mean(-1) if cost == "sum" else c.min(-1).values


@torch.inference_mode()
def cem_clamped(model, z, zg, prefix=None, n=300, topk=30, iters=30, gen=None, init_mean=None, cost="last"):
    """Batched CEM over the free suffix only. z, zg: [E,192]; prefix: [E, m*A] or None.
    Every candidate is prefix (+) suffix and is scored by the unmodified model, so the score is the
    model's score of the full plan. Returns (full plan [E, H*A], model cost of that plan [E])."""
    e, dev = z.shape[0], z.device
    dp = 0 if prefix is None else prefix.shape[1]
    ds = H * A - dp
    mean = torch.zeros(e, ds, device=dev) if init_mean is None else init_mean.clone()
    std = torch.ones(e, ds, device=dev)
    for _ in range(iters):
        suf = torch.randn(e, n, ds, generator=gen, device=dev) * std[:, None] + mean[:, None]
        suf[:, 0] = mean
        cand = suf if dp == 0 else torch.cat([prefix[:, None].expand(e, n, dp), suf], -1)
        c = plan_cost(model, z, zg, cand.view(e, n, H, A), cost)
        _, i = torch.topk(c, topk, dim=1, largest=False)
        el = torch.gather(suf, 1, i[..., None].expand(-1, -1, ds))
        mean, std = el.mean(1), el.std(1)
    plan = mean if dp == 0 else torch.cat([prefix, mean], -1)
    return plan, plan_cost(model, z, zg, plan.view(e, 1, H, A), cost)[:, 0]


@torch.inference_mode()
def predict_latent(model, z, blocks):
    """Model's own prediction of the latent after executing `blocks` [E, m, A] from z [E,192]."""
    emb = z[:, None]
    act = model.action_encoder(blocks, return_last_only=True, latent=emb)
    return model.predict(emb, act)[:, 0]


def simulate(model, pairs, a_mean, a_std, method, m=1, d=5, T=75, iters=30, n=300, topk=30,
             seed=0, chunk=50, perturb=None, log=None, cost="last"):
    """Closed-loop Push-T under a real-time clock. Returns first-success step per episode (T if none).

    method:
      sync      observe, hold still d steps while planning, execute m blocks, repeat
      oracle    sync with d = 0 (not deployable; upper bound for the same feedback schedule)
      naive     async: plan a free 5-block plan from the current observation while executing the
                queue; when it arrives m blocks later, skip its first m blocks and run the next m
      clamp     async, issue #7 V1: the m queued blocks are clamped as the prefix of every candidate
                and only the suffix is searched; run the first m suffix blocks next
      clamp_ws  clamp + warm start of the suffix mean from the previous suffix (shifted)
      reanchor  async, issue #7 V2: predict the latent after the queue with the model, plan a free
                5-block plan from that predicted latent, run its first m blocks next
    Async methods need 5*m >= d (the plan must arrive before the queue runs out). All methods pay
    the same start-up hold of d steps for the first plan.
    perturb: None or (t_p, radius_px, angle_rad): at step t_p every block is displaced by a
      seeded random offset of that radius and rotated by +-angle.
    """
    dev = next(model.parameters()).device
    n_ep = len(pairs["state"])
    if method == "oracle":
        d = 0
    if method not in ("sync", "oracle"):
        assert B * m >= d, "commitment window shorter than the planning delay"
    first = np.full(n_ep, T, dtype=np.int64)
    costs = []
    envs = [make_env() for _ in range(min(chunk, n_ep))]
    mb = m * A
    for c0 in range(0, n_ep, chunk):
        idx = np.arange(c0, min(c0 + chunk, n_ep))
        E = len(idx)
        ev = envs[:E]
        for e, i in zip(ev, idx):
            set_episode(e, pairs["state"][i], pairs["goal_state"][i])
        gen = torch.Generator(device=dev).manual_seed(seed * 100003 + c0)
        prng = np.random.default_rng(seed * 7919 + c0)
        zg = encode_pixels(model, pairs["goal"][idx])
        t = 0
        done = np.zeros(E, dtype=bool)

        def step_all(act):   # act: [E,2] raw env actions
            nonlocal t
            if t >= T:
                return
            if perturb is not None and t == perturb[0]:
                for e in ev:
                    ang = prng.uniform(0, 2 * np.pi)
                    off = perturb[1] * np.array([np.cos(ang), np.sin(ang)])
                    e.block.position = (e.block.position[0] + off[0], e.block.position[1] + off[1])
                    e.block.angle = e.block.angle + prng.choice([-1.0, 1.0]) * perturb[2]
                    e.space.step(e.dt)
            for j, e in enumerate(ev):
                _, _, term, _, _ = e.step(act[j])
                if term and not done[j]:
                    done[j] = True
                    first[idx[j]] = t
            t += 1

        def hold(k):
            for _ in range(k):
                step_all(np.zeros((E, 2)))

        def execute(blocks):   # [E, k*A] normalised
            a = blocks.reshape(E, -1, 2).double().cpu().numpy() * a_std + a_mean
            for s in range(a.shape[1]):
                step_all(a[:, s])

        def observe():
            return encode_pixels(model, np.stack([e.render() for e in ev]))

        kw = dict(n=n, topk=topk, iters=iters, gen=gen, cost=cost)
        if method in ("sync", "oracle"):
            while t < T:
                z = observe()
                hold(d)
                plan, c = cem_clamped(model, z, zg, **kw)
                costs.append(c.mean().item())
                execute(plan[:, :mb])
        else:
            z = observe()
            hold(d)
            plan, c = cem_clamped(model, z, zg, **kw)
            queue = plan[:, :mb]
            carry = plan[:, mb:]
            while t < T:
                z = observe()                      # observation at the start of the cycle
                if method == "naive":
                    new, c = cem_clamped(model, z, zg, **kw)
                    nxt = new[:, mb:2 * mb]
                elif method in ("clamp", "clamp_ws"):
                    init = None
                    if method == "clamp_ws":       # blocks the previous plan placed after the queue
                        init = torch.zeros(E, H * A - mb, device=dev)
                        init[:, :carry.shape[1]] = carry
                    new, c = cem_clamped(model, z, zg, prefix=queue, init_mean=init, **kw)
                    carry = new[:, 2 * mb:]
                    nxt = new[:, mb:2 * mb]
                elif method == "reanchor":
                    zhat = predict_latent(model, z, queue.view(E, m, A))
                    new, c = cem_clamped(model, zhat, zg, **kw)
                    nxt = new[:, :mb]
                else:
                    raise ValueError(method)
                costs.append(c.mean().item())
                execute(queue)                     # the world moves on while the plan is computed
                queue = nxt
        if log:
            log(f"  {method} m={m} d={d} it={iters} {cost}: {c0 + E}/{n_ep} success@50={np.mean(first[:c0 + E] < 50):.3f} @75={np.mean(first[:c0 + E] < 75):.3f}")
    return first, float(np.mean(costs))
