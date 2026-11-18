"""E11: planning latency of sampling-based solvers on two stacks, per environment.
official stack: stable_worldmodel's CEMSolver / ICEMSolver / MPPISolver calling the model's get_cost, which
  re-encodes the images every iteration (what a user of the released code gets);
flash stack: our batched planners on cached latents (flash_lewm.realtime.cem_clamped, eager) and FlashCEM (compiled).
Environment from PUSHT_H5 / FLASH_CKPT / FLASH_ADIM / FLASH_EP_KEY / FLASH_STATE_KEY.
Usage: e11_solver_latency.py <tag> [n_pairs=10]"""
import io, contextlib, json, os, sys, time, types
import numpy as np, torch, gymnasium as gym
from flash_lewm.core import load_model, official_eval_rows, load_pairs, preprocess, encode_pixels, sync
from flash_lewm.search import FlashCEM
import flash_lewm.realtime as rt
from stable_worldmodel.solver.cem import CEMSolver
from stable_worldmodel.solver.icem import ICEMSolver
from stable_worldmodel.solver.mppi import MPPISolver

tag = sys.argv[1]; npairs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
adim = int(os.environ.get("FLASH_ADIM", "2")); dev = "cuda"
model = load_model(); rows = official_eval_rows(50)[:npairs]; pairs = load_pairs(rows)
BUDGETS = {"cem 300x30": ("cem", 300, 30), "icem 300x10": ("icem", 300, 10), "icem 100x5": ("icem", 100, 5), "mppi 300x10": ("mppi", 300, 10)}

def info_for(i):
    return {"pixels": preprocess(pairs["pixels"][i:i + 1]).cpu()[:, None], "goal": preprocess(pairs["goal"][i:i + 1]).cpu()[:, None],
            "action": torch.zeros(1, 1, adim * (5 if getattr(model, "is_lewm", False) else 1))}

def timeit(f):
    for _ in range(2): f(0)
    ts = []
    for i in range(npairs):
        sync(); t = time.perf_counter(); f(i); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)

res = {}
for name, (algo, n, it) in BUDGETS.items():
    cls = {"cem": CEMSolver, "icem": ICEMSolver, "mppi": MPPISolver}[algo]
    kw = dict(batch_size=1, num_samples=n, n_steps=it, device=dev, seed=42)
    if algo in ("cem", "icem"): kw.update(var_scale=1.0, topk=30)
    try:
        sol = cls(model, **kw)
        hz = 5 if getattr(model, "is_lewm", False) else 1          # LeWM plans 5 blocks of 5; Fast-LeWM packs them into one
        sol.configure(action_space=gym.spaces.Box(-1, 1, shape=(1, adim)), n_envs=1, config=types.SimpleNamespace(horizon=hz, action_block=25 // hz))
        def off(i):
            with contextlib.redirect_stdout(io.StringIO()): sol.solve(info_for(i))
        res[f"official {name}"] = timeit(off)
    except Exception as ex:
        res[f"official {name}"] = f"error: {ex!r}"[:200]
    def ours(i, algo=algo, n=n, it=it):
        zz = encode_pixels(model, np.concatenate([pairs["pixels"][i:i + 1], pairs["goal"][i:i + 1]]))
        rt.cem_clamped(model, zz[:1], zz[1:], n=n, iters=it, topk=30, gen=torch.Generator(device=dev).manual_seed(0), cost="min", algo=algo)
    res[f"cached {name}"] = timeit(ours)
    print(name, res[f"official {name}"], res[f"cached {name}"], flush=True)
if not getattr(model, "is_lewm", False):
    fl = FlashCEM(model, n=300, topk=30, iters=30)
    def fl_t(i):
        zz = encode_pixels(model, np.concatenate([pairs["pixels"][i:i + 1], pairs["goal"][i:i + 1]])); fl(zz[0], zz[1], torch.Generator(device=dev).manual_seed(0))
    res["flash cem 300x30 (compiled)"] = timeit(fl_t)
print(tag, res, flush=True)
json.dump(res, open(f"results/e11_solver_latency_{tag}.json", "w"), indent=1)
