"""E0: reproduce the official per-plan path, profile it, and check the cached-latent CEM is equivalent."""
import json, sys, time, types
import numpy as np, torch
from flash_lewm.core import *
from flash_lewm.search import cem
from stable_worldmodel.solver.cem import CEMSolver

torch.backends.cudnn.benchmark = False
dev = "cuda"
model = load_model()
n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
rows = official_eval_rows(50)[:n_pairs]
pairs = load_pairs(rows)

solver = CEMSolver(model, batch_size=1, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device=dev, seed=42)
import gymnasium as gym
solver.configure(action_space=gym.spaces.Box(-1, 1, shape=(1, 2)), n_envs=1,
                 config=types.SimpleNamespace(horizon=1, action_block=25))

def info_for(i):
    px = preprocess(pairs["pixels"][i:i+1]).cpu()[:, None]     # [1,1,3,224,224] on CPU like the policy
    g = preprocess(pairs["goal"][i:i+1]).cpu()[:, None]
    return {"pixels": px, "goal": g, "action": torch.zeros(1, 1, 2)}

# ---- profile hooks
T = {}
def wrap(obj, name, key):
    f = getattr(obj, name)
    def g(*a, **k):
        sync(); t = time.perf_counter(); r = f(*a, **k); sync(); T[key] = T.get(key, 0) + time.perf_counter() - t; return r
    setattr(obj, name, g); return f

import io, contextlib
def official(i, seed=42):
    solver.torch_gen = torch.Generator(device=dev).manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        sync(); t = time.perf_counter(); out = solver.solve(info_for(i)); sync()
    return out, time.perf_counter() - t

for _ in range(2): official(0)           # warm-up
t_off, plans_off = [], []
for i in range(n_pairs):
    out, t = official(i); t_off.append(t); plans_off.append(out["actions"][0, 0].to(dev))

# cached-latent CEM, same RNG stream
t_c, t_enc, diffs, costs = [], [], [], []
for _ in range(2):
    zz = encode_pixels(model, np.concatenate([pairs["pixels"][:1], pairs["goal"][:1]]))
    cem(model, zz[0], zz[1], gen=torch.Generator(device=dev).manual_seed(42))
for i in range(n_pairs):
    sync(); t = time.perf_counter()
    zz = encode_pixels(model, np.concatenate([pairs["pixels"][i:i+1], pairs["goal"][i:i+1]]))
    sync(); t1 = time.perf_counter()
    r = cem(model, zz[0], zz[1], gen=torch.Generator(device=dev).manual_seed(42))
    sync(); t2 = time.perf_counter()
    t_enc.append(t1 - t); t_c.append(t2 - t)
    diffs.append((r["plan"].reshape(-1) - plans_off[i]).abs().max().item())
    c_off = dense_cost(model, zz[0], zz[1], plans_off[i].view(1, 5, 10))[0].item()
    costs.append((c_off, r["cost"]))

# profiled official run (hooks add sync overhead, so only the shares are used)
wrap(model, "encode", "image_encode (ViT+proj, 2 per iter)")
wrap(model.action_encoder, "forward", "action prefix encoder")
wrap(model, "predict", "endpoint predictor")
orig_get_cost = wrap(model, "get_cost", "get_cost total")
_, t_prof = official(0)
res = {
    "n_pairs": n_pairs,
    "official_ms_per_plan": [float(np.mean(t_off) * 1e3), float(np.std(t_off) * 1e3)],
    "cached_ms_per_plan_incl_encode": [float(np.mean(t_c) * 1e3), float(np.std(t_c) * 1e3)],
    "cached_encode_ms": float(np.mean(t_enc) * 1e3),
    "speedup": float(np.mean(t_off) / np.mean(t_c)),
    "max_abs_plan_diff_vs_official": float(np.max(diffs)),
    "cost_official_vs_cached": costs[:4],
    "profile_ms": {k: v * 1e3 for k, v in T.items()},
    "profile_total_ms": t_prof * 1e3,
}
res["profile_ms"]["cem bookkeeping (sample/topk/expand)"] = t_prof * 1e3 - T["get_cost total"] * 1e3
res["profile_ms"]["get_cost other (H2D copies, dict)"] = (T["get_cost total"] - T["image_encode (ViT+proj, 2 per iter)"]
    - T["action prefix encoder"] - T["endpoint predictor"]) * 1e3
print(json.dumps(res, indent=1))
json.dump(res, open("results/e0_profile.json", "w"), indent=1)
