"""OGBench Cube (single) closed loop with the Fast-LeWM OGB checkpoint, following the official
eval config: start state and goal image from the dataset (goal = +25 steps), 50 env steps,
success = cube within 4 cm of the goal cube position at any step.
Usage: cube_eval.py <n_episodes> <tag> <cfg> [<cfg> ...]   cfg = cost:m   (cost last|min, m = blocks
executed before replanning; m=5 is the official schedule). Planning is instantaneous (no delay)."""
import os, sys, json, time
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, hdf5plugin
from flash_lewm.core import load_model, encode_pixels
from flash_lewm.realtime import plan_cost

H5 = "/data1/cgh/dataset/lewm-original/cube/cube_single_expert.h5"
CK = "/data1/cgh/code/flash-lewm/ckpt/Fast-lewm_ogb_object.ckpt"
H, B, DA = 5, 5, 5
A = B * DA
OFF, T = 25, 50
n_ep, tag, cfgs = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
dev = "cuda"
model = load_model(CK)

with h5py.File(H5, "r") as f:
    ep, step, ep_len = f["ep_idx"][:], f["step_idx"][:], f["ep_len"][:]
    valid = np.nonzero(step <= ep_len[ep] - OFF - 1)[0]
    rows = np.sort(valid[np.random.default_rng(42).choice(len(valid) - 1, size=n_ep, replace=False)])
    act = f["action"][:2_000_000:7].astype(np.float64)
    a_mean, a_std = act.mean(0), act.std(0)
    D = {k: [] for k in ["pixels", "goal", "qpos", "qvel", "bpos", "gpos", "gquat", "acts"]}
    for r in rows:
        D["pixels"].append(f["pixels"][r]); D["goal"].append(f["pixels"][r + OFF])
        D["qpos"].append(f["qpos"][r]); D["qvel"].append(f["qvel"][r])
        D["bpos"].append(f["privileged_block_0_pos"][r])
        D["gpos"].append(f["privileged_block_0_pos"][r + OFF]); D["gquat"].append(f["privileged_block_0_quat"][r + OFF])
        D["acts"].append(f["action"][r:r + OFF])
    D = {k: np.stack(v) for k, v in D.items()}
moved = np.linalg.norm(D["gpos"] - D["bpos"], axis=1) > 0.04      # goal not already satisfied at the start
print(f"episodes {n_ep}, non-trivial (cube must move > 4 cm): {moved.sum()}", flush=True)

from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
CH = 25
envs = [CubeEnv(env_type="single", ob_type="states", multiview=False, width=224, height=224,
                visualize_info=False, terminate_at_goal=True) for _ in range(CH)]

def start(e, i):
    e.reset(seed=0)
    e.unwrapped.set_state(D["qpos"][i], D["qvel"][i])
    e.unwrapped.set_target_pos(0, D["gpos"][i], D["gquat"][i])

# sanity: rendering matches the dataset, expert replay reaches the goal
diffs, ok = [], 0
for i in range(min(20, n_ep)):
    start(envs[0], i)
    diffs.append(np.abs(envs[0].render().astype(float) - D["pixels"][i].astype(float)).mean())
    s = False
    for a in D["acts"][i]:
        _, _, term, _, info = envs[0].step(a.astype(np.float64)); s = s or bool(term)
        if term: break
    ok += s
print(f"sanity: render-vs-dataset mean abs px diff {np.mean(diffs):.2f}; expert replay success {ok}/{min(20, n_ep)}", flush=True)

@torch.inference_mode()
def cem(z, zg, gen, cost, n=300, topk=30, iters=30):
    e = z.shape[0]
    mean = torch.zeros(e, H * A, device=dev); std = torch.ones(e, H * A, device=dev)
    for _ in range(iters):
        c = torch.randn(e, n, H * A, generator=gen, device=dev) * std[:, None] + mean[:, None]
        c[:, 0] = mean
        v = plan_cost(model, z, zg, c.view(e, n, H, A), cost)
        _, i = torch.topk(v, topk, dim=1, largest=False)
        el = torch.gather(c, 1, i[..., None].expand(-1, -1, H * A))
        mean, std = el.mean(1), el.std(1)
    return mean

path = f"results/cube_eval_{tag}.json"
try: res = json.load(open(path))
except Exception: res = {}
for cfg in cfgs:
    cost, m = cfg.split(":"); m = int(m)
    first = np.full(n_ep, T, dtype=np.int64); t0 = time.time()
    for c0 in range(0, n_ep, CH):
        idx = np.arange(c0, min(c0 + CH, n_ep)); E = len(idx); ev = envs[:E]
        for e, i in zip(ev, idx): start(e, i)
        gen = torch.Generator(device=dev).manual_seed(c0)
        zg = encode_pixels(model, D["goal"][idx])
        done = np.zeros(E, dtype=bool); t = 0
        while t < T:
            px = np.stack([e.render() for e in ev]) if t > 0 else D["pixels"][idx]
            plan = cem(encode_pixels(model, px), zg, gen, cost)
            a = plan[:, :m * A].reshape(E, -1, DA).double().cpu().numpy() * a_std + a_mean
            for s in range(a.shape[1]):
                if t >= T: break
                for j, e in enumerate(ev):
                    if done[j]: continue
                    _, _, term, _, _ = e.step(a[j, s])
                    if term:
                        done[j] = True; first[idx[j]] = t
                t += 1
        print(f"  {cfg}: {c0 + E}/{n_ep} success={np.mean(first[:c0 + E] < T):.3f}", flush=True)
    res[cfg] = {"n": n_ep, "first_success": first.tolist(), "moved": moved.tolist(), "success": float((first < T).mean()),
                "success_nontrivial": float((first[moved] < T).mean()) if moved.any() else None,
                "mean_first_success_step_nontrivial": float(first[moved][first[moved] < T].mean()) if (first[moved] < T).any() else None}
    print(f"{cfg:10s} n={n_ep} success={100*res[cfg]['success']:.1f}%  non-trivial={100*(res[cfg]['success_nontrivial'] or 0):.1f}% (n={moved.sum()})  t={res[cfg]['mean_first_success_step_nontrivial']}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(path, "w"))
