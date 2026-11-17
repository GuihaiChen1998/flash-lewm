"""E10: wall-clock planning latency per environment: the official CEMSolver path (re-encodes the images every
iteration) vs FlashCEM (latents cached, whole CEM iteration compiled), both 300 samples x 30 iterations.
Environment is chosen by PUSHT_H5 / FLASH_CKPT / FLASH_ADIM / FLASH_EP_KEY as in the other scripts.
Usage: e10_latency.py <tag> [n_pairs=20]"""
import io, contextlib, json, os, sys, time, types
import numpy as np, torch
from flash_lewm.core import load_model, official_eval_rows, load_pairs, preprocess, encode_pixels, sync
from flash_lewm.search import FlashCEM
from stable_worldmodel.solver.cem import CEMSolver
import gymnasium as gym

tag = sys.argv[1]; n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
adim = int(os.environ.get("FLASH_ADIM", "2")); dev = "cuda"
model = load_model()
rows = official_eval_rows(50)[:n_pairs]; pairs = load_pairs(rows)
solver = CEMSolver(model, batch_size=1, num_samples=300, var_scale=1.0, n_steps=30, topk=30, device=dev, seed=42)
solver.configure(action_space=gym.spaces.Box(-1, 1, shape=(1, adim)), n_envs=1, config=types.SimpleNamespace(horizon=1, action_block=25))

def info_for(i):
    px = preprocess(pairs["pixels"][i:i + 1]).cpu()[:, None]
    g = preprocess(pairs["goal"][i:i + 1]).cpu()[:, None]
    return {"pixels": px, "goal": g, "action": torch.zeros(1, 1, adim)}

def official(i):
    solver.torch_gen = torch.Generator(device=dev).manual_seed(42)
    with contextlib.redirect_stdout(io.StringIO()):
        sync(); t = time.perf_counter(); solver.solve(info_for(i)); sync()
    return time.perf_counter() - t

flash = FlashCEM(model, n=300, topk=30, iters=30)
def flash_t(i):
    sync(); t = time.perf_counter()
    zz = encode_pixels(model, np.concatenate([pairs["pixels"][i:i + 1], pairs["goal"][i:i + 1]]))
    flash(zz[0], zz[1], torch.Generator(device=dev).manual_seed(42)); sync()
    return time.perf_counter() - t

for _ in range(3): official(0); flash_t(0)
to = [official(i) for i in range(n_pairs)]; tf = [flash_t(i) for i in range(n_pairs)]
res = {"official_ms": [float(np.median(to) * 1e3), float(np.percentile(to, 90) * 1e3)],
       "flash_ms": [float(np.median(tf) * 1e3), float(np.percentile(tf, 90) * 1e3)], "n": n_pairs, "gpu": torch.cuda.get_device_name()}
print(tag, res, flush=True)
os.makedirs("results", exist_ok=True); json.dump(res, open(f"results/e10_latency_{tag}.json", "w"))
