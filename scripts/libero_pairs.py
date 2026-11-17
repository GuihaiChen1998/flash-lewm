"""Evaluation episodes for one LIBERO task: start = a demo state, goal = the frame/state 25 steps later,
goal2 = 50 steps later. Keep only segments where some object moves more than the success radius for both
goals (and goal2 differs from the start). Usage: libero_pairs.py <demo.hdf5> <out.npz> <n> [offset=25]"""
import sys, os
import numpy as np, h5py

fn, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
off = int(sys.argv[4]) if len(sys.argv) > 4 else 25
os.environ["FLASH_LIBERO_BDDL"] = os.path.join(os.environ["LIBERO_BDDL_DIR"], os.path.basename(h5py.File(fn, "r")["data"].attrs["bddl_file_name"]))
from flash_lewm.libero_env import LiberoNav, XY_THR

nav = LiberoNav()
f = h5py.File(fn, "r"); d = f["data"]
cand = []
for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
    st = d[k]["states"][:]
    P = np.stack([nav.obj_pos(s) for s in st])           # [T, n_obj, 3]
    for t in range(0, len(st) - 2 * off):
        a, b, c = P[t], P[t + off], P[t + 2 * off]
        mv = lambda x, y: np.linalg.norm(x - y, axis=1).max() > XY_THR
        if mv(a, b) and mv(b, c) and mv(a, c):
            cand.append((k, t))
rng = np.random.default_rng(42)
pick = [cand[i] for i in np.sort(rng.choice(len(cand), size=min(n, len(cand)), replace=False))]
R = {k: [] for k in ["state", "pixels", "goal", "goal_state", "goal2", "goal2_state"]}
for k, t in pick:
    g = d[k]
    R["state"].append(g["states"][t]); R["pixels"].append(g["obs/agentview_rgb"][t])
    for key, o in (("goal", off), ("goal2", 2 * off)):
        R[key].append(g["obs/agentview_rgb"][t + o]); R[key + "_state"].append(g["states"][t + o])
R = {k: np.stack(v) for k, v in R.items()}
R["demo"] = np.array([int(k.split("_")[1]) for k, _ in pick]); R["t"] = np.array([t for _, t in pick])
print(f"candidate segments {len(cand)}, picked {len(pick)}, objects {len(nav.obj)}, demos used {len(set(R['demo']))}")
np.savez(out, **R)
