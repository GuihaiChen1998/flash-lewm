"""Evaluation episodes for Reacher from the LeWM dataset: start = dataset qpos/qvel, goal = frame and qpos 25 steps
later, goal2 = 50 steps later; keep only episodes where some joint must move more than the success threshold for
both goals (and goal2 differs from the start). Usage: reacher_pairs.py <out.npz> <n> [threshold=0.1]"""
import sys
import numpy as np, h5py, hdf5plugin  # noqa: F401

out, n = sys.argv[1], int(sys.argv[2]); thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
with h5py.File("/data1/cgh/dataset/lewm-original/reacher/reacher.h5", "r") as f:
    ep, step, ep_len = f["ep_idx"][:], f["step_idx"][:], f["ep_len"][:]
    qpos = f["qpos"][:]
    idx = np.nonzero(step <= ep_len[ep] - 51)[0]
    mv = lambda a, b: np.abs(qpos[a] - qpos[b]).max(1) > thr
    keep = idx[mv(idx + 25, idx) & mv(idx + 50, idx + 25) & mv(idx + 50, idx)]
    rows = np.sort(np.random.default_rng(42).choice(keep, size=n, replace=False))
    R = {k: [] for k in ["state", "pixels", "goal", "goal_state", "goal2", "goal2_state"]}
    for r in rows:
        R["state"].append(np.concatenate([f["qpos"][r], f["qvel"][r]]))
        R["pixels"].append(f["pixels"][r])
        for k, o in (("goal", 25), ("goal2", 50)):
            R[k].append(f["pixels"][r + o]); R[k + "_state"].append(f["qpos"][r + o])
R = {k: np.stack(v) for k, v in R.items()}; R["rows"] = rows
print(f"candidates {len(keep)} of {len(idx)}, picked {n}")
np.savez(out, **R)
