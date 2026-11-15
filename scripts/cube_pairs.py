"""Evaluation episodes for Cube from the LeWM dataset: start = dataset state, goal = frame and cube
pose 25 steps later, goal2 = 50 steps later; only episodes where the cube has to move more than the
4 cm success radius for both goals. Usage: cube_pairs.py <out.npz> <n>"""
import sys
import numpy as np, h5py, hdf5plugin

out, n = sys.argv[1], int(sys.argv[2])
with h5py.File("/data1/cgh/dataset/lewm-original/cube/cube_single_expert.h5", "r") as f:
    ep, step, ep_len = f["ep_idx"][:], f["step_idx"][:], f["ep_len"][:]
    bpos = f["privileged_block_0_pos"][:]
    N = len(ep)
    ok = step <= ep_len[ep] - 51
    idx = np.nonzero(ok)[0]
    moved = (np.linalg.norm(bpos[idx + 25] - bpos[idx], axis=1) > 0.04) & (np.linalg.norm(bpos[idx + 50] - bpos[idx + 25], axis=1) > 0.04)
    rows = np.sort(np.random.default_rng(42).choice(idx[moved], size=n, replace=False))
    R = {k: [] for k in ["state", "pixels", "goal", "goal_state", "goal2", "goal2_state"]}
    for r in rows:
        R["state"].append(np.concatenate([f["qpos"][r], f["qvel"][r]]))
        R["pixels"].append(f["pixels"][r])
        for k, o in (("goal", 25), ("goal2", 50)):
            R[k].append(f["pixels"][r + o])
            R[k + "_state"].append(np.concatenate([f["privileged_block_0_pos"][r + o], f["privileged_block_0_quat"][r + o]]))
R = {k: np.stack(v) for k, v in R.items()}
R["rows"] = rows
print(f"episodes {n}; cube displacement median {np.median(np.linalg.norm(R['goal_state'][:, :3] - bpos[rows], axis=1)):.3f} m to goal, "
      f"{np.median(np.linalg.norm(R['goal2_state'][:, :3] - bpos[rows], axis=1)):.3f} m to goal2")
np.savez(out, **R)
