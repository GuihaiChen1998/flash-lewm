"""Evaluation episodes for visual AntMaze from the OGBench dataset: start = a dataset state,
goal = frame / xy 25 steps later, goal2 = 50 steps later. Usage: ant_pairs.py <out.npz> <n>"""
import sys
import numpy as np
from flash_lewm.ant_env import XY_THR

out, n = sys.argv[1], int(sys.argv[2])
d = np.load("/data1/cgh/dataset/ogbench/visual-antmaze-medium-navigate-v0.npz")
obs, qpos, qvel, term = d["observations"], d["qpos"], d["qvel"], d["terminals"].astype(bool)
N = len(term)
ep = np.concatenate([[0], np.cumsum(term)[:-1]])
ok = np.zeros(N, bool); ok[:N - 50] = ep[:N - 50] == ep[50:]
d25 = np.zeros(N); d25[:N - 25] = np.linalg.norm(qpos[25:, :2] - qpos[:-25, :2], axis=1)
cand = np.nonzero(ok & (d25 > 2 * XY_THR))[0]          # the goal is not already within the success radius
rows = np.sort(np.random.default_rng(42).choice(cand, size=n, replace=False))
R = {"state": np.concatenate([qpos[rows], qvel[rows]], 1).astype(np.float64), "pixels": obs[rows],
     "goal": obs[rows + 25], "goal_state": qpos[rows + 25, :2].astype(np.float64),
     "goal2": obs[rows + 50], "goal2_state": qpos[rows + 50, :2].astype(np.float64), "rows": rows}
print(f"episodes {n}; start->goal xy distance median {np.median(d25[rows]):.2f} (25 steps), "
      f"{np.median(np.linalg.norm(qpos[rows + 50, :2] - qpos[rows, :2], axis=1)):.2f} (50 steps); success radius {XY_THR}")
np.savez(out, **R)
