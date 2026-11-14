"""Build ViZDoom evaluation episodes: start pose reached by replaying k0 exploration actions,
goal = frame / pose 25 steps further along the same exploration run (goal2 = 50 steps further).
Usage: doom_pairs.py <out.npz> <n>"""
import sys
import numpy as np
from flash_lewm.doom_env import *

out, n = sys.argv[1], int(sys.argv[2])
g = make_game(0)
R = {k: [] for k in ["state", "goal", "goal_state", "goal2", "goal2_state", "start_pose", "pixels"]}
seed = 100_000
while len(R["state"]) < n:
    seed += 1
    rng = np.random.default_rng(seed)
    k0 = int(rng.integers(20, KMAX))
    g.set_seed(seed); g.new_episode()
    ex = Explorer(rng); acts = []; ok = True; rec = {}
    for t in range(k0 + 51):
        if g.is_episode_finished():
            ok = False; break
        px, pose = observe(g)
        if t in (k0, k0 + 25, k0 + 50):
            rec[t - k0] = (px, pose)
        a = ex.act(pose); acts.append(a)
        g.make_action([float(a[0]), float(a[1])], SKIP)
    if not ok:
        continue
    d25, _ = pose_error(rec[25][1], rec[0][1])
    if d25 < 2 * POS_THR:            # goal too close to the start: trivial
        continue
    spec = np.zeros(2 + 2 * KMAX); spec[0], spec[1] = seed, k0; spec[2:2 + 2 * k0] = np.array(acts[:k0]).reshape(-1)
    R["state"].append(spec); R["pixels"].append(rec[0][0]); R["start_pose"].append(rec[0][1])
    R["goal"].append(rec[25][0]); R["goal_state"].append(rec[25][1]); R["goal2"].append(rec[50][0]); R["goal2_state"].append(rec[50][1])
R = {k: np.stack(v) for k, v in R.items()}
# determinism check: replaying the spec must reproduce the start pose
env = DoomNav(); errs = []
for i in range(min(20, n)):
    env.set_spec(R["state"][i]); errs.append(pose_error(env.pose(), R["start_pose"][i])[0])
print("replay start-pose error (max over 20):", max(errs))
d25 = np.linalg.norm(R["goal_state"][:, :2] - R["start_pose"][:, :2], axis=1)
d50 = np.linalg.norm(R["goal2_state"][:, :2] - R["start_pose"][:, :2], axis=1)
print(f"episodes {n}; start->goal distance median {np.median(d25):.0f} (25 steps), {np.median(d50):.0f} (50 steps); success radius {POS_THR}, angle {ANG_THR}")
np.savez(out, **R)
