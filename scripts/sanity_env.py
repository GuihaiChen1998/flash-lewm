
import numpy as np, h5py, hdf5plugin
from flash_lewm.core import *
from flash_lewm.closedloop import *
rows = official_eval_rows(50)[:12]
pairs = load_pairs(rows)
env = make_env()
succ, diffs = 0, []
with h5py.File(H5, "r") as f:
    for i, r in enumerate(rows):
        set_episode(env, pairs["state"][i], pairs["goal_state"][i])
        img = env.render()
        diffs.append(np.abs(img.astype(float) - pairs["pixels"][i].astype(float)).mean())
        acts = f["action"][r:r + 25]
        ok = False
        for a in acts:
            _, rew, term, _, _ = env.step(a.astype(np.float64)); ok = ok or term
        succ += ok
        print(i, r, 'render-vs-dataset mean abs px diff', round(diffs[-1], 3), 'expert replay success', ok, 'final dist', round(-rew, 2), img.shape, img.dtype)
print('expert replay success', succ, '/', len(rows)); print(action_stats())
