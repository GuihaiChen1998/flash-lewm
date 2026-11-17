"""Replay check for LIBERO: restore a demo state, replay the recorded actions, compare simulator state and rendered frame with the dataset.
Usage: libero_check.py <demo.hdf5> [n_demos]"""
import sys, os, json
import numpy as np, h5py
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
f = h5py.File(path, "r"); d = f["data"]
bddl = os.path.join(get_libero_path("bddl_files"), "libero_10", os.path.basename(d.attrs["bddl_file_name"]))
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.seed(0)
for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1]))[:n]:
    g = d[k]; st, act, img = g["states"][:], g["actions"][:], g["obs/agentview_rgb"][:]
    env.reset(); obs = env.set_init_state(st[0])
    e0 = [np.abs(obs["agentview_image"].astype(float) - img[0]).mean(), np.abs(obs["agentview_image"][::-1].astype(float) - img[0]).mean()]
    errs = {}
    for t in range(len(act)):
        obs, r, done, info = env.step(act[t])
        if t + 1 < len(st) and t + 1 in (1, 5, 25, 50, 100, len(st) - 1):
            s = env.get_sim_state()
            errs[t + 1] = (float(np.abs(s[1:] - st[t + 1][1:]).max()), float(np.abs(obs["agentview_image"].astype(float) - img[t + 1]).mean()))
    print(k, "len", len(act), "img err at 0 (raw, flipped)", np.round(e0, 2), "state/img err", {a: (round(b, 5), round(c, 2)) for a, (b, c) in errs.items()}, "success", env.check_success(), flush=True)
    # restore from the middle and replay 25 steps
    t0 = len(act) // 2; env.reset(); env.set_init_state(st[t0])
    for t in range(t0, min(t0 + 25, len(act))): obs, *_ = env.step(act[t])
    te = min(t0 + 25, len(st) - 1); s = env.get_sim_state()
    print("   mid restore", t0, "-> state err", round(float(np.abs(s[1:] - st[te][1:]).max()), 5), "img err", round(float(np.abs(obs["agentview_image"].astype(float) - img[te]).mean()), 2), flush=True)
print("obs keys", [k for k in obs.keys()][:40]); print("state dim", len(s))
sim = env.sim; print("nq", sim.model.nq, "nv", sim.model.nv, "bodies", [sim.model.body_id2name(i) for i in range(sim.model.nbody)][:60])
