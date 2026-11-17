"""Upper bound for the LIBERO evaluation episodes: replay the demo's own actions from each start and count how
often the success criterion fires within 25 / 50 steps. Usage: libero_replay_success.py <demo.hdf5> <pairs.npz> [n]"""
import sys, os
import numpy as np, h5py
fn, pp = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
os.environ["FLASH_LIBERO_BDDL"] = os.path.join(os.environ["LIBERO_BDDL_DIR"], os.path.basename(h5py.File(fn, "r")["data"].attrs["bddl_file_name"]))
from flash_lewm.libero_env import LiberoNav
nav = LiberoNav(); d = h5py.File(fn, "r")["data"]; P = np.load(pp)
hit25 = hit50 = 0; first = []
for i in range(min(n, len(P["t"]))):
    k, t = f"demo_{P['demo'][i]}", int(P["t"][i]); act = d[k]["actions"][:]
    nav.set_spec(P["state"][i]); nav._set_goal_state(P["goal_state"][i]); f0 = None
    for s in range(50):
        a = act[t + s] if t + s < len(act) else np.zeros(7)
        _, _, term, _, _ = nav.step(a)
        if term and f0 is None: f0 = s + 1
    first.append(f0 if f0 is not None else 99)
first = np.array(first)
print(f"demo-action replay: success within 25 steps {np.mean(first <= 25):.3f}, within 50 {np.mean(first <= 50):.3f}, n={len(first)}; median first hit {np.median(first[first < 99]) if (first < 99).any() else None}")
