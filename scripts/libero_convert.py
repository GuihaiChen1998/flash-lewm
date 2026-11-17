"""Convert LIBERO demo hdf5 files to the npz layout of flash_lewm.train_min (agentview frames, 7-d actions).
Usage: libero_convert.py <out.npz> <demo1.hdf5> [demo2.hdf5 ...]"""
import sys, os
import numpy as np, h5py

out, files = sys.argv[1], sys.argv[2:]
P, A, E, S, T = [], [], [], [], []
e = 0
for ti, fn in enumerate(files):
    with h5py.File(fn, "r") as f:
        d = f["data"]
        for k in sorted(d.keys(), key=lambda s: int(s.split("_")[1])):
            g = d[k]; n = g["actions"].shape[0]
            P.append(g["obs/agentview_rgb"][:]); A.append(g["actions"][:].astype(np.float32))
            E.append(np.full(n, e)); S.append(np.arange(n)); T.append(np.full(n, ti)); e += 1
    print(os.path.basename(fn), "episodes so far", e, flush=True)
np.savez(out, pixels=np.concatenate(P), action=np.concatenate(A), ep_idx=np.concatenate(E), step_idx=np.concatenate(S),
         task=np.concatenate(T), files=np.array([os.path.basename(x) for x in files]))
print("frames", sum(len(x) for x in A))
