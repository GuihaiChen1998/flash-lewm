
import time, glob, os, subprocess, numpy as np, sys
while subprocess.run("ps -eo args | grep 'doom_collect.py' | grep -v grep | wc -l", shell=True, capture_output=True, text=True).stdout.strip() != '0':
    time.sleep(10)
fs = sorted(glob.glob('/data1/cgh/dataset/doom/mwh_[0-9].npz')); parts = [np.load(f) for f in fs]
off = 0; out = {k: [] for k in ['pixels','action','state','ep_idx','step_idx']}
for p in parts:
    for k in out: out[k].append(p[k] + (off if k == 'ep_idx' else 0))
    off += int(p['ep_idx'].max()) + 1
out = {k: np.concatenate(v) for k, v in out.items()}
np.savez('/data1/cgh/dataset/doom/mwh_all.npz', **out); print('merged', out['pixels'].shape, 'episodes', off, flush=True)
