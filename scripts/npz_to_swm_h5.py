"""Convert a train_min-style npz (pixels, action, ep_idx, step_idx) into the stable-worldmodel HDF5 layout
read by the official Fast-LeWorldModel train.py / eval.py (adds ep_len, ep_offset). Usage: npz_to_swm_h5.py <in.npz> <out.h5>"""
import sys
import numpy as np, h5py

src, dst = sys.argv[1], sys.argv[2]
d = np.load(src, mmap_mode="r")
ep = np.asarray(d["ep_idx"]).astype(np.int32)
order = np.argsort(ep, kind="stable")
assert (order == np.arange(len(ep))).all(), "rows must already be grouped by episode"
uniq, offset, length = np.unique(ep, return_index=True, return_counts=True)
assert (uniq == np.arange(len(uniq))).all(), "episode ids must be 0..E-1"
n = len(ep)
with h5py.File(dst, "w") as f:
    px = d["pixels"]
    ds = f.create_dataset("pixels", shape=px.shape, dtype=np.uint8, chunks=(min(100, n),) + px.shape[1:])
    for i in range(0, n, 20000):
        ds[i:i + 20000] = px[i:i + 20000]
        print("pixels", i, flush=True)
    f.create_dataset("action", data=np.asarray(d["action"], dtype=np.float32), chunks=(min(1000, n), d["action"].shape[1]))
    f.create_dataset("ep_idx", data=ep, chunks=(min(1000, n),))
    f.create_dataset("step_idx", data=np.asarray(d["step_idx"]).astype(np.int64), chunks=(min(1000, n),))
    f.create_dataset("ep_len", data=length.astype(np.int32))
    f.create_dataset("ep_offset", data=offset.astype(np.int64))
    for k in d.files:
        if k in ("task", "state") and len(d[k]) == n:
            f.create_dataset(k, data=np.asarray(d[k]))
print("rows", n, "episodes", len(uniq))
