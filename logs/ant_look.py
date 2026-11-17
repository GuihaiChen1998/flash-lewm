import numpy as np
d=np.load('/data1/cgh/dataset/ogbench/visual-antmaze-medium-navigate-v0.npz')
for k in d.files: print(k, d[k].shape, d[k].dtype)
t=d['terminals']; print('episodes', int(t.sum()), 'first terminal idx', np.nonzero(t)[0][:3]); a=d['actions']; print('act range', a.min(0), a.max(0))
