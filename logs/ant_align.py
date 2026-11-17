
import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, ogbench, mujoco
print('mujoco', mujoco.__version__, 'ogbench', getattr(ogbench,'__version__','?'))
env = ogbench.make_env_and_datasets('visual-antmaze-medium-navigate-v0', env_only=True); env.reset(seed=0); u = env.unwrapped
print('frame_skip', u.frame_skip, 'timestep', u.model.opt.timestep, 'integrator', u.model.opt.integrator, 'solver iters', u.model.opt.iterations)
d = np.load('/data1/cgh/dataset/ogbench/visual-antmaze-medium-navigate-v0.npz')
qpos, qvel, act, term = d['qpos'], d['qvel'], d['actions'], d['terminals'].astype(bool)
N = len(term); ep = np.concatenate([[0], np.cumsum(term)[:-1]])
ok = np.zeros(N, bool); ok[:N-26] = ep[:N-26] == ep[26:]
rows = np.random.default_rng(0).choice(np.nonzero(ok)[0], 300, replace=False)
K = [1, 2, 3, 5, 10, 15, 25]
def rollout(q, v, acts, clean=False):
    if clean: mujoco.mj_resetData(u.model, u.data)
    u.set_state(q.astype(np.float64), v.astype(np.float64)); out = []
    for a in acts:
        u.step(a); out.append(np.concatenate([u.data.qpos.copy(), u.data.qvel.copy()]))
    return np.array(out)
E = {k: [] for k in K}; EX = {k: [] for k in K}; S = {k: [] for k in K}; C = {k: [] for k in K}; M = {k: [] for k in K}
rng = np.random.default_rng(1)
for r in rows:
    A = rollout(qpos[r], qvel[r], act[r:r+25])
    B = rollout(qpos[r] + rng.normal(0, 1e-7, 15).astype(np.float32), qvel[r], act[r:r+25])   # same env, start perturbed by float32-level noise
    Cc = rollout(qpos[r], qvel[r], act[r:r+25], clean=True)
    for k in K:
        E[k].append(np.abs(A[k-1][:15] - qpos[r+k]).max()); EX[k].append(np.linalg.norm(A[k-1][:2] - qpos[r+k][:2]))
        S[k].append(np.linalg.norm(A[k-1][:2] - B[k-1][:2])); C[k].append(np.linalg.norm(Cc[k-1][:2] - qpos[r+k][:2]))
        M[k].append(np.linalg.norm(qpos[r+k][:2] - qpos[r][:2]))
print('k | max|qpos err| median, p90 | xy err median, p90 | xy err after mj_resetData median | self-divergence (1e-7 start noise) median, p90 | true xy displacement median')
for k in K:
    print(k, '| %.2e %.2e | %.2e %.2e | %.2e | %.2e %.2e | %.3f' % (np.median(E[k]), np.percentile(E[k],90), np.median(EX[k]), np.percentile(EX[k],90), np.median(C[k]), np.median(S[k]), np.percentile(S[k],90), np.median(M[k])))
# one-step: which stored quantity mismatches most
r = rows[0]; A = rollout(qpos[r], qvel[r], act[r:r+1])
print('one-step qpos err per dim', np.abs(A[0][:15]-qpos[r+1]).round(6)); print('one-step qvel err per dim', np.abs(A[0][15:]-qvel[r+1]).round(5))
# index-offset hypotheses on one-step error
for name, a in [('act[t]', act[rows]), ('act[t-1]', act[rows-1]), ('act[t+1]', act[rows+1]), ('zero', np.zeros_like(act[rows]))]:
    e = [np.abs(rollout(qpos[r], qvel[r], a[i:i+1])[0][:15] - qpos[r+1]).max() for i, r in enumerate(rows[:100])]
    print('one-step err using', name, 'median %.2e' % np.median(e))
