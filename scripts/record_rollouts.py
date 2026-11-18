"""Record frames of real-time closed-loop rollouts for the project page.
Same simulator and episodes as scripts/e9_realtime.py; every env step is rendered and tagged as acting or
holding still (planning pause). Usage: record_rollouts.py <out.npz> <n_episodes> <scenario> <cfg> [<cfg> ...]
cfg = method:m:d:iters:cost (scenario given separately)."""
import os, sys
import numpy as np, torch
from flash_lewm.core import load_model, official_eval_rows, load_pairs, action_stats
import flash_lewm.realtime as rt

out, n, scn, cfgs = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
SCN = {"base": ({}, 25, 50), "switch": ({"switch": 8}, 25, 75)}
kw, off, T = SCN[scn]
model = load_model()
a_mean, a_std = (model.a_mean, model.a_std) if hasattr(model, "a_mean") else action_stats()

if os.environ.get("FLASH_ENV") in ("doom", "ant", "cube", "libero", "reacher"):
    d = dict(np.load(os.environ["DOOM_PAIRS"])); pairs = {k: v[:n] for k, v in d.items()}
else:
    rows = official_eval_rows(50, seed=42, goal_offset=50)[:n]
    pairs = load_pairs(rows, goal_offset=off); q = load_pairs(rows, goal_offset=50)
    pairs["goal2"], pairs["goal2_state"] = q["goal"], q["goal_state"]


class Rec:
    def __init__(self, env):
        object.__setattr__(self, "_env", env); object.__setattr__(self, "frames", []); object.__setattr__(self, "acting", [])
    def __getattr__(self, k):
        return getattr(self._env, k)
    def __setattr__(self, k, v):
        setattr(self._env, k, v)
    def step(self, a):
        r = self._env.step(a)
        self.frames.append(np.asarray(self._env.render()).copy()); self.acting.append(int(np.any(a)))
        return r


made = []
_mk, _set = rt.make_env, rt.set_episode
def make_env():
    e = Rec(_mk()); made.append(e); return e
def set_episode(e, s, g):
    _set(e._env, s, g); e.frames.clear(); e.acting.clear(); e.frames.append(np.asarray(e._env.render()).copy()); e.acting.append(0)
rt.make_env, rt.set_episode = make_env, set_episode

res = {"goal": pairs["goal"], "goal2": pairs["goal2"], "T": T, "switch": kw.get("switch", -1)}
for cfg in cfgs:
    p = cfg.split(":"); made.clear()
    first, _ = rt.simulate(model, pairs, a_mean, a_std, p[0], m=int(p[1]), d=int(p[2]), iters=int(p[3]), cost=p[4], T=T, chunk=n, **kw)
    L = max(len(e.frames) for e in made)
    fr = np.stack([np.stack(e.frames + [e.frames[-1]] * (L - len(e.frames))) for e in made])
    ac = np.array([e.acting + [0] * (L - len(e.acting)) for e in made])
    key = cfg.replace(":", "_")
    res[key + "_frames"], res[key + "_acting"], res[key + "_first"] = fr, ac, np.asarray(first)
    print(cfg, "success", np.mean(np.asarray(first) < 50), "frames", fr.shape, flush=True)
np.savez_compressed(out, **res)
print("saved", out)
