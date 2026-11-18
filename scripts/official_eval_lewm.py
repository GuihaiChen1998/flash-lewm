"""Run the official LeWM eval.py (third_party/le-wm, Maes et al. 2026) unchanged, with shims only for the
installed stable_worldmodel (the Fast-LeWM pin) and for loading:
1. stable_pretraining is only used for the ImageNet normalisation constants -> a tiny stand-in module;
2. swm.wm.utils.load_pretrained(<env>) -> flash_lewm.lewm.load_lewm(ckpt/lewm/<env>) (HF release, weights strict);
3. World.evaluate(dataset=..., goal_offset=..., video=...) -> World.evaluate_from_dataset (same protocol, renamed
   arguments in the newer library), without video; per-episode successes are written to FLASH_OFFICIAL_JSON.
FLASH_COST: last (official JEPA.get_cost, re-encodes every candidate image every CEM iteration),
            hooklast (our scorer: observation and goal encoded once per call, official terminal cost -> must match),
            min (our scorer with the prefix-min cost).
Everything else (row sampling, environments, success test, CEM 300 x 30, horizon 5 x 5) is the official code and config.
Datasets are read from cache_dir with the official names (symlinks in /data1/cgh/dataset/lewm_cache).
Usage: official_eval_lewm.py --config-name=<pusht|tworoom|cube|reacher> policy=<env> +cache_dir=<dir> seed=<s>"""
import os, sys, types
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LE = os.path.join(ROOT, "third_party", "le-wm")
sys.path.insert(0, LE)
spt = types.ModuleType("stable_pretraining")
spt.data = types.SimpleNamespace(dataset_stats=types.SimpleNamespace(ImageNet=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
sys.modules["stable_pretraining"] = spt
import stable_worldmodel as swm
from flash_lewm.lewm import load_lewm


def load_pretrained(name, *a, **k):
    m = load_lewm(os.path.join(ROOT, "ckpt", "lewm", name))
    c = os.environ.get("FLASH_COST", "last")
    if c in ("min", "hooklast"):
        from flash_lewm.official_hooks import install_min_cost
        install_min_cost(m, "min" if c == "min" else "last")
    return m


utils = types.ModuleType("stable_worldmodel.wm.utils"); utils.load_pretrained = load_pretrained
sys.modules["stable_worldmodel.wm.utils"] = utils; swm.wm.utils = utils


def evaluate(self, dataset, start_steps, goal_offset, eval_budget, episodes_idx, callables=None, video=None, **_):
    res = self.evaluate_from_dataset(dataset, episodes_idx=episodes_idx, start_steps=start_steps, goal_offset_steps=goal_offset,
                                     eval_budget=eval_budget, callables=callables, save_video=False)
    out = os.environ.get("FLASH_OFFICIAL_JSON")
    if out:
        import json, numpy as np
        json.dump({"success_rate": res["success_rate"], "episode_successes": np.asarray(res["episode_successes"]).astype(int).tolist(),
                   "episodes_idx": [int(x) for x in episodes_idx], "start_steps": [int(x) for x in start_steps],
                   "cost": os.environ.get("FLASH_COST", "last")}, open(out, "w"))
    return res


swm.World.evaluate = evaluate
import eval as official_eval
official_eval.run()
