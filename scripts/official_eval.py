"""Run the official Fast-LeWorldModel eval.py unchanged, with two shims:
1. stable_pretraining is only used for the ImageNet normalisation constants -> a tiny stand-in module;
2. swm.policy.AutoCostModel is replaced by flash_lewm.core.load_model, because the released checkpoints were
   pickled with transformers 5.x and need the ViT rebuilt under the installed 4.54 (weights loaded strictly).
Optional: FLASH_COST=min swaps the planning cost for the prefix-minimum cost (our method), FLASH_SOLVER=flash
uses FlashCEM. Everything else (row sampling, environments, success test, CEM settings) is the official code.
Usage: official_eval.py --config-name=<pusht|tworoom|cube|reacher> <hydra overrides...>"""
import os, sys, types
FL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "Fast-LeWorldModel")
sys.path.insert(0, FL)
spt = types.ModuleType("stable_pretraining")
spt.data = types.SimpleNamespace(dataset_stats=types.SimpleNamespace(ImageNet=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])))
sys.modules["stable_pretraining"] = spt
import stable_worldmodel as swm
from flash_lewm.core import load_model


def _auto_cost_model(name, cache_dir=None):
    path = os.path.join(cache_dir, name + "_object.ckpt")
    if not os.path.exists(path):
        path = os.path.join(cache_dir, name + ".ckpt")
    m = load_model(path)
    c = os.environ.get("FLASH_COST", "last")
    if c in ("min", "hooklast"):          # hooklast: our scorer with the official terminal cost (equivalence check)
        from flash_lewm.official_hooks import install_min_cost
        install_min_cost(m, "min" if c == "min" else "last")
    return m


swm.policy.AutoCostModel = _auto_cost_model

# no videos (imageio-ffmpeg is not installed); keep the per-episode successes as JSON next to the results file
_orig_eval = swm.World.evaluate_from_dataset


def _eval_no_video(self, *a, **k):
    k["save_video"] = False
    res = _orig_eval(self, *a, **k)
    import json, numpy as np
    out = os.environ.get("FLASH_OFFICIAL_JSON")
    if out:
        json.dump({"success_rate": res["success_rate"], "episode_successes": np.asarray(res["episode_successes"]).astype(int).tolist(),
                   "episodes_idx": [int(x) for x in k.get("episodes_idx", [])], "start_steps": [int(x) for x in k.get("start_steps", [])],
                   "cost": os.environ.get("FLASH_COST", "last")}, open(out, "w"))
    return res


swm.World.evaluate_from_dataset = _eval_no_video
import eval as official_eval
official_eval.run()
