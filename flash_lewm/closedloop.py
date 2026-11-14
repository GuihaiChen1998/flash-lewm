"""Minimal closed-loop Push-T evaluation that mirrors the official protocol:
start state and goal image from the dataset (goal = same episode, +25 steps), 50 env steps,
replan every 25 steps (horizon 1 x 5 blocks x 5 actions), success = env termination at any step.
"""
import time

import numpy as np
import torch

from .core import encode_pixels, sync


import os

ENV = os.environ.get("FLASH_ENV", "pusht")


def make_env():
    if ENV == "doom":
        from .doom_env import DoomNav

        return DoomNav()
    if ENV == "tworoom":
        from stable_worldmodel.envs.two_room.env import TwoRoomEnv

        env = TwoRoomEnv()
    else:
        from stable_worldmodel.envs.pusht.env import PushT

        env = PushT(resolution=224)
    env.reset(seed=0)
    return env


def set_episode(env, state, goal_state):
    if ENV == "doom":
        env.set_spec(state)
        env._set_goal_state(goal_state)
        return
    env.reset(seed=0)
    dt = np.float32 if ENV == "tworoom" else np.float64
    env._set_state(np.asarray(state, dtype=dt))
    env._set_goal_state(np.asarray(goal_state, dtype=dt))


@torch.inference_mode()
def run_episode(env, model, planner, pair, a_mean, a_std, budget=50, block=25, first_obs="dataset"):
    """planner(z, zg) -> dict(plan=[5,10] normalised). Returns success, final distance, plan latencies."""
    set_episode(env, pair["state"], pair["goal_state"])
    success, lat, costs = False, [], []
    dist = None
    for t in range(budget):
        if t % block == 0:
            px = pair["pixels"] if (t == 0 and first_obs == "dataset") else env.render()
            sync()
            t0 = time.perf_counter()
            if getattr(planner, "needs_pixels", False):   # official path encodes inside get_cost
                out = planner(px, pair["goal"])
            else:
                zz = encode_pixels(model, np.stack([px, pair["goal"]]))
                out = planner(zz[0], zz[1])
            acts = out["plan"].reshape(block, 2).double().cpu().numpy() * a_std + a_mean
            sync()
            lat.append(time.perf_counter() - t0)
            costs.append(out["cost"])
        _, r, term, _, _ = env.step(acts[t % block])
        dist = -r
        success = success or bool(term)
    return {"success": success, "final_dist": float(dist), "latency": lat, "cost": costs}
