"""OGBench visual AntMaze (medium) as goal-image navigation behind the simulator's env interface.
Episode spec = dataset (qpos, qvel); goal = xy of the dataset frame k steps later; success = the
ant's xy within XY_THR of the goal xy.

Simulator version matters: with mujoco 3.1.6 - 3.3.0 replaying the dataset actions from a dataset
state reproduces the recorded trajectory to float32 precision (5e-7 after 25 steps); with 3.6.0
the one-step error is ~1e-2 and the xy error after 25 steps is ~0.44 (median). Run this env with
PYTHONPATH pointing at .mj/3.3.0 first.

Two variants:
  FLASH_ANT_NOISE=0    deterministic execution (aligned with the dataset records)
  FLASH_ANT_NOISE=0.2  Gaussian noise of that std is added to every executed action and the result
                       clipped to [-1, 1], exactly as the navigate dataset was collected."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

NAME = "visual-antmaze-medium-navigate-v0"
XY_THR = 1.0          # maze cells are 4 units; the median 25-step displacement in the data is 2.7
NOISE = float(os.environ.get("FLASH_ANT_NOISE", "0"))
_count = [0]


class AntNav:
    def __init__(self):
        import ogbench
        self.env = ogbench.make_env_and_datasets(NAME, env_only=True)
        self.env.reset(seed=0)
        self.u = self.env.unwrapped
        self.goal = None
        _count[0] += 1
        self.rng = np.random.default_rng(1000 + _count[0])

    def reset(self, seed=0):
        return None

    def set_spec(self, spec):
        spec = np.asarray(spec, dtype=np.float64)
        self.u.set_state(spec[:15], spec[15:29])

    def _set_goal_state(self, xy):
        self.goal = np.asarray(xy, dtype=np.float64)[:2]

    def render(self):
        return self.u.get_ob()

    def step(self, a):
        a = np.asarray(a, dtype=np.float64)
        if NOISE > 0:
            a = a + self.rng.normal(0, NOISE, a.shape)
        self.env.step(np.clip(a, -1.0, 1.0).astype(np.float32))
        d = float(np.linalg.norm(self.u.data.qpos[:2] - self.goal))
        return None, -d, bool(d < XY_THR), False, {}
