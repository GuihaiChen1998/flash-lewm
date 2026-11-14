"""OGBench visual AntMaze (medium) as goal-image navigation behind the simulator's env interface.
Episode spec = dataset (qpos, qvel); goal = xy of the dataset frame k steps later; success = the
ant's xy within XY_THR of the goal xy."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

NAME = "visual-antmaze-medium-navigate-v0"
XY_THR = 1.0          # maze cells are 4 units; the median 25-step displacement in the data is 2.7


class AntNav:
    def __init__(self):
        import ogbench
        self.env = ogbench.make_env_and_datasets(NAME, env_only=True)
        self.env.reset(seed=0)
        self.u = self.env.unwrapped
        self.goal = None

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
        self.env.step(np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0))
        d = float(np.linalg.norm(self.u.data.qpos[:2] - self.goal))
        return None, -d, bool(d < XY_THR), False, {}
