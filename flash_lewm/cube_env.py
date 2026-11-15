"""OGBench Cube (single) behind the real-time simulator's env interface.
Episode spec = dataset (qpos[21], qvel[20]); goal state = goal cube position[3] + quaternion[4];
success = env termination (cube within 4 cm of the goal position). Uses MuJoCo 3.6.0."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np


class CubeNav:
    def __init__(self):
        from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
        self.env = CubeEnv(env_type="single", ob_type="states", multiview=False, width=224, height=224,
                           visualize_info=False, terminate_at_goal=True)
        self.env.reset(seed=0)
        self.u = self.env.unwrapped

    def reset(self, seed=0):
        return None

    def set_spec(self, spec):
        spec = np.asarray(spec, dtype=np.float64)
        self.env.reset(seed=0)
        self.u.set_state(spec[:21], spec[21:41])

    def _set_goal_state(self, g):
        g = np.asarray(g, dtype=np.float64)
        self.u.set_target_pos(0, g[:3], g[3:7])

    def render(self):
        return self.env.render()

    def step(self, a):
        self.u._reset_next_step = False
        _, _, term, _, _ = self.env.step(np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0))
        return None, 0.0, bool(term), False, {}
