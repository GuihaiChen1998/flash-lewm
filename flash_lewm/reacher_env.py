"""DMC Reacher (qpos_match task, as in the official Fast-LeWM eval) for the real-time simulator.
spec = qpos[2] + qvel[2] from the dataset; goal = target qpos[2]; success = every joint within the task's
qpos threshold (the official success test)."""
import numpy as np


class ReacherNav:
    def __init__(self):
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401  (registers swm/ envs)
        self.env = gym.make("swm/ReacherDMControl-v0", task="qpos_match").unwrapped
        self.env.reset(seed=0)

    def set_spec(self, spec):
        spec = np.asarray(spec, dtype=np.float64)
        self.env.reset(seed=0)
        self.env.set_state(spec[:2], spec[2:4])

    def _set_goal_state(self, g):
        self.env.set_target_qpos(np.asarray(g, dtype=np.float64)[:2])

    def render(self):
        return self.env.render()

    def step(self, a):
        out = self.env.step(np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0))
        return None, 0.0, bool(out[2]), False, {}
