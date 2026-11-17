"""LIBERO adapter for the real-time simulator (one task per process, chosen by FLASH_LIBERO_BDDL).

spec  = 45-d (for most scenes) flattened MuJoCo state stored in the LIBERO demo files: [time, qpos, qvel].
goal  = a full state from the same demo some steps later; success = every movable (free-joint) object is
        within XY_THR metres of its position in the goal state. The robot pose is not part of the criterion.
Frames are the 128x128 agentview camera, matching obs/agentview_rgb in the demo files (no flip needed).

An all-zero action from the simulator's "hold" (planning pause) keeps the last gripper command instead of
sending 0, which in robosuite would half-open the gripper and drop a held object.
"""
import os
import numpy as np

XY_THR = float(os.environ.get("FLASH_LIBERO_THR", "0.04"))


def free_joint_slices(model):
    """qpos address of each free joint (movable object) in a mujoco-py style model wrapper."""
    adr = []
    for j in range(model.njnt):
        if model.jnt_type[j] == 0:       # mjJNT_FREE
            adr.append(int(model.jnt_qposadr[j]))
    return adr


class LiberoNav:
    def __init__(self):
        from libero.libero.envs import OffScreenRenderEnv

        self.env = OffScreenRenderEnv(bddl_file_name=os.environ["FLASH_LIBERO_BDDL"], camera_heights=128, camera_widths=128)
        self.env.seed(0)
        self.env.reset()
        m = self.env.sim.model
        self.nq = m.nq
        self.obj = free_joint_slices(m)
        self.grip = -1.0
        self.goal = None

    def obj_pos(self, state):
        q = np.asarray(state)[1:1 + self.nq]
        return np.stack([q[a:a + 3] for a in self.obj])

    def set_spec(self, spec):
        self.env.reset()
        self.obs = self.env.set_init_state(np.asarray(spec, dtype=np.float64))
        g = self.obs.get("robot0_gripper_qpos")
        self.grip = -1.0 if g is None or g[0] - g[1] > 0.06 else 1.0    # open fingers -> last command was "open"

    def _set_goal_state(self, g):
        self.goal = self.obj_pos(g)

    def render(self):
        return self.obs["agentview_image"]

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0).copy()
        if not a.any():
            a[6] = self.grip
        else:
            self.grip = a[6]
        self.obs, _, _, _ = self.env.step(a)
        cur = self.obj_pos(self.env.get_sim_state())
        term = bool(np.all(np.linalg.norm(cur - self.goal, axis=1) < XY_THR))
        return None, 0.0, term, False, {}
