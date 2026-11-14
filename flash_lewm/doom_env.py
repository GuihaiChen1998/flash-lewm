"""ViZDoom goal-image navigation (my_way_home map) behind the small env interface the real-time
simulator uses: render(), step(raw_action) -> (..., terminated, ...), _set_goal_state(pose).

ViZDoom cannot be reset to an arbitrary pose, so an episode is specified by (seed, k0, actions):
re-seed, start a new episode and replay the k0 exploration actions to reach the start pose."""
import os
import numpy as np

S, SKIP = 112, 4
A_LO, A_HI = np.array([-5.0, -10.0]), np.array([20.0, 10.0])
KMAX = 80
POS_THR, ANG_THR = 48.0, 30.0     # success: within POS_THR map units and ANG_THR degrees of the goal pose


def make_game(seed=0):
    import vizdoom as vzd
    g = vzd.DoomGame()
    g.load_config(os.path.join(vzd.scenarios_path, "my_way_home.cfg"))
    g.set_window_visible(False)
    g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_available_buttons([vzd.Button.MOVE_FORWARD_BACKWARD_DELTA, vzd.Button.TURN_LEFT_RIGHT_DELTA])
    g.set_available_game_variables([vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE])
    g.set_mode(vzd.Mode.PLAYER)
    g.set_episode_timeout(1_000_000)
    g.set_seed(seed)
    g.init()
    return g


def observe(g):
    import cv2
    s = g.get_state()
    return cv2.resize(s.screen_buffer, (S, S), interpolation=cv2.INTER_AREA), np.array(s.game_variables, dtype=np.float64)


class Explorer:
    """The data-collection policy: Ornstein-Uhlenbeck forward / turn, turns away when stuck."""

    def __init__(self, rng):
        self.rng, self.a, self.stuck, self.last = rng, np.array([8.0, 0.0]), 0, None

    def act(self, pose):
        self.a = self.a + 0.25 * (np.array([8.0, 0.0]) - self.a) + self.rng.normal(0, [4.0, 3.0])
        self.stuck = self.stuck + 1 if (self.last is not None and np.linalg.norm(pose[:2] - self.last[:2]) < 2.0) else 0
        if self.stuck >= 3:
            self.a[1] = self.rng.choice([-1, 1]) * self.rng.uniform(6, 10)
        self.a = np.clip(self.a, A_LO, A_HI)
        self.last = pose
        return self.a.copy()


def pose_error(p, q):
    d = float(np.linalg.norm(p[:2] - q[:2]))
    a = abs(float(p[2] - q[2])) % 360.0
    return d, min(a, 360.0 - a)


class DoomNav:
    def __init__(self):
        self.g = make_game(0)
        self.goal = None

    def reset(self, seed=0):
        return None

    def set_spec(self, spec):
        seed, k0 = int(spec[0]), int(spec[1])
        acts = np.asarray(spec[2:2 + 2 * k0], dtype=np.float64).reshape(k0, 2)
        self.g.set_seed(seed)
        self.g.new_episode()
        for a in acts:
            self.g.make_action([float(a[0]), float(a[1])], SKIP)

    def _set_goal_state(self, pose):
        self.goal = np.asarray(pose, dtype=np.float64)

    def pose(self):
        return observe(self.g)[1]

    def render(self):
        if self.g.is_episode_finished():
            return np.zeros((S, S, 3), np.uint8)
        return observe(self.g)[0]

    def step(self, a):
        if self.g.is_episode_finished():
            return None, 0.0, False, False, {}
        a = np.clip(np.asarray(a, dtype=np.float64), A_LO, A_HI)
        self.g.make_action([float(a[0]), float(a[1])], SKIP)
        if self.g.is_episode_finished():
            return None, 0.0, False, False, {}
        d, ang = pose_error(self.pose(), self.goal)
        return None, -d, bool(d < POS_THR and ang < ANG_THR), False, {}
