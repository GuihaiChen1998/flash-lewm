"""Collect a ViZDoom navigation dataset (my_way_home map, continuous forward / turn deltas).
Usage: doom_collect.py <out.npz> <n_episodes> <ep_len> [seed]   Frames are stored at 112x112."""
import os, sys, time
import numpy as np, cv2, vizdoom as vzd

S, SKIP = 112, 4
A_LO, A_HI = np.array([-5.0, -10.0]), np.array([20.0, 10.0])


def make_game(seed=0):
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


def frame(g):
    s = g.get_state()
    return cv2.resize(s.screen_buffer, (S, S), interpolation=cv2.INTER_AREA), np.array(s.game_variables, dtype=np.float32)


if __name__ == "__main__":
    out, n_ep, ep_len = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    rng = np.random.default_rng(seed)
    g = make_game(seed)
    P = np.zeros((n_ep * ep_len, S, S, 3), np.uint8); AC = np.zeros((n_ep * ep_len, 2), np.float32)
    ST = np.zeros((n_ep * ep_len, 3), np.float32); EP = np.zeros(n_ep * ep_len, np.int64); SI = np.zeros(n_ep * ep_len, np.int64)
    t0 = time.time(); i = 0
    for e in range(n_ep):
        g.new_episode()
        a = np.array([8.0, 0.0]); stuck = 0; last = None
        for t in range(ep_len):
            if g.is_episode_finished():      # touched the goal item: restart from a new spawn, keep the episode id distinct
                break
            px, st = frame(g)
            # Ornstein-Uhlenbeck exploration; turn harder when not moving (wall)
            a = a + 0.25 * (np.array([8.0, 0.0]) - a) + rng.normal(0, [4.0, 3.0])
            if last is not None and np.linalg.norm(st[:2] - last[:2]) < 2.0:
                stuck += 1
            else:
                stuck = 0
            if stuck >= 3:
                a[1] = rng.choice([-1, 1]) * rng.uniform(6, 10)
            a = np.clip(a, A_LO, A_HI)
            P[i], AC[i], ST[i], EP[i], SI[i] = px, a, st, e, t
            last = st
            g.make_action([float(a[0]), float(a[1])], SKIP)
            i += 1
        if e % 100 == 0:
            print(f"episode {e}/{n_ep} frames {i} ({i / (time.time() - t0):.0f} fps)", flush=True)
    np.savez(out, pixels=P[:i], action=AC[:i], state=ST[:i], ep_idx=EP[:i], step_idx=SI[:i])
    print("saved", out, i, "frames; position range", ST[:i, :2].min(0), ST[:i, :2].max(0))
