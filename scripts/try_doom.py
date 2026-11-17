
import time, numpy as np, traceback
try:
    import vizdoom as vzd, os
    print('vizdoom', vzd.__version__, 'scenarios:', sorted(f for f in os.listdir(vzd.scenarios_path) if f.endswith('.cfg'))[:30])
    g = vzd.DoomGame()
    g.load_config(os.path.join(vzd.scenarios_path, 'my_way_home.cfg'))
    g.set_window_visible(False); g.set_screen_resolution(vzd.ScreenResolution.RES_320X240); g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_available_buttons([vzd.Button.MOVE_FORWARD_BACKWARD_DELTA, vzd.Button.TURN_LEFT_RIGHT_DELTA, vzd.Button.MOVE_LEFT_RIGHT_DELTA])
    g.set_available_game_variables([vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE])
    g.set_mode(vzd.Mode.PLAYER); g.set_episode_timeout(100000); g.init()
    g.new_episode(); t=time.time(); n=0; rng=np.random.default_rng(0); pos=[]
    while n < 2000 and not g.is_episode_finished():
        s = g.get_state(); pos.append(list(s.game_variables))
        g.make_action([float(rng.uniform(-5,15)), float(rng.uniform(-8,8)), 0.0], 4); n+=1
    pos=np.array(pos); print('frame', s.screen_buffer.shape, s.screen_buffer.dtype, 'steps', n, 'steps/s', n/(time.time()-t), 'pos range', pos.min(0), pos.max(0))
    g.close()
except Exception: traceback.print_exc()
