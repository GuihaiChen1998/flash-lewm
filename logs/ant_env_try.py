
import os; os.environ["MUJOCO_GL"]="egl"
import numpy as np, ogbench, traceback
try:
    env = ogbench.make_env_and_datasets('visual-antmaze-medium-navigate-v0', env_only=True)
    ob, info = env.reset(seed=0)
    print('obs', type(ob), getattr(ob,'shape',None), getattr(ob,'dtype',None), 'info keys', list(info.keys())[:12])
    u = env.unwrapped; print('unwrapped', type(u).__mro__[:4]); print([m for m in dir(u) if any(s in m for s in ['set_state','get_xy','set_xy','set_goal','get_ob','render','_ob_type','frame_skip','cur_goal'])])
    print('action space', env.action_space, 'nq', u.model.nq, 'nv', u.model.nv)
    d = np.load('/data1/cgh/dataset/ogbench/visual-antmaze-medium-navigate-v0.npz', mmap_mode=None)
    obs, qpos, qvel, act = d['observations'], d['qpos'], d['qvel'], d['actions']
    for r in [5000, 123456, 700123]:
        u.set_state(qpos[r].astype(np.float64), qvel[r].astype(np.float64))
        img = u.get_ob() if hasattr(u,'get_ob') else env.render()
        img2 = env.render()
        print(r, 'get_ob', getattr(img,'shape',None), 'diff vs dataset', float(np.abs(np.asarray(img).astype(float)-obs[r].astype(float)).mean()) if getattr(img,'shape',None)==obs[r].shape else None, '| render', img2.shape, float(np.abs(img2.astype(float)-obs[r].astype(float)).mean()) if img2.shape==obs[r].shape else None)
        # replay 25 dataset actions and compare xy
        for a in act[r:r+25]: o,_,_,_,_ = env.step(a)
        print('   xy after 25 expert steps', u.data.qpos[:2], 'dataset', qpos[r+25][:2], 'start', qpos[r][:2], 'obs diff', float(np.abs(o.astype(float)-obs[r+25].astype(float)).mean()))
    dd = np.linalg.norm(qpos[25:200000:997,:2]-qpos[0:200000-25:997,:2],axis=1); d50 = np.linalg.norm(qpos[50:200000:997,:2]-qpos[0:200000-50:997,:2],axis=1)
    print('xy displacement median: 25 steps', np.median(dd), '50 steps', np.median(d50))
except Exception: traceback.print_exc()
