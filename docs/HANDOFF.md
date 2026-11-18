# 会话交接文档（每轮更新）

最后更新：2026-11-17，第 14 轮。

## 项目
仓库 `GuihaiChen1998/flash-lewm`，目标是加速 Fast-LeWM 的潜空间规划。issue #6 是初始方案。

## 现状
- issue #6 方案已实现并在 Push-T 上验证，结果见 `docs/issue6_review_and_results_v1.md`。
- 主要产出：`FlashCEM`（`flash_lewm/search.py`），467 ms → 65 ms，成功率 89.0% → 88.9%（n=1010）。
- 共享前缀树搜索、异步调度：已实现，结果为负。

## 机器与路径（Linux#1，用户 cgh）
- 代码 `/data1/cgh/code/flash-lewm`（`third_party/Fast-LeWorldModel` 为官方代码，提交 2fc3bca；`.pydeps` 为项目内依赖；`ckpt/`；`results/`；`logs/`）
- 数据 `/data1/cgh/dataset/lewm-original/pusht_expert_train.h5`（已解压）；cube / reacher / tworoom 仍是压缩包
- Python `/home/cgh/anaconda3/envs/sglang/bin/python`；Jupyter MCP 工具名 `jupyter-mcp_execute_python`
- 代理端口会变：环境变量里的 `127.0.0.1:23333` 一直没监听；11-14 用 `17897`；**11-16 起 `17897` 也不通，改用 `127.0.0.1:11111`**。不通时先 `ss -ltnp | grep 127.0.0.1` 找在听的端口
- 系统 git 用 `/usr/bin/git`（conda 里的 git 不支持 https）
- Jupyter 网关单次调用超过约 60 秒容易超时：长任务用 `nohup` 后台跑，再轮询日志
- GPU 计时要独占一张卡，否则数字会被污染

## 注意
- checkpoint 是 pickle 的整个模型对象，由 transformers 5.x 保存；`core.load_model` 会用已装的 4.54 重建 ViT 并严格加载权重。
- SDPA 在 bf16 下、批 × 头数很大时报 `invalid configuration argument`。
- 原始结果 JSON 只在 Linux#1 的 `results/` 下，仓库里是汇总表。
- Linux#2 上有一个误建的 `/data1/ekyang/cgh/CODE/flash-lewm`（只有官方代码克隆），可删。

## 方向调整（第 3 轮）
用户要算法层面的创新，不要纯工程优化。思路：对比 LeWM / Fast-LeWM 伪代码找可消除的串行结构。
当前判断：Fast-LeWM 剩下的串行维度是 CEM 轮数 T，候选想法 A–D 见 `docs/discussion_log_v1.md` 第 3 轮，等用户选定。
FlashCEM 降级为「优化后的同步基线」。

## issue #7（第 4 轮）
承诺前缀 / 异步重规划已实现并验证，结果见 `docs/issue7_review_and_results_v1.md`。
要点：官方终点代价在频繁重规划下失效；改用「前缀终点最小」代价（`plan_cost(..., cost="min")`）后前缀钳制成立，
但不优于「整段执行 + min 代价」。min 代价本身让到达更快、抗扰动更好，是目前最值得跟进的发现。
第 3 轮的候选想法 A–D 尚未动手。

运行注意：
- 远端更新代码：`cd /data1/cgh/code/flash-lewm && /usr/bin/git fetch origin main && /usr/bin/git reset --hard origin/main`（需代理，端口见上）。
- 用户要求 sleep / timeout 不超过 20 秒；长实验用 nohup 后台跑，用定时回访查看。
- 在远端用 `pkill -f` 会连带杀掉发起它的 shell，改用 ps 取 pid 再 kill。
- GPU 与他人共用时，并行开太多进程会互相拖慢；计时类实验要注明。

## min 代价与新场景（第 5–6 轮）
结果见 `docs/min_cost_and_scenarios_v1.md`。min 代价在 Push-T（官方协议 1010 回合、四个新场景）和 Cube（200 回合）上都不降成功率、到达更快；
承诺前缀在 switch / far 场景显著优于整段执行。
- Cube：数据 `/data1/cgh/dataset/lewm-original/cube/cube_single_expert.h5`（102 GB），权重 `ckpt/Fast-lewm_ogb_object.ckpt`，
  脚本 `scripts/cube_eval.py`（需 `MUJOCO_GL=egl`，退出时的 EGL 报错可忽略）。数据集动作列有 NaN 行，拟合归一化前要去掉。
- ViZDoom 1.3.2 已装在 `.pydeps`，无头可跑；尚无世界模型，等用户确认后再训练。
- 场景预设在 `scripts/e9_realtime.py` 的 `SCN`，cfg 格式 `method:m:d:iters:cost:scenario`。

## 多环境（第 8–9 轮）
结果见 `docs/multi_env_results_v1.md`。目前最强的两条结论：min 代价在 Push-T / Cube / Two-Room / ViZDoom 上都成立；
前缀钳制在「停顿有代价或勤重规划有用」的场景胜出（ViZDoom 全部场景、Two-Room 与 Push-T 的远目标和目标切换）。

各环境的运行前缀（都配合 `scripts/e9_realtime.py <n_extra> <tag> <cfg...>`，cfg = `method:m:d:iters:cost:scenario`）：
- Two-Room：`FLASH_ENV=tworoom PUSHT_H5=/data1/cgh/dataset/lewm-original/tworoom/tworoom.h5 FLASH_CKPT=ckpt/Fast-lewm_tworoom_object.ckpt FLASH_EP_KEY=ep_idx FLASH_STATE_KEY=proprio`
- ViZDoom：`FLASH_ENV=doom FLASH_CKPT=ckpt/doom_mwh.pt DOOM_PAIRS=/data1/cgh/dataset/doom/eval_pairs.npz`（每个进程会开 50 个 Doom 实例）
- AntMaze：`FLASH_ENV=ant FLASH_ADIM=8 FLASH_ANT_NOISE=0|0.2 FLASH_CKPT=ckpt/ant_medium.pt DOOM_PAIRS=/data1/cgh/dataset/ant/eval_pairs.npz MUJOCO_GL=egl`，
  且 `PYTHONPATH` 必须以 `.mj/3.3.0` 开头（MuJoCo ≥3.5 与 OGBench 数据集动力学不一致）。
- 结果文件 `results/e9_realtime_<tag>*.json` 的键在不同环境间重名，按标签前缀（TR / DOOM / 无前缀为 Push-T）分开读。

其他：
- 训练：`python -m flash_lewm.train_min <data.npz> <out.pt> steps=30000 [img=112]`，数据整体放进 GPU 显存；约 9 步/秒。
- `.mj/<版本>/` 是各版本 MuJoCo 的独立安装目录，只用于 AntMaze 和版本对比。
- Reacher：`/data1/cgh/dataset/lewm-original/reacher/reacher.h5`（99 GB）、`ckpt/Fast-lewm_dmc_object.ckpt` 已就位，评估脚本未写。
- Jupyter 连接（Claude_MCP_Portal）会过期，需要用户重新登录。

## Cube / ViZDoom 加固（第 11 轮，2026-11-17）
结果见 `docs/cube_doom_solid_v1.md`，汇总数字在 `results/summary_cube_doom_solid.json`。
- 承诺前缀在 Cube（300 回合）和 ViZDoom（两个种子各 500 回合）、延迟 5 / 10 步下都显著优于同步规划和朴素异步。
- min 代价的结论收窄：Cube 普遍有效；ViZDoom 只在频繁重规划时有效，整段执行 + 静态目标时与官方代价无显著差别。
- Cube 的 switch 场景：原评测集有缺陷（105 / 300 回合换目标即成功）；已生成 `eval_pairs_v2.npz` 并重跑（标签 `C2sw*`）：
  延迟 5 步 clamp − sync min +3.7 [+0.3, +7.0]，延迟 10 步无差别。静态场景仍用原评测集。
- ViZDoom 零延迟 m=2 参照已补跑：静态 57.5%，是最好的参照；clamp（d=5）比它低 5.1 [−7.7, −2.4]，所以是「接近」而非「追平」零延迟。

运行前缀（配合 `scripts/e9_realtime.py <n_extra> <tag> <cfg...>`，实际回合数 = 50 + n_extra）：
- Cube：`FLASH_ENV=cube FLASH_ADIM=5 FLASH_CHUNK=25 MUJOCO_GL=egl FLASH_CKPT=ckpt/Fast-lewm_ogb_object.ckpt PUSHT_H5=/data1/cgh/dataset/lewm-original/cube/cube_single_expert.h5 DOOM_PAIRS=/data1/cgh/dataset/cube_eval/eval_pairs.npz`
  （`FLASH_CHUNK` 是每批并行的环境数，Cube 用 25；适配器在 `flash_lewm/cube_env.py`；日志里的 EGL / xcb 报错可忽略）
- ViZDoom 500 回合：`DOOM_PAIRS=/data1/cgh/dataset/doom/eval_pairs500.npz`；第二个模型种子 `ckpt/doom_mwh_s1.pt`
  （`python -m flash_lewm.train_min /data1/cgh/dataset/doom/mwh_all.npz ckpt/doom_mwh_s1.pt steps=30000 img=112 seed=1`）
- 结果标签：`CUBE*`（Cube）、`DA*`（Doom 种子 A）、`DB*`（Doom 种子 B）；每跑完一个配置就写一次 JSON。
- 耗时：24 个进程并行时每个配置 8–25 分钟；异步配置（clamp / naive）和 m=1 的配置最慢。要快就少开进程。

## 资源存放位置（2026-11-17 起全部入库）
- **git 仓库**：代码、文档、`results/`（全部逐回合结果 JSON）、`logs/`、`requirements-pydeps.txt`（`.pydeps` 的包清单）。
- **GitHub Release `assets-v1`**（https://github.com/GuihaiChen1998/flash-lewm/releases/tag/assets-v1）：8 个权重
  （官方 `Fast-lewm_{pusht,ogb,tworoom,dmc}_object.ckpt`；自训练 `doom_mwh.pt`、`doom_mwh_10k.pt`、`doom_mwh_s1.pt`、`ant_medium.pt`）
  和 5 个评测回合文件（`cube_eval_pairs.npz`、`cube_eval_pairs_v2.npz`、`doom_eval_pairs.npz`、`doom_eval_pairs500.npz`、`ant_eval_pairs.npz`）。
- **Hugging Face（账号 UUUooUUU，私有）**：模型仓库 `UUUooUUU/flash-lewm`（`ckpt/`、`results/`、`logs/`）；
  数据集仓库 `UUUooUUU/flash-lewm-data`（`eval_pairs/` 同上 5 个文件；`doom/mwh_all.npz` 是 11 GB 的 Doom 训练数据，
  上传状态看 Linux#1 的 `logs/upload_hf_big.log`，末尾出现 `ALL DONE` 才算完成）。
- 上传脚本：`scripts/upload_release.sh`（环境变量 `GH`）、`scripts/upload_hf.py small|big|all`（环境变量 `HF`；
  需 `HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_XET=1`，代理下新存储后端连不上）。令牌不入库，用完即删。
- 推送权限：chancanmakeit 账号的令牌对本仓库只读；能推送的是 GuihaiChen1998 账号的令牌。
- 没有上传的：官方数据集原件（Push-T / Cube / Two-Room / Reacher / OGBench AntMaze，可从官方下载）、
  `.pydeps/` 与 `.mj/`（按清单重装）、`third_party/Fast-LeWorldModel`（官方代码，提交 2fc3bca）。
- Linux#1 上文件名与 Release 资产名的对应：`/data1/cgh/dataset/cube_eval/eval_pairs{,_v2}.npz`、
  `/data1/cgh/dataset/doom/eval_pairs{,500}.npz`、`/data1/cgh/dataset/ant/eval_pairs.npz`。

## LIBERO-10（第 13 轮）
结果见 `docs/libero_results_v1.md`。单任务（STUDY_SCENE1）：承诺前缀对同步规划 +2 到 +6（全部显著），对朴素异步无差别；基础成功率只有 5–20%。
10 任务（多任务模型 `ckpt/libero10.pt`）：零延迟静态成功率 8.7%，各方法差别 ≤2 个百分点，几乎无信息量；任务 8 的判据有漏洞（停顿时夹爪猜错 → 杯子掉进微波炉算成功），已从合并数字剔除。
- 评测并行注意：每个 LIBERO 场景约占 0.8 GB 内存；20 进程 × 25 场景曾把 250 GB 吃满，现在用 `xargs -P 10` + `FLASH_CHUNK=13`（约 110 GB）。
- 数据 `/data1/cgh/dataset/libero/*.hdf5`（10 个任务，13.7 GB；下载脚本 `scripts/libero_download.sh <代理端口> <0|1>` 支持续传，
  清单 `files.txt`）；转换后的训练数据 `study1.npz`；评测回合 `pairs_study1.npz`。
- 适配器 `flash_lewm/libero_env.py`，一个进程只跑一个任务（`FLASH_LIBERO_BDDL`）。运行前缀见结果文档「复现」一节。
- 依赖目录 `.pydeps_libero/`、配置 `.libero_cfg/config.yaml`（不入库，按复现一节重建）。
- **注意**：不要在 Linux#1 上对有进程正在写的已跟踪日志做 `git stash -u` / `reset`，会把文件换掉，进程继续写已删除的 inode
  （可从 `/proc/<pid>/fd/1` 读回）。这次因此丢了 LIBERO 训练第 12000 步之后的日志。

## 官方协议对比与官方训练（第 14 轮）
- 官方 `eval.py` 已打通（`scripts/official_eval.py`），四个环境复现论文数字在 ±2 以内；min 代价在 Reacher +11.4，其余持平。结果见 `docs/official_protocol_v1.md`。
- 官方 `train.py` 在 ViZDoom、LIBERO 数据上训练中：数据用 `scripts/npz_to_swm_h5.py` 转到 `/data1/cgh/dataset/official_h5/{doom_mwh,libero10}.h5`，
  输出 `ckpt/official_train/{doom,libero10}/`；依赖在 `.pydeps_train`（stable-pretraining **0.1.6**、lightning 2.6.1、torchmetrics 1.9.0；
  0.1.8 会拒绝从头训练），PYTHONPATH 末尾要加 `.pydeps_libero`（matplotlib）。命令：
  `STABLEWM_HOME=/data1/cgh/dataset/official_h5 WANDB_MODE=disabled python train.py data=tworoom data.dataset.name=<doom_mwh|libero10> 'data.dataset.keys_to_load=[pixels,action]' 'data.dataset.keys_to_cache=[action]' wandb.enabled=False hydra.run.dir=<输出目录>`（在 `third_party/Fast-LeWorldModel` 下运行）。
- 官方训练的 LIBERO 模型（`ckpt/official_train/libero10/Fast-lewm_epoch_10_object.ckpt`）评测完：零延迟只有 6.7%，不比精简训练好，结论见 `docs/libero_results_v1.md`。读它时 `PUSHT_H5` 要指向 `/data1/cgh/dataset/official_h5/libero10.h5`（动作归一化取全部 10 个任务）。
- 官方训练的 ViZDoom 模型（`ckpt/official_train/doom/Fast-lewm_epoch_10_object.ckpt`，读它时 `PUSHT_H5=/data1/cgh/dataset/official_h5/doom_mwh.h5`）与精简训练成功率几乎一样，真实延迟对比结论不变（`DOFF*`）。第 14 轮计划 6 项已全部完成。
- 项目展示页 `docs/site/`（index.html + assets/*.gif）。GIF 由 `scripts/record_rollouts.py`（录帧，原始数据在 `/data1/cgh/dataset/rollouts/*.npz`）+ `scripts/make_gifs.py`（左右对比）+ `scripts/make_grid_gif.py`（前 8 回合不挑选）生成；大 GIF 是挑过的成功回合，页面上已注明。
- 并行注意：评测进程要设 `OMP_NUM_THREADS=4`，否则多进程时 CPU 严重超订。
- 实测耗时 + 实时对比完成（`docs/real_latency_v1.md`）：官方求解器约 600 ms（6 步）、FlashCEM 65 ms（1 步）；完整方法对官方部署全部显著，换目标 +8 到 +62。结果标签 `RL<env><场景>`。

## 下一步候选（第 11 轮后）
1. 给 Cube 加 far / kick 场景；查清官方代价配置在换目标那一步就成功的 21–29 个回合是怎么回事。
2. 用真实规划耗时代替按步数注入的延迟（FlashCEM 65 ms 对官方 467 ms），把「求解器变快」和「承诺前缀」两条线合起来比较。
3. LIBERO-10：先提基础成功率（目标距离 25 → 10 步、修夹爪猜测）再比较方法；Reacher 评估脚本未写。
4. 第 3 轮的算法候选 A–D、iCEM 基线仍未动手。

## 更早的候选（第 1 轮遗留）
1. FlashCEM 接入官方 `eval.py`，在 Cube 上复核。
2. 测 `cem_multi` 多环境合批吞吐。
3. 编译图像编码器。
