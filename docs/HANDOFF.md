# 会话交接文档（每轮更新）

最后更新：2026-11-14，第 6 轮结束。

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
- 机器环境变量里的代理 `127.0.0.1:23333` 当时没有监听，下载用的是 `127.0.0.1:17897`
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
- 远端更新代码：`cd /data1/cgh/code/flash-lewm && /usr/bin/git fetch origin main && /usr/bin/git reset --hard origin/main`（需代理 127.0.0.1:17897）。
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

## 下一步候选（第 1 轮遗留）
1. FlashCEM 接入官方 `eval.py`，在 Cube 上复核。
2. 测 `cem_multi` 多环境合批吞吐。
3. 编译图像编码器。
