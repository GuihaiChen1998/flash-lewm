# 原版 LeWM 在官方 eval.py 下：实现等价性与 min 代价（v1，2026-11-18）

问题：原版 LeWM 实验里用了我们的快速实现（观测只编码一次、编译），审稿人可能质疑我们改了 LeWM 的官方配置。
这里用 LeWM 官方 `eval.py`（`third_party/le-wm`）和官方配置直接回答。

## 结论

1. **我们的实现与官方实现等价**：观测和目标只编码一次、代价仍用官方终点代价（`hooklast`），与官方原始代码（`last`，每轮把 300 个候选的图像重新编码）
   在 4 个环境 × 10 个种子 × 50 回合上成功率完全相同，逐回合一致率 Two-Room、Cube、Reacher 100%，Push-T 99.2%（500 回合里 4 个翻转，方向相反，均值不变）。
   翻转来自约 1e-4 量级的浮点差异偶尔改变 CEM 选中的候选。
2. **min 代价在 LeWM 官方协议下 4 个环境全部显著提升**：+3.4（Push-T）到 +11.4（Reacher）个百分点。
   这比在 Fast-LeWM 上更一致（Fast-LeWM 只有 Reacher 显著，其余持平）。
3. 所以「编码一次 + 编译」只是实现上的加速，不改变规划结果；规划层面的改动只有我们的两个部件（min 代价、承诺前缀），可以单独消融。

## 设定

- 代码：LeWM 官方 `eval.py` 与 `config/eval/{pusht,tworoom,cube,reacher}.yaml` 不改；权重 `quentinll/lewm-*`（HF 发布版，严格加载）。
- CEM 300 候选 × 30 轮、取前 30、时域 5 块、每块 5 步、每次执行 5 块；50 回合、目标偏移 25 步、预算 50 步；种子 42–51。
- 包装 `scripts/official_eval_lewm.py` 只做三件事：用 stable_pretraining 的归一化常数代替整个库；`swm.wm.utils.load_pretrained` 换成我们的加载器
  （装的是 Fast-LeWM 固定的 stable_worldmodel 版本，没有这个函数）；`World.evaluate` 映射到同一协议的 `evaluate_from_dataset`（新版库改了参数名），不存视频。
- 数据集按官方名字从软链接目录 `/data1/cgh/dataset/lewm_cache` 读取（`pusht_expert_train`、`tworoom`、`ogbench/cube_single_expert`、`dmc/reacher_random`）。
- 三种模式（环境变量 `FLASH_COST`）：`last` 官方 `JEPA.get_cost`；`hooklast` 我们的打分（`flash_lewm/official_hooks.py` + `flash_lewm/lewm.py`），官方终点代价；
  `min` 我们的打分 + min 代价。求解器都是官方 `CEMSolver`，只换打分函数。

## 结果（成功率 %，10 个种子 × 50 回合的均值）

| 环境 | 官方原始 | 我们的实现（终点代价） | 逐回合一致 | 我们的实现 + min 代价 | min − 官方（500 回合配对） | 种子 42 三种模式 |
|---|---|---|---|---|---|---|
| Push-T | 87.4 | 87.4 | 99.2% | 90.8 | +3.4 [+1.0, +5.8] | 96 / 96 / 100 |
| Two-Room | 88.2 | 88.2 | 100% | 93.0 | +4.8 [+2.4, +7.4] | 88 / 88 / 96 |
| OGBench Cube | 68.4 | 68.4 | 100% | 72.8 | +4.4 [+1.2, +7.6] | 70 / 70 / 76 |
| Reacher | 79.8 | 79.8 | 100% | 91.2 | +11.4 [+7.2, +15.6] | 76 / 76 / 92 |

各种子的官方成功率（42 → 51）：Push-T 96 90 84 88 86 88 88 90 82 82；Two-Room 88 84 88 88 88 92 86 94 94 80；
Cube 70 78 68 60 70 68 70 66 66 68；Reacher 76 82 86 88 80 84 74 76 70 82。种子间标准差 4–5 个百分点，单种子数字不宜直接比较。

对照（Fast-LeWM，同一协议，`docs/official_protocol_v1.md`）：官方 93.4 / 81.8 / 90.0 / 72.6（Two-Room / Reacher / Push-T / Cube），
min 代价 +1.0 / +11.4 / +0.4 / +1.8。

## 耗时

官方 `last` 模式每次规划约 5.5 s（每轮重新编码 300 张图），`hooklast` / `min` 约 0.6 s（`docs/lewm_plugin_v1.md`）。本协议是同步评测、不计耗时，
所以耗时不影响这里的成功率，只影响跑完的时间。

## 局限

- 没有和 LeWM 论文表格逐项对照（论文数字未在本仓库中核对）；这里的「官方」指用官方代码与权重在本机复跑的结果。
- 承诺前缀需要实时设定才有意义，官方协议下无法评测，见 `docs/lewm_plugin_v1.md`。

## 复现

```bash
# 软链接：/data1/cgh/dataset/lewm_cache/{pusht_expert_train.h5, tworoom.h5, ogbench/cube_single_expert.h5, dmc/reacher_random.h5}
PYTHONPATH=.:.pydeps_off:.pydeps MUJOCO_GL=egl FLASH_COST=last FLASH_OFFICIAL_JSON=results/official_lewm/lewm_pusht_last_s42.json \
python scripts/official_eval_lewm.py --config-path=$PWD/third_party/le-wm/config/eval --config-name=pusht policy=pusht \
  +cache_dir=/data1/cgh/dataset/lewm_cache seed=42 eval.num_eval=50
# FLASH_COST=hooklast / min 同上；结果 results/official_lewm/lewm_<env>_<mode>_s<seed>.json
```
