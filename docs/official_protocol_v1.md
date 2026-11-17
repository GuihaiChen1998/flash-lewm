# 官方评测协议下的对比（v1，2026-11-17）

目的：在 Fast-LeWM 官方 `eval.py` 下复现官方基线，并把「前缀终点最小代价」（min）接入官方评测做配对对比。

## 结论

1. **官方基线复现成功**：官方种子 42（每个环境 50 回合）下，我们得到 Two-Room 96%、Reacher 86%、Push-T 98%、Cube 78%，
   论文表 1 的 Fast-LeWM 是 98 / 88 / 96 / 80，四个环境都在 ±2 个百分点以内。
2. **单种子会高估**：种子 42 恰好是偏好的种子。10 个种子（500 回合）平均下官方基线是 93.4 / 81.8 / 90.0 / 72.6，
   每种子之间的标准差 3–6 个百分点。论文只报了一个种子。
3. **min 代价在官方协议下不降成功率，在 Reacher 上显著提升**：
   Reacher +11.4 [+7.6, +15.2]（81.8% → 93.2%）；其余三个环境 +0.4 到 +1.8，区间都含 0。

## 方法

- `scripts/official_eval.py` 直接调用官方 `eval.py` 的 `run()`，不改官方代码，只加两层替换：
  - `stable_pretraining` 在评测里只用来取 ImageNet 归一化常数，用一个极小的替身模块代替；
  - `swm.policy.AutoCostModel` 换成 `flash_lewm.core.load_model`：官方权重由 transformers 5.x 保存，需要在已装的 4.54 下重建 ViT（权重严格加载）。
  - 另外关掉了视频保存（缺 imageio-ffmpeg），并把逐回合成功写到 `FLASH_OFFICIAL_JSON` 指定的文件。
- 回合抽样、环境、成功判据、CEM 参数（300 样本、30 轮、top-30）、规划配置全部来自官方配置文件 `config/eval/{pusht,tworoom,cube,reacher}.yaml`。
- `FLASH_COST=min` 把 `JEPA.get_cost` 换成 `flash_lewm/official_hooks.py`：按官方方式编码当前帧与目标帧，
  再用 `flash_lewm.realtime.plan_cost(..., "min")` 打分。
- **等价性核对**：`FLASH_COST=hooklast` 用同一个替换函数但取官方终点代价，Two-Room 种子 42 的 50 个回合逐个与官方一致
  （都是 48 / 50），CEM 末轮代价均值也完全相同（12.804136548…）。
- 每个环境跑种子 42–51，每个种子 50 回合；同一种子下 last 与 min 的回合完全相同，做配对 bootstrap（95% 置信区间）。

## 结果（50 步预算内成功率，%）

| 环境 | 论文 Fast-LeWM（单种子） | 我们，种子 42 | 官方代价，10 种子 | min 代价，10 种子 | min − 官方 |
|---|---|---|---|---|---|
| Two-Room | 98 | 96 | 93.4 | 94.4 | +1.0 [−0.4, +2.4] |
| Reacher | 88 | 86 | 81.8 | 93.2 | +11.4 [+7.6, +15.2] |
| Push-T | 96 | 98 | 90.0 | 90.4 | +0.4 [−1.4, +2.2] |
| Cube | 80 | 78 | 72.6 | 74.4 | +1.8 [−1.2, +5.0] |

每种子成功率（官方代价）：Two-Room 90–98，Reacher 76–92，Push-T 82–98，Cube 64–82。全部数字在 `results/summary_official_protocol.json`，
逐回合结果在 `results/official/<环境>_<代价>_s<种子>.json`。

## 说明

- 官方协议是同步的（规划时环境等待），所以这里只能比代价函数；承诺前缀需要实时协议，见 `docs/issue7_review_and_results_v1.md` 等。
- 与论文的差异可能来自依赖版本：我们用 transformers 4.54（论文 5.3）、MuJoCo 3.6（与论文一致）、torch 2.x（论文 2.10）。
- 运行注意：每个评测进程默认会开很多计算线程；8 个并行时负载到 150、几乎不前进。现在用 `OMP_NUM_THREADS=4`、最多 4 个并行，
  Two-Room 每个种子约 1 分钟。

## 复现

```bash
cd /data1/cgh/code/flash-lewm
PY=/home/cgh/anaconda3/envs/sglang/bin/python
export PYTHONPATH=$PWD:$PWD/third_party/Fast-LeWorldModel:$PWD/.pydeps_off:$PWD/.pydeps MUJOCO_GL=egl OMP_NUM_THREADS=4
CFG=$PWD/third_party/Fast-LeWorldModel/config/eval
FLASH_COST=min FLASH_OFFICIAL_JSON=results/official/reacher_min_s42.json \
$PY scripts/official_eval.py --config-path=$CFG --config-name=reacher seed=42 eval.num_eval=50 world.max_episode_steps=100 \
  eval.dataset_path=/data1/cgh/dataset/lewm-original/reacher/reacher.h5 eval.ckpt_path=$PWD/ckpt/Fast-lewm_dmc_object.ckpt \
  hydra.run.dir=$PWD/results/official/reacher_min_s42
```
`.pydeps_off` 里是 hydra-core 1.3.2、omegaconf 2.3.0、scikit-learn 1.5.2 等（`pip install --no-deps --target .pydeps_off ...`）。
