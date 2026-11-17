# LIBERO-10：单任务与 10 任务实验（v1，2026-11-17）

目的：在 LIBERO-10 上检验承诺前缀（clamp）和前缀终点最小代价（min）。这是第一轮，只用一个任务。

## 结论

1. **承诺前缀显著优于同步规划**：延迟 5 步时静态目标 +6.0 [+1.3, +10.7]、换目标 +6.3 [+3.0, +10.0]；
   延迟 10 步时 +4.0 [+0.7, +7.3] 和 +2.3 [+0.7, +4.3]。方向与 Cube、ViZDoom 一致。
2. **承诺前缀与朴素异步没有显著差别**（四个格子都在 ±1 个百分点内，区间含 0）。这和 Cube、ViZDoom 不同，
   那两个环境里承诺前缀对朴素异步也有 +4 到 +14 的优势。
3. **基础成功率很低**：零延迟最好只有 19%（示范动作回放的上限是 93%），所以各差值的绝对量都小，只能算初步证据。
4. min 代价只在零延迟时有效（m=1 +6.7、m=5 +4.3，静态目标）；同步规划下与官方代价无差别。
5. 一个反常现象：换目标场景里，延迟 5 步的承诺前缀（9.3%）比零延迟、每块重规划（5.0%）还高 +4.3 [+0.7, +8.0]。
   猜测是频繁重规划在这个模型上造成动作抖动，承诺前缀等于强制执行一段已排队的动作，起了平滑作用；没有验证。

## 设置

- 任务：`STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`（50 条示范，9470 帧）。
- 世界模型：用 `flash_lewm.train_min` 在这 50 条示范上训练 30000 步，输入 agentview 128×128 图像缩放到 112，动作 7 维。
  第 12000 步时验证集预测误差 0.06，「当前帧当预测」基线 0.22。权重 `ckpt/libero_study1.pt`。
- 评测回合（`scripts/libero_pairs.py`）：起点取示范某一帧的完整仿真状态，目标取 25 步后那一帧，换目标场景的第二目标取 50 步后；
  只保留起点→目标、目标→第二目标、起点→第二目标都有物体移动超过 4 cm 的片段（4028 个候选里抽 300 个）。
- 成功判据（`flash_lewm/libero_env.py`）：所有可动物体（书、杯子）都在目标帧位置 4 cm 以内，不看机械臂。
  这不是 LIBERO 官方的语言任务成功率，不能和 TC-LeWM 等论文的数字比较。
- 判据核对：从示范状态复位后，渲染图与数据集图平均每像素差 1–4（满分 255）；
  用示范自己的动作从 100 个评测起点回放，93% 在 25 步内触发成功（中位数 19 步）。
- 规划停顿时的全零动作会让 LIBERO 的夹爪半张开，适配器里改为保持上一次的夹爪指令。
- 协议、配置写法、统计方法同 `docs/cube_doom_solid_v1.md`；300 回合，配对 bootstrap 95% 置信区间。

## 成功率（50 步内，%）

| 配置 | 静态，d=5 | 静态，d=10 | 换目标，d=5 | 换目标，d=10 |
|---|---|---|---|---|
| clamp + min | 18.3 | 9.3 | 9.3 | 3.3 |
| naive + min | 17.7 | 10.0 | 8.3 | 3.7 |
| sync（m=5）+ min | 12.3 | 5.3 | 3.0 | 1.0 |
| sync（m=5）+ 官方代价 | 13.0 | 5.3 | 2.7 | 1.3 |

零延迟参照（静态 / 换目标）：m=5 min 19.0 / 5.3，m=5 官方代价 14.7 / 7.0，m=2 min 15.0 / 7.0，m=1 min 15.0 / 5.0，m=1 官方代价 8.3 / 4.3。

## 配对差（百分点，50 步内）

| 对比 | 静态，d=5 | 静态，d=10 | 换目标，d=5 | 换目标，d=10 |
|---|---|---|---|---|
| clamp − sync min | +6.0 [+1.3, +10.7] | +4.0 [+0.7, +7.3] | +6.3 [+3.0, +10.0] | +2.3 [+0.7, +4.3] |
| clamp − sync 官方代价 | +5.3 [+1.0, +9.7] | +4.0 [+0.7, +7.3] | +6.7 [+3.3, +10.7] | +2.0 [+0.0, +4.0] |
| clamp − naive | +0.7 [−3.3, +4.7] | −0.7 [−3.7, +2.3] | +1.0 [−2.0, +4.0] | −0.3 [−2.3, +1.7] |
| sync：min − 官方代价 | −0.7 [−4.3, +2.7] | +0.0 [−1.7, +1.7] | +0.3 [−1.3, +2.0] | −0.3 [−1.0, +0.0] |

其他：
- clamp（d=5）− 零延迟 m=5 min：静态 −0.7 [−5.7, +4.3]，换目标 +4.0 [+0.3, +8.0]。
- 零延迟下 min − 官方代价：m=5 静态 +4.3 [+0.3, +8.3]、换目标 −1.7 [−4.0, +0.7]；m=1 静态 +6.7 [+3.0, +10.7]、换目标 +0.7 [−1.3, +2.7]。
- 放宽到 75 步，clamp − sync min 为 +6.7、+4.7、+6.0、+6.0，区间都不含 0。

## 10 任务实验（多任务模型）

- 模型：10 个任务的全部 500 条示范（13.8 万帧）合训一个 Fast-LeWM，60000 步，`ckpt/libero10.pt`。
  最终验证预测误差约 0.01–0.02，「当前帧当预测」基线 0.08–0.11。
- 每个任务 100 个评测回合（`/data1/cgh/dataset/libero/pairs10_<i>.npz`，任务编号见 `pairs10_index.json`），
  10 个配置 × 2 个场景，标签 `L10t<i><场景>`。

**结论：基本没有信息量。** 静态目标下 10 个任务合并后零延迟成功率只有 8.7%，换目标 3.1%（9 个任务，见下），
各方法之间的差别都在 2 个百分点以内。多任务模型的规划成功率没有比单任务模型高（单任务那个任务上零延迟 m=5 是 19%，
这里同一任务是 15%）。

**一个任务的判据有漏洞（已剔除）**：任务 8（把杯子放进微波炉并关门）里，带延迟的方法有 37–41 个回合在前 10 步就成功了，
零延迟只有 6 个。这些「成功」发生在第一次规划停顿期间：适配器按指间距猜测停顿时的夹爪指令，猜错时夹爪张开，
杯子直接掉进微波炉，就满足了「物体到目标位置」。下面的合并数字剔除了任务 8。

合并 9 个任务（900 回合）的配对差（百分点，50 步内）：

| 对比 | 静态，d=5 | 静态，d=10 | 换目标，d=5 | 换目标，d=10 |
|---|---|---|---|---|
| clamp − sync min | +1.9 [+0.4, +3.3] | +1.4 [+0.3, +2.6] | +0.0 [−0.8, +0.8] | −0.2 [−0.8, +0.2] |
| clamp − naive | +0.0 [−1.4, +1.4] | −0.7 [−2.0, +0.7] | −0.2 [−1.0, +0.6] | −0.3 [−0.9, +0.2] |

零延迟下 min − 官方代价（m=5）：静态 +0.4 [−1.2, +2.0]，换目标 −0.7 [−1.8, +0.6]。
合并成功率（静态 / 换目标，%）：零延迟 m=5 min 8.7 / 3.1，sync m=5 min 5.6 / 3.0，naive 7.4 / 3.2，clamp 7.4 / 3.0（d=5）。

各任务成功率见 `results/summary_libero10.json`（含任务 8）。只有任务 1（碗放进抽屉并关上）和任务 6、7 在两个场景下都有 10% 以上的成功率。

## 局限与下一步

- 只有一个任务、一个模型种子；成功率低，差值量级只有几个百分点。
- 训练日志在第 12000 步后丢失（同步代码时日志文件被替换），最终训练曲线没有记录；权重正常保存。
- 10 任务实验受地板效应限制，几乎分不出方法。要在 LIBERO 上得到有分辨力的比较，先要把基础成功率提上去：
  缩短目标距离（25 → 10 步）、加大模型 / 训练步数，或换用规划能力更强的求解器。
- 适配器在停顿时猜测夹爪指令的做法有漏洞（任务 8）；应改为记录并沿用评测回合起点之前那一步示范的夹爪动作。

## 复现

```bash
cd /data1/cgh/code/flash-lewm
PY=/home/cgh/anaconda3/envs/sglang/bin/python
export LIBERO_CONFIG_PATH=$PWD/.libero_cfg MUJOCO_GL=egl PYTHONWARNINGS=ignore
export PYTHONPATH=$PWD/.mj/3.3.0:$PWD/.pydeps_libero:$PWD/third_party/LIBERO:$PWD/third_party/Fast-LeWorldModel:$PWD:$PWD/.pydeps
export LIBERO_BDDL_DIR=$PWD/third_party/LIBERO/libero/libero/bddl_files/libero_10
D=/data1/cgh/dataset/libero; T=STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy
$PY scripts/libero_convert.py $D/study1.npz $D/${T}_demo.hdf5
$PY -m flash_lewm.train_min $D/study1.npz ckpt/libero_study1.pt steps=30000 img=112
$PY scripts/libero_pairs.py $D/${T}_demo.hdf5 $D/pairs_study1.npz 300
$PY scripts/libero_replay_success.py $D/${T}_demo.hdf5 $D/pairs_study1.npz 100
FLASH_ENV=libero FLASH_ADIM=7 FLASH_CHUNK=25 FLASH_LIBERO_BDDL=$LIBERO_BDDL_DIR/$T.bddl FLASH_CKPT=ckpt/libero_study1.pt \
DOOM_PAIRS=$D/pairs_study1.npz $PY scripts/e9_realtime.py 250 LIB1base6 clamp:1:5:30:min:base
```

依赖：LIBERO 代码在 `third_party/LIBERO`（GitHub master 压缩包）；robosuite 1.4.1、bddl 1.0.1、gym 0.25.2、matplotlib 等
用 `pip install --no-deps --target .pydeps_libero` 装在项目目录内；MuJoCo 用 `.mj/3.3.0`。
逐回合结果 `results/e9_realtime_LIB1*.json`（`LIBmid*` 是训练中途权重的冒烟）。
