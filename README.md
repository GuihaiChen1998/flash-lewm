# flash-lewm

Fast-LeWM（[论文](https://arxiv.org/abs/2606.26217)，[官方代码](https://github.com/Yuntian-Gao/Fast-LeWorldModel)）规划求解器的加速实验。

当前状态：issue #6 的方案已在 Push-T 上实现并验证，结论和数据见
[docs/issue6_review_and_results_v1.md](docs/issue6_review_and_results_v1.md)。
简要说：共享前缀缓存是精确的，但在官方配置（300 候选）下更慢；分支搜索和异步在单 GPU 上不成立；
把图像潜向量缓存到 CEM 循环外并编译整轮迭代（`FlashCEM`）后，单次规划从 467 ms 降到 65 ms，闭环成功率不变（n=1010）。

后续进展（按时间）：

| 文档 | 内容 |
|---|---|
| [docs/issue7_review_and_results_v1.md](docs/issue7_review_and_results_v1.md) | issue #7：承诺前缀 / 异步重规划，实时仿真协议 |
| [docs/min_cost_and_scenarios_v1.md](docs/min_cost_and_scenarios_v1.md) | 「前缀终点最小」代价与扰动、目标切换、远目标场景 |
| [docs/multi_env_results_v1.md](docs/multi_env_results_v1.md) | Two-Room、ViZDoom、Visual AntMaze（含数据集对齐） |
| [docs/cube_doom_solid_v1.md](docs/cube_doom_solid_v1.md) | Cube 与 ViZDoom 的大样本复核 |
| [docs/libero_results_v1.md](docs/libero_results_v1.md) | LIBERO-10 单任务与 10 任务 |
| [docs/official_protocol_v1.md](docs/official_protocol_v1.md) | 官方 eval.py 协议下复现与 min 代价对比 |
| [docs/real_latency_v1.md](docs/real_latency_v1.md) | 按实测规划耗时的实时对比：完整方法 vs 官方部署（最新） |
| [docs/discussion_log_v1.md](docs/discussion_log_v1.md) | 逐轮讨论记录 |
| [docs/HANDOFF.md](docs/HANDOFF.md) | 交接文档：机器、路径、运行前缀、未入库资源的位置 |

目前最稳的结论：按实测规划耗时，完整方法（FlashCEM + 承诺前缀 + min 代价）在 Push-T、Two-Room、Cube、ViZDoom 上
全部显著优于官方 Fast-LeWM 的实际部署（换目标场景 +8 到 +62 个百分点）；官方协议下 min 代价不降成功率、Reacher +11.4。

## 代码

| 文件 | 内容 |
|---|---|
| `flash_lewm/core.py` | 模型加载、图像编码、稠密评分、`PrefixScorer`（Prefill / Append 精确 K/V 共享） |
| `flash_lewm/search.py` | `cem`（与官方 CEMSolver 同规则）、`tree_search`（issue #6 算法 1，同步/异步/自适应）、`FlashCEM` |
| `flash_lewm/closedloop.py` | 按官方协议的 Push-T 闭环评估 |
| `scripts/e0_profile.py` | 官方路径剖析 |
| `scripts/e1_prefix_cache.py` | 前缀缓存精确性、吞吐、延迟–批大小曲线 |
| `scripts/e3_search.py` | 离线搜索质量对照 |
| `scripts/e5_closedloop.py`、`scripts/paired.py` | 闭环成功率与配对 bootstrap |
| `scripts/e7_flash.py` | FlashCEM 离线延迟与质量 |
| `flash_lewm/realtime.py`、`scripts/e9_realtime.py` | 实时仿真器（延迟注入、承诺前缀、min 代价、场景预设） |
| `flash_lewm/{doom,ant,cube}_env.py`、`flash_lewm/train_min.py` | ViZDoom / AntMaze / Cube 适配器，最小训练脚本 |

## 运行（Linux#1）

```bash
cd /data1/cgh/code/flash-lewm
export PYTHONPATH=$PWD:$PWD/third_party/Fast-LeWorldModel:$PWD/.pydeps SDL_VIDEODRIVER=dummy
PY=/home/cgh/anaconda3/envs/sglang/bin/python
CUDA_VISIBLE_DEVICES=1 $PY scripts/e0_profile.py 8
CUDA_VISIBLE_DEVICES=1 $PY scripts/e5_closedloop.py 960 big "official|flash 300x30 mean"
$PY scripts/paired.py results/e5_closedloop_big*.json
```

依赖：`sglang` 环境本身没有改动；`stable-worldmodel==0.0.6`、gymnasium、pymunk、pygame、shapely、h5py、hdf5plugin 等用
`pip install --no-deps --target .pydeps` 装在项目目录内。数据 `/data1/cgh/dataset/lewm-original/pusht_expert_train.h5`
（由 `.h5.zst` 解压，43 GB），checkpoint 在 `ckpt/`。

用法示例：

```python
from flash_lewm.core import load_model, encode_pixels
from flash_lewm.search import FlashCEM
model = load_model()
planner = FlashCEM(model, n=300, topk=30, iters=30)   # 首次调用编译约 10 秒
z, zg = encode_pixels(model, pixels_uint8)            # [2,224,224,3] -> [2,192]
plan = planner(z, zg)["plan"]                         # [5,10]，归一化动作空间
```
