# flash-lewm

Fast-LeWM（[论文](https://arxiv.org/abs/2606.26217)，[官方代码](https://github.com/Yuntian-Gao/Fast-LeWorldModel)）规划求解器的加速实验。

当前状态：issue #6 的方案已在 Push-T 上实现并验证，结论和数据见
[docs/issue6_review_and_results_v1.md](docs/issue6_review_and_results_v1.md)。
简要说：共享前缀缓存是精确的，但在官方配置（300 候选）下更慢；分支搜索和异步在单 GPU 上不成立；
把图像潜向量缓存到 CEM 循环外并编译整轮迭代（`FlashCEM`）后，单次规划从 467 ms 降到 65 ms，闭环成功率不变（n=1010）。

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
