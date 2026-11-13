# 讨论记录 v1

## 第 1 轮（2026-11-13）

**用户要求**：读 flash-lewm 仓库 issue #6，给出看法；可行就实现并验证，需要改进就改进后实现。计算用 Linux#1
（最初说 Linux#2，随后更正），conda 环境 sglang 或 ms-swift，代码放 `/data1/cgh/code/`，数据 `/data1/cgh/dataset/`，
HF 缓存 `/data1/cgh/cache/huggingface`。

**做了什么**
- 读 issue #6（GPT 写的研究方案：共享动作前缀 + 异步分支搜索，假设 H1/H2/H3，实验 E0–E7）和 Fast-LeWM 官方代码。
- 在 Linux#1 建 `/data1/cgh/code/flash-lewm`，解压 Push-T 数据，下载官方 checkpoint，复现官方求解器。
- 实现并跑了 E0（剖析）、E1（前缀缓存）、E3（离线搜索）、E5（闭环，最多 1010 回合）、E7（FlashCEM）。

**结论**：issue 的三条假设里只有「精确性」成立；瓶颈其实是每轮 CEM 重复编码图像。改成 FlashCEM 后
7.2× 加速且闭环成功率不变。详见 `issue6_review_and_results_v1.md`。

**待用户决定**
- 是否把方向从「共享前缀 + 异步」改为「FlashCEM + 少轮次 / 宽批」。
- 是否继续做 Cube 等其它环境（需解压 46 GB 的 cube 数据并下载对应 checkpoint）。
