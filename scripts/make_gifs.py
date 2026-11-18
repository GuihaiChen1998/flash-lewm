"""Side-by-side GIFs for the project page from scripts/record_rollouts.py output.
Left: official Fast-LeWM deployment (sync, ~600 ms per plan = 6 env steps frozen). Right: ours (FlashCEM 65 ms,
commitment prefix, min cost). Played in real time: 1 env step = 0.1 s.
Usage: make_gifs.py <rollouts.npz> <out_prefix> <n_gifs> [left_cfg right_cfg]"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, pre, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
L = sys.argv[4] if len(sys.argv) > 4 else "sync_5_6_30_last"
R = sys.argv[5] if len(sys.argv) > 5 else "clamp_1_1_30_min"
d = np.load(src)
T, sw = int(d["T"]), int(d["switch"])
fL, fR = d[L + "_frames"], d[R + "_frames"]; aL, aR = d[L + "_acting"], d[R + "_acting"]
sL, sR = d[L + "_first"], d[R + "_first"]
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
f13, f15, f17 = ImageFont.truetype(F, 13), ImageFont.truetype(FB, 15), ImageFont.truetype(FB, 17)
import os
P, G, PAD, HEAD, FOOT = int(os.environ.get('GIF_PANEL', '240')), 64, 14, 58, 34
NCOL = int(os.environ.get('GIF_COLORS', '128'))
INK, MUTE, BG = (20, 24, 31), (96, 102, 112), (255, 255, 255)
RED, GRN = (226, 72, 60), (24, 160, 110)

# episodes: ours succeeds and official fails first, then both succeed
order = [i for i in np.argsort(sR) if sR[i] < T and sL[i] >= T] + [i for i in np.argsort(sR) if sR[i] < T and sL[i] < T]
order = order[:k]
print("chosen", [(int(i), int(sL[i]), int(sR[i])) for i in order])

def panel(img, frame, acting, x, title, sub, t, succ_t, goal):
    fr = Image.fromarray(frame).resize((P, P), Image.NEAREST if frame.shape[0] < 150 else Image.BILINEAR)
    col = GRN if acting else RED
    dr = ImageDraw.Draw(img)
    dr.text((x, 8), title, font=f17, fill=INK); dr.text((x, 31), sub, font=f13, fill=MUTE)
    y = HEAD
    dr.rectangle([x - 4, y - 4, x + P + 3, y + P + 3], fill=col)
    img.paste(fr, (x, y))
    g = Image.fromarray(goal).resize((G, G), Image.BILINEAR)
    dr.rectangle([x + P - G - 7, y + 3, x + P - 3, y + G + 7], fill=(255, 255, 255)); img.paste(g, (x + P - G - 5, y + 5))
    dr.text((x + P - G - 5, y + G + 9), "goal", font=f13, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    if succ_t is not None and t >= succ_t:
        dr.rectangle([x, y + P - 30, x + P, y + P], fill=GRN)
        dr.text((x + 8, y + P - 25), f"reached at {succ_t / 10:.1f} s", font=f15, fill=(255, 255, 255))
    elif not acting:
        dr.rectangle([x, y + P - 30, x + P, y + P], fill=RED)
        dr.text((x + 8, y + P - 25), "planning... robot frozen", font=f15, fill=(255, 255, 255))

paths = []
for n, i in enumerate(order):
    frames = []
    for t in range(T + 1):
        W = 2 * P + 3 * PAD + 6
        img = Image.new("RGB", (W, HEAD + P + FOOT + 8), BG)
        goal = d["goal2"][i] if (sw >= 0 and t >= sw) else d["goal"][i]
        tl, tr = min(t, fL.shape[1] - 1), min(t, fR.shape[1] - 1)
        panel(img, fL[i, tl], aL[i, tl], PAD + 3, "Fast-LeWM (official)", os.environ.get("GIF_LSUB", "CEM 581 ms / plan, waits 6 steps"), t, int(sL[i]) if sL[i] < T else None, goal)
        panel(img, fR[i, tr], aR[i, tr], 2 * PAD + P + 6, "Ours", os.environ.get("GIF_RSUB", "FlashCEM 65 ms + commitment prefix"), t, int(sR[i]) if sR[i] < T else None, goal)
        dr = ImageDraw.Draw(img)
        foot = f"t = {t / 10:.1f} s"
        if sw >= 0 and sw <= t < sw + 12:
            foot += "    goal changed!"
        dr.text((PAD + 3, HEAD + P + 10), foot, font=f15, fill=RED if (sw >= 0 and sw <= t < sw + 12) else INK)
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=NCOL))
    frames += [frames[-1]] * 15
    p = f"{pre}_{n}.gif"
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    paths.append(p)
print("wrote", paths)
