"""Uncurated grid GIF: the first N evaluation episodes for both methods, side by side.
Usage: make_grid_gif.py <rollouts.npz> <out.gif> [n=8]"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
src, out = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
d = np.load(src); T, sw = int(d["T"]), int(d["switch"])
F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15); f12 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
C = 4; Rr = (n + C - 1) // C; S = 104; GAP = 4; HEAD = 30
import os
cfg = [("Fast-LeWM (official)", os.environ.get("GIF_L", "sync_5_6_30_last")), ("Ours", os.environ.get("GIF_R", "clamp_1_1_30_min"))]
gw = C * S + (C - 1) * GAP
W = 2 * gw + 3 * 16; H = HEAD + Rr * S + (Rr - 1) * GAP + 30
frames = []
for t in range(T + 1):
    img = Image.new("RGB", (W, H), (255, 255, 255)); dr = ImageDraw.Draw(img)
    for c, (name, k) in enumerate(cfg):
        x0 = 16 + c * (gw + 16)
        fr, ac, first = d[k + "_frames"], d[k + "_acting"], d[k + "_first"]
        ok = sum(1 for i in range(n) if first[i] <= t)
        dr.text((x0, 6), f"{name}: {ok}/{n} reached", font=F, fill=(20, 24, 31))
        for i in range(n):
            r, q = divmod(i, C); x, y = x0 + q * (S + GAP), HEAD + r * (S + GAP)
            tt = min(t, fr.shape[1] - 1)
            col = (24, 160, 110) if first[i] <= t else ((226, 72, 60) if not ac[i, tt] else (210, 214, 220))
            dr.rectangle([x - 2, y - 2, x + S + 1, y + S + 1], fill=col)
            img.paste(Image.fromarray(fr[i, tt]).resize((S, S), Image.BILINEAR), (x, y))
    msg = f"t = {t / 10:.1f} s" + ("   goal changed" if sw >= 0 and sw <= t < sw + 12 else "")
    dr.text((16, H - 22), msg + "     red border = frozen while planning, green = reached", font=f12, fill=(96, 102, 112))
    frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=96))
frames += [frames[-1]] * 15
frames[0].save(out, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
print("wrote", out)
