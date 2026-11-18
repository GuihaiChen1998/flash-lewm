"""Check that our LeWM scorer matches the official JEPA.get_cost on random candidates, and print the encoder output."""
import sys, numpy as np, torch
from flash_lewm.core import load_model, official_eval_rows, load_pairs, preprocess, encode_pixels
from flash_lewm.realtime import plan_cost
m = load_model(sys.argv[1]); rows = official_eval_rows(50)[:2]; p = load_pairs(rows)
g = torch.Generator(device="cuda").manual_seed(0)
cand = torch.randn(1, 64, 5, 10, generator=g, device="cuda")
info = {"pixels": preprocess(p["pixels"][:1])[:, None, None].expand(1, 64, 1, 3, 224, 224).contiguous(),
        "goal": preprocess(p["goal"][:1])[:, None, None].expand(1, 64, 1, 3, 224, 224).contiguous(),
        "action": torch.zeros(1, 64, 1, 10, device="cuda")}
with torch.inference_mode():
    off = m.get_cost(info, cand)
    zz = encode_pixels(m, np.concatenate([p["pixels"][:1], p["goal"][:1]]))
    ours = plan_cost(m, zz[:1], zz[1:], cand, "last")
print("max rel diff", ((off - ours).abs() / off.abs()).max().item(), off[0, :4].tolist(), ours[0, :4].tolist())
