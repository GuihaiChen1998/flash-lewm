"""Minimal Fast-LeWM trainer (no stable-pretraining / lightning): same modules and the same loss as
the official train.py (dense prefix supervision + SIGReg), data held in GPU memory.

Dataset file: .npz with pixels [N,S,S,3] uint8, action [N,d] float32, ep_idx [N], step_idx [N].
A training sample is 6 frames spaced `k` env steps apart and the 5*k actions between them.
"""
import json, math, sys, time
import numpy as np, torch
from module import ARPredictor, Embedder, MLP, SIGReg
from jepa import JEPA

MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)


def build_model(action_dim, img_size, k=5):
    from transformers import ViTConfig, ViTModel
    enc = ViTModel(ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3, intermediate_size=768,
                             image_size=img_size, patch_size=14, qkv_bias=True), add_pooling_layer=False)
    pred = ARPredictor(input_dim=192, hidden_dim=192, output_dim=192, depth=6, value_heads=16, mlp_dim=2048,
                       value_dim_head=64, dropout=0.1, emb_dropout=0.0, action_fusion_hidden_dim=768,
                       action_fusion_zero_init=True, token_processing="batch")
    ae = Embedder(input_dim=action_dim * k, emb_dim=192, use_positional_encoding=True, temporal_mixer_type="transformer",
                  transformer_depth=3, transformer_heads=6, transformer_dim_head=32, transformer_mlp_dim=768,
                  use_latent_condition=True, latent_dim=192)
    return JEPA(encoder=enc, predictor=pred, action_encoder=ae,
                projector=MLP(192, 2048, 192, norm_fn=torch.nn.BatchNorm1d),
                pred_proj=MLP(192, 2048, 192, norm_fn=torch.nn.BatchNorm1d))


def load_trained(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = build_model(ck["action_dim"], ck["img_size"], ck["k"])
    m.load_state_dict(ck["state_dict"])
    m = m.to(device).eval(); m.requires_grad_(False)
    return m, ck


def norm_px(px):
    x = px.permute(0, 3, 1, 2).float() / 255.0
    return (x - torch.tensor(MEAN, device=px.device).view(1, 3, 1, 1)) / torch.tensor(STD, device=px.device).view(1, 3, 1, 1)


@torch.inference_mode()
def encode(model, px_uint8):
    """uint8 [B,S,S,3] (numpy or tensor) at the training resolution -> [B,192]."""
    dev = next(model.parameters()).device
    x = norm_px(torch.as_tensor(np.ascontiguousarray(px_uint8) if isinstance(px_uint8, np.ndarray) else px_uint8).to(dev))
    return model.projector(model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])


def main(data, out, steps=30000, bs=128, lr=1e-4, k=5, H=5, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    d = np.load(data)
    px = torch.from_numpy(d["pixels"]).to(dev)
    act = d["action"].astype(np.float32)
    good = ~np.isnan(act).any(1)
    a_mean, a_std = act[good].mean(0), act[good].std(0) + 1e-6
    actn = torch.from_numpy(np.nan_to_num((act - a_mean) / a_std)).to(dev)
    ep, st = d["ep_idx"], d["step_idx"]
    n = len(ep)
    span = k * H
    ok = np.zeros(n, bool)
    ok[:n - span] = ep[:n - span] == ep[span:]
    starts_all = np.nonzero(ok)[0]
    rng = np.random.default_rng(seed)
    val_ep = set(rng.choice(np.unique(ep), size=max(1, len(np.unique(ep)) // 20), replace=False).tolist())
    is_val = np.isin(ep[starts_all], list(val_ep))
    tr, va = torch.from_numpy(starts_all[~is_val]).to(dev), torch.from_numpy(starts_all[is_val]).to(dev)
    print(f"frames {n}, train starts {len(tr)}, val starts {len(va)}, action dim {act.shape[1]}, img {px.shape[1]}", flush=True)
    model = build_model(act.shape[1], px.shape[1], k).to(dev)
    sig = SIGReg(knots=17, num_proj=1024).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    offs = torch.arange(0, span + 1, k, device=dev)
    aoff = torch.arange(span, device=dev)

    def batch(idx):
        frames = norm_px(px[(idx[:, None] + offs).reshape(-1)]).view(len(idx), H + 1, 3, px.shape[1], px.shape[2])
        a = actn[(idx[:, None] + aoff)].view(len(idx), H, -1)
        return {"pixels": frames, "action": a}

    def loss_fn(b):
        o = model.encode(b)
        emb, ap = o["emb"], o["act_emb"]
        pred = model.predict(emb[:, 0], ap)
        pl = (pred - emb[:, 1:]).pow(2).mean()
        return pl, sig(emb.transpose(0, 1))

    t0 = time.time(); log = []
    for s in range(steps):
        model.train()
        idx = tr[torch.randint(0, len(tr), (bs,), device=dev)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pl, sl = loss_fn(batch(idx))
            loss = pl + 0.09 * sl
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if s % 500 == 0 or s == steps - 1:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vs = [loss_fn(batch(va[torch.randint(0, len(va), (bs,), device=dev)])) for _ in range(4)]
                vp = float(np.mean([v[0].item() for v in vs]))
                # copy baseline: predicting "nothing changes" (how much the prediction beats it shows the model uses actions)
                b = batch(va[torch.randint(0, len(va), (bs,), device=dev)]); e = model.encode(b)["emb"]
                copy = (e[:, :1] - e[:, 1:]).pow(2).mean().item()
            log.append({"step": s, "pred": pl.item(), "sigreg": sl.item(), "val_pred": vp, "val_copy": copy, "t": time.time() - t0})
            print(json.dumps(log[-1]), flush=True)
        if (s + 1) % 5000 == 0 or s == steps - 1:
            torch.save({"state_dict": model.state_dict(), "action_dim": int(act.shape[1]), "img_size": int(px.shape[1]), "k": k,
                        "a_mean": torch.tensor(a_mean), "a_std": torch.tensor(a_std), "step": s + 1, "log": json.dumps(log)}, out)


if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[3:])
    main(sys.argv[1], sys.argv[2], **{k: (float(v) if k == "lr" else int(v)) for k, v in kw.items()})
