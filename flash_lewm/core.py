"""Flash-LeWM core: model loading, cached latents, dense and prefix-cached scoring.

All scorers work on a single (observation, goal) pair unless stated otherwise.
Shapes (Push-T): z, zg: [192]; plans: [N, H=5, 10] in the normalised action space.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

H5 = os.environ.get("PUSHT_H5", "/data1/cgh/dataset/lewm-original/pusht_expert_train.h5")
CKPT = os.environ.get("FLASH_CKPT", "/data1/cgh/code/flash-lewm/ckpt/Fast-lewm_pusht_object.ckpt")
EP_KEY = os.environ.get("FLASH_EP_KEY", "episode_idx")       # "ep_idx" in the tworoom / cube / reacher files
STATE_KEY = os.environ.get("FLASH_STATE_KEY", "state")       # what the env is reset from ("proprio" for tworoom)
IMN_MEAN = (0.485, 0.456, 0.406)
IMN_STD = (0.229, 0.224, 0.225)


def load_model(path=CKPT, device="cuda"):
    if os.path.isdir(str(path)) and os.path.exists(os.path.join(str(path), "weights.pt")):   # original LeWM release
        from .lewm import load_lewm
        return load_lewm(str(path), device)
    # The checkpoint is a pickled nn.Module. Its pickle globals were listed with
    # pickletools before first use (only torch / transformers.ViT / jepa / module classes).
    if str(path).endswith(".pt"):   # a model trained with flash_lewm.train_min (state dict + config)
        from .train_min import load_trained
        m, ck = load_trained(path, device)
        m.img_size, m.a_mean, m.a_std = ck["img_size"], ck["a_mean"].double().numpy(), ck["a_std"].double().numpy()
        return m
    m = torch.load(path, weights_only=False, map_location="cpu")
    m.encoder = _rebuild_vit(m.encoder)
    m = m.to(device).eval()
    m.requires_grad_(False)
    return m


def _rebuild_vit(old):
    """The checkpoint pickles a transformers-5.x ViTModel object. Rebuild it with the installed
    transformers version from the stored hyper-parameters and copy the weights (strict)."""
    from transformers import ViTConfig, ViTModel

    d = old.config.__dict__
    keys = ["hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "hidden_act",
            "hidden_dropout_prob", "attention_probs_dropout_prob", "initializer_range", "layer_norm_eps",
            "image_size", "patch_size", "num_channels", "qkv_bias", "encoder_stride"]
    cfg = ViTConfig(**{k: d[k] for k in keys if k in d})
    sd = old.state_dict()
    new = ViTModel(cfg, add_pooling_layer=any(k.startswith("pooler") for k in sd))
    missing, unexpected = new.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return new


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Accumulating wall-clock timer with CUDA synchronisation."""

    def __init__(self):
        self.t = {}

    def add(self, key, t0):
        sync()
        self.t[key] = self.t.get(key, 0.0) + time.perf_counter() - t0

    @staticmethod
    def now():
        sync()
        return time.perf_counter()


@torch.inference_mode()
def preprocess(px, device="cuda", size=224):
    """uint8 [B,H,W,3] -> normalised float [B,3,H,W] (same as the official eval transform at 224 px)."""
    x = torch.as_tensor(np.ascontiguousarray(px)).to(device).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMN_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMN_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    if x.shape[-1] != size or x.shape[-2] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    return x


@torch.inference_mode()
def encode_pixels(model, px):
    """uint8 [B,H,W,3] -> latent [B,192]."""
    x = preprocess(px, next(model.parameters()).device, getattr(model, "img_size", 224))
    out = model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
    return model.projector(out)


@torch.inference_mode()
def dense_cost(model, z, zg, plans):
    """Reference scorer. Same computation as JEPA.get_cost for horizon=1, with z / zg already encoded."""
    n = plans.shape[0]
    emb = z.view(1, 1, -1).expand(n, 1, -1)
    act = model.action_encoder(plans, return_last_only=True, latent=emb)
    pred = model.predict(emb, act)
    return (pred[:, 0] - zg.view(1, -1)).pow(2).sum(-1)


@torch.inference_mode()
def dense_cost_multi(model, z, zg, plans):
    """Batched over environments. z, zg: [E,192]; plans: [E,N,H,10] -> [E,N]."""
    e, n = plans.shape[:2]
    emb = z.view(e, 1, 1, -1).expand(e, n, 1, -1).reshape(e * n, 1, -1)
    act = model.action_encoder(plans.reshape(e * n, *plans.shape[2:]), return_last_only=True, latent=emb)
    pred = model.predict(emb, act)
    return (pred[:, 0].view(e, n, -1) - zg.view(e, 1, -1)).pow(2).sum(-1)


class PrefixScorer:
    """Exact shared-prefix scoring (Prefill / Append) for the causal action-prefix encoder.

    Sequence fed to the encoder: [latent token, a_1, ..., a_H] with sinusoidal positions 0..H.
    A branch fixes the latent token plus the first s action blocks. Its per-layer K/V are computed
    once (prefill) and every suffix in the branch attends to them (append). s = 0 shares only the
    latent token, which is valid for all candidates of one observation.
    """

    def __init__(self, model):
        self.m = model
        self.ae = model.action_encoder.impl
        self.layers = self.ae.temporal_mixer.layers
        assert self.ae.temporal_mixer_type == "transformer" and self.ae.use_latent_condition
        assert not self.ae.use_learnable_latent_token

    def _tok(self, a):
        return self.ae.embed(self.ae.token_proj(a.float()))

    def _pos(self, start, length, ref):
        pe = self.ae._sinusoidal_pos_encoding(start + length, ref.size(-1), ref.device, ref.dtype)
        return pe[:, start:start + length]

    @torch.inference_mode()
    def prefill(self, z, prefixes):
        """z: [192]; prefixes: [L, s, 10] (s may be 0). Returns per-layer (K, V) of the prefix tokens."""
        L, s = prefixes.shape[:2]
        lat = self.ae.latent_proj(z.view(1, 1, -1).float()).expand(L, 1, -1)
        x = torch.cat([lat, self._tok(prefixes)], dim=1) if s > 0 else lat
        x = x + self._pos(0, 1 + s, x)
        cache = []
        for blk in self.layers:
            at = blk.attn
            h = at.heads
            n = at.norm(blk.norm1(x))
            q, k, v = at.to_qkv(n).chunk(3, dim=-1)
            q, k, v = (t.view(L, 1 + s, h, -1).transpose(1, 2) for t in (q, k, v))  # [L,h,Tp,d]
            cache.append((k, v))
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + at.to_out(o.transpose(1, 2).reshape(L, 1 + s, -1))
            x = x + blk.mlp(blk.norm2(x))
        return {"kv": cache, "s": s, "L": L, "z": z}

    @torch.inference_mode()
    def append(self, cache, suffix):
        """suffix: [L, K, Ts, 10] -> last prefix state p_H: [L, K, 192]."""
        L, K, Ts = suffix.shape[:3]
        s = cache["s"]
        Tp = 1 + s
        x = self._tok(suffix)
        x = x + self._pos(Tp, Ts, x)
        causal = torch.ones(Ts, Ts, dtype=torch.bool, device=x.device).tril()
        nl = len(self.layers)
        for li, (blk, (kp, vp)) in enumerate(zip(self.layers, cache["kv"])):
            at = blk.attn
            h = at.heads
            n = at.norm(blk.norm1(x))
            q, k, v = at.to_qkv(n).chunk(3, dim=-1)
            q, k, v = (t.view(L, K, Ts, h, -1).permute(0, 1, 3, 2, 4) for t in (q, k, v))  # [L,K,h,Ts,d]
            last = li == nl - 1
            if last:  # only the final token is read out
                q = q[:, :, :, -1:]
            lp = torch.einsum("lkhtd,lhpd->lkhtp", q, kp) * at.scale
            ls = torch.einsum("lkhtd,lkhsd->lkhts", q, k) * at.scale
            ls = ls.masked_fill(~(causal[-1:] if last else causal), float("-inf"))
            wgt = torch.cat([lp, ls], dim=-1).softmax(dim=-1)
            o = torch.einsum("lkhtp,lhpd->lkhtd", wgt[..., :Tp], vp) + torch.einsum(
                "lkhts,lkhsd->lkhtd", wgt[..., Tp:], v)
            o = o.permute(0, 1, 3, 2, 4).reshape(L, K, o.shape[3], -1)
            if last:
                x = x[:, :, -1:]
            x = x + at.to_out(o)
            x = x + blk.mlp(blk.norm2(x))
        x = self.ae.temporal_mixer.norm(x)
        return self.ae.out_norm(x)[:, :, -1]

    @torch.inference_mode()
    def cost(self, cache, suffix, zg):
        """Cost of every leaf: [L, K]."""
        L, K = suffix.shape[:2]
        p = self.append(cache, suffix).reshape(L * K, 1, -1)
        emb = cache["z"].view(1, 1, -1).expand(L * K, 1, -1)
        pred = self.m.predict(emb, p)
        return (pred[:, 0] - zg.view(1, -1)).pow(2).sum(-1).view(L, K)


# ----------------------------------------------------------------------------- data


def official_eval_rows(n=50, seed=42, goal_offset=25, h5=H5):
    """Replicates the start-row sampling in eval.py (seed 42, 50 rows by default)."""
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(h5, "r") as f:
        ep = f[EP_KEY][:]
        step = f["step_idx"][:]
        ep_len = f["ep_len"][:]
    max_start = ep_len[ep] - goal_offset - 1
    valid = np.nonzero(step <= max_start)[0]
    g = np.random.default_rng(seed)
    idx = g.choice(len(valid) - 1, size=n, replace=False)
    return np.sort(valid[idx])


def load_pairs(rows, goal_offset=25, h5=H5):
    """Returns dict with pixels / goal pixels (uint8) and state / goal state for the given rows."""
    import h5py
    import hdf5plugin  # noqa: F401

    out = {"pixels": [], "goal": [], "state": [], "goal_state": []}
    with h5py.File(h5, "r") as f:
        for r in rows:
            out["pixels"].append(f["pixels"][r])
            out["goal"].append(f["pixels"][r + goal_offset])
            out["state"].append(f[STATE_KEY][r])
            out["goal_state"].append(f[STATE_KEY][r + goal_offset])
    return {k: np.stack(v) for k, v in out.items()}


def action_stats(h5=H5, cache=None):
    """Mean / std of the action column (what sklearn StandardScaler fits in eval.py)."""
    if cache is None:
        name = os.path.basename(h5).split(".")[0].replace("_expert_train", "")
        cache = f"/data1/cgh/code/flash-lewm/ckpt/{name}_action_stats.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        return d["mean"], d["std"]
    import h5py

    with h5py.File(h5, "r") as f:
        a = f["action"][:].astype(np.float64)
    a = a[~np.isnan(a).any(axis=1)]
    mean, std = a.mean(0), a.std(0)
    np.savez(cache, mean=mean, std=std)
    return mean, std
