"""Original LeWorldModel (Maes et al. 2026) for our planners.

load_lewm(dir) builds the model from the Hugging Face release (config.json + weights.pt, e.g. quentinll/lewm-pusht)
with the official code in third_party/le-wm (imported under private module names so it does not clash with the
Fast-LeWM code, which also has jepa.py / module.py). The ViT encoder is rebuilt exactly as
stable_pretraining.backbone.utils.vit_hf does; weights load strictly.

lewm_preds(model, z, plans) returns the latent after every block of the autoregressive rollout, computed exactly as
JEPA.rollout in the official code (history of model.history frames, one predictor call per block), so the official
terminal cost is preds[..., -1, :] and the prefix-min cost is a min over the block axis at no extra cost."""
import importlib.util, json, os, sys
import torch

_LE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "le-wm")


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod


def load_lewm(d, device="cuda"):
    from transformers import ViTConfig, ViTModel
    mod = _load("lewm_module", os.path.join(_LE, "module.py"))
    jp = _load("lewm_jepa", os.path.join(_LE, "jepa.py"))
    cfg = json.load(open(os.path.join(d, "config.json")))
    sizes = {"tiny": (192, 12, 3), "small": (384, 12, 6), "base": (768, 12, 12)}
    h, l, nh = sizes[cfg["encoder"]["size"]]
    enc = ViTModel(ViTConfig(hidden_size=h, num_hidden_layers=l, num_attention_heads=nh, intermediate_size=4 * h,
                             image_size=cfg["encoder"]["image_size"], patch_size=cfg["encoder"]["patch_size"]),
                   add_pooling_layer=False)
    clean = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
    mlp = lambda k: mod.MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"], hidden_dim=cfg[k]["hidden_dim"],
                            norm_fn=torch.nn.BatchNorm1d)
    m = jp.JEPA(encoder=enc, predictor=mod.ARPredictor(**clean(cfg["predictor"])), action_encoder=mod.Embedder(**clean(cfg["action_encoder"])),
                projector=mlp("projector"), pred_proj=mlp("pred_proj"))
    m.load_state_dict(torch.load(os.path.join(d, "weights.pt"), map_location="cpu", weights_only=False), strict=True)
    m.is_lewm, m.history = True, int(cfg["predictor"].get("num_frames", 3))
    m = m.to(device).eval(); m.requires_grad_(False)
    return m


def lewm_preds_raw(model, z, plans):
    """Same as lewm_preds, without the inference-mode decorator (used inside torch.compile)."""
    e, n, h = plans.shape[:3]
    emb = z.view(e, 1, 1, -1).expand(e, n, 1, -1).reshape(e * n, 1, -1)
    act = plans.reshape(e * n, h, -1)
    hs, out = model.history, []
    for t in range(h):
        ae = model.action_encoder(act[:, :t + 1])
        pr = model.predict(emb[:, -hs:], ae[:, -hs:])[:, -1:]
        emb = torch.cat([emb, pr], 1); out.append(pr)
    return torch.cat(out, 1).view(e, n, h, -1)


@torch.inference_mode()
def lewm_preds(model, z, plans):
    """z [E,D]; plans [E,N,H,A] (normalised blocks) -> latents after 1..H blocks [E,N,H,D]."""
    e, n, h = plans.shape[:3]
    emb = z.view(e, 1, 1, -1).expand(e, n, 1, -1).reshape(e * n, 1, -1)
    act = plans.reshape(e * n, h, -1)
    hs, out = model.history, []
    for t in range(h):
        ae = model.action_encoder(act[:, :t + 1])
        pr = model.predict(emb[:, -hs:], ae[:, -hs:])[:, -1:]
        emb = torch.cat([emb, pr], 1); out.append(pr)
    return torch.cat(out, 1).view(e, n, h, -1)
