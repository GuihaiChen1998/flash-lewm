"""Hooks into the official JEPA.get_cost (third_party/Fast-LeWorldModel/jepa.py).

install_min_cost(model, cost="min") replaces get_cost with: encode goal and current frame exactly as the
official code does, then score every candidate by flash_lewm.realtime.plan_cost (min over the endpoints of
all prefix lengths). With cost="last" it must reproduce the official numbers (checked by check_equivalence).
"""
import types
import torch


def _encode_pair(self, info_dict, n_cand):
    device = next(self.parameters()).device
    for k in list(info_dict.keys()):
        if torch.is_tensor(info_dict[k]):
            v = info_dict[k]
            if v.dim() >= 2 and v.size(1) == n_cand:
                v = v[:, :1]
            info_dict[k] = v.to(device)
    goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
    goal["pixels"] = goal["goal"]
    for k in list(goal.keys()):
        if k.startswith("goal_"):
            goal[k[len("goal_"):]] = goal.pop(k)
    goal.pop("action", None)
    zg = self.encode(goal)["emb"][:, -1]                      # [B, D]
    init = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
    init.pop("action", None)
    z = self.encode(init)["emb"][:, -1]                       # [B, D]
    return z, zg


def install_min_cost(model, cost="min"):
    from .realtime import plan_cost
    enc_dim = model.action_encoder.impl.input_dim if hasattr(model.action_encoder, "impl") else None

    def get_cost(self, info_dict, action_candidates):
        b, s = action_candidates.shape[:2]
        z, zg = _encode_pair(self, info_dict, s)
        d = enc_dim or self._action_encoder_input_dim()
        plans = action_candidates.reshape(b, s, -1, d)        # [B, S, blocks, block_dim]
        return plan_cost(self, z, zg, plans, cost)

    model._official_get_cost = model.get_cost
    model.get_cost = types.MethodType(get_cost, model)
    return model
