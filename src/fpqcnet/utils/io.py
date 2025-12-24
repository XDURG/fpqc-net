from __future__ import annotations
import os
import torch

def load_checkpoint_safely(model: torch.nn.Module,
                           ckpt_path: str | None,
                           map_location: str = "cpu",
                           strict: bool = False) -> dict:
    if not ckpt_path:
        return {}

    if not os.path.isfile(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}. Skip loading.")
        return {}

    try:
        obj = torch.load(ckpt_path, map_location=map_location)
        if isinstance(obj, dict) and "state_dict" in obj:
            state = obj["state_dict"]
        else:
            state = obj

        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_state[nk] = v

        missing, unexpected = model.load_state_dict(new_state, strict=strict)
        if missing:
            print(f"[WARN] Missing keys when loading: {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading: {len(unexpected)}")

        meta = {}
        if isinstance(obj, dict):
            for k in ["epoch", "best_metric", "cfg"]:
                if k in obj:
                    meta[k] = obj[k]
        print(f"[OK] Loaded checkpoint: {ckpt_path}")
        return meta
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint ({ckpt_path}): {e}. Skip loading.")
        return {}
