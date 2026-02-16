import torch

from .args import Hparams

def check_finite(model):
    bad = []
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            bad.append(name)
    if bad:
        raise RuntimeError(f"Non-finite values detected in {model}: {', '.join(bad)}")

def deep_copy_full_state(obj):
    # Tensor case
    if torch.is_tensor(obj):
        t = obj.detach().clone()
        # optional: preserve requires_grad flag
        t.requires_grad_(obj.requires_grad)
        return t

    # Dict
    if isinstance(obj, dict):
        return {k: deep_copy_full_state(v) for k, v in obj.items()}

    # List / Tuple
    if isinstance(obj, list):
        return [deep_copy_full_state(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(deep_copy_full_state(v) for v in obj)

    # Primitives
    return obj