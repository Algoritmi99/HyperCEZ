import torch

from .args import Hparams

def check_finite(model):
    bad = []
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            bad.append(name)
    if bad:
        raise RuntimeError(f"Non-finite values detected in {model}: {', '.join(bad)}")
