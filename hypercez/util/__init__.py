import math
import torch
from numpy import ndarray

from .args import Hparams

def check_finite(model):
    bad = []
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            bad.append(name)
    if bad:
        raise RuntimeError(f"Non-finite values detected in {model}: {', '.join(bad)}")

def check_finite_grads(model):
    """Check that all existing parameter gradients of `model` are finite.

    Raises:
        RuntimeError: if any parameter has a gradient that contains NaN or Inf.
    """
    bad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad.append(name)
    if bad:
        raise RuntimeError(f"Non-finite gradients detected in {model}: {', '.join(bad)}")

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

def make_scheduler(optimizer, hparams):
    if hparams.optimizer["lr_decay_type"] == 'cosine':
        max_steps = hparams.train["training_steps"] - int(
            hparams.train["training_steps"] * hparams.optimizer["lr_warm_up"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps * 3, eta_min=0
        )
    elif hparams.optimizer["lr_decay_type"] == 'full_cosine':
        max_steps = hparams.train["training_steps"] - int(
            hparams.train["training_steps"] * hparams.optimizer["lr_warm_up"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps // 2, eta_min=0
        )
    else:
        scheduler = None

    return scheduler


def adjust_lr(hparams, optimizer, step_count, scheduler):
    lr_warm_step = int(hparams.train["training_steps"] * hparams.optimizer["lr_warm_up"])
    optimize_config = hparams.optimizer

    # adjust learning rate, step lr every lr_decay_steps
    if step_count < lr_warm_step:
        lr = optimize_config["lr"] * step_count / lr_warm_step
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        if hparams.optimizer["lr_decay_type"] == 'cosine':
            if scheduler is not None:
                scheduler.step()
            lr = scheduler.get_last_lr()[0] # return a list
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = optimize_config["lr"] * optimize_config["lr_decay_rate"] ** (
                        (step_count - lr_warm_step) // optimize_config["lr_decay_steps"])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    return lr

def has_nan(model_state):
    for component, state in model_state.items():
        for k, v in state.items():
            if torch.isnan(v).any():
                return True, component
    return False, None

def clone_ez_state(ez_agent):
    out_state = {}
    for component_name in ez_agent.get_model_list():
        out_state[component_name] = {}
        for param_name, param in ez_agent.model.__getattr__(component_name).named_parameters():
            out_state[component_name][param_name] = param.clone().detach()
    return out_state

def clip_list(values, max_norm, eps=1e-6, use_safe_max=False, safe_max=1e2):
    """Clip a list of tensors/floats to have L2 norm at most max_norm.

    Optionally, also bound each individual tensor's absolute values by
    ``safe_max`` to keep entries in a numerically safe range.

    Mirrors torch.nn.utils.clip_grad_norm_ with norm_type=2.0, but operates
    on a list that may contain both tensors and scalars.
    """
    if values is None:
        return None
    if len(values) == 0:
        return values

    # Compute total L2 norm^2 over all entries
    total_norm_sq = 0.0
    for v in values:
        if torch.is_tensor(v):
            # Use tensor 2-norm
            total_norm_sq += float(v.norm(2).item() ** 2)
        else:
            # Assume numeric scalar (float/int)
            fv = float(v)
            total_norm_sq += fv * fv

    total_norm = math.sqrt(total_norm_sq)

    result = values
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        clipped = []
        for v in values:
            if torch.is_tensor(v):
                clipped.append(v * clip_coef)
            else:
                clipped.append(float(v) * clip_coef)
        result = clipped

    if not use_safe_max:
        return result

    # Per-entry safety clipping based on max absolute value.
    safe_result = []
    for v in result:
        if torch.is_tensor(v):
            if v.numel() == 0:
                safe_result.append(v)
                continue
            max_abs = v.abs().max()
            if max_abs > safe_max:
                scale = safe_max / (max_abs + eps)
                safe_result.append(v * scale)
            else:
                safe_result.append(v)
        else:
            fv = float(v)
            if fv > safe_max:
                fv = safe_max
            elif fv < -safe_max:
                fv = -safe_max
            safe_result.append(fv)

    return safe_result

def list_has_nonfinite(tensors):
    for t in tensors:
        if not torch.isfinite(t).all():
            return True
    return False

def np_non_imag(inp: ndarray | tuple):
    if isinstance(inp, tuple):
        return inp[0]
    return inp