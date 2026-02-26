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
