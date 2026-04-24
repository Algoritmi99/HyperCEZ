import copy
import os
import random
import tempfile
from enum import IntEnum

import numpy as np
import torch
import torch.nn as nn


class ActType(IntEnum):
    INITIAL = 0
    TRAIN = 1


class Agent:
    CHECKPOINT_VERSION = 1

    def __init__(self, hparams):
        self.model_name = hparams.model
        self.env_name = hparams.env
        self.reward_discount = hparams.reward_discount
        self.control_dim = copy.deepcopy(hparams.control_dim)
        self.obs_shape = copy.deepcopy(hparams.state_dim)
        self.model = None
        self.input_shape = None
        self.hparams = hparams
        self.training_mode = False
        self.device = torch.device("cpu")

    def act(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        raise NotImplementedError

    def act_init(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        raise NotImplementedError

    def is_ready_for_training(self, task_id=None, pbar=None):
        return self.model is not None

    def done_training(self, task_id=None, pbar=None):
        return False

    def init_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def reset_trained_steps(self):
        pass

    def get_trained_steps(self, task_id=None):
        return 0

    def learn(self, task_id: int, verbose=False):
        raise NotImplementedError

    def eval(self):
        pass

    def collect(self, x_t, u_t, reward, x_tt, task_id, done=False):
        pass

    def reset(self, obs):
        pass

    def to(self, device=torch.device("cpu")):
        for name, value in self.__dict__.items():
            if hasattr(value, "to") and (isinstance(value, torch.Tensor) or isinstance(value, nn.Module) or isinstance(value, Agent)):
                value.to(device)
        self.device = device

    def save(self, path):
        if isinstance(path, (str, os.PathLike)):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "__hypercez_checkpoint__": True,
            "version": self.CHECKPOINT_VERSION,
            "agent_class": self.__class__.__name__,
            "agent": self,
            "rng_state": self._capture_rng_state(),
        }
        self._atomic_torch_save(ckpt, path)

    @classmethod
    def load(cls, path, map_location='cpu', verbose=False):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(ckpt, dict) and ckpt.get("__hypercez_checkpoint__", False):
            out = ckpt["agent"]
            rng_state = ckpt.get("rng_state")
            if rng_state is not None:
                cls._restore_rng_state(rng_state)
            if verbose:
                print(f"Loaded new-format checkpoint for {ckpt.get('agent_class', type(out).__name__)}")
            return out

        # Legacy checkpoint fallback.
        if verbose:
            print([(i, v) for i, v in ckpt.items()])
        out = cls(ckpt["hparams"])
        for i, v in ckpt.items():
            out.__dict__[i] = v
        return out

    @staticmethod
    def _atomic_torch_save(obj, path):
        path = os.fspath(path)
        target_dir = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_ckpt_", suffix=".pth", dir=target_dir)
        os.close(fd)
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _capture_rng_state():
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": None,
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return rng_state

    @staticmethod
    def _restore_rng_state(rng_state):
        if not isinstance(rng_state, dict):
            return
        if "python" in rng_state and rng_state["python"] is not None:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state and rng_state["numpy"] is not None:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state and rng_state["torch"] is not None:
            torch_state = rng_state["torch"]
            if not isinstance(torch_state, torch.Tensor):
                torch_state = torch.as_tensor(torch_state, dtype=torch.uint8)
            torch_state = torch_state.detach().to(device="cpu", dtype=torch.uint8)
            torch.set_rng_state(torch_state)
        if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
            cuda_states = []
            for state in rng_state["torch_cuda"]:
                if not isinstance(state, torch.Tensor):
                    state = torch.as_tensor(state, dtype=torch.uint8)
                state = state.detach().to(device="cpu", dtype=torch.uint8)
                cuda_states.append(state)
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception as exc:
                print(f"Warning: could not restore CUDA RNG state ({exc}); continuing without it.")
