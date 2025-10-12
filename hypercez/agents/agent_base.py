import copy
from enum import IntEnum

import torch
import torch.nn as nn


class ActType(IntEnum):
    INITIAL = 0
    TRAIN = 1


class Agent:
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

    def learn(self, task_id: int):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

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
        ckpt = {copy.deepcopy(i): copy.deepcopy(v) for i, v in self.__dict__.items()}
        torch.save(ckpt, path)

    @classmethod
    def load(cls, path, map_location='cpu'):
        ckpt = torch.load(path, map_location=map_location)
        out = cls(ckpt["hparams"])
        for i, v in ckpt.items():
            out.__dict__[i] = v
        return out
