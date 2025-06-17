import copy
from enum import IntEnum


class ActType(IntEnum):
    INITIAL = 0
    RECURRENT = 1


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

    def act(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError
