import copy


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

    def act(self, state):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError
