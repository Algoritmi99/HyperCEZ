class Agent:
    def __init__(self, hparams):
        self.model_name = hparams.model
        self.env_name = hparams.env
        self.reward_discount = hparams.reward_discount
        self.control_dim = hparams.control_dim
        self.state_dim = hparams.state_dim
        self.model = None

    def act(self, state):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

