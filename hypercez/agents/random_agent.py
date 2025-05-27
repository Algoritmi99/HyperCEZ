import numpy as np

from hypercez.agents.agent_base import Agent


class RandomAgent(Agent):
    def __init__(self, hparams):
        super(RandomAgent, self).__init__(hparams)

    def act(self, state, task_id=None):
        return np.random.randn(self.control_dim, 1)

