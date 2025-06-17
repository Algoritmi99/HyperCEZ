import numpy as np

from hypercez.agents.agent_base import Agent, ActType


class RandomAgent(Agent):
    def __init__(self, hparams):
        super(RandomAgent, self).__init__(hparams)

    def act(self, obs, task_id=None, act_type: ActType = ActType.INITIAL):
        return np.random.randn(self.control_dim, 1)
